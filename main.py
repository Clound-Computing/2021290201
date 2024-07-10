import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    BertModel,
    BertPreTrainedModel,
)

# from utils_checkthat import *
import json
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import random


# 配置参数

# 配置信息
model_name_or_path = "models/bert-base-uncased"  # 模型名称或路径
datasets_path = (
    "datasets/gossipcop_v3-3_integration_based_fake_tn200.json"  # 数据集路径
)
# datasets_path = "datasets/mini.json" # 数据集路径
max_steps = 8000  # 训练的最大步数
per_device_train_batch_size = 2  # 每个设备的训练批次bacth大小
logging_steps = 200  # 日志步数

# 随机种子
RANDOM_STATE = 49

# 0 = 不使用摘要，1 = 使用抽象摘要，2 = 使用抽取摘要
IS_USE_SUMMARY_TRAIN = 0
IS_USE_SUMMARY_TEST = 0


# 设置随机数种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_STATE)


# 数据加载


# 读取JSON文件
# with open('datasets/mini.json', 'r') as file:
with open(datasets_path, "r") as file:
    data = json.load(file)

# 将JSON数据转换为DataFrame
orgin_df = pd.DataFrame.from_dict(data, orient="index")


orgin_df.head()


# 数据预处理


# 提取 doc_1 相关列，并重命名
df_doc1 = orgin_df[["doc_1_id", "doc_1_text", "doc_1_label"]].rename(
    columns={"doc_1_id": "doc_id", "doc_1_text": "doc_txt", "doc_1_label": "our rating"}
)

# 提取 doc_2 相关列，并重命名
df_doc2 = orgin_df[["doc_2_id", "doc_2_text", "doc_2_label"]].rename(
    columns={"doc_2_id": "doc_id", "doc_2_text": "doc_txt", "doc_2_label": "our rating"}
)

# 合并两个DataFrame
df = pd.concat([df_doc1, df_doc2])

# 重置索引
df = df.reset_index(drop=True)


df.head()


# 统计数据集中的标签分布
df["our rating"].value_counts()


def convert_to_int(rating):
    if rating == "TRUE" or rating == "true" or rating == True or rating == "legitimate":
        return 0
    if rating == "FALSE" or rating == "false" or rating == False or rating == "fake":
        return 1
    if rating == "partially false":
        return 2
    else:
        return 3


df["label"] = df["our rating"].apply(convert_to_int)


# 首先将数据划分为训练集和临时集（验证集+测试集）
df_train, df_temp = train_test_split(
    df, test_size=0.05, random_state=RANDOM_STATE, stratify=df["label"]
)

# 一步划分为验证集和测试集
df_val, df_test = train_test_split(
    df_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=df_temp["label"]
)

# 检查划分的比例
print(f"Training set size: {len(df_train)}")
print(f"Validation set size: {len(df_val)}")
print(f"Test set size: {len(df_test)}")


df_train.head()


# 分词、实例化数据集和训练参数


tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)


def get_encodings(dataframe, tokenizer, summary=0):
    encodings = []
    labels = []
    for idx in range(len(dataframe)):
        if summary == 2:
            sum_text = (
                str(dataframe.iloc[idx]["title"])
                + ". "
                + dataframe.iloc[idx]["text_extractive"]
            )
        if summary == 1:
            sum_text = (
                str(dataframe.iloc[idx]["title"])
                + ". "
                + dataframe.iloc[idx]["text_abstractive"]
            )
        elif summary == 0:
            sum_text = str(dataframe.iloc[idx]["doc_txt"])

        # 对长文本进行拆分，以便于模型处理
        if len(sum_text.split()) > 50:
            text_parts = get_split(sum_text, 500, 50)
            tensors = tokenizer(
                text_parts, padding="max_length", truncation="only_first"
            )
            encodings.append(tensors)
            labels.append(dataframe.iloc[idx]["label"])
        else:
            encodings.append(
                tokenizer(sum_text, padding="max_length", truncation="only_first")
            )
            labels.append(dataframe.iloc[idx]["label"])

    return encodings, labels


def get_split(text, split_length, stride_length=50):
    l_total = []
    l_partial = []
    text_length = len(text.split())
    partial_length = split_length - stride_length
    if text_length // partial_length > 0:
        n = text_length // partial_length
    else:
        n = 1
    for w in range(n):
        if w == 0:
            l_partial = text.split()[:split_length]
            l_total.append(" ".join(l_partial))
        else:
            l_partial = text.split()[
                w * partial_length : w * partial_length + split_length
            ]
            l_total.append(" ".join(l_partial))
    return l_total


train_encodings, train_labels = get_encodings(df_train, tokenizer, IS_USE_SUMMARY_TRAIN)


val_encodings, val_labels = get_encodings(df_val, tokenizer, IS_USE_SUMMARY_TEST)
test_encodings, test_labels = get_encodings(df_val, tokenizer, IS_USE_SUMMARY_TEST)


class CheckThatLabDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {}
        internal_counter = 0
        if type(self.encodings[idx]["input_ids"][0]) == list:
            for encoding in self.encodings[idx]["input_ids"]:
                if internal_counter < 4:
                    if internal_counter != 0:
                        item["input_ids_" + str(internal_counter)] = encoding
                    else:
                        item["input_ids"] = encoding
                internal_counter += 1
        else:
            item["input_ids"] = self.encodings[idx]["input_ids"]

        internal_counter = 0
        if type(self.encodings[idx]["attention_mask"][0]) == list:
            for encoding in self.encodings[idx]["attention_mask"]:
                if internal_counter < 4:
                    if internal_counter != 0:
                        item["attention_mask_" + str(internal_counter)] = encoding
                    else:
                        item["attention_mask"] = encoding
                internal_counter += 1
        else:
            item["attention_mask"] = self.encodings[idx]["attention_mask"]

        for i in range(1, 4):
            if not "input_ids_" + str(i) in item:
                item["input_ids_" + str(i)] = np.zeros(512, dtype=int)
                item["attention_mask_" + str(i)] = np.zeros(512, dtype=int)

        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = CheckThatLabDataset(train_encodings, train_labels)
val_dataset = CheckThatLabDataset(val_encodings, val_labels)
test_dataset = CheckThatLabDataset(test_encodings, test_labels)


# 构建模型


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# logging_steps = 200

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",  # 模型结构的输出目录
    max_steps=max_steps,  # 最大训练步数
    evaluation_strategy="steps",
    save_strategy="steps",
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=1,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=logging_steps,
    load_best_model_at_end=True,
    seed=RANDOM_STATE,
    save_steps=logging_steps * 2,
)


from transformers import BertModel, BertPreTrainedModel
import torch.nn.functional as F


class BertClassifier(BertPreTrainedModel):
    def __init__(
        self,
        config,
        labels_count=4,
        hidden_dim=4 * 768,
        dropout=0.1,
        freeze_emb=False,
        freeze_all=False,
    ):
        super().__init__(config)

        self.num_labels = labels_count
        self.bert = BertModel(config)
        if freeze_all:
            for param in self.bert.parameters():
                param.requires_grad = False

        if freeze_emb:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        self.pre_classifier = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, labels_count)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        input_ids_1=None,
        attention_mask_1=None,
        input_ids_2=None,
        attention_mask_2=None,
        input_ids_3=None,
        attention_mask_3=None,
    ):
        tensors = []

        hidden_state = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )[
            0
        ]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        tensors.append(pooled_output)

        hidden_state = self.bert(
            input_ids_1, attention_mask=attention_mask_1, token_type_ids=token_type_ids
        )[
            0
        ]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        tensors.append(pooled_output)

        hidden_state = self.bert(
            input_ids_2, attention_mask=attention_mask_2, token_type_ids=token_type_ids
        )[
            0
        ]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        tensors.append(pooled_output)

        hidden_state = self.bert(
            input_ids_3, attention_mask=attention_mask_3, token_type_ids=token_type_ids
        )[
            0
        ]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        tensors.append(pooled_output)

        pooled_output = torch.cat(tensors, dim=1)

        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = F.relu(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,)

        if loss is not None:
            output = (loss,) + output

        return output


def model_init():
    return BertClassifier.from_pretrained(model_name_or_path, labels_count=2)


# 初始化Trainer
trainer = Trainer(
    model_init=model_init,
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset,  # evaluation dataset
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# 训练模型
trainer.train()


# 评估模型


evaluation_results = trainer.evaluate()
for key, value in evaluation_results.items():
    print(f"{key}: {value}")


# 生成预测结果
pred = trainer.predict(val_dataset)
preds = pred.predictions.argmax(-1)


def convert_to_rating(int):
    if int == 0:
        return "true"
    if int == 1:
        return "false"


df_val["preds"] = preds
df_val["preds"] = df_val["preds"].apply(convert_to_rating)
df_val["ture"] = df_val["label"].apply(convert_to_rating)
columns = ["doc_id", "ture", "preds"]

# 保存预测结果
if IS_USE_SUMMARY_TRAIN == 0:
    df_val.to_csv("predictions.csv", columns=columns, index=False)
elif IS_USE_SUMMARY_TRAIN == 1:
    df_val.to_csv("predictions_abstractive.csv", columns=columns, index=False)
else:
    df_val.to_csv("predictions_extractive.csv", columns=columns, index=False)