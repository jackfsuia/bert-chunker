from dataclasses import dataclass, field
import json
import time
import logging
import os
from typing import Dict, Optional, List, Union
import torch
from torch.utils.data import Dataset
import re
import torch
from torch import nn
from torch.utils.data import DataLoader
import transformers
from transformers import Trainer, GPTQConfig, AutoModel, AutoTokenizer, AutoConfig
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from datasets import load_dataset
import safetensors.torch
import random

from datasets import load_dataset,Dataset
IGNORE_TOKEN_ID = -100 
START_SYM = "<--start-->"
tokenizer_path = "/data/all-MiniLM-L6-v2"
max_len = 255
max_use_len = max_len-2

def tokenize_function(item, tokenizer, max_len):

    chunks=re.split(START_SYM,item['chunk'])

    #加上，随机5%去掉句号
    # precess_chunks=[]
    # for c in chunks:
    #     text = c.rstrip()        
    #     chance = random.random()
    #     if chance < 0.05 and text.endswith('.'):
    #         text = text[:-1]+'\n'
    #         precess_chunks.append(text)
    #     else:
    #         precess_chunks.append(c)
    # chunks = precess_chunks

    input_ids=[]
    labels=[]
    for _,c in enumerate(chunks):
        c=tokenizer(c,truncation=False)['input_ids']
        #去掉头尾，最后两端再加上special tokens在头尾
        c=c[1:-1]
        if c:
            input_ids+=c

            cl=len(c)*[0]

            cl[0]=1
            labels+=cl
    #判断数据是否是空，抛出异常
    if not labels:
        print('--------')
        print(item)
    labels[0]=0

    full_texts = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
        }
        
    start_idx = 0

    while(1):

        if start_idx>len(input_ids)-1:
            break

        end_idx = start_idx + max_use_len



        window_input_ids = input_ids[start_idx:end_idx]
        window_labels = labels[start_idx:end_idx]

        # 加入bert 头尾
        window_input_ids = [tokenizer.cls_token_id] + window_input_ids + [tokenizer.sep_token_id]

        window_labels=[IGNORE_TOKEN_ID] + window_labels + [IGNORE_TOKEN_ID]

        # 填充到max_len
        len_now = len(window_input_ids)
        window_input_ids = window_input_ids + [0] * (max_len - len_now)
        window_labels = window_labels + [IGNORE_TOKEN_ID] * (max_len - len_now)
        attention_mask = [1] * len_now + [0] * (max_len - len_now)

        full_texts["input_ids"].append(window_input_ids)
        full_texts["labels"].append(window_labels)
        full_texts["attention_mask"].append(attention_mask)


        label_chunk = labels[start_idx:end_idx]
        last_cut_id = end_idx
        for i in range(len(label_chunk)):
            j= len(label_chunk) - i - 1
            if label_chunk[j]  == 1 and j !=0:
                last_cut_id = start_idx + j
                break

        start_idx = last_cut_id

    return full_texts


def chunkenize_dataset(dataset, sava_name):

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        padding_side="right",
        model_max_length=max_len,
        trust_remote_code=True,
    )


    train_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer, "max_len": max_len},
        num_proc=6,
    )

    train_dataset = train_dataset.to_pandas()
    train_dataset = train_dataset.explode(['input_ids',"labels","attention_mask"], ignore_index=True)
    train_dataset.to_parquet(sava_name)

    return train_dataset


def merge_batch(batch, group_size=3):
    START_SYM="<--start-->"
    new_data = {"input": [],}
    for i in range(0, len(batch['input']), group_size):
        # 获取当前组的 4 条数据
        texts = batch['input'][i:i+group_size]
        
        # 合并数据
        merged_text = ' '.join(texts)
        merged_text = merged_text[:-len(START_SYM)]
        
        # 添加到新数据中
        new_data["input"].append(merged_text)
    return new_data


def batch_process(dataset):

    new_dataset = dataset.map(
        merge_batch,
        batched=True,  # 启用批处理
        batch_size=1000,  # 每批次处理 1000 条原始数据
        remove_columns=dataset.column_names,  # 移除原始列
        num_proc=6,  # 使用 6 个进程并行处理
    )

    return new_dataset

def merge_model(safetensors1,w1,safetensors2,w2,save_path):
    from safetensors.torch import load_file, save_file

    # 加载两个模型的权重
    weights1 = load_file(safetensors1)
    weights2 = load_file(safetensors2)

    new_weights = {}

    # 遍历所有参数并计算加权平均
    for key in weights1.keys():
        if key in weights2:  # 确认两个模型都有这个参数
            new_weights[key] = w1 * weights1[key] + w2 * weights2[key]

    # 保存新的权重到一个新的 .safetensors 文件
    save_file(new_weights, save_path)