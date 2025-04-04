# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import time
import logging
import os
from typing import Dict, Optional, List,Union
import torch
from torch.utils.data import Dataset
import re
import torch
from torch import nn
from torch.utils.data import DataLoader
import transformers
from transformers import Trainer, GPTQConfig,AutoModel,AutoTokenizer,AutoConfig
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from datasets import load_dataset
import safetensors.torch
from modeling_bertchunker import BertChunker
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler

@dataclass
class ModelArguments:
    base_model_name_or_path: Optional[str] = field(default="huggingface.co/nreimers/MiniLM-L6-H384-uncased")
    model_checkpoint_bin: Optional[str] = None

@dataclass
class DataArguments:
    data_path: str = field(
        default="/hy-tmp/train_data.jsonl", metadata={"help": "Path to the training data."}
    )

    eval_data_path: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj","up_proj","down_proj"]
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    num_proc: Optional[int] = None
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=500,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    output_dir: str ="/hy-tmp/"
    slide_window: bool = False
    API_syn_dataset:bool= False
    resume_from_checkpoint: Optional[str] = None
    evaluate_before_train:bool = False
    project_name:str = "bert-chunker"
@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj","up_proj","down_proj"]
    )
    modules_to_save:List[str] = field(
        default_factory=lambda:["embed_tokens", "lm_head"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
@dataclass
class DebugArguments:
    debugpy: bool = False

parser = transformers.HfArgumentParser(
    (ModelArguments, DataArguments, TrainingArguments, LoraArguments, DebugArguments)
)
(
    model_args,
    data_args,
    training_args,
    lora_args,
    debugpy_args,
) = parser.parse_args_into_dataclasses()
if debugpy_args.debugpy:
    import debugpy
    try:
        # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
        debugpy.listen(("localhost", 9501))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    except Exception as e:
            pass
os.environ["WANDB_PROJECT"] = training_args.project_name
IGNORE_TOKEN_ID =-100 # LabelSmoother.ignore_index

START_SYM="<--start-->"


def tokenize_function(item, tokenizer, max_len):

    chunks=re.split(START_SYM,item['input'])
    input_ids=[]
    labels=[]
    for i,c in enumerate(chunks):
        c=tokenizer(c,truncation=False)['input_ids']
        #去掉头尾，最后两端再加上special tokens在头尾
        c=c[1:-1]
        if c:
            input_ids+=c

            cl=len(c)*[0]
            
            cl[0]=1
            labels+=cl
    if not labels:
        print('--------')
        print(item)
    labels[0]=0

    #截断

    input_ids=input_ids[:max_len-2]
    labels=labels[:max_len-2]

    #加上bert头尾

    input_ids = [tokenizer.cls_token_id]+input_ids+[tokenizer.sep_token_id]

    labels=[-100]+labels+[-100]

    #填充到max_len
    lennow=len(input_ids)
    input_ids=input_ids+[0]*(max_len-lennow)
    labels=labels+[-100]*(max_len-lennow)
    attention_mask=[1]*lennow+[0]*(max_len-lennow)

    
    full_text={}
    full_text["input_ids"]=input_ids
    full_text["labels"]=labels
    full_text["attention_mask"]=attention_mask

    return full_text


class deterSampleTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return SequentialSampler(self.train_dataset)
    
    
def train():

    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED


    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are incompatible with QLoRA."
            )



    tokenizer = AutoTokenizer.from_pretrained(
        model_args.base_model_name_or_path,
        padding_side="right",
        model_max_length=training_args.model_max_length,
        trust_remote_code=True,
    )

    # load MiniLM-L6-H384-uncased bert config
    config = AutoConfig.from_pretrained(
        model_args.base_model_name_or_path,
        trust_remote_code=True,
    )

    model = BertChunker(config)


    model.model=transformers.AutoModel.from_pretrained(
        model_args.base_model_name_or_path,
        config=config,
       
        torch_dtype="auto",
        trust_remote_code=True,
        quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=True,
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
    )

    if model_args.model_checkpoint_bin:
        state_dict = safetensors.torch.load_file(model_args.model_checkpoint_bin)
        model.load_state_dict(state_dict)



    # dataset = load_dataset("json", data_files=data_args.data_path, num_proc=training_args.num_proc,split="train",cache_dir=training_args.cache_dir)# train[:15%]
    train_dataset = load_dataset(os.path.splitext(os.path.basename(data_args.data_path))[1][1:], data_files=data_args.data_path,num_proc=training_args.num_proc, split="train",cache_dir=training_args.cache_dir)# train[:15%]
    if training_args.slide_window or training_args.API_syn_dataset:
        train_dataset = train_dataset
        eva_data ={}
        for evd in data_args.eval_data_path:
            eval_dataset=load_dataset(os.path.splitext(os.path.basename(evd))[1][1:], data_files=evd,num_proc=training_args.num_proc, split='train',cache_dir=training_args.cache_dir)
            eva_data[os.path.splitext(os.path.basename(evd))[0]] = eval_dataset
    else:
        train_dataset = train_dataset.map(tokenize_function, fn_kwargs={"tokenizer": tokenizer, "max_len":training_args.model_max_length},num_proc=training_args.num_proc,cache_dir=training_args.cache_dir)
        eva_data ={}
        for evd in data_args.eval_data_path:
            eval_dataset=load_dataset(os.path.splitext(os.path.basename(evd))[1][1:], data_files=evd,num_proc=training_args.num_proc, split='train')
            eval_dataset=eval_dataset.map(tokenize_function, fn_kwargs={"tokenizer": tokenizer, "max_len":training_args.model_max_length},num_proc=training_args.num_proc,cache_dir=training_args.cache_dir)
            eva_data[os.path.splitext(os.path.basename(evd))[0]] = eval_dataset

    
    if training_args.slide_window:
        trainer = deterSampleTrainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=train_dataset,eval_dataset=eva_data)
    else:  
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=train_dataset,eval_dataset=eva_data)
    if training_args.evaluate_before_train:
        trainer.evaluate()
    trainer.train(resume_from_checkpoint = training_args.resume_from_checkpoint)
    

if __name__ == "__main__":
    train()
