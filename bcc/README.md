---
language:
- en
- zh
pipeline_tag: token-classification
---
# bert-chunker-chinese

## Introduction

bert-chunker-chinese is a chinese text chunker based on BERT with a classifier head to predict the start token of chunks (for use in RAG, etc), and using a sliding window it cuts documents of any size into chunks. It was finetuned on top of [bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5).

This repo includes model checkpoint, BertChunker class definition file and all the other files needed.

## Quickstart
Download this repository. Then enter it. Run the following:

```python
# -*- coding: utf-8 -*-
import safetensors
from transformers import AutoConfig,AutoTokenizer
from modeling_bertchunke_zh import BertChunker

# load config and tokenizer
config = AutoConfig.from_pretrained(
    "tim1900/bert-chunker-chinese",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "tim1900/bert-chunker-chinese",
    padding_side="right",
    model_max_length=config.max_position_embeddings,
    trust_remote_code=True,
)

# initialize model
model = BertChunker(config)
device='cpu' # or 'cuda'
model.to(device)

# load tim1900/bert-chunker-chinese/model.safetensors
state_dict = safetensors.torch.load_file(f"./model.safetensors")
model.load_state_dict(state_dict)

# text to be chunked
text='''起点中文网(www.qidian.com)创立于2002年5月，是国内知名的原创文学网站，隶属于阅文集团旗下。起点中文网以推动中国原创文学事业为宗旨，长期致力于原创文学作者的挖掘与培养，并取得了巨大成果：2003年10月，起点中文网开启“在线收费阅读”服务，成为真正意义上的网络文学赢利模式的先锋之一，就此奠定了原创文学的行业基础。此后，起点又推出了作家福利、文学交互、内容发掘推广、版权管理等机制和体系，为原创文学的发展注入了巨大活力，有力推动了中国文学原创事业的发展。在清晨的微光中，一只孤独的猫头鹰在古老的橡树上低声吟唱，它的歌声如同夜色的回声，穿越了时间的迷雾。树叶在微风中轻轻摇曳，仿佛在诉说着古老的故事，每一个音符都带着森林的秘密。一位年轻的程序员正专注地敲打着键盘，代码的海洋在他眼前展开。他的手指在键盘上飞舞，如同钢琴家在演奏一曲复杂的交响乐。屏幕上的光标闪烁，仿佛在等待着下一个指令，引领他进入未知的数字世界。'''

# chunk the text. The lower threshold is, the more chunks will be generated. Can be negative or positive.
chunks=model.chunk_text(text, tokenizer, threshold=0.5)

# print chunks
for i, c in enumerate(chunks):
    print(f'-----chunk: {i}------------')
    print(c)
```