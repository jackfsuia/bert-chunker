# bert-chunker-3
[bert-chunker-3](https://huggingface.co/tim1900/bert-chunker-3) uses a totally different data synthetizing technique and training technique from [bert-chunker-2](https://huggingface.co/tim1900/bert-chunker-2) and [bert-chunker](https://huggingface.co/tim1900/bert-chunker), which **make it more reliable and stable**. Specifically, bert-chunker-3

1. **uses a LLM to label training data**. This is to alleviate the data distribution shift where the synthetized chunk data is totally different from real data distribution, and as observed in bert-chunker-2 and bert-chunker, this can cause unnatural chunking decision sometimes. 
2. **uses a slide window data generating and deterministic batch sampler for training**. This is to make sure the training aligns well with the inference.

### To do:
1. create more messy text (the ones without period or so) for the train dataset, so unstructured text will be better handled.

All the code will be released here...
