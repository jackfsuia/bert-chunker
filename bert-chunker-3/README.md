# bert-chunker-3
bert-chunker-3 uses a totally different data synthetizing technique and training technique from bert-chunker-2 and bert-chunker, which make it more reliable and stable. Specifically, bert-chunker-3

1. **uses a LLM to label training data**. This is to alleviate the data distribution shift where the synthetized chunk data is totally different from real data distribution,and as observed in bert-chunker-2 and bert-chunker, this can cause unnatural chunking decision sometimes. 
2. **uses a deterministic batch sampler when training**. This is to keep the chunks of one article in one batch as much as possible so it aligns well with the inference. Because during inference, chunker is often applied to one continuous article in a slide window manner.

All the code will be released...
