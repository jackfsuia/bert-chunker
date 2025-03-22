# bert-chunker-Chinese-2
[bert-chunker-Chinese-2](https://huggingface.co/tim1900/bert-chunker-Chinese-2) uses a totally different data synthetizing technique and training technique from [bert-chunker-Chinese](https://huggingface.co/tim1900/bert-chunker-Chinese), which **make it more reliable and stable**. Specifically, bert-chunker-Chinese-2

1. **synthesize training data by filtering out the short articles, and use line breaks as the start tokens of chunks**. This is to alleviate the data distribution shift where the synthetized chunk data is totally different from real data distribution, and as observed in previous versions of bert-chunker, this can cause unnatural chunking decision sometimes.
2. **uses a slide window data generating and deterministic batch sampler for training**. This is to make sure the training aligns well with the inference.

# Usage
generate_dataset -> train -> convert_to_hf_safetensors -> test