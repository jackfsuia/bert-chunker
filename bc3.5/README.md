# bert-chunker-3.5
[bert-chunker-3.5](https://huggingface.co/tim1900/bert-chunker-3.5) differs from [bc3](https://huggingface.co/tim1900/bert-chunker-3) it:

1. **was trained on a larger model length (255->512).**
2. **left 50 tokens on both edges of the context window.**
3. **scans the whole doc first, then greedy search split points according to the given max chunk size.**

# Usage
process_dataset.ipynb -> train -> convert_to_hf_safetensors -> test

# to do
1. more data
