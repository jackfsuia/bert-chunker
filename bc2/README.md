# bert-chunker-2
[bert-chunker-2](https://huggingface.co/tim1900/bert-chunker-2) is a 0.1:0.9 linear weight merging of our [bert-chunker-1](https://huggingface.co/tim1900/bert-chunker) (a semantic chunker) and a trained structure chunker (trained by simply using line breaks as the start tokens of chunks). It is designed so that it reach a balance between structure chunking and semantic chunking.
# Usage
generate_dataset -> train -> 模型融合 -> convert_to_hf_safetensors -> test