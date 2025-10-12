# bert-chunker-3
[bert-chunker-3](https://huggingface.co/tim1900/bert-chunker-3) uses a totally different data synthetizing technique and training technique from [bert-chunker-2](https://huggingface.co/tim1900/bert-chunker-2) and [bert-chunker](https://huggingface.co/tim1900/bert-chunker), which **make it more reliable and stable**. Specifically, bert-chunker-3

1. **uses a LLM to label training data**. This is to alleviate the data distribution shift where the synthetized chunk data is totally different from real data distribution, and as observed in bert-chunker-2 and bert-chunker, this can cause unnatural chunking decision sometimes. 
2. **uses a slide window data generating and deterministic batch sampler for training**. This is to make sure the training aligns well with the inference.

# Usage
API_labeling_data_pipeline -> process_dataset -> train -> convert_to_hf_safetensors -> test

# to do
1. Create more noisy data by randomly removing "." of sentences.
2. Add a "max_token_of_chunks" button to it. (Done, at [here](https://huggingface.co/tim1900/bert-chunker-3#experemental))
3. more data
4. edgcut
5. evaluation frame work
6. fix bugs
   
