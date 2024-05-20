# BertChunker: Efficient and Trained Chunking for Retrieval Augmented Generation
[Model](https://huggingface.co/tim1900/BertChunker) | [Paper](https://github.com/jackfsuia/BertChunker/blob/main/main.pdf)

Code for generating dataset and training of BertChunker, a semantic chunker for RAG. 

## Generating dataset
See [generate_dataset.ipynb](generate_dataset.ipynb)
## Train from the base model all-MiniLM-L6-v2
Run
 ``` shell
 bash train.sh
 ```
## Inference
See [test.py](test.py)
## Citation

If this work is helpful, please kindly cite as:

```bibtex
@article{BertChunker,
  title={BertChunker: Efficient and Trained Chunking for Retrieval Augmented Generation}, 
  author={Yannan Luo},
  year={2024},
  url={https://github.com/jackfsuia/BertChunker}
}
```