# BertChunker: Efficient and Trained Chunking for Unstructured Documents
[Model](https://huggingface.co/tim1900/BertChunker) | [Paper](https://github.com/jackfsuia/BertChunker/blob/main/main.pdf)

BertChunker is a text chunker based on BERT with a classifier head to predict the start token of chunks (for use in RAG, etc). It is finetuned based on [nreimers/MiniLM-L6-H384-uncased](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased), and the whole training lasted for 10 minutes on a Nvidia P40 GPU on a 50 MB synthetized dataset. This repo includes codes for model defining, generating dataset, training and testing.

## Generate dataset
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
  title={BertChunker: Efficient and Trained Chunking for Unstructured Documents}, 
  author={Yannan Luo},
  year={2024},
  url={https://github.com/jackfsuia/BertChunker}
}
```
