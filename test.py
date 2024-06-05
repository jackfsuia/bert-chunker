import safetensors
from transformers import AutoConfig,AutoTokenizer
from modeling_bertchunker import BertChunker

# load bert tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    padding_side="right",
    model_max_length=255,
    trust_remote_code=True,
)

# load MiniLM-L6-H384-uncased bert config
config = AutoConfig.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    trust_remote_code=True,
)

# initialize model
model = BertChunker(config)
device='cuda'
model.to(device)

# load parameters
state_dict = safetensors.torch.load_file("outputModels/checkpoint-3750/model.safetensors")
model.load_state_dict(state_dict)

# text to be chunked
text="In the heart of the bustling city, where towering skyscrapers touch the clouds and the symphony \
    of honking cars never ceases, Sarah, an aspiring novelist, found solace in the quiet corners of the ancient library. \
    Surrounded by shelves that whispered stories of centuries past, she crafted her own world with words, oblivious to the rush outside.\
    Dr. Alexander Thompson, aboard the spaceship 'Pandora's Venture', was en route to the newly discovered exoplanet Zephyr-7. \
    As the lead astrobiologist of the expedition, his mission was to uncover signs of microbial life within the planet's subterranean ice caves. \
    With each passing light year, the anticipation of unraveling secrets that could alter humanity's\
     understanding of life in the universe grew ever stronger."

# chunk the text. The lower threshold is, the more chunks will be generated. Can be negative or positive.
chunks=model.chunk_text(text, tokenizer, threshold=0)

# print chunks
for i, c in enumerate(chunks):
    print(f'-----chunk: {i}------------')
    print(c)

print('----->Here is the result of fast chunk method<------:')
# chunk the text faster with a fixed context window, batchsize is the number of windows run per batch.
chunks=model.chunk_text_fast(text, tokenizer, batchsize=20, threshold=0)

# print chunks
for i, c in enumerate(chunks):
    print(f'-----chunk: {i}------------')
    print(c)