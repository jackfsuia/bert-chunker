import torch
from transformers import AutoConfig,AutoTokenizer,BertForTokenClassification
import math

model_path=r"D:\github-my-project\bert-chunker-2"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    padding_side="right",
    model_max_length=255,
    trust_remote_code=True,
)

config = AutoConfig.from_pretrained(
    model_path,
    trust_remote_code=True,
)

device = 'cpu'

model = BertForTokenClassification.from_pretrained(model_path, ).to(device)


def chunk_text(model,text:str, tokenizer, prob_threshold=0.5)->list[str]:
# slide context window
    MAX_TOKENS=255
    tokens=tokenizer(text, return_tensors="pt",truncation=False)
    input_ids=tokens['input_ids']
    attention_mask=tokens['attention_mask'][:,0:MAX_TOKENS]
    attention_mask=attention_mask.to(model.device)
    CLS=input_ids[:,0].unsqueeze(0)
    SEP=input_ids[:,-1].unsqueeze(0)
    input_ids=input_ids[:,1:-1]
    model.eval()
    split_str_poses=[]
    
    token_pos = []

    windows_start =0
    windows_end= 0
    logits_threshold = math.log(1/prob_threshold-1)
    
    print(f'Processing {input_ids.shape[1]} tokens...')
    while windows_end <= input_ids.shape[1]:
        windows_end= windows_start + MAX_TOKENS-2

        ids=torch.cat((CLS, input_ids[:,windows_start:windows_end],SEP),1)

        ids=ids.to(model.device)
        
        output=model(input_ids=ids,attention_mask=torch.ones(1, ids.shape[1],device=model.device))
        logits = output['logits'][:, 1:-1,:]
        chunk_decision = (logits[:,:,1]>(logits[:,:,0]-logits_threshold))
        greater_rows_indices = torch.where(chunk_decision)[1].tolist()

        # null or not
        if len(greater_rows_indices)>0 and (not (greater_rows_indices[0] == 0 and len(greater_rows_indices)==1)):

            split_str_pos=[tokens.token_to_chars(sp + windows_start + 1).start for sp in greater_rows_indices]
            token_pos +=[sp + windows_start + 1 for sp in greater_rows_indices]
            split_str_poses += split_str_pos

            windows_start = greater_rows_indices[-1] + windows_start

        else:

            windows_start = windows_end

    substrings = [text[i:j] for i, j in zip([0] + split_str_poses, split_str_poses+[len(text)])]
    token_pos = [0] + token_pos
    return substrings,token_pos



text='''In the heart of the bustling city, where towering skyscrapers touch the clouds and the symphony 
    of honking cars never ceases, Sarah, an aspiring novelist, found solace in the quiet corners of the ancient library 
    Surrounded by shelves that whispered stories of centuries past, she crafted her own world with words, oblivious to the rush outside Dr.Alexander Thompson, aboard the spaceship 'Pandora's Venture', was en route to the newly discovered exoplanet Zephyr-7. 
    As the lead astrobiologist of the expedition, his mission was to uncover signs of microbial life within the planet's subterranean ice caves. 
    With each passing light year, the anticipation of unraveling secrets that could alter humanity's
     understanding of life in the universe grew ever stronger.'''

chunks, token_pos=chunk_text(model,text, tokenizer, prob_threshold=0.5)

# print chunks
for i, (c,t) in enumerate(zip(chunks,token_pos)):
    print(f'-----chunk: {i}----token_idx: {t}--------')
    print(c)






