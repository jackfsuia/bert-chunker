from chunking_evaluation import BaseChunker, GeneralEvaluation
from chunking_evaluation.chunking import ClusterSemanticChunker,KamradtModifiedChunker
from chromadb.utils import embedding_functions
from chromadb import Embeddings,EmbeddingFunction,Documents
import semchunk
import time
import torch
from transformers import AutoConfig, AutoTokenizer, BertForTokenClassification,AutoModel
import math
import os
import torch.nn.functional as F
from collections import deque
model_path = r"model_checkpoint\checkpoint-1200"
# model_path = r'D:\github-my-project\bert-chunker-3'
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

device = "cpu"  # or 'cuda'

model = BertForTokenClassification.from_pretrained(
    model_path,
).to(device)


api=""
os.environ['OPENAI_API_KEY'] = api
# Instantiate evaluation
evaluation = GeneralEvaluation()

# Choose embedding function
default_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api,
    model_name="text-embedding-3-large"
)



class MyEmbeddingFunction(EmbeddingFunction):
    
    
    def __init__(self):
        super().__init__()
        self.model=AutoModel.from_pretrained(r'D:\github_others\sentence-transformers\all-MiniLM-L6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained(r'D:\github_others\sentence-transformers\all-MiniLM-L6-v2')

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def __call__(self, input: Documents) -> Embeddings:
        # Tokenize sentences
        encoded_input = self.tokenizer(input, padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings.tolist()

# Instantiate instance of ef
all_MiniLM_L6_v2 = MyEmbeddingFunction()


class bertChunker3(BaseChunker):
    def __init__(self,prob_threshold=0.50543):
        super().__init__()
        self.prob_threshold = prob_threshold
    def chunk_text_with_max_chunk_size(self,model, text, tokenizer, prob_threshold=0.5,max_tokens_per_chunk = 400):
        with torch.no_grad():
            
            # slide context window chunking
            MAX_TOKENS = 512
            tokens = tokenizer(text, return_tensors="pt", truncation=False)
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"][:, 0:MAX_TOKENS]
            attention_mask = attention_mask.to(model.device)
            CLS = input_ids[:, 0].unsqueeze(0)
            SEP = input_ids[:, -1].unsqueeze(0)
            input_ids = input_ids[:, 1:-1]
            model.eval()
            split_str_poses = []
            token_pos = []
            windows_start = 0
            windows_end = 0
            logits_threshold = math.log(1 / prob_threshold - 1)
            
            unchunk_tokens = 0
            backup_pos = None
            best_logits = torch.finfo(torch.float32).min         
            STEP = round(((MAX_TOKENS - 2)//2)*1.75 )
            print(f"Processing {input_ids.shape[1]} tokens...")
            # while windows_end <= input_ids.shape[1]:
            while windows_start < input_ids.shape[1]:    
                windows_end = windows_start + MAX_TOKENS - 2 
                ids = torch.cat((CLS, input_ids[:, windows_start:windows_end], SEP), 1)
                ids = ids.to(model.device)
                output = model(
                    input_ids=ids,
                    attention_mask=torch.ones(1, ids.shape[1], device=model.device),
                )
                logits = output["logits"][:, 1:-1, :]
                
                
                logit_diff = logits[:, :, 1] - logits[:, :, 0]
                        
                        
                chunk_decision = logit_diff > - logits_threshold
                greater_rows_indices = torch.where(chunk_decision)[1].tolist()

                # null or not
                if len(greater_rows_indices) > 0 and (
                    not (greater_rows_indices[0] == 0 and len(greater_rows_indices) == 1)
                ):

                    
                    unchunk_tokens_this_window = greater_rows_indices[0] if greater_rows_indices[0]!=0 else greater_rows_indices[1]#exclude the fist index

                    # manually chunk
                    if unchunk_tokens + unchunk_tokens_this_window > max_tokens_per_chunk:   #change ">" to ">=" if buggy for the moment
                        big_windows_end = max_tokens_per_chunk - unchunk_tokens
                        max_value, max_index= logit_diff[:,1:big_windows_end].max(),  logit_diff[:,1:big_windows_end].argmax() + 1
                        if best_logits < max_value:
                            backup_pos = windows_start + max_index
                        
                        windows_start = backup_pos
                        
                        
                        split_str_pos = [tokens.token_to_chars(backup_pos + 1).start]
                        split_str_poses = split_str_poses + split_str_pos
                        token_pos = token_pos + [backup_pos]
                        best_logits = torch.finfo(torch.float32).min
                        backup_pos = -1
                        unchunk_tokens = 0
                        
                    # auto chunk    
                    else:
                        
                        if len(greater_rows_indices) >= 2:
                            for gi, (gri0,gri1) in enumerate(zip(greater_rows_indices[:-1],greater_rows_indices[1:])):
                                
                                if gri1 - gri0 > max_tokens_per_chunk:
                                    greater_rows_indices=greater_rows_indices[:gi+1]
                                    break
                                    
                        split_str_pos = [tokens.token_to_chars(sp + windows_start + 1).start for sp in greater_rows_indices if sp > 0]
                        split_str_poses = split_str_poses + split_str_pos
                        token_pos = token_pos+ [sp + windows_start for sp in greater_rows_indices if sp > 0]
                        
                        windows_start = greater_rows_indices[-1] + windows_start
                        best_logits = torch.finfo(torch.float32).min
                        backup_pos = -1
                        unchunk_tokens = 0

                else:

                    # unchunk_tokens_this_window = min(windows_end - windows_start,STEP)
                    unchunk_tokens_this_window = min(windows_start+STEP,input_ids.shape[1]) - windows_start

                    # manually chunk
                    if unchunk_tokens + unchunk_tokens_this_window > max_tokens_per_chunk:  #change ">" to ">=" if buggy for the moment
                        big_windows_end =  max_tokens_per_chunk - unchunk_tokens
                        if logit_diff.shape[1] > 1:
                            
                            max_value, max_index= logit_diff[:,1:big_windows_end].max(),  logit_diff[:,1:big_windows_end].argmax() + 1
                            if best_logits < max_value:
                                backup_pos = windows_start + max_index
                            
                            
                        windows_start = backup_pos
                        split_str_pos = [tokens.token_to_chars(backup_pos + 1).start]
                        split_str_poses = split_str_poses + split_str_pos
                        token_pos = token_pos + [backup_pos]
                        best_logits = torch.finfo(torch.float32).min
                        backup_pos = -1
                        unchunk_tokens = 0
                    else:
                    # auto leave
                        if logit_diff.shape[1] > 1:
                            max_value, max_index= logit_diff[:,1:].max(),  logit_diff[:,1:].argmax() + 1
                            if best_logits < max_value:
                                best_logits = max_value
                                backup_pos = windows_start + max_index
                        
                        unchunk_tokens = unchunk_tokens + STEP
                        windows_start = windows_start + STEP

            substrings = [
                text[i:j] for i, j in zip([0] + split_str_poses, split_str_poses + [len(text)])
            ]
            token_pos = [0] + token_pos
        return substrings, token_pos

    def split_text(self, text):

        chunks, token_pos = self.chunk_text_with_max_chunk_size(model, text, tokenizer)

        return chunks    
class bertChunker4(BaseChunker):
    def __init__(self,max_tokens_per_chunk=200,prob_threshold=0.52):
        super().__init__()
        self.prob_threshold = prob_threshold
        self.max_tokens_per_chunk = max_tokens_per_chunk

    def split_text(self, text):

        max_len = 512
        max_use_len = max_len-2

        mask_edge =50

        effective_window = max_use_len - mask_edge*2
        max_tokens_per_chunk = self.max_tokens_per_chunk
        with torch.no_grad():

            unk_id = tokenizer.unk_token_id
            
            tokens = tokenizer(text, return_tensors="pt", truncation=False)
            
            
            input_ids = tokens['input_ids'].squeeze()[1:-1]
            
            len_of_input_ids = len(input_ids)
            
            
            left =  effective_window - len_of_input_ids%effective_window 
            
            
            full_input_ids = torch.cat([ torch.tensor([unk_id]*mask_edge), input_ids, torch.tensor([unk_id]*(mask_edge + left))])
                    
            
            
            # assert len(input_ids)%effective_window == 0
            #判断数据是否是空，抛出异常
            prob_list = []

            start_idx = 0

            while(1):
                
                end_idx = start_idx + max_use_len
                window_input_ids =torch.cat([ torch.tensor([tokenizer.cls_token_id]),  full_input_ids[start_idx:end_idx], torch.tensor([tokenizer.sep_token_id])])
                attention_mask = [1] * len(window_input_ids)
                window_input_ids =  window_input_ids.to(model.device)
                
                output = model(
                    input_ids=window_input_ids.unsqueeze(0),
                    attention_mask=torch.ones(1, window_input_ids.shape[0], device=model.device),
                )
                logits = output["logits"][:, 1+mask_edge:-1-mask_edge, :]
                
                logit_diff = logits[:, :, 1] - logits[:, :, 0]
                logit_diff = logit_diff.squeeze().tolist()
                
                prob_list = prob_list + logit_diff
                
                start_idx = start_idx + effective_window
                if end_idx==len(full_input_ids):
                    break
        prob_list = prob_list[:len_of_input_ids]

        def find_split_points(numbers, m):
            if m >= len(numbers): return [0]
            def sliding_window_max_indices(arr, M):
                
                N = len(arr)
                
                dq = deque()  # 存储索引
                result = []
                
                for i in range(N):
                    # 移除队列中超出当前窗口范围的索引（窗口是 [i-M+1, i]）
                    while dq and dq[0] < i - M + 1:
                        dq.popleft()
                    
                    # 从队尾移除所有对应值 <= arr[i] 的索引（保持单调递减）
                    while dq and arr[dq[-1]] <= arr[i]:
                        dq.pop()
                    
                    dq.append(i)
                    
                    # 当窗口形成（即 i >= M - 1），记录最大值的索引
                    if i >= M - 1:
                        result.append(dq[0])
                
                return result

            max_pos = sliding_window_max_indices(numbers, m+1)
            
            splits = [0]
            st = 1

            while st < len(max_pos):
                split = max_pos[st]
                splits.append(split)
                st = split + 1
                
            return splits

        # prob_list = prob_list[:1000]
        token_split_points = find_split_points(prob_list, max_tokens_per_chunk)

        str_split_points = [tokens.token_to_chars(pos + 1).start for pos in token_split_points]

        if str_split_points[0] != 0:
            str_split_points[0] = 0

        substrings = [
            text[i:j] for i, j in zip(str_split_points, str_split_points[1:] + [len(text)])
        ]

        return substrings    

class semChunker(BaseChunker):
    def __init__(self,chunk_size):
        super().__init__()
        # self.chunker =semchunk.chunkerify(lambda text: len(text.split()), chunk_size)
        self.chunker =semchunk.chunkerify('isaacus/kanon-tokenizer', chunk_size)#0.5054 0.50543
    def split_text(self, text):

        chunks = self.chunker(text)
        return chunks
semchunker = semChunker(chunk_size=400)
bert_chunker = bertChunker4()#0.50543 
# default_ef = all_MiniLM_L6_v2
cluster_chunker = ClusterSemanticChunker(default_ef, max_chunk_size=400)#400
k_chunker=KamradtModifiedChunker()
results = evaluation.run(bert_chunker, default_ef)
print(results)