from transformers.modeling_utils import PreTrainedModel
from torch import nn
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertModel
import torch
import torch.nn.functional as F
class BertChunker(PreTrainedModel):

    config_class = BertConfig

    def __init__(self, config, ):
        super().__init__(config)

        self.model = BertModel(config)
        self.chunklayer = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids=None, attention_mask=None,labels=None, **kwargs):
        model_output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        token_embeddings = model_output[0]
        logits = self.chunklayer(token_embeddings)
        model_output["logits"]=logits
        loss = None
        logits = logits.contiguous()
        if labels!=None:
            labels = labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()#用-100
            # loss_fct = nn.CrossEntropyLoss(ignore_index=50257)
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(labels.device)
            loss = loss_fct(logits, labels)
            model_output["loss"]=loss

        return model_output

    def chunk_text(self, text:str, tokenizer,threshold=0.5)->list[str]:
    # slide context window
        MAX_TOKENS=self.model.config.max_position_embeddings
        tokens=tokenizer(text, return_tensors="pt",truncation=False)
        input_ids=tokens['input_ids'].to(self.device)
        attention_mask=tokens['attention_mask'][:,0:MAX_TOKENS]
        attention_mask=attention_mask.to(self.device)
        CLS=input_ids[:,0].unsqueeze(0)
        SEP=input_ids[:,-1].unsqueeze(0)
        input_ids=input_ids[:,1:-1]
        self.eval()
        split_str_poses=[]

        windows_start =0
        windows_end= 0

        while windows_end <= input_ids.shape[1]:
            windows_end= windows_start + MAX_TOKENS-2

            ids=torch.cat((CLS, input_ids[:,windows_start:windows_end],SEP),1)

            ids=ids.to(self.device)
            
            output=self(input_ids=ids,attention_mask=torch.ones(1, ids.shape[1],device=self.device))
            logits = output['logits'][:, 1:-1,:]
            chunk_probabilities = F.softmax(logits, dim=-1)[:,:,1]
            chunk_decision = (chunk_probabilities>threshold)
            greater_rows_indices = torch.where(chunk_decision)[1].tolist()

            # null or not
            if len(greater_rows_indices)>0 and (not (greater_rows_indices[0] == 0 and len(greater_rows_indices)==1)):

                split_str_pos=[tokens.token_to_chars(sp + windows_start + 1).start for sp in greater_rows_indices]

                split_str_poses += split_str_pos

                windows_start = greater_rows_indices[-1] + windows_start

            else:

                windows_start = windows_end

        substrings = [text[i:j] for i, j in zip([0] + split_str_poses, split_str_poses+[len(text)])]
        return substrings
    

    def chunk_text_smooth(self, text:str, tokenizer,threshold=0)->list[str]:
    # slide context window
        MAX_TOKENS=self.model.config.max_position_embeddings
        tokens=tokenizer(text, return_tensors="pt",truncation=False)
        input_ids=tokens['input_ids'].to(self.device)
        attention_mask=tokens['attention_mask'][:,0:MAX_TOKENS]
        attention_mask=attention_mask.to(self.device)
        CLS=input_ids[:,0].unsqueeze(0)
        SEP=input_ids[:,-1].unsqueeze(0)
        input_ids=input_ids[:,1:-1]
        self.eval()
        split_str_poses=[]

        windows_start =0
        windows_end= 0
        prob_pair_list=[]

        for j in range(input_ids.shape[1]):

            prob_pair_list.append([])


        while windows_start <= input_ids.shape[1]:
            windows_end= windows_start + MAX_TOKENS-2

            ids=torch.cat((CLS, input_ids[:,windows_start:windows_end],SEP),1)

            ids=ids.to(self.device)
            
            output=self(input_ids=ids,attention_mask=torch.ones(1, ids.shape[1],device=self.device))
            logits = output['logits'][:, 1:-1,:]


            chunk_probabilities = F.softmax(logits, dim=-1).tolist()


            # is_left_greater = ((logits[:,:, 0] + threshold) < logits[:,:, 1])


            for i in range(windows_start, windows_start + len(chunk_probabilities[0])):
                prob_pair_list[i].append(chunk_probabilities[0][i-windows_start][1])


            # split_str_pos=[tokens.token_to_chars(sp + windows_start + 1).start for sp in greater_rows_indices]

            # split_str_poses += split_str_pos

            windows_start = windows_start + MAX_TOKENS//2-1

        split_str_poses=[]
        for i in range(len(prob_pair_list)):
            if sum(prob_pair_list[i])/len(prob_pair_list[i])>threshold:
                split_str_poses+=[tokens.token_to_chars(i + 1).start]



        substrings = [text[i:j] for i, j in zip([0] + split_str_poses, split_str_poses+[len(text)])]
        return substrings
    


    def chunk_text_fast(
        self, text: str, tokenizer, batchsize=20, threshold=0
    ) -> list[str]:
    # chunk the text faster with a fixed context window, batchsize is the number of windows run per batch.
        self.eval()

        split_str_poses=[]
        MAX_TOKENS = self.model.config.max_position_embeddings
        USEFUL_TOKENS = MAX_TOKENS - 2 # delete cls and sep
        tokens = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = tokens["input_ids"]


        CLS = tokenizer.cls_token_id

        SEP = tokenizer.sep_token_id

        input_ids = input_ids[:, 1:-1].squeeze().contiguous()# delete cls and sep

        token_num = input_ids.shape[0]
        seq_num = input_ids.shape[0] // (USEFUL_TOKENS)
        left_token_num = input_ids.shape[0] % (USEFUL_TOKENS)

        if seq_num > 0:

            reshaped_input_ids = input_ids[: seq_num * USEFUL_TOKENS].view( seq_num, USEFUL_TOKENS )

            i = torch.arange(seq_num).unsqueeze(1)
            j = torch.arange(USEFUL_TOKENS).repeat(seq_num, 1)

            bias = 1 # 1 bias by cls token
            position_id = i * (USEFUL_TOKENS) + j + bias 
            position_id = position_id.to(self.device)
            reshaped_input_ids = torch.cat(
                (
                    torch.full((reshaped_input_ids.shape[0], 1), CLS),
                    reshaped_input_ids,
                    torch.full((reshaped_input_ids.shape[0], 1), SEP),
                ),
                1,
            )

            batch_num = seq_num // batchsize
            left_seq_num = seq_num % batchsize
            for i in range(batch_num):
                batch_input = reshaped_input_ids[i : i + batchsize, :].to(self.device)
                attention_mask = torch.ones(batch_input.shape[0], batch_input.shape[1]).to(self.device)
                output = self(input_ids=batch_input, attention_mask=attention_mask)
                logits = output['logits'][:, 1:-1,:]#delete cls and sep
                is_left_greater = ((logits[:,:, 0] + threshold) < logits[:,:, 1])
                pos = is_left_greater * position_id[i : i + batchsize, :]
                pos = pos[pos>0].tolist()
                split_str_poses += [tokens.token_to_chars(p).start for p in pos]
            if left_seq_num > 0:
                batch_input = reshaped_input_ids[-left_seq_num:, :].to(self.device)
                attention_mask = torch.ones(batch_input.shape[0], batch_input.shape[1]).to(self.device)
                output = self(input_ids=batch_input, attention_mask=attention_mask)
                logits = output['logits'][:, 1:-1,:]#delete cls and sep
                is_left_greater = ((logits[:,:, 0] + threshold) < logits[:,:, 1])
                pos = is_left_greater * position_id[-left_seq_num:, :]
                pos = pos[pos>0].tolist()
                split_str_poses += [tokens.token_to_chars(p).start for p in pos]

        if left_token_num > 0:
            left_input_ids = torch.cat([torch.tensor([CLS]), input_ids[-left_token_num:], torch.tensor([SEP])])
            left_input_ids = left_input_ids.unsqueeze(0).to(self.device)
            attention_mask = torch.ones(left_input_ids.shape[0], left_input_ids.shape[1]).to(self.device)
            output = self(input_ids=left_input_ids, attention_mask=attention_mask)
            logits = output['logits'][:, 1:-1,:]#delete cls and sep
            is_left_greater = ((logits[:,:, 0] + threshold) < logits[:,:, 1])
            bias = token_num - (left_input_ids.shape[1] - 2) + 1
            pos = (torch.where(is_left_greater)[1] + bias).tolist()
            split_str_poses += [tokens.token_to_chars(p).start for p in pos]
     
        substrings = [text[i:j] for i, j in zip([0] + split_str_poses, split_str_poses+[len(text)])]
        return substrings
