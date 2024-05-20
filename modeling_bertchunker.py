from transformers.modeling_utils import PreTrainedModel
from torch import nn
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertModel
import torch
import safetensors
from transformers import AutoConfig,AutoTokenizer
class BertChunker(PreTrainedModel):

    config_class = BertConfig

    def __init__(self, config, ):
        super().__init__(config)

        self.model = BertModel(config)
        self.chunklayer = nn.Linear(384, 2)
      
    def forward(self, input_ids=None, attention_mask=None,labels=None, **kwargs):
        model_output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        token_embeddings = model_output[0]
        logits = self.chunklayer(token_embeddings)
        model_output["logits"]=logits
        loss = None
        logits = logits.contiguous()
        if labels:
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

    def chunk_text(self, text:str, tokenizer,threshold=0)->list[str]:

        MAX_TOKENS=255
        tokens=tokenizer(text, return_tensors="pt",truncation=False)
        input_ids=tokens['input_ids']
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

            output=self(input_ids=ids,attention_mask=attention_mask[:,:len(ids)])
            logits = output['logits'][:, 1:-1,:]
            is_left_greater = ((logits[:,:, 0] + threshold) < logits[:,:, 1])
            greater_rows_indices = torch.where(is_left_greater)[1].tolist()

            # null or not
            if len(greater_rows_indices)>0 and (not (greater_rows_indices[0] == 0 and len(greater_rows_indices)==1)):

                
                split_str_pos=[tokens.token_to_chars(sp + windows_start + 1).start for sp in greater_rows_indices]

                split_str_poses += split_str_pos

                windows_start = greater_rows_indices[-1] + windows_start

            else:

                windows_start = windows_end

        substrings = [text[i:j] for i, j in zip([0] + split_str_poses, split_str_poses+[len(text)])]
        return substrings