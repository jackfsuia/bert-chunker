# bert-chunker: efficient and trained chunking for unstructured documents
[bert-chunker-3.5](https://huggingface.co/tim1900/bert-chunker-3.5) is a text chunker based on BertForTokenClassification to predict the start token of chunks (for use in RAG, etc), and using a sliding window it cuts documents of any size into chunks. We see it as an alternative of [semantic chunker](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb), specially, it not only works for the structured texts, but also the **unstructured and messy texts**.

## Updates
- 2026.1.1: [bert-chunker-3.5](https://huggingface.co/tim1900/bert-chunker-3.5) is released. It has a simpler chunk strategy, less bugs, trained on larger model length, better performance.
- 2025.5.12: an experimental script that supports specifying the maximum tokens per chunk is available now at [here](https://huggingface.co/tim1900/bert-chunker-3#experemental).
- 2025.2.23: [bert-chunker-Chinese-2](https://huggingface.co/tim1900/bert-chunker-Chinese-2).
- 2025.2.09：[bert-chunker-3](https://huggingface.co/tim1900/bert-chunker-3).
- 2024.12：[bert-chunker-2](https://huggingface.co/tim1900/bert-chunker-2), [bert-chunker-chinese](https://huggingface.co/tim1900/bert-chunker-chinese).
- 2024.5: [bert-chunker](https://huggingface.co/tim1900/bert-chunker).
## Usage
Run the following:

```python
import torch
from transformers import AutoTokenizer, BertForTokenClassification
from collections import deque

model_path =r"tim1900/bert-chunker-3.5"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    padding_side="right",
    trust_remote_code=True,
)

device = "cpu"  # or 'cuda'

model = BertForTokenClassification.from_pretrained(model_path).to(device)

def split_text(text, max_tokens_per_chunk):
    MAX_LEN = 512
    MAX_USE_LEN = MAX_LEN-2
    MASK_EDGE =50
    EFFECT_SIZE = MAX_USE_LEN - MASK_EDGE*2
    with torch.no_grad():
        unk_id = tokenizer.unk_token_id
        print("\n>>>>>>>>> Tokenizing text...")
        tokens = tokenizer(text, return_tensors="pt", truncation=False)
        print("\n>>>>>>>>> Chunking text...")
        input_ids = tokens['input_ids'].squeeze()[1:-1]
        len_of_input_ids = len(input_ids)
        left =  EFFECT_SIZE - len_of_input_ids%EFFECT_SIZE 
        full_input_ids = torch.cat([ torch.tensor([unk_id]*MASK_EDGE), input_ids, torch.tensor([unk_id]*(MASK_EDGE + left))])
        prob_list = []
        start_idx = 0
        while(1):
            end_idx = start_idx + MAX_USE_LEN
            window_input_ids =torch.cat([ torch.tensor([tokenizer.cls_token_id]),  full_input_ids[start_idx:end_idx], torch.tensor([tokenizer.sep_token_id])])
            window_input_ids =  window_input_ids.to(model.device)
            output = model(
                input_ids=window_input_ids.unsqueeze(0),
                attention_mask=torch.ones(1, window_input_ids.shape[0], device=model.device),
            )
            logits = output["logits"][:, 1+MASK_EDGE:-1-MASK_EDGE, :]
            logit_diff = logits[:, :, 1] - logits[:, :, 0]
            logit_diff = logit_diff.squeeze().tolist()
            prob_list = prob_list + logit_diff
            start_idx = start_idx + EFFECT_SIZE
            if end_idx==len(full_input_ids):
                break
    prob_list = prob_list[:len_of_input_ids]
    def find_split_points(numbers, m):
        if m >= len(numbers): return [0]
        def sliding_window_max_indices(arr, M):
            dq = deque()
            result = []
            for i in range(len(arr)):
                while dq and dq[0] < i - M + 1:
                    dq.popleft()
                while dq and arr[dq[-1]] <= arr[i]:
                    dq.pop()
                dq.append(i)
                if i >= M - 1:
                    result.append(dq[0])
            return result

        max_pos = sliding_window_max_indices(numbers, m)

        splits = [0]
        st = 1

        while st < len(max_pos):
            split = max_pos[st]
            splits.append(split)
            st = split + 1

        return splits

    
    token_split_points = find_split_points(prob_list, max_tokens_per_chunk)

    str_split_points = [tokens.token_to_chars(pos + 1).start for pos in token_split_points]

    if str_split_points[0] != 0:
        str_split_points[0] = 0

    substrings = [
        text[i:j] for i, j in zip(str_split_points, str_split_points[1:] + [len(text)])
    ]

    return substrings, token_split_points

# chunking 

txt = r"""The causes and effects of dropouts in vocational and professional education are more pressing than ever. A decreasing attractiveness of vocational education, particularly in payment and quality, causes higher dropout rates while hitting ongoing demographic changes resulting in extensive skill shortages for many regions. Therefore, tackling the internationally high dropout rates is of utmost political and scientific interest. This thematic issue contributes to the conceptualization, analysis, and prevention of vocational and professional dropouts by bringing together current research that progresses to a deeper processual understanding and empirical modelling of dropouts. It aims to expand our understanding of how dropout and decision processes leading to dropout can be conceptualized and measured in vocational and professional contexts. Another aim is to gather empirical studies on both predictors and dropout consequences. Based on this knowledge, the thematic issue intends to provide evidence of effective interventions to avoid dropouts and identify promising ways for future dropout research in professional and vocational education to support evidence-based vocational education policy.

We thus welcome research contributions (original empirical and conceptual/measurement-related articles, literature reviews, meta-analyses) on dropouts (e.g., premature terminations, intentions to terminate, vertical and horizontal dropouts) that are situated in vocational and professional education at workplaces, schools, or other tertiary professional education institutions. 


Part 1 of the thematic series outlines central theories and measurement concepts for vocational and professional dropouts. Part 2 outlines measurement approaches for dropout. Part 3 investigates relevant predictors of dropout. Part 4 analyzes the effects of dropout on an individual, organizational, and systemic level. Part 5 deals with programs and interventions for the prevention of dropouts.

We welcome papers that include but are not limited to:

Theoretical papers on the concept and processes of vocational and professional dropout or retention
Measurement approaches to assess dropout or retention
Quantitative and qualitative papers on the causes of dropout or retention
Quantitative and qualitative papers on the effects of dropout or retention on learners, providers/organizations and the (educational) system
Design-based research and experimental papers on dropout prevention programs or retention
Submission instructions
Before submitting your manuscript, please ensure you have carefully read the Instructions for Authors for Empirical Research in Vocational Education and Training. The complete manuscript should be submitted through the Empirical Research in Vocational Education and Training submission system. To ensure that you submit to the correct thematic series please select the appropriate section in the drop-down menu upon submission. In addition, indicate within your cover letter that you wish your manuscript to be considered as part of the thematic series on series title. All submissions will undergo rigorous peer review, and accepted articles will be published within the journal as a collection.

Lead Guest Editor:
Prof. Dr. Viola Deutscher, University of Mannheim
viola.deutscher@uni-mannheim.de

Guest Editors:
Prof. Dr. Stefanie Findeisen, University of Konstanz
stefanie.findeisen@uni-konstanz.de 

Prof. Dr. Christian Michaelis, Georg-August-University of Göttingen
christian.michaelis@wiwi.uni-goettingen.de

Deadline for submission
This Call for Papers is open from now until 29 February 2023. Submitted papers will be reviewed in a timely manner and published directly after acceptance (i.e., without waiting for the accomplishment of all other contributions). Thanks to the Empirical Research in Vocational Education and Training (ERVET) open access policy, the articles published in this thematic issue will have a wide, global audience.

Option of submitting abstracts: Interested authors should submit a letter of intent including a working title for the manuscript, names, affiliations, and contact information for all authors, and an abstract of no more than 500 words to the lead guest editor Viola Deutscher (viola.deutscher@uni-mannheim.de) by July, 31st 2023. Due to technical issues, we also ask authors who already submitted an abstract before May, 30th to send their abstracts again to the address stated above. However, abstract submission is optional and is not mandatory for the full paper submission.

Different dropout directions in vocational education and training: the role of the initiating party and trainees’ reasons for dropping out
The high rates of premature contract termination (PCT) in vocational education and training (VET) programs have led to an increasing number of studies examining the reasons why adolescents drop out. Since adol...

Authors:Christian Michaelis and Stefanie Findeisen
Citation:Empirical Research in Vocational Education and Training 2024 16:15
Content type:Research
Published on: 6 August 2024"
"""
chunks, token_pos = split_text(txt, max_tokens_per_chunk = 400)

# print chunks
for i, (c, t) in enumerate(zip(chunks, token_pos)):
    print(f"========================== CHUNK: {i}  TOKEN_IDX: {t} ==========================")
    print(c)
```
## Citation

If this work is helpful, please kindly cite as:

```bibtex
@article{bert-chunker,
  title={bert-chunker: Efficient and Trained Chunking for Unstructured Documents}, 
  author={Yannan Luo},
  year={2024},
  url={https://github.com/jackfsuia/BertChunker}
}
```
