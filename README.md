# bert-chunker: efficient and trained chunking for unstructured documents
[bert-chunker-3](https://huggingface.co/tim1900/bert-chunker-3) is a text chunker based on BertForTokenClassification to predict the start token of chunks (for use in RAG, etc), and using a sliding window it cuts documents of any size into chunks. We see it as an alternative of [semantic chunker](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb). Specially, it not only works for the structured texts, but also the **unstructured and messy texts**.

Different from [bert-chunker-2](https://huggingface.co/tim1900/bert-chunker-2) and [bert-chunker](https://huggingface.co/tim1900/bert-chunker), to overcome the data distribution shift, our training data were labeled by a LLM and trainng pipeline was improved, therefore it is **more stable**.

## Usage
Run the following:

```python
import torch
from transformers import AutoConfig, AutoTokenizer, BertForTokenClassification
import math

model_path = "tim1900/bert-chunker-3"

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

def chunk_text(model, text, tokenizer, prob_threshold=0.5):
    # slide context window chunking
    MAX_TOKENS = 255
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
    print(f"Processing {input_ids.shape[1]} tokens...")
    while windows_end <= input_ids.shape[1]:
        windows_end = windows_start + MAX_TOKENS - 2

        ids = torch.cat((CLS, input_ids[:, windows_start:windows_end], SEP), 1)

        ids = ids.to(model.device)

        output = model(
            input_ids=ids,
            attention_mask=torch.ones(1, ids.shape[1], device=model.device),
        )
        logits = output["logits"][:, 1:-1, :]
        chunk_decision = logits[:, :, 1] > (logits[:, :, 0] - logits_threshold)
        greater_rows_indices = torch.where(chunk_decision)[1].tolist()

        # null or not
        if len(greater_rows_indices) > 0 and (
            not (greater_rows_indices[0] == 0 and len(greater_rows_indices) == 1)
        ):

            split_str_pos = [
                tokens.token_to_chars(sp + windows_start + 1).start
                for sp in greater_rows_indices
                if sp > 0
            ]
            token_pos += [
                sp + windows_start + 1 for sp in greater_rows_indices if sp > 0
            ]
            split_str_poses += split_str_pos

            windows_start = greater_rows_indices[-1] + windows_start

        else:

            windows_start = windows_end

    substrings = [
        text[i:j] for i, j in zip([0] + split_str_poses, split_str_poses + [len(text)])
    ]
    token_pos = [0] + token_pos
    return substrings, token_pos


# chunking code docs
print("\n>>>>>>>>> Chunking code docs...")
doc = r"""
Of course, as our first example shows, it is not always _necessary_ to declare an expression holder before it is created or used. But doing so provides an extra measure of clarity to models, so we strongly recommend it.

## Chapter 4 The Basics

## Chapter 5 The DCP Ruleset

### 5.1 A taxonomy of curvature

In disciplined convex programming, a scalar expression is classified by its _curvature_. There are four categories of curvature: _constant_, _affine_, _convex_, and _concave_. For a function \(f:\mathbf{R}^{n}\rightarrow\mathbf{R}\) defined on all \(\mathbf{R}^{n}\)the categories have the following meanings:

\[\begin{array}{llll}\text{constant}&f(\alpha x+(1-\alpha)y)=f(x)&\forall x,y\in \mathbf{R}^{n},\;\alpha\in\mathbf{R}\\ \text{affine}&f(\alpha x+(1-\alpha)y)=\alpha f(x)+(1-\alpha)f(y)&\forall x,y\in \mathbf{R}^{n},\;\alpha\in\mathbf{R}\\ \text{convex}&f(\alpha x+(1-\alpha)y)\leq\alpha f(x)+(1-\alpha)f(y)&\forall x,y \in\mathbf{R}^{n},\;\alpha\in[0,1]\\ \text{concave}&f(\alpha x+(1-\alpha)y)\geq\alpha f(x)+(1-\alpha)f(y)&\forall x,y \in\mathbf{R}^{n},\;\alpha\in[0,1]\end{array}\]

Of course, there is significant overlap in these categories. For example, constant expressions are also affine, and (real) affine expressions are both convex and concave.

Convex and concave expressions are real by definition. Complex constant and affine expressions can be constructed, but their usage is more limited; for example, they cannot appear as the left- or right-hand side of an inequality constraint.

### Top-level rules

CVX supports three different types of disciplined convex programs:

* A _minimization problem_, consisting of a convex objective function and zero or more constraints.
* A _maximization problem_, consisting of a concave objective function and zero or more constraints.
* A _feasibility problem_, consisting of one or more constraints and no objective.

### Constraints

Three types of constraints may be specified in disciplined convex programs:

* An _equality constraint_, constructed using \(==\), where both sides are affine.
* A _less-than inequality constraint_, using \(<=\), where the left side is convex and the right side is concave.
* A _greater-than inequality constraint_, using \(>=\), where the left side is concave and the right side is convex.

_Non_-equality constraints, constructed using \(\sim=\), are never allowed. (Such constraints are not convex.)

One or both sides of an equality constraint may be complex; inequality constraints, on the other hand, must be real. A complex equality constraint is equivalent to two real equality constraints, one for the real part and one for the imaginary part. An equality constraint with a real side and a complex side has the effect of constraining the imaginary part of the complex side to be zero."""
# chunk the text. The prob_threshold should be between (0, 1). The lower it is, the more chunks will be generated.
chunks, token_pos = chunk_text(model, doc, tokenizer, prob_threshold=0.5)

# print chunks
for i, (c, t) in enumerate(zip(chunks, token_pos)):
    print(f"-----chunk: {i}----token_idx: {t}--------")
    print(c)


# chunking ads
print("\n>>>>>>>>> Chunking ads...")

ad = r"""The causes and effects of dropouts in vocational and professional education are more pressing than ever. A decreasing attractiveness of vocational education, particularly in payment and quality, causes higher dropout rates while hitting ongoing demographic changes resulting in extensive skill shortages for many regions. Therefore, tackling the internationally high dropout rates is of utmost political and scientific interest. This thematic issue contributes to the conceptualization, analysis, and prevention of vocational and professional dropouts by bringing together current research that progresses to a deeper processual understanding and empirical modelling of dropouts. It aims to expand our understanding of how dropout and decision processes leading to dropout can be conceptualized and measured in vocational and professional contexts. Another aim is to gather empirical studies on both predictors and dropout consequences. Based on this knowledge, the thematic issue intends to provide evidence of effective interventions to avoid dropouts and identify promising ways for future dropout research in professional and vocational education to support evidence-based vocational education policy.

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
# chunk the text. The prob_threshold should be between (0, 1). The lower it is, the more chunks will be generated.
chunks, token_pos = chunk_text(model, ad, tokenizer, prob_threshold=0.5)

# print chunks
for i, (c, t) in enumerate(zip(chunks, token_pos)):
    print(f"-----chunk: {i}----token_idx: {t}--------")
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
