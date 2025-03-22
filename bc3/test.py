import torch
from transformers import AutoConfig, AutoTokenizer, BertForTokenClassification
import math
import os

model_path = r"D:\github-my-project\bert-chunker-3"
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



# quantized_model = torch.quantization.quantize_dynamic(
#     model, {torch.nn.Linear}, dtype=torch.qint8
# )
# # print(quantized_model)
# def print_size_of_model(model):
#     torch.save(model.state_dict(), "temp.p")
#     print('Size (MB):', os.path.getsize("temp.p")/1e6)
#     os.remove('temp.p')

# # print_size_of_model(model)
# # print_size_of_model(quantized_model)
# model = quantized_model

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
# print("\n>>>>>>>>> Chunking code docs...")
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
# # chunk the text. The prob_threshold should be between (0, 1). The lower it is, the more chunks will be generated.
# chunks, token_pos = chunk_text(model, doc, tokenizer, prob_threshold=0.5)

# # print chunks
# for i, (c, t) in enumerate(zip(chunks, token_pos)):
#     print(f"-----chunk: {i}----token_idx: {t}--------")
#     print(c)


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

testing_text: str = """
Trees play a crucial role in maintaining ecological balance and supporting life on Earth. 
Their importance extends beyond mere aesthetics, as they are integral to environmental health, economic stability, and social well-being. 
Protecting trees is essential for several reasons, each of which underscores their multifaceted contributions to life on our planet.

Environmental Benefits   

Trees are vital for the environment as they produce oxygen, absorb carbon dioxide, and help mitigate climate change. 
Through photosynthesis, trees convert carbon dioxide into oxygen, a process that is essential for the survival of most living organisms. 
Moreover, trees act as carbon sinks, storing carbon that would otherwise contribute to global warming. 
By reducing the concentration of greenhouse gases in the atmosphere, trees help stabilize the climate, making them a natural ally in the fight against climate change.
Additionally, trees play a significant role in maintaining biodiversity. 
They provide habitat and food for a wide range of species, from insects to birds and mammals. 
Forests, which are large expanses of trees, are home to the majority of the world's terrestrial biodiversity. 
Protecting trees, therefore, means preserving the habitats of countless species, some of which are endangered.

Economic Importance   

Trees also have substantial economic value. 
They contribute to the economy by providing raw materials for various industries, including timber, paper, and pharmaceuticals. 
Sustainable forestry practices ensure that these resources are available for future generations without depleting the forests. 
Furthermore, trees enhance property values and attract tourism, which can be a significant source of income for many communities.
Urban areas benefit economically from trees as well. 
They reduce the need for air conditioning by providing shade and cooling the environment, which can lead to significant energy savings. 
Trees also help manage stormwater, reducing the costs associated with water treatment and flood damage.

Social and Health Benefits   

On a social level, trees contribute to the well-being of communities. 
They enhance the beauty of landscapes and provide spaces for recreation and relaxation. 
Studies have shown that green spaces with abundant trees can reduce stress, improve mood, and promote physical activity, all of which contribute to better mental and physical health.
Trees also play a role in social cohesion. 
Community tree-planting projects can bring people together, fostering a sense of community and shared purpose. 
These activities not only improve the local environment but also strengthen social bonds and promote environmental stewardship.

Conclusion   

In conclusion, the protection of trees is of paramount importance due to their extensive environmental, economic, and social benefits. 
As vital components of the Earth's ecosystem, trees help combat climate change, support biodiversity, and contribute to human health and well-being. 
Efforts to protect and preserve trees, whether through conservation, sustainable forestry, or urban planning, are essential for ensuring a sustainable future for all living beings. 
By recognizing and acting upon the importance of trees, we can create a healthier, more balanced world.
The history of Namibia is a complex tapestry woven from the threads of indigenous heritage, colonial conquest, and the struggle for independence. 
Situated on the southwestern coast of Africa, Namibia's history is marked by significant events and transitions that have shaped its path to becoming a sovereign nation.

Pre-Colonial Era    

The earliest inhabitants of Namibia were the San and Khoe peoples, who lived as nomadic hunters and gatherers for thousands of years. 
These indigenous groups managed the land's limited resources and developed a rich cultural heritage. 
Over time, Bantu-speaking groups such as the Owambo, Herero, and Kavango migrated into the region, establishing agricultural and pastoral societies. 
The interactions among these groups laid the foundation for Namibia's diverse cultural landscape.

German Colonial Rule   

In the late 19th century, Namibia became a German colony known as German South West Africa. 
The German colonial administration was marked by harsh policies and brutal suppression of uprisings, most notably the Herero and Namaqua genocide from 1904 to 1908, where tens of thousands of Herero and Nama people were killed. 
This dark chapter in Namibia's history left a lasting impact on its indigenous populations and remains a poignant reminder of the atrocities of colonialism.

South African Administration   

Following Germany's defeat in World War I, the League of Nations mandated South Africa to administer the territory. 
South Africa imposed its apartheid policies, further marginalizing the indigenous peoples and intensifying the struggle for independence. 
Despite international condemnation, South African control persisted, leading to decades of resistance and conflict.

The Struggle for Independence

The fight for Namibian independence gained momentum in the mid-20th century with the formation of the South West Africa People's Organization (SWAPO) in 1960. 
SWAPO led an armed struggle against South African rule, drawing international attention to Namibia's plight. 
The United Nations played a crucial role in advocating for Namibia's independence, culminating in a UN-supervised election in 1989.

Independence and Beyond   

On March 21, 1990, Namibia achieved independence, with Sam Nujoma becoming its first president. 
The transition was marked by efforts to foster national reconciliation and build a democratic society. 
Namibia adopted a constitution that emphasized human rights and sought to address the legacies of colonialism and apartheid.
In the years following independence, Namibia has made strides in economic development and social progress, although challenges remain. 
The government has focused on land reform, addressing inequalities, and promoting sustainable development. 
Namibia's journey from colonial subjugation to independence is a testament to the resilience and determination of its people.

Conclusion   

Namibia's history is a narrative of endurance and transformation. 
From its indigenous roots to colonial oppression and eventual liberation, Namibia's past has shaped its identity and continues to influence its future. 
As Namibia moves forward, it carries the lessons of its history, striving to build a nation that honors its diverse heritage and fosters unity and prosperity for all its citizens
Trees play a pivotal role in the field of medicine, serving as a vital source of both traditional and modern pharmaceutical products. 
Their protection is crucial not only for maintaining biodiversity and ecological balance but also for ensuring the continued availability of medicinal resources that are fundamental to human health and well-being.

Historical Significance of Medicinal Trees   

Historically, trees have been integral to traditional medicine systems across the globe. 
Ancient cultures, such as those in China, India, and Egypt, have long utilized tree-derived substances for healing purposes. 
For instance, the bark of the willow tree, which contains salicylic acid, has been used for centuries to alleviate pain and inflammation, eventually leading to the development of aspirin, one of the most widely used drugs today. 
Similarly, the opium poppy, another plant-based source, has given rise to powerful painkillers like morphine and codeine, highlighting the long-standing relationship between trees and medicinal advancements.

Modern Pharmaceutical Applications   

In contemporary medicine, trees continue to be a cornerstone for drug discovery and development. 
A significant portion of pharmaceuticals are derived from plant compounds, with trees providing essential chemical templates for synthesizing new drugs. 
These natural products often serve as the basis for treatments of serious ailments, including cancer, heart disease, and malaria. 
For example, the Pacific yew tree is the source of paclitaxel, a chemotherapy drug used to treat various cancers. 
This underscores the necessity of preserving tree species that may harbor potential cures for diseases yet to be fully understood.

Biodiversity and Ecosystem Health   

The preservation of trees is also critical for maintaining the biodiversity necessary for medicinal plant research. 
Studies have shown that old-growth forests, with their dense canopy cover, are rich in medicinal plant diversity, providing habitats for numerous species that could hold the key to future medical breakthroughs. 
Deforestation and habitat destruction threaten these ecosystems, potentially leading to the loss of invaluable plant species before their medicinal properties can be explored.

Socioeconomic and Cultural Importance    

Beyond their direct medicinal applications, trees support traditional medicine practices that are vital to the healthcare systems of many indigenous and rural communities. 
In regions where access to modern pharmaceuticals is limited, trees offer an accessible and affordable source of treatment, often forming the backbone of local healthcare. 
Protecting these natural resources ensures that communities can continue to rely on traditional knowledge and practices that have been passed down through generations.
    """
text = """19th Century: Military and Social Transformations in Africa  
The most effective armies of the 19th century relied on local resources at every level—training, equipment, and leadership. These forces stood in contrast to those modeled after European armies, which adopted European-style uniforms, barracks life, training methods, and military ranks. In Madagascar, reforms went as far as introducing the purchase of military ranks, mirroring practices found in early 19th-century European armies.  
The influence of European armies was significant, leading to the widespread adoption of European muskets. Many new African armies adopted these firearms, some for their firepower (such as Enfield rifles) and others for the psychological impact of their loud volleys, which could spread panic among enemy cavalry. Many African rulers also sought to acquire artillery, but due to the weight of cannons and the large quantities of gunpowder required, attempts to manufacture them locally were largely unsuccessful. By the 1870s, more advanced weaponry—including breech-loading rifles, repeating carbines, machine guns, and artillery—began to be imported. The Maxim gun, in particular, became the pinnacle of military technology, though European powers maintained a monopoly over it. Some African leaders, such as the Igbo chiefs, used European cannons more for their psychological impact than for actual warfare.  
Even before the 19th century, African rulers sought to import European firearms. However, it was only in the second half of the century, when more effective firearms became available, that European weaponry clearly became a decisive factor in African warfare. As European arms became a critical advantage, securing access to them during wartime became essential, making arms procurement a central aspect of trade, diplomacy, and governance.  
Ethiopian Military Strategies and African Arms Trade  
The shift in military power brought significant changes, illustrated by the contrasting strategies of Ethiopian emperors Tewodros II and Yohannes IV. Tewodros attempted to manufacture cannons and gunpowder domestically using European technicians and proved that he could defeat better-equipped armies even without such weapons. However, Yohannes IV—and later, Menelik II—recognized that only by acquiring superior European firearms could they overcome their enemies in Tigray and the coastal plains. Likewise, North African, Saharan, and Sudanese rulers sought to stockpile European arms through trade and diplomacy, mainly via North Africa and coastal trade routes in West and East Africa. Madagascar's rulers followed a similar strategy.  
In central and eastern Africa, the ivory trade was a major economic driver, further increasing the demand for firearms and contributing to the growing militarization of society. One major consequence of this increasing reliance on European weaponry was the decline of cavalry as an elite military force, with European-style infantry replacing it. In forested regions and areas influenced by Nguni military traditions, infantry had long been the dominant force. In the 19th century, these infantry units underwent more intensive training, gradually becoming professional armies equipped with European weapons.  
Large-scale population movements, such as those following the Mfecane or the collapse of the Oyo Empire, further accelerated the decline of cavalry. As formerly cavalry-based societies moved into forested regions where horse warfare was impractical, cavalry forces diminished. However, in some African states directly confronting European colonial expansion, cavalry remained essential. These states began breeding small horses for mobile warfare and imported European firearms, maintaining temporary military superiority."""
# chunk the text. The prob_threshold should be between (0, 1). The lower it is, the more chunks will be generated.
chunks, token_pos = chunk_text(model, doc, tokenizer, prob_threshold=0.5)

# print chunks
for i, (c, t) in enumerate(zip(chunks, token_pos)):
    print(f"-----chunk: {i}----token_idx: {t}--------")
    print(c)
