
import semchunk
import tiktoken                        # `transformers` and `tiktoken` are not required.
from transformers import AutoTokenizer # They're just here for demonstration purposes.

chunk_size = 4 # A low chunk size is used here for demonstration purposes. Keep in mind, `semchunk`
               # does not know how many special tokens, if any, your tokenizer adds to every input,
               # so you may want to deduct the number of special tokens added from your chunk size.
text = 'The quick brown fox jumps over the lazy dog.'

# You can construct a chunker with `semchunk.chunkerify()` by passing the name of an OpenAI model,
# OpenAI `tiktoken` encoding or Hugging Face model, or a custom tokenizer that has an `encode()`
# method (like a `tiktoken`, `transformers` or `tokenizers` tokenizer) or a custom token counting
# function that takes a text and returns the number of tokens in it.
# chunker = semchunk.chunkerify('isaacus/kanon-tokenizer', chunk_size)
        #   semchunk.chunkerify('gpt-4', chunk_size) or \
        #   semchunk.chunkerify('cl100k_base', chunk_size) or \
        #   semchunk.chunkerify(AutoTokenizer.from_pretrained('isaacus/kanon-tokenizer'), chunk_size) or \
        #   semchunk.chunkerify(tiktoken.encoding_for_model('gpt-4'), chunk_size) or \
        #   semchunk.chunkerify(lambda text: len(text.split()), chunk_size)
# If you give the resulting chunker a single text, it'll return a list of chunks. If you give it a
# list of texts, it'll return a list of lists of chunks.

text = r"""Of course, as our first example shows, it is not always _necessary_ to declare an expression holder before it is created or used. But doing so provides an extra measure of clarity to models, so we strongly recommend it.

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
chunk_size = 120
chunker =semchunk.chunkerify(lambda text: len(text.split()), chunk_size)

a= chunker(text)
for i,a_i in enumerate(a):
    print(f'--------------------Chunk {i}:---------------------------------------\n{a_i}')
# assert chunker([text], progress = True) == [['The quick brown fox', 'jumps over the', 'lazy dog.']]

# If you have a lot of texts and you want to speed things up, you can enable multiprocessing by
# setting `processes` to a number greater than 1.
# assert chunker([text], processes = 2) == [['The quick brown fox', 'jumps over the', 'lazy dog.']]

# You can also pass a `offsets` argument to return the offsets of chunks, as well as an `overlap`
# argument to overlap chunks by a ratio (if < 1) or an absolute number of tokens (if >= 1).
# chunks, offsets = chunker(text, offsets = True, overlap = 0.5)