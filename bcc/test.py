# -*- coding: utf-8 -*-
import safetensors
from transformers import AutoConfig,AutoTokenizer
from modeling_bertchunke_zh import BertChunker

# load config and tokenizer
config = AutoConfig.from_pretrained(
    "./",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "./",
    padding_side="right",
    model_max_length=config.max_position_embeddings,
    trust_remote_code=True,
)

# initialize model
model = BertChunker(config)
device='cpu' # or 'cuda'
model.to(device)

# load tim1900/bert-chunker-chinese/model.safetensors
state_dict = safetensors.torch.load_file(f"./model.safetensors")
model.load_state_dict(state_dict)


# text to be chunked 
text='''起点中文网(www.qidian.com)创立于2002年5月，是国内知名的原创\n\n文学 网站，隶属于阅文集团  \n  旗下。起点中文 网以推动中国  原创文学事业为宗旨，长期致力于原创文学作者的挖掘与培养，并取得了巨大成果：2003年10月，起点中文网开启“在线收费阅读”服务，成为真正意义上的网络文学赢利模式的先锋之一，就此奠定了原创文学的行业基础。此后，起点又推出了作家福利、文学交互、内容发掘推广、版权管理等机制和体系，为原创文学的发展注入了巨大活力，有力推动了中国文学原创事业的发展。在清晨的微光中，一只孤独的猫头鹰在古老的橡树上低声吟唱，它的歌声如同夜色的回声，穿越了时间的迷雾。树叶在微风中轻轻摇曳，仿佛在诉说着古老的故事，每一\\$#%#3 \n 个音符都带着\njkdadw\n 森林  的秘密。一位年轻的程序员正专注地敲打着键盘，代码的海洋在他眼前展开。他的手指在键盘上飞舞，如同钢琴家在演奏一曲复杂的交响乐。屏幕上的光标闪烁，仿佛在等待着下一个指令，引领他进入未知的数字世界。'''
# text='''“小说”一词最早出现于《庄子·外物》：「饰小说以干县令，其于大达亦远矣。」庄子所谓的「小说」，是指琐碎的言论，与小说观念相差甚远。直至东汉桓谭《新论》：「小说家合残丛小语，近取譬喻，以作短书，治身理家，有可观之辞。」班固《汉书·艺文志》将「小说家」列为十家之后，其下的定义为：「小说家者流，盖出于稗官，街谈巷语，道听途说者之所造也。」才稍与小说的意义相近。而中国小说最大的特色，便自宋代开始具有文言小说与白话小说两种不同的小说系统。文言小说起源于先秦的街谈巷语，是一种小知小道的纪录。在历经魏晋南北朝及隋唐长期的发展，无论是题材或人物的描写，文言小说都有明显的进步，形成笔记与传奇两种小说类型。而白话小说则起源于唐宋时期说话人的话本，故事的取材来自民间，主要表现了百姓的生活及思想意识。但不管文言小说或白话小说都源远流长，呈现各自不同的艺术特色。”。'''

# text='''二舅是我家亲戚中的怪人。

# 二舅琴棋书画样样精通，才华横溢，生活中却笨手笨脚。传说二舅年轻时英俊潇洒，现实中的他却普普通通，甚至有些邋遢。我见到二舅时他已经老了。

# 二舅一生未娶，即便是亲友聚会也难得见到他的身影。也许我与二舅有缘，毕业后我留在了城市，暂且寄居在姥姥留下的老房子里。这样，我就经常有机会与住在后院的二舅打交道了。

# 那年冬天，二舅滑倒摔坏了股骨头。我先是背他回屋，后来又送他去医院，那之后我们的关系发生了改变。二舅出院后经常叫我陪他喝酒，他酒量不大，但酒瘾很大。

# 我陪二舅“喝两口”的时候，他会讲一些陈年旧事，比如早年在县城的博物馆整理文物，他教过谁鉴定文物，谁还拜他为师。他说的那人可是个了不起的大名人，如今已成大师，风光多年，而二舅却默默无闻。我当然不能驳他的面子，说他吹牛。二舅很聪明，从我的眼神里看出了怀疑，主动回答：“你认为二舅在吹牛吗？二舅说的都是实话，有据为证。”说着他从覆满灰尘的书堆里翻找起一封信。二舅说那封信是大师写给自己的，里面涉及请教和致谢的内容。可惜，他找了半天也没找到。对此，我半信半疑。'''


# chunk the text. The lower threshold is, the more chunks will be generated.
chunks=model.chunk_text(text, tokenizer, threshold=0.5)

# print chunks
for i, c in enumerate(chunks):
    print(f'-----chunk: {i}------------')
    print(c)