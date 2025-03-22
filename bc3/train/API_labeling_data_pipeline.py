
def doubao_factory():
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key="apikey", base_url="https://ark.cn-beijing.volces.com/api/v3")
    async def thread_func(p:str)->str:
        try:
            response = await client.chat.completions.create(
                model="ep-20250108194306-w44z6",
                messages=[
                    {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
                    {"role": "user", "content": p},
                ],
                stream=False,
            )
            result = response.choices[0].message.content
        except Exception as e:
            print(e)
            result = "Network Error!"
            print('-->one network error')
        return result
    
    import asyncio
    from tqdm.asyncio import tqdm
    def llm_response(prompts: list[str]) -> list[str]:
        async def main():
            tasks = [thread_func(p) for p in prompts]
            results = await asyncio.gather(*tasks,)
            return results
        
        results = asyncio.run(main())

        return results
    
    
    return llm_response

dou = doubao_factory()
template = '''<para>\n\n请把上面的文章合理分段，用来做RAG。指出你分段的地方，用的格式是：
段落总结：xxxx
段落起点：xxxx （该段的第一句话）'''

import datasets
from datasets import load_dataset
import json
from tqdm import tqdm

dataset = load_dataset("/data/bert-chunker-v2/dataset/",data_files=["para4000.parquet"],split="train",num_proc=6,cache_dir='/data/bert-chunker-v2/dataset/cache')
batchsize=250
for i in tqdm(range(0,len(dataset),batchsize),desc='Processing batches'):
    
    prompts = [template.replace("<para>",t) for t in dataset[i:i+batchsize]['text']]
    responses = dou(prompts)
    with open(f'/data/bert-chunker-v2/dataset/doubao_responses/para4000_{i}.json','w',encoding='utf-8') as f:
        f.write(json.dumps(responses))
    
#------------------合流------------------   
for i in range(0,len(dataset),batchsize):
    with open(f'/data/bert-chunker-v2/dataset/doubao_responses/para4000_{i}.json','r',encoding='utf-8') as f:
        a+=json.load(f)
a=[{'doubao':i} for i in a]
a = Dataset.from_list(a)
a.to_parquet('/data/bert-chunker-v2/dataset/doubao_responses/para4000_doubao_data.parquet')
print(len(a))


#------------------合并原始数据集和后来豆包-----------------  
import json,re
from datasets import load_dataset,Dataset
dataset2 = load_dataset("/data/bert-chunker-v2/dataset/doubao_responses/",data_files=["para4000_doubao_data.parquet"],split="train",num_proc=6,cache_dir='/data/bert-chunker-v2/dataset/cache')


# 将 dataset2 的列添加到 dataset1 中
for column in dataset2.column_names:
    dataset = dataset.add_column(name=column, column=dataset2[column])


dataset.to_parquet('/data/bert-chunker-v2/dataset/doubao_responses/para4000_data_and_doubao.parquet')
# ----------------cut------------------
import re

def preprocess_function(item):
    def cut(text, dou_resp):
        START_SYM = "<--start-->"
        sp = re.split(r'段落总结：.*?\n段落起点：', dou_resp)
        sp = [i.strip() for i in sp if len(i.strip()) > 0]

        chunk = text
        check = ''
        for i in sp[1:]:
            pos = chunk.find(i[:30])
            if pos == -1:
                check = text+'\n-------------wrong---------\n'+i
                
                return '', check

            chunk = chunk[:pos] + START_SYM + chunk[pos:]

        return chunk, check

    chunk, check = cut(item['text'], item['doubao'])

    item['chunk'] = chunk
    item['check'] = check

    return item
d = dataset.map(preprocess_function,num_proc=6)
print(d)

#------过滤没cut好的-----
d = d.filter(lambda x: x['check']=='', num_proc=6)
d.to_parquet("/data/bert-chunker-v2/dataset/doubao_responses/para4000_cut.parquet")
print(d)
# #---------------------------------------------
# d = dataset.map(preprocess_function,num_proc=6)
# print(d)