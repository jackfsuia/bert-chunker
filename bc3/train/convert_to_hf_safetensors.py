import safetensors
from safetensors.torch import load_file, save_file

# 加载 safetensors 文件并提取元数据
file_path = r"D:\github-my-project\bert-chunker-3\一些重要的用于可能融合的checkpoint\BertChunker第一版\model.safetensors"
state_dict = load_file(file_path)

# 提取元数据
with  safetensors.safe_open(file_path, framework="pt") as f:
    metadata = f.metadata()
# 创建一个新的字典来存储修改后的参数
new_state_dict = {}

# 遍历原始参数字典
for key, value in state_dict.items():
    # 将 "model." 替换为 "bert."
    new_key = key.replace("model.", "bert.")
    
    # 将 "chunklayer." 替换为 "classifier."
    new_key = new_key.replace("chunklayer.", "classifier.")
    
    # 丢弃 "model.pooler.dense.bias" 和 "model.pooler.dense.weight"
    if "model.pooler.dense.bias" in key or "model.pooler.dense.weight" in key:
        continue
    
    # 将修改后的键值对添加到新的字典中
    new_state_dict[new_key] = value

# 保存修改后的参数字典到新的 safetensors 文件，并保留元数据
save_file(new_state_dict, r"D:\github-my-project\bert-chunker-3\model.safetensors", metadata=metadata)