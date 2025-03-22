# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, BertForTokenClassification
import math

model_path = r"D:\github-my-project\bert-chunker-Chinese-2\data\bc-chinese-2"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    padding_side="right",
    model_max_length=512,
    trust_remote_code=True,
)

device = "cpu"  # or 'cuda'

model = BertForTokenClassification.from_pretrained(
    model_path,
).to(device)

def chunk_text(model, text, tokenizer, prob_threshold=0.5):
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


# chunking
print("\n>>>>>>>>> Chunking...")
doc = r'''9. 类
*****

类提供了把数据和功能绑定在一起的方法。创建新类时创建了新的对象 *类型*
，从而能够创建该类型的新 *实例*。实例具有能维持自身状态的属性，还具有
能修改自身状态的方法（由其所属的类来定义）。

和其他编程语言相比，Python 的类只使用了很少的新语法和语义。Python 的类
有点类似于 C++ 和 Modula-3 中类的结合体，而且支持面向对象编程（OOP）的
所有标准特性：类的继承机制支持多个基类、派生的类能覆盖基类的方法、类的
方法能调用基类中的同名方法。对象可包含任意数量和类型的数据。和模块一样
，类也支持 Python 动态特性：在运行时创建，创建后还可以修改。

如果用 C++ 术语来描述的话，类成员（包括数据成员）通常为 *public* （例
外的情况见下文 私有变量），所有成员函数都为 *virtual* 。与 Modula-3 中
一样，没有用于从对象的方法中引用本对象成员的简写形式：方法函数在声明时
，有一个显式的第一个参数代表本对象，该参数由方法调用隐式提供。与在
Smalltalk 中一样，Python 的类也是对象，这为导入和重命名提供了语义支持
。与 C++ 和 Modula-3 不同，Python 的内置类型可以用作基类，供用户扩展。
此外，与 C++ 一样，具有特殊语法的内置运算符（算术运算符、下标等）都可
以为类实例重新定义。

由于缺乏关于类的公认术语，本章中偶尔会使用 Smalltalk 和 C++ 的术语。本
章还会使用 Modula-3 的术语，Modula-3 的面向对象语义比 C++ 更接近
Python，但估计听说过这门语言的读者很少。


9.1. 名称和对象
===============

对象之间相互独立，多个名称（甚至是多个作用域内的多个名称）可以绑定到同
一对象。这在其他语言中通常被称为别名。Python 初学者通常不容易理解这个
概念，处理数字、字符串、元组等不可变基本类型时，可以不必理会。但是，对
于涉及可变对象（如列表、字典，以及大多数其他类型）的 Python 代码的语义
，别名可能会产生意料之外的效果。这样做，通常是为了让程序受益，因为别名
在某些方面就像指针。例如，传递对象的代价很小，因为实现只传递一个指针；
如果函数修改了作为参数传递的对象，调用者就可以看到更改——无需像 Pascal
那样用两个不同的机制来传参。


9.2. Python 作用域和命名空间
============================

在介绍类前，首先要介绍 Python 的作用域规则。类定义对命名空间有一些巧妙
的技巧，了解作用域和命名空间的工作机制有利于加强对类的理解。并且，即便
对于高级 Python 程序员，这方面的知识也很有用。

接下来，我们先了解一些定义。

*namespace* （命名空间）是从名称到对象的映射。现在，大多数命名空间都使
用 Python 字典实现，但除非涉及到性能优化，我们一般不会关注这方面的事情
，而且将来也可能会改变这种方式。命名空间的例子有：内置名称集合（包括
"abs()" 函数以及内置异常的名称等）；一个模块的全局名称；一个函数调用中
的局部名称。对象的属性集合也是命名空间的一种形式。关于命名空间的一个重
要知识点是，不同命名空间中的名称之间绝对没有关系；例如，两个不同的模块
都可以定义 "maximize" 函数，且不会造成混淆。用户使用函数时必须要在函数
名前面加上模块名。

点号之后的名称是 **属性**。例如，表达式 "z.real" 中，"real" 是对象 "z"
的属性。严格来说，对模块中名称的引用是属性引用：表达式
"modname.funcname" 中，"modname" 是模块对象，"funcname" 是模块的属性。
模块属性和模块中定义的全局名称之间存在直接的映射：它们共享相同的命名空
间！ [1]

属性可以是只读的或者可写的。 在后一种情况下，可以对属性进行赋值。 模块
属性是可写的：你可以写入 "modname.the_answer = 42" 。  也可以使用
"del" 语句删除可写属性。 例如，"del modname.the_answer" 将从名为
"modname" 对象中移除属性 "the_answer"。

命名空间是在不同时刻创建的，且拥有不同的生命周期。内置名称的命名空间是
在 Python 解释器启动时创建的，永远不会被删除。模块的全局命名空间在读取
模块定义时创建；通常，模块的命名空间也会持续到解释器退出。从脚本文件读
取或交互式读取的，由解释器顶层调用执行的语句是 "__main__" 模块调用的一
部分，也拥有自己的全局命名空间。内置名称实际上也在模块里，即
"builtins" 。
'''
# chunk the text. The prob_threshold should be between (0, 1). The lower it is, the more chunks will be generated.
chunks, token_pos = chunk_text(model, doc, tokenizer, prob_threshold=0.5)

# print chunks
for i, (c, t) in enumerate(zip(chunks, token_pos)):
    print(f"-----chunk: {i}----token_idx: {t}--------")
    print(c)