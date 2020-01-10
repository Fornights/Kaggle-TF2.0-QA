# Kaggle-TF2.0-QA

主要看transformers里面的run_squad.py 

## 目前的工作

已经完成训练文本到feature的转化，可以直接作为模型的输入

## TODO
- 初步的训练，看看结果
- HTML 标签的去除，去除的同时改变answer的start_pos 和 end_pos的变化

## 数据集信息

### 基本情况
1. 公开数据量: 307,373 training examples with single annotations, 7,830 examples with 5-way annotations for development data, and 7,842 5-way annotated items sequestered as test data
2. 任务定义: close to end2end form
   1. input: a question together with entire Wikipedia page
   2. output: long answer + short anwer
3. 参考标注精度: long answer/short answer 90%/84%
4. 包含数据: question, wikipedia page, long answer, short answer
5. 问题格式：8 words or more, fact seeking questions
   1. start with ‘who’, ‘when’, or ‘where’ directly followed by: a) a finite form of ‘do’ or a modal verb; or b) a finite form of ‘be’ or ‘have’ with a verb in some later position;
   2. start with ‘who’ directly followed by a verb that is not a finite form of ‘be’;
   3. contain multiple entities as well as an adjective, adverb, verb or determiner;
   4. contain a categorical noun phrase immediately preceded by a preposition or relative clause;
   5. end with a categorical noun phrase, and do not contain a preposition or relative clause.
   6. 'how to' questions excluded

### 标注规约

<div align=center>
<img src=https://i.loli.net/2019/11/25/8mrqWftHeVb4gSd.png>
</div>
<center>标注流程与样本统计</center>

1. Long Answer(l): A HTML bounding box on the Wikipedia page, anotationor select the earlist bounding box containing enough information. 
   1.  typically a paragraph or table or list items or whole lists, contains the information required to answer the question
   2.  return NULL if there is no answer on the page, or if the information required to answer the question is spread across many paragraphs.
2. Short Answer(s):  
   1. a span or set of spans (typically entities) within long answer that answer the question
   2. a boolean ‘yes’ or ‘no’ answer
   3. NULL, if l = NULL -> s = NULL

### 标注质量评估

1. post-hoc evaluation of correctness of non-null answers, according to 4 experts
2. k-way annotations (k=25) on a subset of data, [0, 0.2)，[0.2, 0.4)，[0.4, 0.6)，[0.6, 0.8)，[0.8, 1.0]

### 样本分布形式化

数据由 (q, d, l, s) 四元元组组成，q -> question，d -> document，l -> long answer，s -> short answer。对相应的随机变量 Q, D, L, S。

将四元组分割为三元组 (q, d, l) 和 (q, d, s)。每个数据项 (q, d, l) 都是独立同分布从状态空间采样

$$p(l, q, d) = p(q, d) \times p(l|q, d)$$

$p(q, d)$ 是样本分布（问题-文档对的概率质量函数）。
$p(l|q, d)$ 是条件分布，l 受两种随机因素的影响：随机选择的标注人员和特定标注人员的随机选择

## 基线参考

1. Naive solution, First paragraph, Most paragraph annotation applied on, Closest question TF-IDF.
2. DocumentQA
3. DecAtt + DocReader

## 参考
1. pytorch(https://github.com/pytorch)
2. transformers(https://github.com/huggingface/transformers)
3. [Google Research NQ Github repo](https://github.com/google-research-datasets/natural-questions)
4. [Dataset Details](https://ai.google/research/pubs/pub47761)
