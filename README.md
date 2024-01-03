# 基于文本内容的违规信息识别



## 背景与挖掘目标

​		随着短视频内容的快速增长，出现了大量低俗、暴力和违法内容，这些不仅破坏了社会道德和网络生态，还提出了紧迫的内容审核需求。本研究着重于收集和清洗这些平台的文本数据，并通过深入的数据分析来识别不适当内容。本研究将合并适当和不适当内容的数据集，分析文本长度和不适当内容的比例，创建词云以揭示常见的敏感词汇，构建主题模型进行分析，最终开发一个有效的敏感信息识别模型。这些工作旨在提高内容审核的效率和准确性，为维护网络环境的清洁提供支持。

## 文件的读取和合并

首先读取保存到本地的文本数据，如代码 1 所示。通过获取`config.py`中的文件保存路径使用`pandas`库对文件进行读取和合并，文件路径的相关配置在`config.py`中，如下所示：

```python
# 数据文件路径
SEN_FILE_PATH = pathlib.Path("../data/train_sensitiveness.csv")
INSEN_FILE_PATH = pathlib.Path("../data/train_insensitiveness.csv")
TEST_FILE_PATH = pathlib.Path("../data/test.csv")
```

文件读取和合并代码如下：

```python
import sys
sys.path.append("../config")
import pathlib
from plotly.subplots import make_subplots
import pandas as pd
from config import *
from my_log import logger


def load_file(file_path: pathlib.Path):
    """
    读取csv文件中的数据(编码格式为-简体中文GB2312)，无法编码的字符使用反斜线转义。
    :param file_path: csv文件路径
    :return: 读取后的数据
    """
    try:
        df = pd.read_csv(file_path, encoding="GB2312", encoding_errors="backslashreplace")
        logger.info(f"读取文件{file_path}成功!")
        return df
    except Exception as e:
        logger.error(e.args[0])
        raise e


def concat_file():
    """
    将两组数据进行合并
    :return: 合并后的数据
    """
    sen_df = load_file(SEN_FILE_PATH)
    insen_df = load_file(INSEN_FILE_PATH)

    concat_df = pd.concat((insen_df, sen_df), axis="index", ignore_index=True)
    logger.info("合并数据成功！")

    return concat_df


def save_concat_data():
    """
    将合并后的数据保存为训练集数据
    :return: None
    """
    concat_df = concat_file()
    concat_df.to_csv("../data/train.csv")
    logger.info("数据保存成功！")

if __name__ == '__main__':
    save_concat_data()
```

## 绘制词长统计图

通过`plotly`库对文本的词长进行统计图绘制，绘图代码如下：

```python
def analyse_data_plot():
    """
    绘制统计图分析所有文本内容的词长； 分析违规信息和违规信息数量的占比。
    :return:
    """
    df = concat_file()
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.6, 0.4],
        row_heights=[0.5, 0.5],
        specs=[[{"type": "box", "rowspan": 2}, {"type": "bar"}],
               [None, {"type": "pie"}]]
    )

    x_bar = ["违规", "非违规"]
    y_bar = [df[df["label"] == 1].shape[0], df[df["label"] == 0].shape[0]]
    fig.add_bar(x=x_bar, y=y_bar, marker_color=["#72DCC6", "#C7A7F8"],
                showlegend=False, row=1, col=2)

    labels_pie = x_bar.copy()
    values_pie = y_bar.copy()
    fig.add_pie(labels=labels_pie, values=values_pie, marker={"colors": ["#72DCC6", "#C7A7F8"]},
                hole=0.4, showlegend=False, row=2, col=2)

    s_y_box = df[df["label"] == 1]["0"].apply(len)
    ins_y_box = df[df["label"] == 0]["0"].apply(len)

    fig.add_box(y=s_y_box, name="违规", row=1, col=1)
    fig.add_box(y=ins_y_box, name="非违规", row=1, col=1)

    fig.update_layout(
        yaxis1_title="词长",
        yaxis2_title="信息数量"
    )

    fig.show()


if __name__ == '__main__':
    analyse_data_plot()
```

绘制的图像如下：

![](./image/1.png)

### 统计结果的分析

- 非违规评论的平均词长约为48.39个字符。
- 违规评论的平均词长约为37.65个字符。

统计结果显示违规评论的平均词长比非违规评论的平均词长短。这可能有几个原因：

1. **违规评论的简洁性**：违规评论可能更倾向于使用简短、直接的语言表达负面情绪或攻击性内容。
2. **非违规评论的详细性**：相比之下，非违规评论可能包含更多的解释、论述或情感表达，这通常需要使用更长的词语和句子。

- 非违规评论的词长方差约为998.19。
- 违规评论的词长方差约为738.85。

非违规评论的词长方差高于违规评论，这意味着非违规评论中词长的分布更加广泛，词长变化更大。相比之下，违规评论中词长的一致性更高，变化较小。这可能表明：

1. **非违规评论的多样性**：非违规评论可能涉及更广泛的话题和表达方式，从而导致词长的差异性更大。
2. **违规评论的集中性**：违规评论可能更倾向于使用一组相对固定的词汇，这些词汇在长度上相对一致。

对于违规评论和非违规评论的占比，违规评论占比为**26.9%**；非违规评论占比为**73.1%**。

## 违规评论词云图的绘制

### 分词与删除停用词

对评论文本内容进行分词，代码如下：

```python
import sys
from collections import Counter

import pandas as pd
import numpy as np
from PIL import Image
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt

sys.path.append("../config")

from config import STOPWORD_FILE, MY_STOPWORDS
from my_log import logger


def cut_sentences(text):
    text = [i for i in text if 0x4e00 <= ord(i) <= 0x9fa5]
    text = "".join(text)
    seg_list = jieba.cut(text, cut_all=False)
    word_list = np.array(list(seg_list))
    return word_list
```

在这一步中将非中文字符删除，使用`jieba`库对中文文本进行分词。

接着对分词后的文本数据进行停用词的删除，停用词删除的相关配置在`config.py`中，如下所示：

```python
# 停用词路径
STOPWORD_FILE = ["baidu_stopwords.txt", "cn_stopwords.txt", "hit_stopwords.txt", "scu_stopwords.txt"]
STOPWORD_FILE = ["../data/stopwords/" + i for i in STOPWORD_FILE]
MY_STOPWORDS = ["中国", "哈哈哈"]
```

停用词的删除采用四种停用词库加上自己设定的停用词共同合并而成。

停用词删除及分词结果的保存代码如下：

```python
def del_stopwords(data):
    ss = []
    for stopword in STOPWORD_FILE:
        s = np.loadtxt(stopword, encoding="utf-8", dtype=str)
        ss.append(s)
    ss = np.hstack(ss)
    ss = np.unique(ss)

    bollen = np.isin(data, ss)
    data = data[~bollen]
    data = [i for i in data if len(i) > 1]

    return data


def save_words():
    length = len(df["0"])
    for pos, i in enumerate(df["0"]):
        word_list = cut_sentences(i)
        word_list = del_stopwords(word_list)
        df.loc[pos, "cut_words"] = " ".join(word_list)
        print(f"{pos + 1}/{length}")
    df.drop("Unnamed: 0", inplace=True, axis=1)
    df.to_csv("../data/train_words.csv")
    logger.info(f"分词数据train_words.csv保存成功！")
```

### 绘制词云图

使用`python`库`wordcloud`进行词云图的绘制：

```python
def plot_wordcloud(label):
    df = pd.read_csv("../data/train_words.csv")
    sen_text = df["cut_words"][df["label"] == label]
    sen_text = [str(i) for i in sen_text]
    sen_texts = " ".join(sen_text).split(" ")
    sen_dict = Counter(sen_texts)

    for w in MY_STOPWORDS:
        if w in sen_dict:
            del sen_dict[w]

    mask = np.array(Image.open("../data/mask/enlarged_image.jpg"))
    mask = np.where(mask >= 128, 0, 255)
    wordcloud = WordCloud(font_path="../data/font/STSONG.TTF", background_color="white",
                          max_words=500, width=10000, height=10000, mask=mask, colormap="cividis")
    wordcloud.generate_from_frequencies(sen_dict)

    plt.figure(figsize=(10, 5), dpi=300)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    sort_dict = sorted(list(sen_dict.keys()), key=lambda x: list(sen_dict.values())[list(sen_dict.keys()).index(x)])
    print(sort_dict)


if __name__ == '__main__':
    plot_wordcloud(1)
```

绘制违规评论词云图结果如下：

<img src="./image/2.png" style="zoom:50%;" />

得到违规评论中出现最多的10个词及其出现次数如下：

```shell
词语：'东北', '白人', '感觉', '美国', '亚裔', '真的', '喜欢', '印度人', '黄种人', '黑人'
出现次数：158, 174, 178, 179, 202, 257, 290, 342, 344, 860
```

## 主题模型的构建

本研究选用`LDA`模型进行主题词的构建，模型生成及保存代码如下：

```python
import pickle

import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim

from my_log import logger


def create_lda_model(num_topics=3):
    df = pd.read_csv("../data/train_words.csv")
    documents = df["cut_words"]
    documents = [str(d) for d in documents]

    tokenized_documents = [document.split(" ") for document in documents]
    dictionary = corpora.Dictionary(tokenized_documents)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]

    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    lda_model.save("../model/lda_model/LDA_model.lda")
    with open("../model/lda_model/matrix.pkl", "wb") as file:
        pickle.dump({
            "corpus": corpus,
            "dictionary": dictionary
        }, file)
```

由于模型较大因此在训练后将模型保存为`pkl`文件，使用特定函数对保存的模型进行读取，模型读取代码如下：

```python
def load_model():
    with open("../model/lda_model/matrix.pkl", "rb") as file:
        matrix_dict = pickle.load(file)
    lda_model = LdaModel.load("../model/lda_model/LDA_model.lda")

    return lda_model, matrix_dict["corpus"], matrix_dict["dictionary"]
```

话题的展示如下：

```python
def show_topics(num_topics=3, num_words=6):
    lda_model, _, _ = load_model()
    topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words)
    for topic in topics:
        print(topic)
    return topics
```

展示结果如下：

```shell
(0, '0.013*"男人" + 0.012*"女人" + 0.008*"恶心" + 0.008*"真的" + 0.007*"女性" + 0.007*"喜欢"')
(1, '0.013*"东北" + 0.012*"河南人" + 0.011*"上海" + 0.009*"北京" + 0.009*"河南" + 0.007*"真的"')
(2, '0.053*"黑人" + 0.027*"中国" + 0.026*"歧视" + 0.012*"美国" + 0.011*"黄种人" + 0.011*"白人"')
```

使用`LDA`专门绘图库`pyLDAvis`进行主题词的可视化，代码如下：

```python
def plot_topics():
    """
    绘制主题可视化图像
    :return:
    """
    lda_model, corpus, dictionary = load_model()
    vis_data = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, '../data/lda_pass.html')
```

可视化结果如下：

![](./image/3.png)

其中右上角的$\lambda$调节词语主题的相关性，如果λ接近1，那么在该主题下更频繁出现的词，跟主题更相关； 如果λ越接近0，那么该主题下更特殊、更独有（exclusive）的词，跟主题更相关。

左侧显示的图像表示二维化后的三个主题，距离越远，重叠越小表示主题之间的相似度越小。可见三个主题相似度很小，分类效果良好。

可以通过选择指定的词语查看该词语与主题的关联程度，也可以选择某一主题查看其下词语的分布。

例如，选择“黑人”一词：

![](./image/4.png)

可见“黑人”与第一个主题密切相关而与其它两个主题几乎无关。

三个主题下的词语分别如下：

+ 主题1

![](./image/5.png)

+ 主题2

![](./image/6.png)

+ 主题3

![](./image/7.png)

通过分析三个主题各部分词语及权重分布，得到如下结论：

1. **主题0 - 性别和社会态度**:
   - 关键词：男人，女人，恶心，真的，女性，喜欢
   - 分析：这个主题聚焦于性别相关的话题，涉及男性和女性的比较、态度和感情。词汇“恶心”和“真的”可能表明了一些对性别议题的强烈情感或争议性看法。
2. **主题1 - 地域和身份认同**:
   - 关键词：东北，河南人，上海，北京，河南，真的
   - 分析：这个主题显然涉及中国的不同地区，可能反映了对特定地区或其居民的看法和偏见。提到多个地区名可能指出了地域差异或地域性的刻板印象。
3. **主题2 - 种族和歧视**:
   - 关键词：黑人，中国，歧视，美国，黄种人，白人
   - 分析：这个主题中包含了种族和国籍的元素，特别是关于种族歧视的议题。词汇的选择显示了讨论可能集中在种族差异、民族身份，以及不同种族群体之间的动态和冲突。

## 预测模型

本研究使用`bert-base-chinese`模型对文本内容进行编码，使用`torch`库进行神经网络的搭建，训练生成了一个精准度较高的评论二分类模型，模型训练代码如下：

```python
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from DataProcess import load_file
from word_cloud import del_stopwords, cut_sentences
from my_log import logger

df = pd.read_csv("../data/train_words.csv")
X = df["cut_words"].astype(str)
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.to_numpy()
X_val = X_val.to_numpy()
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()

model_name = '../model/bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)


def tokenize_text(text):
    inputs = tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
    return inputs


def train_model(num_epochs=5):
    input_ids = []
    attention_mask = []
    for text in X_train:
        inputs = tokenize_text(text)
        input_ids.append(inputs["input_ids"])
        attention_mask.append(inputs["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    labels = torch.tensor(y_train)

    dataset = TensorDataset(input_ids, attention_mask, labels)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')

    model.save_pretrained('../model/bert_model')
```

模型测试代码如下：

```python
def test_model():
    model = BertForSequenceClassification.from_pretrained('../model/bert_model')

    input_ids = []
    attention_mask = []
    for text in X_val:
        inputs = tokenize_text(text)
        input_ids.append(inputs["input_ids"])
        attention_mask.append(inputs["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    labels = torch.tensor(y_val)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs["logits"]
    probabilities = F.softmax(logits, dim=1)
    predicted_labels = torch.argmax(probabilities, dim=1)

    predicted_labels = np.array(predicted_labels)
    labels = np.array(labels)

    test_df = pd.DataFrame(data=np.vstack((labels, predicted_labels)).T, columns=["predict_y_val", "y_val"])
    test_df.to_csv("../output/test.csv")
```

获取模型预测分数的代码如下：

```python
def get_model_score():
    s_df = pd.read_csv("../output/test.csv")
    p_y_val = s_df["predict_y_val"]
    y_val = s_df["y_val"]

    p = precision_score(y_val, p_y_val)
    r = recall_score(y_val, p_y_val)
    f1 = f1_score(y_val, p_y_val)
    a = accuracy_score(y_val, p_y_val)

    print(f"precision score: {p: .4f}\nrecall score: {r: .4f}\nf1 score: {f1: .4f}\naccuracy score: {a: .4f}")
```

结果如下：

| 名称   | 数值   |
| ------ | ------ |
| 精确度 | 0.7583 |
| 召回率 | 0.8669 |
| F1分数 | 0.8090 |
| 准确度 | 0.9035 |

1. **精确度（Precision**）：精确度是指在所有被模型预测为正类的样本中，实际上为正类的样本所占的比例。。0.7583 的精确度意味着模型在正例预测方面表现良好。
2. **召回率（Recall）**：召回率是指在所有实际为正类的样本中，被模型正确预测为正类的样本所占的比例。0.8669 的召回率表明模型在捕捉正例方面表现良好。
3. **F1 分数**：它综合了精确度和召回率，通常在需要平衡两者时使用。0.8090 的 F1 分数表明模型在精确度和召回率之间取得了一种平衡。
4. **准确度（Accuracy**）：它表示模型正确分类所有样本的能力。0.9035 的准确度表明模型总体上在分类方面表现良好。

对题目中所给的待分类文本进行分类的代码如下：

```python
def predict_output():
    df = load_file("../data/test.csv")
    texts = df["0"]
    texts = np.array([" ".join(del_stopwords(cut_sentences(t))) for t in texts], dtype=str)
    model = BertForSequenceClassification.from_pretrained('../model/bert_model')

    input_ids = []
    attention_mask = []
    for text in texts:
        inputs = tokenize_text(text)
        input_ids.append(inputs["input_ids"])
        attention_mask.append(inputs["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs["logits"]
    probabilities = F.softmax(logits, dim=1)
    predicted_labels = torch.argmax(probabilities, dim=1)

    predicted_labels = np.array(predicted_labels)

    df["label"] = predicted_labels
    df.to_csv("../output/output.csv")
```

