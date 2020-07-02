import numpy as np
import pandas as pd
import jieba
import subset as subset

news = pd.read_csv('sqlResult.csv', encoding='gb18030')
print(news.shape)
print(news.head(5))

# 处理缺失值
print(news[news.content.isna()].head(5))
news = news.dropna(subset=['content'])
print(news.shape)

# 加载停用词
with open('chinese_stopwords.txt', 'r', encoding='utf-8') as file:
    stopwords = [i[:-1] for i in file.readlines()]
    print(stopwords)

# 分词
def split_text(text):
    text = text.replace(' ', '')
    text = text.replace('\n', '')
    text2 = jieba.cut(text.strip())
    result = ' '.join([w for w in text2 if w not in stopwords])
    return result
print(news.iloc[0].content)
# 再对第一行进行分词
print(split_text(news.iloc[0].content))
# 发现OK，则对所有文本进行分词
import pickle,os
if not os.path.exists('corpus.pkl'):
    corpus = list(map(split_text,[str(i) for i in news.content])
    print(corpus[0])
    # 保存到文件
    with open('corpus.pkl','wb') as file:
        pickle.dump(corpus,file)
else:
    # 调用上次保存的结果
    with open('corpus.pkl','rb') as file:
        corpus = pickle.load(file)

import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# 计算corpus的TF——IDF矩阵(即分词的重要程度)
countvectorizer = CountVectorizer(encoding ='gb18030',min_df = 0.015) # 最小阈值是0.015，也就是不重要的词不予考虑
tfidftransformer = TfidfTransformer()
countvectorizer = countvectorizer.fit_transform(corpus)
tfidf = tfidftransformer.fit_transformer(corpus)

# 标记是否为自己的新闻 # map会根据提供的函数对指定序列做映射
label = list(map(lambda source:1 if '新华' in str(source) else 0, news.source)) # labmbda 定义一个简单的函数
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
# 数据集切分
X_train,X_text,y_train,y_test = train_test_split(tfidf.toarray(),label,test_size=0.3) # test_size 为比例
model = MultinomialNB()
model.fit(X_train,y_train)
# 使用model来检测新闻风格
# y_predict = model.predict(X_text)
prediction = model.predict(tfidf.toarray())
labels = np.array(label) # 把得到的列表转化一下
compare_news-index = pd.DataFrame({'prediction':prediction,'labels':labels})
# 可能的抄袭的内容特征： labels不同，但是预测值是一样的
copy_news_index = compare_news_index[(compare_news_index['prediction']==1) & (compare_news_index['labels'=0])]

# 定义新华社新闻
xinhuashe_news_index = compare_news_index[(compare_news_index['labels'] == 1)].index
# k-means 聚类
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
# 把tfidf的特征进行了规范化
scaled_array = normalizer.fit_transform(tfidf.toarray())

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 25)
k_labels = kmeans.fit_predict(scaled_array)

# 创建Id 做索引
id_class = {index:class_ for index,class_ in enumerate(k_labels)}
from collections import  defaultdict
class_id = defaultdict(set)
for index ,class_ in id_class.items():
    # 只统计标签为“新华社”的内容
    if index in xinhuashe_news_index.t0list():
        class_id[class_].add(index)

from sklearn.metrics.pairwise import cosine_similarty
# 找到像是文本 cpindex
def find_similar_text(cpindex,top=10):
    # 只在新华社的类别里找，而且只找第一个桶的
    dist_dict ={i:cosine_similarty(tfidf[cpindex],tfidf[i] for i in class_id[id_class[cpindex]]}
    return sorted(dist_dict.item(), key=lambda x:x[1][0],reverse=True)[:top]

cpindex = 3352
similarty_list = find_similar_text(cpindex)
print(similarty_list)
print('怀疑抄袭:\n',news_iloc[cpindex].content)
# 找一篇相似的
similar2 = similarty_list[0][0]
print('相似原文:\n',news_iloc[similar2.content])