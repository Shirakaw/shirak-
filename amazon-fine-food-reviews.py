#!/usr/bin/env python
# coding: utf-8

# # **亚马逊食品评论**

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import sqlite3
import seaborn as sns
import re
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('C:/Users/12499/Desktop/亚马逊食品评价/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## 读取数据

# In[6]:


review=pd.read_csv('C:/Users/12499/Desktop/亚马逊食品评价/Reviews.csv')
review.shape


# # Part A : 探索性数据分析

# ## 基本数据信息

# In[4]:


#Printing dataset information
review.info()


# In[5]:


#Printing few rows
review.head(5)


# In[6]:


# Removing duplicate entries
reviews=review.drop_duplicates(subset=["UserId","ProfileName","Time","Text"], keep='first', inplace=False)
print("The shape of the data set after removing duplicate reviews : {}".format(reviews.shape))


# In[7]:


# Helpfulness Numerator ：认为该评论有用的用户数
# Helpfulness Denominator - 表示他们认为该评论有用的用户数
#score 打分
reviews[["HelpfulnessNumerator", "HelpfulnessDenominator", "Score"]].describe()


# ## 评分分布

# In[8]:


reviews['Score'].value_counts()


# In[9]:


reviews['Score'].value_counts().plot(kind='bar', 
                                     color=['r', 'b', 'g', 'y', 'm'], 
                                     title='Ratings distribution', 
                                     xlabel='Score', ylabel='Number of Users')


# **探索信息**:
# * 打5分的用户最多: = 2,50,962
# * 打2分的用户最少 : = 20,802

# ## 帮助性分布

# In[10]:


#计算帮助性的方法，将分数换算成帮助等级（No Indication helpful intermediate Not Helpful

# (D) == 0        : No Indication
# (N/D) > 75 %    : Helpful
# (N/D) 75 - 25 % : Intermediate
# (N/D) < 25 %    : Not Helpful


def helpCalc(n, d):
    if d==0:
        return 'No Indication'
    elif n > (75.00/100*d):
        return 'Helpful'
    elif n < (25.00/100*d):
        return 'Not Helpful'
    else:
        return 'Intermediate'

reviews['Helpfulness'] = reviews.apply(lambda x : helpCalc(x['HelpfulnessNumerator'], x['HelpfulnessDenominator']), axis=1)
reviews.head(5)


# In[11]:


reviews['Helpfulness'].value_counts()


# In[12]:


reviews['Helpfulness'].value_counts().plot(kind='bar', 
                                           color=['r', 'b', 'g', 'y'], 
                                           title='Distribution of Helpfulness', 
                                           xlabel='Helpfulness', ylabel='Number of Users')


# In[13]:


print('Percentage of No Indication reviews %.2f %%' % ((reviews['Helpfulness'].value_counts()['No Indication'])*100.0/len(reviews)))
print('Percentage of Helpful reviews %.2f %%' % ((reviews['Helpfulness'].value_counts()['Helpful'])*100.0/len(reviews)))
print('Percentage of Intermediate reviews %.2f %%' % ((reviews['Helpfulness'].value_counts()['Intermediate'])*100.0/len(reviews)))
print('Percentage of Not Helpful reviews %.2f %%' % ((reviews['Helpfulness'].value_counts()['Not Helpful'])*100.0/len(reviews)))


# **探索信息**:
# 
# * 评论有用的用户数 : 1,44,901 - (36.78 %)
# * 评论无帮助的用户数: 25,647 - (6.51 %)

# ## 评论分数如何影响帮助 ?

# In[14]:


helpfulness_score = pd.crosstab(reviews['Score'], reviews['Helpfulness'])
helpfulness_score


# In[15]:


helpfulness_score.plot(kind='bar', figsize=(10,6), title='Helpfulness Score')


# ## 每月为时间度量进行分析

# In[16]:


reviews['DateTime'] = pd.to_datetime(reviews['Time'], unit='s')
monthly_review = reviews.groupby([reviews['DateTime'].dt.year, reviews['DateTime'].dt.month, reviews['Score']]).count()['ProductId'].unstack().fillna(0)
monthly_review.head(30)


# In[17]:


monthly_review.plot(figsize=(25,8), xlabel='Year,Month', ylabel='Review Counts', title='Monthly Review Counts')


# ## 评论的文本的长度会因为等级而变化

# In[18]:


def WordLength(text):
    words = str(text).split(" ")
    return len(words)


reviews['TextLength'] = reviews['Text'].apply(lambda x : WordLength(x))

print('最大评论文本长度:', reviews['TextLength'].max())
print('评论文本平均长度:', reviews['TextLength'].mean())
print('最小评论文本长度:', reviews['TextLength'].min())

plt.figure(figsize=(12,10))
ax = sns.boxplot(x='Score',y='TextLength', data=reviews)


# ## 评论文本长度与rating之间的关系

# In[19]:


plt.figure(figsize=(12,10))
ax = sns.violinplot(x='Helpfulness', y='TextLength', data=reviews)


# ## 摘要文本的长度与rating关系

# In[20]:


reviews['SummaryLength'] = reviews['Summary'].apply(lambda x : WordLength(x))

print('最大摘要文本长度:', reviews['SummaryLength'].max())
print('评论摘要文本长度:', reviews['SummaryLength'].mean())
print('最小摘要文本长度:', reviews['SummaryLength'].min())

plt.figure(figsize=(12,10))
ax = sns.boxplot(x='Score',y='SummaryLength', data=reviews)


# ## 摘要文本长度与rating之间的关系

# In[21]:


plt.figure(figsize=(12,10))
ax = sns.violinplot(x='Helpfulness', y='SummaryLength', data=reviews)


# ## 被评论商品的次数展示

# In[22]:


rf = reviews.groupby(['UserId', 'ProfileName']).count()['ProductId']
y = rf.to_frame()
x = y.sort_values('ProductId', ascending=False)
x.head(20).plot(kind='bar', figsize=(20,5), title='Frequency of Top 20 Reviewers', xlabel='(UserId, Profile Name)', ylabel='Number of Reviews')


# ## 将分数编码分为positive和negative两类

# In[23]:


reviews['ScoreClass'] = reviews['Score'].apply(lambda x : 'Positive' if x > 3 else 'Negative')
reviews['ScoreClass'].value_counts()


# In[24]:


reviews['ScoreClass'].value_counts().plot(kind='bar', color=['g','r'], title='Score Class Distribution', xlabel='Score Class', ylabel='Score Count')


# In[25]:


print('乐观评论占比 %.2f %%' % ((reviews['ScoreClass'].value_counts()['Positive'])*100.0/len(reviews)))
print('消极评论占比 %.2f %%' % ((reviews['ScoreClass'].value_counts()['Negative'])*100.0/len(reviews)))


# # Part B : 模型构建

# In[26]:


reviews[['Text', 'ScoreClass']].head(5)


# ## 基于正负类的数据分割

# In[27]:


def splitPosNeg(reviews):
    neg = reviews.loc[reviews['ScoreClass']=='Negative']
    pos = reviews.loc[reviews['ScoreClass']=='Positive']
    return [pos,neg]

[pos,neg] = splitPosNeg(reviews)

print("评论总数: ", len(pos)+len(neg))
print("积极评论总数 : ", len(pos))
print("消极评论总数 : ", len(neg))


# In[28]:


# 输出积极评论
print('Positive Polarity Review :', pos['Text'].values[0])
print('Polarity Sentiment :', pos['ScoreClass'].values[0])


# In[29]:


# Printing a negative review and polarity
print('Negative Polarity Review :', neg['Text'].values[0])
print('Negative Sentiment :', neg['ScoreClass'].values[0])


# ## 数据预处理

# In[30]:


# To-do : Lemmatization

# 评论文本作为x轴为任意长度的string类型，将其全部转换为小写字母
def decontracted(x):
    x = str(x).lower()
    x = x.replace(",000,000", " m").replace(",000", " k").replace("′", "'").replace("’", "'")                           .replace("won't", " will not").replace("cannot", " can not").replace("can't", " can not")                           .replace("n't", " not").replace("what's", " what is").replace("it's", " it is")                           .replace("'ve", " have").replace("'m", " am").replace("'re", " are")                           .replace("he's", " he is").replace("she's", " she is").replace("'s", " own")                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")                           .replace("€", " euro ").replace("'ll", " will").replace("how's"," how has").replace("y'all"," you all")                           .replace("o'clock"," of the clock").replace("ne'er"," never").replace("let's"," let us")                           .replace("finna"," fixing to").replace("gonna"," going to").replace("gimme"," give me").replace("gotta"," got to").replace("'d"," would")                           .replace("daresn't"," dare not").replace("dasn't"," dare not").replace("e'er"," ever").replace("everyone's"," everyone is")                           .replace("'cause'"," because")
    
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    return x

# Removing the html tags
def removeHTML(text):
    pattern = re.compile('<.*?>')
    cleanText = re.sub(pattern,' ',text)
    return cleanText

# 删除所有标点符号或有限的特殊字符集，例如 , or . or # etc.
def removePunctuations(text):
    cleanText  = re.sub('[^a-zA-Z]',' ',text)
    return (cleanText)

# 删除所有数字
def removeNumbers(text):
    cleanText = re.sub("\S*\d\S*", " ", text).strip()
    return (cleanText)

#删除文本中的url
def removeURL(text):
    textModified = re.sub(r"http\S+", " ", text)
    cleanText = re.sub(r"www.\S+", " ", textModified)
    return (cleanText)

#删除像 'zzzzzzzzzzzzzzzzzzzzzzz', 'testtting', 'grrrrrrreeeettttt' etc. 保留像'looks', 'goods', 'soon' etc. 
#删除所有具有三个连续重复字符的单词。
def removePatterns(text): 
    cleanText  = re.sub("\\s*\\b(?=\\w*(\\w)\\1{2,})\\w*\\b",' ',text)
    return (cleanText)

# 删除Stopwords
defaultStopwordList = set(stopwords.words('english'))
remove_not = set(['not','no','nor'])
stopwordList = defaultStopwordList - remove_not

# Snowball Stemming
stemmer = SnowballStemmer(language='english')


# In[31]:


# 数据预处理考虑了整个评论中的所有单词
def preprocessing(text):
    total_words = []
    text = decontracted(text)
    text = removeHTML(text)
    text = removePunctuations(text)
    text = removeNumbers(text)
    text = removeURL(text)
    text = removePatterns(text)
    
    line = nltk.word_tokenize(text)
    for word in line:
        if (word not in stopwordList):
            stemmed_word = stemmer.stem(word)
            total_words.append(stemmed_word)
    return ' '.join(total_words)

#-----------------------------------------------------------------------------------------------------------------------------------------------
# 分别对正面评论和负面评论进行预处理
pos_data = [] # 预处理好的正面评价列表
neg_data = [] # 预处理好的负面评论列表

for p in tqdm(pos['Text']):
    pos_data.append(preprocessing(p))
    
for n in tqdm(neg['Text']):
    neg_data.append(preprocessing(p))
    
#-----------------------------------------------------------------------------------------------------------------------------------------------
# 结合预处理的正面评论和负面评论
data = pos_data + neg_data # 将预处理好的正面和负面评论的列表进行组合
labels = np.concatenate((pos['ScoreClass'].values,neg['ScoreClass'].values)) # 评价分数组合

#------------------------------------------------------------------------------------------------------------------------------------------------
# 标记数据并创建标记列表
token_list = []
for line in tqdm(data):
    l = nltk.word_tokenize(line)
    for w in l:
        token_list.append(w)
        
#------------------------------------------------------------------------------------------------------------------------------------------------
# 从整个评论中获取独特单词的列表
total_words = list(set(token_list))
print("Total unique words in whole reviews : ", len(total_words))

#------------------------------------------------------------------------------------------------------------------------------------------------
# 在整个评论中保存总计独特词
with open('unique_words_in_whole_reviews.pkl', 'wb') as file:
    pickle.dump(total_words, file)


# In[ ]:


# Load the unique word from whole reviews从整个评论中加载特殊词
with open('unique_words_in_whole_reviews.pkl', 'rb') as file:
    total_words = pickle.load(file)
    
#----------------------------------------------------------------------------------------------------------------------------------    
# 从整个评论中查找特殊单词的长度分布
word_length_dist = []

for word in tqdm(total_words):
    length = len(word)
    word_length_dist.append(length)

plt.figure(figsize=(20,10))
plt.hist(word_length_dist, color='green', bins =90)
plt.title('Distribution of the length of all unique words across whole reviews')
plt.xlabel('Word Lengths')
plt.ylabel('Number of Words')


# **探索信息:**
# * 我们可以看到，评论中大多数假词的长度在3到10之间。长度超过15的单词与其他单词相比非常非常少。所以，当我们处理它们时，我们将从评论中丢弃这些词。这意味着我们将只考虑那些长度大于2且小于16的单词。

# In[37]:


# 在整个评论中仅考虑长度大于2且小于16的单词进行数据预处理
def _preprocessing(text):
    total_words_reduced = []
    text = decontracted(text)
    text = removeHTML(text)
    text = removePunctuations(text)
    text = removeNumbers(text)
    text = removeURL(text)
    text = removePatterns(text)
    
    line = nltk.word_tokenize(text)
    for word in line:
        if (word not in stopwordList) and (2 < len(word) < 16):
            stemmed_word = stemmer.stem(word)
            total_words_reduced.append(stemmed_word)
    return ' '.join(total_words_reduced)

#-----------------------------------------------------------------------------------------------------------------------------------------------
# 分别对正面评论和负面评论进行预处理。
pos_data_reduced = [] # 预处理好的正面评价列表
neg_data_reduced = [] # 预处理好的负面评论列表

for p in tqdm(pos['Text']):
    pos_data_reduced.append(_preprocessing(p))
    
for n in tqdm(neg['Text']):
    neg_data_reduced.append(_preprocessing(p))
    
#-----------------------------------------------------------------------------------------------------------------------------------------------
# Combining preprocessed positive review and negative review
data_final = pos_data_reduced + neg_data_reduced # A list of combined preprocessed positive and negative reviews
# An array of combined positive score class and negative score class
labels_final = np.concatenate((pos['ScoreClass'].values,neg['ScoreClass'].values)) 

#------------------------------------------------------------------------------------------------------------------------------------------------
# Tokenizing the data and creating a token list
token_list_reduced = []
for line in tqdm(data_final):
    l = nltk.word_tokenize(line)
    for w in l:
        token_list_reduced.append(w)
        
#------------------------------------------------------------------------------------------------------------------------------------------------
# Get list of unique words from train token list
total_words_reduced = list(set(token_list_reduced))
print("Total unique words in whole reviews of length > 2 and < 16 : ", len(total_words_reduced))

#------------------------------------------------------------------------------------------------------------------------------------------------
# Save Total unique words reduced in whole reviews
with open('unique_words_reduced_in_whole_reviews.pkl', 'wb') as file:
    pickle.dump(total_words_reduced, file)

# Save final data of whole reviews
with open('data_final.pkl', 'wb') as file:
    pickle.dump(data_final, file)
    
# Save final labels of whole reviews
with open('labels_final.pkl', 'wb') as file:
    pickle.dump(labels_final, file)


# ## 使用分层策略将数据集划分为训练和测试

# In[39]:


# Load the final data and final labels
with open('data_final.pkl', 'rb') as file:
    data_final = pickle.load(file)
    
with open('labels_final.pkl', 'rb') as file:
    labels_final = pickle.load(file)
    
#------------------------------------------------------------------------------------------------------------------------------------------------    
# Splitting the datasets into train-test in 80:20 ratio
[train_data,test_data,train_label,test_label] = train_test_split(data_final, labels_final, test_size=0.20, random_state=20160121, stratify=labels)


# In[40]:


# Get list of unique tokens in train data
train_token = []

for line in tqdm(train_data):
    l = nltk.word_tokenize(line)
    for w in l:
        train_token.append(w)
        
x = len(list(set(train_token)))        
print("Unique token in train data : ", x)


# In[14]:


from sklearn.model_selection import GridSearchCV

clf = LogisticRegression()
param_grid = {'C':[1000,500,100,50,10,5,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001],
             'penalty':['l1','l2']} 
tscv = TimeSeriesSplit(n_splits=10) 
gsv = GridSearchCV(clf,param_grid,cv=tscv, scoring = 'f1_micro', verbose=1, n_jobs = -1)
gsv.fit(data_final,abels_final)

print("Best HyperParameter: ",gsv.best_params_)
print("Best Accuracy: %.2f%%"%(gsv.best_score_*100))


# ## 特征提取

# ### [1]. Bag of Words - Unigrams

# In[42]:


cv_object = CountVectorizer(min_df=10, max_features=50000, dtype='float')
cv_object.fit(train_data)

print("Some BOW Unigram features are : ", cv_object.get_feature_names()[100:110])
print("="*145)

print("\nCreating BOW Unigram vectors...")
train_data_cv = cv_object.transform(train_data)
test_data_cv = cv_object.transform(test_data)

print("\nThe type of BOW Unigram Vectorizer ", type(train_data_cv))
print("Shape of train BOW Unigram Vectorizer ", train_data_cv.get_shape())
print("Shape of test BOW Unigram Vectorizer ", test_data_cv.get_shape())


with open('train_data_BOW_Uni.pkl', 'wb') as file:
    pickle.dump(train_data_cv, file)

with open('test_data_BOW_Uni.pkl', 'wb') as file:
    pickle.dump(test_data_cv, file)
    
with open('train_label_BOW_Uni.pkl', 'wb') as file:
    pickle.dump(train_label, file)

with open('test_label_BOW_Uni.pkl', 'wb') as file:
    pickle.dump(test_label, file)


# ### [2]. Bag of Words - Bigrams

# In[43]:


cv_object = CountVectorizer(ngram_range=(1,2), min_df=10, max_features=50000, dtype='float')
cv_object.fit(train_data)

print("Some BOW Bigram features are : ", cv_object.get_feature_names()[100:110])
print("="*145)

print("\nCreating BOW Bigram vectors...")
train_data_cv = cv_object.transform(train_data)
test_data_cv = cv_object.transform(test_data)

print("\nThe type of BOW Bigram Vectorizer ", type(train_data_cv))
print("Shape of train BOW Bigram Vectorizer ", train_data_cv.get_shape())
print("Shape of test BOW Bigram Vectorizer ", test_data_cv.get_shape())

with open('train_data_BOW_Bi.pkl', 'wb') as file:
    pickle.dump(train_data_cv, file)

with open('test_data_BOW_Bi.pkl', 'wb') as file:
    pickle.dump(test_data_cv, file)
    
with open('train_label_BOW_Bi.pkl', 'wb') as file:
    pickle.dump(train_label, file)

with open('test_label_BOW_Bi.pkl', 'wb') as file:
    pickle.dump(test_label, file)


# ### [3]. TF-IDF - Unigram

# In[44]:


tv_object = TfidfVectorizer(min_df=10, max_features=50000, dtype='float')
tv_object.fit(train_data)

print("Some Tf-idf Unigram features are : ", tv_object.get_feature_names()[100:110])
print("="*145)

print("\nCreating Tf-idf Unigram vectors...")
train_data_tv = tv_object.transform(train_data)
test_data_tv = tv_object.transform(test_data)

print("\nThe type of Tf-idf Unigram Vectorizer ", type(train_data_tv))
print("Shape of train Tf-idf Unigram Vectorizer ", train_data_tv.get_shape())
print("Shape of test Tf-idf Unigram Vectorizer ", test_data_tv.get_shape())

with open('train_data_TF-IDF_Uni.pkl', 'wb') as file:
    pickle.dump(train_data_tv, file)

with open('test_data_TF-IDF_Uni.pkl', 'wb') as file:
    pickle.dump(test_data_tv, file)
    
with open('train_label_TF-IDF_Uni.pkl', 'wb') as file:
    pickle.dump(train_label, file)

with open('test_label_TF-IDF_Uni.pkl', 'wb') as file:
    pickle.dump(test_label, file)


# ### [4]. TF-IDF - Bigram

# In[45]:


tv_object = TfidfVectorizer(ngram_range=(1,2), min_df=10, max_features=50000, dtype='float')
tv_object.fit(train_data)

print("Some Tf-idf Bigram features are : ", tv_object.get_feature_names()[100:110])
print("="*145)

print("\nCreating Tf-idf Bigram vectors...")
train_data_tv = tv_object.transform(train_data)
test_data_tv = tv_object.transform(test_data)

print("\nThe type of Tf-idf Bigram Vectorizer ", type(train_data_tv))
print("Shape of train Tf-idf Bigram Vectorizer ", train_data_tv.get_shape())
print("Shape of test Tf-idf Bigram Vectorizer ", test_data_tv.get_shape())

with open('train_data_TF-IDF_Bi.pkl', 'wb') as file:
    pickle.dump(train_data_tv, file)

with open('test_data_TF-IDF_Bi.pkl', 'wb') as file:
    pickle.dump(test_data_tv, file)
    
with open('train_label_TF-IDF_Bi.pkl', 'wb') as file:
    pickle.dump(train_label, file)

with open('test_label_TF-IDF_Bi.pkl', 'wb') as file:
    pickle.dump(test_label, file)


# ## 基于机器学习的分类模型

# ## LOGISTIC REGRESSION

# 1. Logistic Regression - L1 Regularizer with Grid Search - BoW - Unigram
# 2. Logistic Regression - L2 Regularizer with Grid Search - BoW - Unigram
# 3. Logistic Regression - L1 Regularizer with Grid Search - BoW - Bigram
# 4. Logistic Regression - L2 Regularizer with Grid Search - BoW - Bigram

# In[1]:


# 导入包
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
# def top_features()

#标准化数据矩阵
def standardize(data, with_mean):
    scalar = StandardScaler(with_mean=with_mean)
    std=scalar.fit_transform(data)
    return (std)

#------------------------------------------------------------------------------------------------------------------------------------------------
# 绘制ROC曲线
def plot_roc_curve(clf, train_data, train_label, test_data, test_label):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    train_prob = clf.predict_proba(train_data)
    train_label_prob = train_prob[:,1]
    fpr["Train"], tpr["Train"], threshold = roc_curve(train_label, train_label_prob, pos_label='Positive')
    roc_auc["Train"] = auc(fpr["Train"], tpr["Train"])
    
    test_prob = clf.predict_proba(test_data)
    test_label_prob = test_prob[:,1]
    fpr["Test"], tpr["Test"], threshold = roc_curve(test_label, test_label_prob, pos_label='Positive')
    roc_auc["Test"] = auc(fpr["Test"], tpr["Test"])
    
    plt.figure(figsize=(5,5))
    linewidth = 2
    plt.plot(fpr["Test"], tpr["Test"], color='green', lw=linewidth, label='ROC curve Test Data (area = %0.2f)' % roc_auc["Test"])
    plt.plot(fpr["Train"], tpr["Train"], color='red', lw=linewidth, label='ROC curve Train Data (area = %0.2f)' % roc_auc["Train"])
    plt.plot([0, 1], [0, 1], color='navy', lw=linewidth, linestyle='--', label='Baseline ROC curve (area = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
#------------------------------------------------------------------------------------------------------------------------------------------------
# 绘制召回曲线
def plot_pr_curve(clf, train_data, train_label, test_data, test_label):
    precision = dict()
    recall = dict()
    pr_auc = dict()
    
    train_prob = clf.predict_proba(train_data)
    train_label_prob = train_prob[:,1]
    precision["Train"], recall["Train"], threshold = precision_recall_curve(train_label, train_label_prob, pos_label='Positive')
    pr_auc["Train"] = auc(recall["Train"], precision["Train"])
    
    test_prob = clf.predict_proba(test_data)
    test_label_prob = test_prob[:,1]
    precision["Test"], recall["Test"], threshold = precision_recall_curve(test_label, test_label_prob, pos_label='Positive')
    pr_auc["Test"] = auc(recall["Test"], precision["Test"])
    
    plt.figure(figsize=(5,5))
    linewidth = 2
    plt.plot(recall["Test"], precision["Test"], color='green', lw=linewidth, label='Precision-Recall Curve Test Data (area = %0.2f)' % pr_auc["Test"])
    plt.plot(recall["Train"], precision["Train"], color='red', lw=linewidth, label='Precision-Recall curve Train Data (area = %0.2f)' % pr_auc["Train"])
    #plt.plot([0, 1], [0, 1], color='navy', lw=linewidth, linestyle='--', label='Baseline Precision-Recall curve (area = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Plot')
    plt.legend(loc="lower right")
    plt.show()
#------------------------------------------------------------------------------------------------------------------------------------------------
# 计算指标性能
def metrics_performance(grid, train_data, train_label, test_data, test_label):
    clf = grid.best_estimator_
    clf.fit(train_data, train_label)
    test_label_pred = clf.predict(test_data)
    
    test_prob = clf.predict_proba(test_data)
    test_label_prob = test_prob[:,1]
   
    print("Accuracy : ", accuracy_score(test_label, test_label_pred, normalize=True) * 100)
    print("Points : ", accuracy_score(test_label, test_label_pred, normalize=False))
    print("Precision : ", np.round(precision_score(test_label ,test_label_pred, pos_label='Positive'),4))
    print("Recall : ", recall_score(test_label, test_label_pred, pos_label='Positive'))
    print("F1-score : ", f1_score(test_label,test_label_pred, pos_label='Positive'))
    print("AUC : ", np.round(roc_auc_score(test_label, test_label_prob),4))
    print ('\nClasification Report :')
    print(classification_report(test_label,test_label_pred))
    
    print('\nConfusion Matrix :')
    cm = confusion_matrix(test_label ,test_label_pred)
       
    df_cm = pd.DataFrame(cm, index = [' (0)',' (1)'], columns = [' (0)',' (1)'])
    plt.figure(figsize = (5,5))
    ax = sns.heatmap(df_cm, annot=True, fmt='d')
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title('Confusion Matrix')
    
    plot_roc_curve(clf, train_data, train_label, test_data, test_label)
    plot_pr_curve(clf, train_data, train_label, test_data, test_label)
#------------------------------------------------------------------------------------------------------------------------------------------------    
# 绘制网格搜索结果
def plot_gridsearch_result(grid):
    cv_result = grid.cv_results_
    auc_train = list(cv_result['mean_train_score'])
    auc_test = list(cv_result['mean_test_score'])
    params = cv_result['params']
    
    hp_values = [p['C'] for p in params] #获取列表C
    
    # 绘制 GridSearchCV 结果的 Heapmap
    cv_result = {'C':hp_values, 'Mean Train Score':auc_train, 'Mean Test Score':auc_test} # dataframe of cv_result
    cv = pd.DataFrame(cv_result)
    sns.heatmap(cv, annot=True)
    
    #绘制误差与 C 值
    plt.figure(figsize=(15,6))
    plt.plot(hp_values , auc_train, color='red', label='Train AUC')
    plt.plot(hp_values , auc_test, color='blue', label='Validation AUC')
    plt.title('ROC 曲线下面积与 C 值 ')
    plt.xlabel('超参数：C 的值')
    plt.ylabel('ROC 曲线下面积（AUC 分数）')
    plt.legend()
    plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------    
# 应用 GridSearchCV 的函数
def gridSearchCV(train_data, train_label, regularization):
    param = {'C' : [1000,500,100]}
    model = LogisticRegression(penalty=regularization, solver='liblinear', random_state=0)
    cv = TimeSeriesSplit(n_splits=10).split(train_data)
    grid = GridSearchCV(estimator=model, param_grid=param, cv=cv, n_jobs=-1, scoring='roc_auc', verbose=40, return_train_score=True)
    grid.fit(train_data, train_label)
    
    print("最佳参数 : ", grid.best_params_)
    print("最佳得分 : ", grid.best_score_)
    print("最佳估算器 : ", grid.best_estimator_)
    
    return grid


# In[2]:


def logisticRegression(train_data, train_label, test_data, test_label, regularization):
    grid = gridSearchCV(train_data, train_label, regularization)
    plot_gridsearch_result(grid)
    metrics_performance(grid, train_data, train_label, test_data, test_label)


# In[3]:


# Load BOW Unigram Datasets
with open('train_data_BOW_Uni.pkl', 'rb') as file:
    train_data_uni = pickle.load(file)
    
with open('test_data_BOW_Uni.pkl', 'rb') as file:
    test_data_uni = pickle.load(file)

with open('train_label_BOW_Uni.pkl', 'rb') as file:
    train_label_uni = pickle.load(file)

with open('test_label_BOW_Uni.pkl', 'rb') as file:
    test_label_uni = pickle.load(file)
    
# Load BOW Bigram Datasets
with open('train_data_BOW_Bi.pkl', 'rb') as file:
    train_data_bi = pickle.load(file)
    
with open('test_data_BOW_Bi.pkl', 'rb') as file:
    test_data_bi = pickle.load(file)

with open('train_label_BOW_Bi.pkl', 'rb') as file:
    train_label_bi = pickle.load(file)

with open('test_label_BOW_Bi.pkl', 'rb') as file:
    test_label_bi = pickle.load(file)
    
# Standardize the data
train_data_uni=standardize(train_data_uni, False)
test_data_uni=standardize(test_data_uni, False)
train_data_bi=standardize(train_data_bi, False)
test_data_bi=standardize(test_data_bi, False)


# ### [1].  基于BOW Unigram的L1正则化逻辑回归模型

# In[27]:


logisticRegression(train_data_uni, train_label_uni, test_data_uni, test_label_uni, 'l1')


# ### [2]. Logistic Regression with L2 Regularization on BOW Unigram

# In[22]:


logisticRegression(train_data_uni, train_label_uni, test_data_uni, test_label_uni, 'l2')


# ### [3]. Logistic Regression with L1 Regularization on BOW Bigram

# In[23]:


logisticRegression(train_data_bi, train_label_bi, test_data_bi, test_label_bi, 'l1')


# ### [4]. Logistic Regression with L2 Regularization on BOW Bigram

# In[145]:


logisticRegression(train_data_bi, train_label_bi, test_data_bi, test_label_bi, 'l2')


# 1. Logistic Regression - L1 Regularizer with Grid Search - BoW - TF-IDF
# 2. Logistic Regression - L2 Regularizer with Grid Search - BoW - TF-IDF
# 3. Logistic Regression - L1 Regularizer with Grid Search - BoW - TF-IDF
# 4. Logistic Regression - L2 Regularizer with Grid Search - BoW - TF-IDF

# In[146]:


# Load TF-IDF Unigram Datasets
with open('train_data_TF-IDF_Uni.pkl', 'rb') as file:
    train_data_uni = pickle.load(file)
    
with open('test_data_TF-IDF_Uni.pkl', 'rb') as file:
    test_data_uni = pickle.load(file)

with open('train_label_TF-IDF_Uni.pkl', 'rb') as file:
    train_label_uni = pickle.load(file)

with open('test_label_TF-IDF_Uni.pkl', 'rb') as file:
    test_label_uni = pickle.load(file)
   
#--------------------------------------------------------------------------------------------------------------------------------------
# Load TF-IDF Bigram Datasets
with open('train_data_TF-IDF_Bi.pkl', 'rb') as file:
    train_data_bi = pickle.load(file)
    
with open('test_data_TF-IDF_Bi.pkl', 'rb') as file:
    test_data_bi = pickle.load(file)

with open('train_label_TF-IDF_Bi.pkl', 'rb') as file:
    train_label_bi = pickle.load(file)

with open('test_label_TF-IDF_Bi.pkl', 'rb') as file:
    test_label_bi = pickle.load(file)


# ### [1]. Logistic Regression with L1 Regularization on TF-IDF Unigram

# In[147]:


logisticRegression(train_data_uni, train_label_uni, test_data_uni, test_label_uni, 'l1')


# ### [2]. Logistic Regression with L2 Regularization on TF-IDF Unigram

# In[148]:


logisticRegression(train_data_uni, train_label_uni, test_data_uni, test_label_uni, 'l2')


# ### [3]. Logistic Regression with L1 Regularization on TF-IDF Bigram

# In[ ]:


logisticRegression(train_data_bi, train_label_bi, test_data_bi, test_label_bi, 'l1')


# ### [4]. Logistic Regression with L2 Regularization on TF-IDF Bigram

# In[ ]:


logisticRegression(train_data_bi, train_label_bi, test_data_bi, test_label_bi, 'l2')


# ## Feature Importance

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import RandomizedSearchCV
parameter_space = {
    'hidden_layer_sizes': [(1024), (50,), (50,100, 50), (48,), (48, 48, 48), (96,), (144,), (192,), (96, 144, 192), (240,), (144, 192, 240)],
    'activation': ['tanh', 'logistic', 'relu'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001, 0.05, 0.1, 1],
    'beta_1': [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
    'beta_2': [0.990, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999],
    'learning_rate': ['constant','adaptive'],
                }


# In[3]:


mlp = MLPClassifier(max_iter=10000, random_state=42)


# In[4]:


score = ['accuracy', 'precision']
clf = RandomizedSearchCV(mlp, parameter_space, n_jobs = -1, n_iter = 15,  cv=3, refit='precision', scoring=score, random_state=0)


# In[7]:


y = review.Score
x = review.Text
cv = CountVectorizer()
X = cv.fit_transform(x)
cv.get_feature_names()[:5]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
print("Validation Accuracy",score*100,"%")


# In[ ]:


plot_confusion_matrix(clf, x_test, y_test)

