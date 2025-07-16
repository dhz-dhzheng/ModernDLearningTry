# Datawhale AI夏令营 用户新增预测挑战赛 基于带货视频评论的用户洞察挑战赛 2.2

## 尝试使用模型进行embedding，conda使用以及模型下载
```shell
eval "$(/mnt/workspace/conda/bin/conda shell.bash hook)"

conda activate newuser
python -m ipykernel install --user --name=newuser --display-name='Environment (newuser)'

# modelscope download --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local_dir ./all_models/sentence-transformers---paraphrase-multilingual-MiniLM-L12-v2

# jupyter nbconvert --to script vc_insights_2_try.ipynb
```

## 尝试优化聚类的0分，但似乎提交数据的格式出现问题，未解决
```python
#!/usr/bin/env python
# coding: utf-8

# 需进行聚类的字段包括：
# 
# - positive_cluster_theme：基于训练集和测试集中正面倾向（sentiment_category=1 或 sentiment_category=3）的评论进行聚类并提炼主题词，聚类数范围为 5~8。
# - negative_cluster_theme：基于训练集和测试集中负面倾向（sentiment_category=2 或 sentiment_category=3）的评论进行聚类并提炼主题词，聚类数范围为 5~8。
# - scenario_cluster_theme：基于训练集和测试集中用户场景相关评论（user_scenario=1）进行聚类并提炼主题词，聚类数范围为 5~8。
# - question_cluster_theme：基于训练集和测试集中用户疑问相关评论（user_question=1）进行聚类并提炼主题词，聚类数范围为 5~8。
# - suggestion_cluster_theme：基于训练集和测试集中用户建议相关评论（user_suggestion=1）进行聚类并提炼主题词，聚类数范围为 5~8。

# 
# - 评论聚类（100分）
# 
# 结果评估采用轮廓系数（仅计算商品识别和情感分析均正确的评论聚类结果），衡量聚类结果的紧密性和分离度。该阶段总评分计算公式如下：
# 
# ![img](https://openres.xfyun.cn/xfyundoc/2025-06-09/06d30dd5-471b-4762-95fa-ce097e9ca912/1749469510941/579-3.bmp)
# 
# 其中Silhouette coefficientᵢ为维度i的聚类结果的轮廓系数，M为需聚类的维度总数。



# In[1]:


import pandas as pd
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sentence_transformers import SentenceTransformer


# In[2]:


# 1. 数据加载
video_data = pd.read_csv("origin_videos_data.csv")
comments_data = pd.read_csv("origin_comments_data.csv")

# 合并视频文本
video_data["text"] = video_data["video_desc"].fillna("") + " " + video_data["video_tags"].fillna("")

# 清洗评论，保留有效字段
comments_data = comments_data.dropna(subset=[
    'sentiment_category', 'user_scenario', 'user_question', 'user_suggestion'
])


# In[5]:


# prompt + SBERT 向量化
model = SentenceTransformer('../all_models/sentence-transformers---paraphrase-multilingual-MiniLM-L12-v2/')

def embed_texts(texts):
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# 产品预测训练
mask = ~video_data["product_name"].isnull()
X_vid = embed_texts(video_data.loc[mask, "text"].tolist())
y_vid = video_data.loc[mask, "product_name"].values

clf_prod = SGDClassifier(loss='log_loss').fit(X_vid, y_vid)
# 预测全部
X_all = embed_texts(video_data["text"].tolist())
video_data["product_name"] = clf_prod.predict(X_all)


# In[6]:


# 构造联合标签
comments_data['y_joint'] = comments_data.apply(
    lambda r: f"{int(r.sentiment_category)}_{int(r.user_scenario)}_{int(r.user_question)}_{int(r.user_suggestion)}",
    axis=1
)

X_com = embed_texts(comments_data["comment_text"].tolist())
y_com = comments_data["y_joint"]

clf_com = SGDClassifier(loss='log_loss').fit(X_com, y_com)
preds = clf_com.predict(X_com)
pcs = np.array([p.split('_') for p in preds], dtype=int)
comments_data[['sentiment_category',
               'user_scenario',
               'user_question',
               'user_suggestion']] = pcs


# In[13]:


comments_data = pd.read_csv("origin_comments_data.csv")
cols_to_fill = list(comments_data.columns)
comments_data[cols_to_fill] = comments_data[cols_to_fill].fillna(0)

comments_data['y_joint'] = comments_data.apply(
    lambda r: f"{int(r.sentiment_category)}_{int(r.user_scenario)}_{int(r.user_question)}_{int(r.user_suggestion)}",
    axis=1
)
X_com = embed_texts(comments_data["comment_text"].tolist())

preds = clf_com.predict(X_com)
pcs = np.array([p.split('_') for p in preds], dtype=int)
comments_data[['sentiment_category',
               'user_scenario',
               'user_question',
               'user_suggestion']] = pcs


# In[14]:


EMBED_MODEL = SentenceTransformer('../all_models/sentence-transformers---paraphrase-multilingual-MiniLM-L12-v2/')

def cluster_and_keywords(df, filter_cond, field_prefix):
    texts = df.loc[filter_cond, "comment_text"].tolist()
    if len(texts) < 10:
        return df  # 样本太少不聚类

    embeds = EMBED_MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    best_k, best_score = 0, -1
    for k in range(5, 9):
        km = KMeans(n_clusters=k, random_state=42).fit(embeds)
        score = silhouette_score(embeds, km.labels_)
        if score > best_score:
            best_score, best_k = score, k

    km = KMeans(n_clusters=best_k, random_state=42).fit(embeds)
    df2 = df.loc[filter_cond].copy()
    df2['cluster'] = km.labels_
    keywords = {}
    for c in range(best_k):
        segs = df2.loc[df2.cluster == c, "comment_text"].tolist()
        vec = TfidfVectorizer(tokenizer=jieba.lcut, max_features=50).fit_transform(segs)
        tfidf_means = np.asarray(vec.mean(axis=0)).ravel()
        top_idx = tfidf_means.argsort()[::-1][:10]
        keywords[c] = " ".join(np.array(TfidfVectorizer(tokenizer=jieba.lcut).fit(segs).get_feature_names_out())[top_idx])

    df.loc[filter_cond, f"{field_prefix}_cluster_theme"] = df2['cluster'].map(keywords)
    return df

conditions = [
    (comments_data['sentiment_category'].isin([1,3]), "positive"),
    (comments_data['sentiment_category'].isin([2,3]), "negative"),
    (comments_data['user_scenario']==1, "scenario"),
    (comments_data['user_question']==1, "question"),
    (comments_data['user_suggestion']==1, "suggestion"),
]
for cond, prefix in conditions:
    comments_data = cluster_and_keywords(comments_data, cond, prefix)


# In[15]:


video_data[["video_id", "product_name"]].to_csv("submit_4/submit_videos.csv", index=None)
comments_data[['video_id', 'comment_id', 'sentiment_category',
               'user_scenario', 'user_question', 'user_suggestion',
               'positive_cluster_theme', 'negative_cluster_theme',
               'scenario_cluster_theme', 'question_cluster_theme',
               'suggestion_cluster_theme']].to_csv(
    "submit_4/submit_comments.csv", index=None)


# In[ ]:





# In[19]:


tmpdf = pd.read_csv('submit_1/submit_videos.csv')
print(tmpdf.shape)
print(tmpdf.head())

tmpdf = pd.read_csv('submit_4/submit_videos.csv')
print(tmpdf.shape)
print(tmpdf.head())

tmpdf = pd.read_csv('submit_1/submit_comments.csv')
print(tmpdf.shape)
print(tmpdf.head())

tmpdf = pd.read_csv('submit_4/submit_comments.csv')
print(tmpdf.shape)
print(tmpdf.head())


# In[17]:


# !zip -r vc_insights_submit_4.zip submit_4/


# In[18]:


# import pickle

# with open('vci_clf_prod.pkl','wb') as tmpfile:
#     pickle.dump(clf_prod, tmpfile)
# with open('vci_clf_com.pkl','wb') as tmpfile:
#     pickle.dump(clf_com, tmpfile)
# with open('vci_EMBED_MODEL.pkl','wb') as tmpfile:
#     pickle.dump(EMBED_MODEL, tmpfile)


# In[ ]:





# In[20]:


tmpdf1 = pd.read_csv('submit_1/submit_videos.csv')

tmpdf2 = pd.read_csv('submit_4/submit_videos.csv')

tmpdf3 = pd.read_csv('submit_1/submit_comments.csv')

tmpdf4 = pd.read_csv('submit_4/submit_comments.csv')

tmpdf2 = tmpdf2.astype(tmpdf1.dtypes.to_dict())

tmpdf4 = tmpdf4.astype(tmpdf3.dtypes.to_dict())


# In[23]:


tmpdf2[tmpdf1.isna()] = np.nan
tmpdf4[tmpdf3.isna()] = np.nan


# In[24]:


print(tmpdf1.shape)
print(tmpdf1.head())
print(tmpdf2.shape)
print(tmpdf2.head())
print(tmpdf3.shape)
print(tmpdf3.head())
print(tmpdf4.shape)
print(tmpdf4.head())


# In[25]:


# tmpdf2.to_csv('submit_5/submit_videos.csv')

# tmpdf4.to_csv('submit_5/submit_comments.csv')


# In[26]:


# !zip -r vc_insights_submit_5.zip submit_5/


```
