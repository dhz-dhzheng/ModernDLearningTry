# Datawhale AI夏令营 用户新增预测挑战赛 基于带货视频评论的用户洞察挑战赛

## 使用电商提示模板+SBERT微调模型

```python
# prompt + SBERT 向量化
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def embed_texts(texts):
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# 产品预测训练
mask = ~video_data["product_name"].isnull()
X_vid = embed_texts(video_data.loc[mask, "text"].tolist())
y_vid = video_data.loc[mask, "product_name"].values

clf_prod = SGDClassifier(loss='log').fit(X_vid, y_vid)
# 预测全部
X_all = embed_texts(video_data["text"].tolist())
video_data["product_name"] = clf_prod.predict(X_all)
```

## 多维情感分析：SBERT + Prompt + 微调

```python

# 构造联合标签
comments_data['y_joint'] = comments_data.apply(
    lambda r: f"{int(r.sentiment_category)}_{int(r.user_scenario)}_{int(r.user_question)}_{int(r.user_suggestion)}",
    axis=1
)

X_com = embed_texts(comments_data["comment_text"].tolist())
y_com = comments_data["y_joint"]

clf_com = SGDClassifier(loss='log').fit(X_com, y_com)
preds = clf_com.predict(X_com)
pcs = np.array([p.split('_') for p in preds], dtype=int)
comments_data[['sentiment_category',
               'user_scenario',
               'user_question',
               'user_suggestion']] = pcs
```
