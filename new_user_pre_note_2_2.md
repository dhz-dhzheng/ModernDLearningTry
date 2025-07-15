# Datawhale AI夏令营 用户新增预测挑战赛

## 任务目标 Task Objective

预测用户 `did` 是否为首次出现（是否是新用户）。目标是最大化 **F1-score**，适合于正负样本不均衡的场景。

---

## 数据加载与处理 Data Loading & Processing

```python
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./testA_data.csv')
full_df = pd.concat([train_df, test_df], axis=0)
```

* `train.csv`: 包含标签 `is_new_did`
* `testA_data.csv`: 没有标签，需要预测
* 合并后便于统一编码和特征处理

---

## 时间特征提取 Time Features

```python
df['ts'] = pd.to_datetime(df['common_ts'], unit='ms')
df['day'] = df['ts'].dt.day
df['dayofweek'] = df['ts'].dt.dayofweek
df['hour'] = df['ts'].dt.hour
```

* 从毫秒时间戳中提取 `hour`（小时）、`dayofweek`（星期几）、`day`（日期）
* 时间特征有助于识别用户行为模式

---

## 类别特征编码 Categorical Feature Encoding

```python
for feature in cat_features:
    le = LabelEncoder()
    all_values = pd.concat([train_df[feature], test_df[feature]]).astype(str)
    le.fit(all_values)
    train_df[feature] = le.transform(train_df[feature].astype(str))
    test_df[feature] = le.transform(test_df[feature].astype(str))
```

* 用 `LabelEncoder` 将类别型特征转换为数字
* 避免训练集和测试集编码不一致，必须用全体数据拟合

---

## 特征构造 Feature Selection

```python
features = [...所有数值特征...]
X_train = train_df[features]
y_train = train_df['is_new_did']
X_test = test_df[features]
```

* 仅保留数值特征
* 若有 object 类型（如 `'did'`），必须剔除

---

## 阈值优化函数 Find Best Threshold

```python
def find_best_threshold(y_true, y_probs):
    ...
    return best_thr, best_f1
```

* 找出预测概率下能获得最大 F1-score 的阈值
* 对于分类任务，合理选择阈值比默认的 0.5 更重要

---

## 模型训练与交叉验证 LightGBM + StratifiedKFold

```python
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    ...
    model = lgb.train(params, ...)
```

* 5 折交叉验证，确保每一折中标签分布均衡
* 使用 `early_stopping` 防止过拟合

---

## 模型评估与预测 Model Evaluation & Inference

```python
final_thr = np.mean(thresholds)
final_score = f1_score(y_train, oof_preds)
test_preds = model.predict(X_test) / 5
```

* 使用交叉验证得到的平均最佳阈值用于测试集预测
* 最终结果保存在 `submit.csv`

---

## 总结 Summary

| 步骤               | 功能            |
| ---------------- | ------------- |
| 时间特征             | 捕捉用户活跃时段      |
| 类别编码             | 保证模型能处理离散特征   |
| 阈值优化             | 提高 F1-score   |
| Stratified KFold | 保证训练/验证标签比例一致 |
| LightGBM         | 高效的 GBDT 模型   |

---

## 完整代码

```python
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 1. 数据加载
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./testA_data.csv')
submit = test_df[['did']]
full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)


# In[2]:


# 2. 时间处理
for df in [train_df, test_df, full_df]:
    df['ts'] = pd.to_datetime(df['common_ts'], unit='ms')
    df['day'] = df['ts'].dt.day
    df['dayofweek'] = df['ts'].dt.dayofweek
    df['hour'] = df['ts'].dt.hour
    df.drop(columns=['ts'], inplace=True)

# 3. 类别特征编码
cat_features = ['device_brand', 'ntt', 'operator', 'common_country', 'common_province', 'common_city', 'appver', 'channel', 'os_type', 'udmap']
label_encoders = {}
for feat in cat_features:
    le = LabelEncoder()
    le.fit(full_df[feat].astype(str))
    train_df[feat] = le.transform(train_df[feat].astype(str))
    test_df[feat] = le.transform(test_df[feat].astype(str))
    full_df[feat] = le.transform(full_df[feat].astype(str))


# In[3]:


# 4. TF-IDF 相关特征（按 did 聚合 mid/eid 为文本）
for key in ['mid', 'eid']:
    user_seq = full_df.groupby('did')[key].agg(list).astype(str)
    user_freq = user_seq.apply(lambda x: len(set(x.split())))
    user_len = user_seq.apply(lambda x: len(x.split()))
    full_df = full_df.merge(user_freq.rename(f'{key}_unique_cnt'), left_on='did', right_index=True, how='left')
    full_df = full_df.merge(user_len.rename(f'{key}_total_cnt'), left_on='did', right_index=True, how='left')

# 5. 目标编码（对类别特征做 did 粒度编码）
def add_target_encoding(df, target_col, features, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for feat in features:
        df[f'{feat}_te'] = 0
        for train_idx, val_idx in skf.split(df, df[target_col]):
            mean = df.iloc[train_idx].groupby(feat)[target_col].mean()
            df.iloc[val_idx, df.columns.get_loc(f'{feat}_te')] = df.iloc[val_idx][feat].map(mean)
        df[f'{feat}_te'].fillna(df[target_col].mean(), inplace=True)


# In[4]:


train_df_copy = train_df.copy()
add_target_encoding(train_df_copy, 'is_new_did', cat_features)

# 6. 时间差特征（同 did 内事件时间差、排序）
train_df['event_index'] = train_df.groupby('did').cumcount()
train_df['did_event_count'] = train_df.groupby('did')['common_ts'].transform('count')
train_df['ts_diff'] = train_df.groupby('did')['common_ts'].diff().fillna(0)
train_df['ts_diff'] = train_df['ts_diff'].clip(upper=3600000)  # 最大1小时差值

# 7. 聚合统计特征（每个 did 的点击数量、唯一值等）
agg_funcs = {
    'mid': ['nunique', 'count'],
    'eid': ['nunique'],
    'channel': ['nunique'],
    'appver': ['nunique'],
    'os_type': ['nunique'],
    'hour': ['nunique', 'mean'],
}
agg_df = train_df.groupby('did').agg(agg_funcs)
agg_df.columns = ['did_' + '_'.join(col) for col in agg_df.columns]
agg_df.reset_index(inplace=True)
train_df = train_df.merge(agg_df, on='did', how='left')
test_df = test_df.merge(agg_df, on='did', how='left')


# In[8]:


print(train_df.columns)
print(test_df.columns)


# In[12]:


# 8. 特征列选择
new_features = [
    'mid', 'eid', 'device_brand', 'ntt', 'operator',
    'common_country', 'common_province', 'common_city', 'appver', 'channel',
    'common_ts', 'os_type', 'udmap', 'day', 'dayofweek', 'hour',
    'did_mid_nunique', 'did_mid_count', 'did_eid_nunique',
    'did_channel_nunique', 'did_appver_nunique', 'did_os_type_nunique',
    'did_hour_nunique', 'did_hour_mean',
]

X_train = train_df[new_features]
y_train = train_df['is_new_did']
X_test = test_df[new_features]

# 9. 模型训练与预测
def find_best_threshold(y_true, y_probs):
    best_thr, best_f1 = 0.5, 0
    for thr in np.linspace(0.1, 0.4, 7):
        preds = (y_probs >= thr).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_thr, best_f1 = thr, score
    return best_thr, best_f1


# In[13]:


print("非数值列:", X_train.select_dtypes(exclude=['number']).columns.tolist())


# In[14]:


params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'max_depth': 10,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 10,
    'verbose': -1,
    'seed': 42
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))
oof_preds = np.zeros(len(X_train))
thresholds = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"\nFold {fold+1}")
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_val = lgb.Dataset(X_val, y_val)

    model = lgb.train(params, lgb_train, valid_sets=[lgb_val], num_boost_round=1000,
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

    val_probs = model.predict(X_val)
    thr, f1 = find_best_threshold(y_val, val_probs)
    print(f"Best F1: {f1:.5f}, Best threshold: {thr:.2f}")

    thresholds.append(thr)
    oof_preds[val_idx] = (val_probs >= thr).astype(int)
    test_preds += model.predict(X_test) / 5

# 最终评估
final_thr = np.mean(thresholds)
final_score = f1_score(y_train, oof_preds)
print(f"\nFinal F1 Score: {final_score:.5f}, Avg Threshold: {final_thr:.2f}")


# In[15]:


submit['is_new_did'] = (test_preds >= final_thr).astype(int)
submit[['is_new_did']].to_csv('newuser_submit3.csv', index=False)
print("保存成功：submit3.csv")


# In[17]:


import pickle
with open('new_user_model.pkl', 'wb') as tmpfile:
    pickle.dump(model, tmpfile)



```
