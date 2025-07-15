Datawhale AI夏令营 用户新增预测挑战赛

类别特征目标编码（Target Encoding）
```
# 目标编码函数：每个类别的 is_new_did 平均值
def target_encode(df, feature, target):
    stats = df.groupby(feature)[target].agg(['mean', 'count']).reset_index()
    stats.columns = [feature, f'{feature}_target_mean', f'{feature}_count']
    return stats

# 添加目标编码特征
target_enc_features = ['device_brand', 'channel', 'operator']

for feat in target_enc_features:
    enc_df = target_encode(train_df, feat, 'is_new_did')
    train_df = train_df.merge(enc_df, on=feat, how='left')
    test_df = test_df.merge(enc_df[[feat, f'{feat}_target_mean']], on=feat, how='left')
    test_df[f'{feat}_target_mean'].fillna(train_df[f'{feat}_target_mean'].mean(), inplace=True)
```

聚合统计特征（基于 did, mid）
```
# 统计 did 出现次数
did_count = train_df['did'].value_counts().reset_index()
did_count.columns = ['did', 'did_count']
train_df = train_df.merge(did_count, on='did', how='left')
test_df = test_df.merge(did_count, on='did', how='left')
test_df['did_count'].fillna(1, inplace=True)

# 统计每个 device_brand 的 user 数量（频次）
brand_user_cnt = train_df.groupby('device_brand')['did'].nunique().reset_index()
brand_user_cnt.columns = ['device_brand', 'brand_user_cnt']
train_df = train_df.merge(brand_user_cnt, on='device_brand', how='left')
test_df = test_df.merge(brand_user_cnt, on='device_brand', how='left')
test_df['brand_user_cnt'].fillna(1, inplace=True)
```

TF-IDF 风格的序列统计（简化为计数特征）
```
# 每个 did 的 eid 数量、唯一值数
eid_stats = train_df.groupby('did')['eid'].agg(['count', 'nunique']).reset_index()
eid_stats.columns = ['did', 'eid_count', 'eid_nunique']
train_df = train_df.merge(eid_stats, on='did', how='left')
test_df = test_df.merge(eid_stats, on='did', how='left')
test_df[['eid_count', 'eid_nunique']] = test_df[['eid_count', 'eid_nunique']].fillna(0)
```
