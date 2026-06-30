import lightgbm as lgb
import numpy as np
from train_learned_combiner import extract_features_and_targets

X, y, cids = extract_features_and_targets()
y_int = np.round(y * 30.0).astype(int)
group = [len(X)]

train_data = lgb.Dataset(X, label=y_int, group=group)

feature_names = [
    'bm25_norm', 'crossencoder_score', 'co_norm',
    'skill_trust', 'activity_decay', 'recruiter_rr',
    'rt_norm', 'intent_score', 'market_validation',
    'company_scale', 'icr', 'notice_norm',
    'github_norm', 'contact_verified', 'oar'
]

# Set feature_fraction (colsample_bytree) to force using different features
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [10, 20],
    'num_leaves': 16,
    'min_data_in_leaf': 10,
    'learning_rate': 0.05,
    'feature_fraction': 0.7,  # force tree diversity
    'verbose': -1,
    'seed': 204
}

bst = lgb.train(params, train_data, num_boost_round=100)
importance = bst.feature_importance(importance_type='gain')

print("\n=== Feature Importance with feature_fraction = 0.7 ===")
sorted_idx = np.argsort(importance)[::-1]
for idx in sorted_idx:
    print(f"  {feature_names[idx]:<20} : {importance[idx]:.4f}")
