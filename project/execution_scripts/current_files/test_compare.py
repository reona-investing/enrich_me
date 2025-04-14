LASSO_NEW_PATH = r"C:\Users\ryosh\enrich_me\project\ml_datasets\48sectors_LASSO_learned_in_250413"


from machine_learning.core.collection import ModelCollection
import pandas as pd
from datetime import datetime

lasso_collection = ModelCollection.load(LASSO_NEW_PATH)
test_start_date = datetime(2022, 1, 1)


after_collection = lasso_collection
after_result = after_collection.get_result_df()
after_result['PredRank'] = after_result.groupby('Date')['Pred'].rank(ascending=False)
after_raw = after_collection.get_raw_targets()
after_raw = after_raw.rename(columns={'Target': 'RawTarget'})
after_df = pd.merge(after_result, after_raw, left_index=True, right_index=True, how='left')
after_df = after_df[after_df.index.get_level_values('Date')>=test_start_date]
after_result = (after_df[after_df['PredRank']<=3].groupby('Date')[['RawTarget']].mean() -\
                after_df[after_df['PredRank']>=46].groupby('Date')[['RawTarget']].mean()) / 2
after_agg = after_result.describe().T
after_agg['SR'] = after_agg['mean'] / after_agg['std']
after_agg = after_agg.T

after_features_df = []
for sector, model in after_collection.models.items():
    after_features_df.append(after_collection.models[sector].features_test_df)
after_features_df = pd.concat(after_features_df, axis=0).sort_index()


from models.dataset import MLDataset
before_dataset = MLDataset(r'C:\Users\ryosh\enrich_me\project\ml_datasets\48sectors_LASSO_learned_in_250308')
before_features_df = before_dataset.train_test_materials.features_test_df



sector = 'ITインフラ'
after = after_features_df[after_features_df.index.get_level_values('Sector')==sector]
before = before_features_df[before_features_df.index.get_level_values('Sector')==sector]

# 表示
#print(after[['GasOil_1d_return', 'USbond10_1d_return', 'DJ_MedicalEquip_1d_return']].tail(5))

# 値が異なる箇所を検出
diff = before != after

# Trueの位置を抽出してインデックスとカラムを取得
differences = [(idx, col) for idx, row in diff.iterrows() for col, val in row.items() if val]

# 表示
print(differences)