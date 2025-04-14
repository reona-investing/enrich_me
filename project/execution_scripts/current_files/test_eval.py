LASSO_MODEL_PATH = r"C:\Users\ryosh\enrich_me\project\ml_datasets\48sectors_LASSO_learned_in_250413"
LGBM_MODEL_PATH = r"C:\Users\ryosh\enrich_me\project\ml_datasets\48sectors_LGBM_learned_in_250413"
ENSEMBLE_MODEL_PATH = r"C:\Users\ryosh\enrich_me\project\ml_datasets\48sectors_Ensembled_learned_in_250413"


from machine_learning.core.collection import ModelCollection
import pandas as pd
from datetime import datetime
from machine_learning.ensemble import EnsembleUtility
from machine_learning.strategies import EnsembleStrategy

lasso_collection = ModelCollection.load(LASSO_MODEL_PATH)
lgbm_collection = ModelCollection.load(LGBM_MODEL_PATH)

ensemble_collection = EnsembleUtility.ensemble_collections([[lasso_collection, 6.7], [lgbm_collection, 1.3]], method='rank',
                                                           output_path = ENSEMBLE_MODEL_PATH)
ensemble_collection.set_order_price_for_all(lgbm_collection.get_order_prices())
ensemble_collection.set_raw_target_for_all(lgbm_collection.get_raw_targets(), separate_by_sector=False)
test_start_date = datetime(2022, 1, 1)


for name, model_collection in [('LASSO', lasso_collection), ('LGBM', lgbm_collection), ('Ensemble', ensemble_collection)]:
    model_result = model_collection.get_result_df()
    model_result['PredRank'] = model_result.groupby('Date')['Pred'].rank(ascending=False)
    model_raw = model_collection.get_raw_targets()
    model_raw = model_raw.rename(columns={'Target': 'RawTarget'})
    model_df = pd.merge(model_result, model_raw, left_index=True, right_index=True, how='left')
    model_df = model_df[model_df.index.get_level_values('Date')>=test_start_date]
    model_result = (model_df[model_df['PredRank']<=3].groupby('Date')[['RawTarget']].mean() -\
                    model_df[model_df['PredRank']>=46].groupby('Date')[['RawTarget']].mean()) / 2
    model_agg = model_result.describe().T
    model_agg['SR'] = model_agg['mean'] / model_agg['std']
    model_agg = model_agg.T
    print(name)
    print(model_agg)
    print('------------------------------------------')
