# facades/data_pipeline のクラス仕様書

## data_update_facade.py

### class DataUpdateFacade
データの更新及び読み込みを担当するファサード
- __init__: 

## machine_learning_facade.py

### class MachineLearningFacade
- __init__: 
- execute: 
- _get_necessary_dfs: 
- _load_model: 
- _train_1st_model: 
- _predict_1st_model: 
- _train_2nd_model: 
- _predict_2nd_model: 
- _ensemble: 
- _update_ensembled_model: 
- _get_features_df: 
- _append_pred_in_1st_model: 

## order_execution_facade.py

### class OrderExecutionFacade
SBI証券でオーダーする
- __init__: 

## trade_data_facade.py

### class TradeDataFacade
取引データを取得するコード
- __init__: 

