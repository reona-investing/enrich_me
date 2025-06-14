# facades/mode_setting のクラス仕様書

## mode_collection.py

### class DataUpdateMode
データ更新に関するモード

### class MachineLearningMode
機械学習に関するモード

### class OrderExecutionMode
注文実行に関するモード

### class TradeDataFetchMode
データ更新に関するモード

### class ModeCollection
フラグ管理用クラス
- adjust_modes: 
- model_copy: モデルをコピーし、更新後にバリデーションを実行する。

## mode_factory.py

### class ModeFactory
- create_mode_collection: 

## mode_for_strategy.py

### class ModeForStrategy
- generate_mode: 現在時刻に基づいたModeCollectionを返す。
- _select_mode: 現在時刻に基づいてモードを決定する。
- _is_between: 現在時刻が指定した時間範囲内にあるかを判定

