# acquisition/features_updater/mergers のクラス仕様書

## feature_data_merger.py

### class FeatureDataMerger
特徴量データの結合に特化したクラス
- __init__: 
- merge_feature_data: 既存の特徴量データと新しくスクレイピングしたデータを結合
- _merge_data: 2つのデータフレームを結合
- _format_merged_df: 結合されたデータフレームを整形

