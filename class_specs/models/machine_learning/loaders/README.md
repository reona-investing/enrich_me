# models/machine_learning/loaders のクラス仕様書

## loader.py

### class DatasetLoader
``MLDatasets`` の作成・読み込みをまとめたユーティリティクラス.
- __init__: 
- create_grouped_datasets: グループ単位で ``SingleMLDataset`` を作成し ``MLDatasets`` として保存する。
- load_datasets: 保存済み ``SingleMLDataset`` 群から ``MLDatasets`` を復元する.
- load_pred_results: dataset_root内の全モデルの ``pred_result_df`` を読み込み結合して返す.
- load_raw_targets: dataset_root内の全モデルの ``raw_target_df`` を読み込み結合して返す.

