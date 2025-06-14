# models/machine_learning/loaders のクラス仕様書

## loader.py

### class DatasetLoader
``MLDatasets`` の作成・読み込みをまとめたユーティリティクラス.
- __init__: 
- create_grouped_datasets: グループ単位で ``SingleMLDataset`` を作成し ``MLDatasets`` として保存する。
- load_datasets: 保存済み ``SingleMLDataset`` 群から ``MLDatasets`` を復元する.

