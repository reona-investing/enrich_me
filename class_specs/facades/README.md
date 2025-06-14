# facades のクラス仕様書

## evaluation_facade.py

### class EvaluationFacade
- __init__: 
- display: 

## sector_ml_datasets_facade.py

### class SectorMLDatasetsFacade
業種単位での学習用データセット作成をまとめて実行するファサード。
- create_datasets: 業種ごとの ``SingleMLDataset`` を生成し ``MLDatasets`` として保存する。

各種パラメータは ``FeaturesSet`` や ``TargetCalculator`` へ渡され、
特徴量計算からデータセットの保存までを一括で行う。

