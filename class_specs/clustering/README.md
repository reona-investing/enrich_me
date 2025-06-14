# clustering のクラス仕様書

## hdbscan_cluster.py

### class HDBSCANCluster
HDBSCAN を用いたクラスタリング処理を担うクラス
- fit: 複数パラメータでクラスタリングを実行し、シルエット係数で最良の結果を返す
- fit_recursive: 再帰的にクラスタリングを実行し、各段階のラベルを保持する

## pca_residuals.py

### class PCAResidualExtractor
指定した主成分数を除去した残差を返す簡易クラス
- __init__: 
- fit: 
- transform: 
- fit_transform: 

## pipeline.py

### class SectorClusteringPipeline
UMAP -> HDBSCAN -> 距離解析 をまとめたパイプライン
- __init__: 
- execute: パイプライン全体を実行し、最終的なクラスタ付与結果を返す

## reducer.py

### class UMAPReducer
UMAP を用いた次元削減処理を行うクラス
- fit_transform: UMAP により次元削減を行い、結果の DataFrame を返す

## sector_clusterer.py

### class SectorClusterer
セクタークラスタリングを補助する互換用クラス
- __init__: 
- apply_umap: 
- apply_hdbscan: 
- apply_recursive_hdbscan: HDBSCAN を再帰的に適用しクラスタを細分化する

