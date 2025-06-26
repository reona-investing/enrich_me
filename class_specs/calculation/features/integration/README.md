# calculation/features/integration のクラス仕様書

## features_set.py

### class FeaturesSet
特徴量データの結合管理に特化したクラス
- __init__: 初期化時はインスタンス生成のみ
- combine_features: 計算済み特徴量を持つ計算器インスタンスから特徴量を結合

Args:
    index_calculator: インデックス系特徴量計算器（計算済み）
    price_calculator: 価格系特徴量計算器（計算済み）
    
Returns:
    結合された特徴量データフレーム
- combine_from_calculators: 計算器インスタンスから特徴量を計算して結合

Args:
    index_calculator: インデックス特徴量計算器
    price_calculator: 価格特徴量計算器
    new_sector_price: セクター価格データ（価格系計算時に必要）
    new_sector_list: セクターリストデータ（価格系計算時に必要）
    stock_dfs_dict: 株価データ辞書（価格系計算時に必要）
    index_params: インデックス系特徴量計算パラメータ
    price_params: 価格系特徴量計算パラメータ
    
Returns:
    結合された特徴量データフレーム
- _merge_features: インデックス系と価格系特徴量の結合

Args:
    indices_features: インデックス系特徴量データフレーム
    price_features: 価格系特徴量データフレーム
    
Returns:
    結合された特徴量データフレーム

