# calculation/features/implementation のクラス仕様書

## index_features.py

### class IndexFeatures
インデックス系特徴量計算クラス
- __init__: 初期化
- calculate_features: インデックス系特徴量を計算し、self.features_dfを更新

Args:
    groups_setting: 特徴量グループの採否設定
    names_setting: 特徴量の採否設定
    currencies_type: 通貨の処理方法
    commodity_type: コモディティの処理方法
    preprocessing_pipeline: 前処理パイプライン (任意)
    
Returns:
    計算された特徴量データフレーム
- apply_preprocessing: 前処理パイプラインを適用し、self.features_dfを更新

Args:
    pipeline: 前処理パイプライン
    
Returns:
    前処理後の特徴量データフレーム
- _select_features_to_scrape: スクレイピング対象特徴量の選択
- _calculate_indices_features: インデックス特徴量の計算メイン処理
- _calculate_1day_return: 1日リターンの計算
- _calculate_1day_return_commodity_JPY: コモディティの円建て1日リターン計算
- _post_process_features: 通貨・債券特徴量の後処理
- _process_currency_relative_strength: 通貨の相対強度計算
- _process_bond_spreads: 債券スプレッドの計算

## price_features.py

### class PriceFeatures
価格系特徴量計算クラス
- __init__: 初期化
- calculate_features: 価格系特徴量を計算し、self.features_dfを更新

Args:
    new_sector_price: セクター価格データ
    new_sector_list: セクターリストデータ
    stock_dfs_dict: 株価データ辞書
    adopt_1d_return: 1日リターンを採用するか
    mom_duration: モメンタム計算期間
    vola_duration: ボラティリティ計算期間
    adopt_size_factor: サイズファクターを採用するか
    adopt_eps_factor: EPSファクターを採用するか
    adopt_sector_categorical: セクターカテゴリを採用するか
    add_rank: ランキングを追加するか
    preprocessing_pipeline: 前処理パイプライン (任意)
    
Returns:
    計算された特徴量データフレーム
- apply_preprocessing: 前処理パイプラインを適用し、self.features_dfを更新

Args:
    pipeline: 前処理パイプライン
    
Returns:
    前処理後の特徴量データフレーム
- _add_return_features: リターン特徴量をself.features_dfに追加
- _add_momentum_features: モメンタム特徴量をself.features_dfに追加
- _add_volatility_features: ボラティリティ特徴量をself.features_dfに追加
- _add_size_factor: サイズファクターをself.features_dfに追加
- _add_eps_factor: EPSファクターをself.features_dfに追加
- _add_sector_categorical: セクターカテゴリ変数をself.features_dfに追加

