# calculation/features/facades のクラス仕様書

## features_facade.py

### class FeaturesFacade
特徴量データフレーム作成のファサードクラス
- __init__: 初期化
- create_features_dataframe: 特徴量データフレームを作成

Args:
    stock_filter: 株式データフィルタ条件
    sector_redefinitions_csv: セクター再定義CSVファイルパス
    sector_index_parquet: セクターインデックス出力パーケットファイルパス
    use_index_features: インデックス系特徴量を使用するか
    use_price_features: 価格系特徴量を使用するか
    groups_setting: インデックス特徴量グループ設定
    names_setting: インデックス特徴量名称設定
    currencies_type: 通貨処理タイプ
    commodity_type: コモディティ処理タイプ
    adopt_1d_return: 1日リターンを採用するか
    mom_duration: モメンタム計算期間
    vola_duration: ボラティリティ計算期間
    adopt_size_factor: サイズファクターを採用するか
    adopt_eps_factor: EPSファクターを採用するか
    adopt_sector_categorical: セクターカテゴリを採用するか
    add_rank: ランキングを追加するか
    index_preprocessing: インデックス系前処理パイプライン
    price_preprocessing: 価格系前処理パイプライン
    
Returns:
    結合された特徴量データフレーム
    
Raises:
    ValueError: 必要なパラメータが不足している場合
- _get_stock_data: 株式データの取得
- _prepare_sector_data: セクターデータの準備
- _create_index_features: インデックス系特徴量の作成
- _create_price_features: 価格系特徴量の作成

