# calculation/facades のクラス仕様書

## calculator_facade.py

### class CalculatorFacade
SectorIndexとFeaturesCalculatorを組み合わせて
一連の処理を実行するファサードクラス
- calculate_all: セクターインデックス計算と特徴量計算を一連で実行する

Args:
    stock_dfs (dict): 'list', 'fin', 'price'キーを持つ株価データ辞書
    sector_redefinitions_csv (str): セクター定義CSVファイルのパス
    sector_index_parquet (str): セクターインデックス出力用parquetファイルのパス
    adopts_features_indices (bool): インデックス系特徴量の採否
    adopts_features_price (bool): 価格系特徴量の採否
    groups_setting (dict, optional): 特徴量グループの採否設定
    names_setting (dict, optional): 特徴量の採否設定
    currencies_type (str): 通貨処理方法 ('relative' or 'raw')
    commodity_type (str): コモディティ処理方法 ('JPY' or 'raw')
    adopt_1d_return (bool): 1日リターンを特徴量とするか
    mom_duration (list, optional): モメンタム算出日数リスト
    vola_duration (list, optional): ボラティリティ算出日数リスト
    adopt_size_factor (bool): サイズファクターを特徴量とするか
    adopt_eps_factor (bool): EPSを特徴量とするか
    adopt_sector_categorical (bool): セクターをカテゴリ変数として採用するか
    add_rank (bool): 各日・各指標の業種別ランキングを追加するか
    indices_preprocessing_pipeline (PreprocessingPipeline, optional): インデックス系特徴量の前処理パイプライン
    price_preprocessing_pipeline (PreprocessingPipeline, optional): 価格系特徴量の前処理パイプライン
    
Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        - new_sector_price: セクターインデックス価格データ
        - order_price_df: 発注用個別銘柄データ
        - features_df: 特徴量データ

