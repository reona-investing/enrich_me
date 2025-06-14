# calculation のクラス仕様書

## features_calculator.py

### class FeaturesCalculator
- calculate_features: 特徴量を計算する。
:param pd.DataFrame new_sector_price: セクター価格
:param pd.DataFrame new_sector_list: 各銘柄の業種設定
:param dict stock_dfs_dict: J-quants API由来のデータを格納した辞書
:param bool adopt_features_indices: インデックス系特徴量の採否
:param bool adopt_features_price: 価格系特徴量の採否
:param dict groups_setting: （インデックス）特徴量グループの採否
:param dict names_setting: （インデックス）特徴量の採否
:param str currencies_type: 通貨を'relative'なら相対強度(例：'JPY')、'raw'ならそのまま(例：'USDJPY')
:param str commodity_type: コモディティを円建てに補正するか否か
:param list return_duration: （価格）何日間のリターンを特徴量とするか
:param list mom_duration: （価格）何日間のモメンタムを特徴量とするか
:param list vola_duration: （価格）何日間のボラティリティを特徴量とするか
:param bool adopt_size_factor: （価格）サイズファクターを特徴量とするか
:param bool adopt_eps_factor: （価格）EPSを特徴量とするか
:param bool adopt_sector_categorical: （価格）セクターをカテゴリ変数として採用するか
:param bool add_rank: （価格）各日・各指標のの業種別ランキング
:param Optional[PreprocessingPipeline] indices_preprocessing_pipeline: インデックス系特徴量の前処理パイプライン
:param Optional[PreprocessingPipeline] price_preprocessing_pipeline: 価格系特徴量の前処理パイプライン
- select_features_to_scrape: 
- calculate_features_indices: 特徴量の算出
- _calculate_1day_return: 
- _calculate_1day_return_commodity_JPY: 
- _process_currency: いずれ1d_return以外にも対応したい
- _process_bond: いずれ1d_return以外にも対応したい
- calculate_features_price: 価格系の特徴量を生成する関数。
new_sector_price: 業種別インデックスの価格情報
new_sector_list: 各銘柄の業種設定
stock_dfs_dict: J-Quants API由来のデータを含んだ辞書
return_duration: リターン算出日数をまとめたリスト
mom_duration: モメンタム算出日数をまとめたリスト
vola_duration: ボラティリティ算出日数をまとめたリスト
adopt_size_factor: サイズファクターを特徴量とするか
adopt_eps_factor: EPSを特徴量とするか
adopt_sector_categorical: セクターをカテゴリカル変数として採用するか
add_rank: 各日・各指標のランクを特徴量として追加するか
- merge_features: features_indicesとfeatures_priceを結合

## sector_fin_calculator.py

### class SectorFinCalculator
- __init__: 
- _get_column_names: 
- calculate: Args:
    fin_df (pd.DataFrame): 財務情報
    price_df (pd.DataFrame): 価格情報
    sector_dif_info (pd.DataFrame): セクター
- _weighted_average: 
- _column_name_getter: 指定したカラム名の変換後の名称を取得。

Args:
    yaml_info (dict[str | dict[str | Any]])
    raw_name (str): 変換前のカラム名

Returns:
    str: 変換後のカラム名

