# models/evaluation のクラス仕様書

## evaluate.py

### class DatetimeManager
- __init__: 対象日を管理するサブモジュール
Args:
    start_date (datetime): 評価開始日
    end_date (datetim): 評価終了日
- extract_duration: 指定した日付のデータを抽出する。

### class LSControlHandler
コントロールデータを読み込む。
type (Literal[str]): コントロールの種類を指定。デフォルトはTOPIX。
- __init__: 
- _load_topix_data: TOPIXデータをpandas.DataFrame形式で読み込む。
- _get_topix_path: TOPIXデータのファイルパスを取得する。
- _calculate_return: リターンを計算する。

### class LSDataHandler
コントロールデータを読み込む
type (Literal[str]): コントロールの種類を指定。デフォルトはTOPIX。
- __init__: 
- _append_rank_cols: データフレームの各列をランク化します。
- _append_raw_target_col: 

### class MetricsCalculator
指標計算を担当
- __init__: Long-Short戦略のモデルの予測結果を格納し、各種指標を計算します。
Args:
    ls_data_handler (LSDataHandler): ロングショート戦略の各種データを保持するクラス
    sectors_to_trade_count (int): 上位（下位）何sectorを取引の対象とするか。
    quantile_bin_count (int):分位リターンのbin数。指定しない場合はsector数が割り当てられる。
    top_sector_weight (float): 最上位（最下位）予想業種にどれだけの比率で傾斜を掛けるか？
- _calculate_all_metrics: 
- calculate_return_by_quantile_bin: 指定したbin数ごとのリターンを算出。
- calculate_longshort_performance: ロング・ショートそれぞれの結果を算出する。
- calculate_longshort_probability: Long, Short, LSの各リターンを算出する。
- calculate_longshort_minus_control: コントロールを差し引いたリターン結果を表示する。
- calculate_monthly_longshort: 月次のリターンを算出
- calculate_daily_sector_performances: 業種ごとの成績を算出する
- calculate_monthly_sector_performances: 業種ごとの月次リターンの算出
- calculate_total_sector_performances: 
- calculate_spearman_correlation: 日次のSpearman順位相関係数と、その平均や標準偏差を算出
- calculate_numerai_correlationss: 日次のnumerai_corrと，その平均や標準偏差を算出
numerai_corr：上位・下位の予測に重みづけされた，より実践的な指標．
- _calc_daily_numerai_corr: 日次のnumerai_corrを算出
- _calc_daily_numerai_rank_corr: 日次のnumerai_corr（Targetをランク化）を算出

### class Visualizer
計算結果の可視化を担当
- __init__: 
- _output_basic_data: 
- display_result: 任意のデータフレームを動的に描画

