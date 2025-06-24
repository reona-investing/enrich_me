# evaluation のクラス仕様書

## trade_history_analyzer.py

### class TradeHistoryAnalyzer
- __init__: 
- calculate_daily_profit: 日次の利益を算出する。
- calculate_daily_profit_by_sector: 日次の業種別利益を算出する。
- calculate_daily_leveraged: 
- calculate_daily_cumulative: 
- calculate_daily_leveraged_cumurative: 
- calculate_monthly_profit: 
- calculate_monthly_profit_by_sector: 
- calculate_monthly_leveraged: 
- calculate_total_profit: 
- calculate_total_profit_by_sector: 
- calculate_total_leveraged: 
- display_result: 任意のデータフレームを動的に描画(改善版)
- plot_graphs: 任意のグラフを動的に描画

## returns_analyzer.py

### class ReturnDataHandler
- __init__: リターンデータを読み込み、期間で抽出する。

### class ReturnMetricsCalculator
- __init__: ReturnDataHandlerを受け取り、各種指標を計算する。税率(`tax_rate`)とレバレッジ(`leverage`)を指定可能。
- calculate_daily_returns: 日次統計量を計算する。
- calculate_cumulative_returns: 累積リターンとドローダウンを計算する。
- calculate_monthly_returns: 月次リターンを計算する。

計算される主な指標は以下の通り。
* 日次平均リターン
* 年率換算リターン
* 日次リターン標準偏差
* 年率換算標準偏差
* シャープレシオ
* 最大ドローダウン（実績）
* 最大ドローダウン（理論）
これらは税引前、税引後、税引後＆レバレッジ込みの3パターンで計算される。

### class ReturnVisualizer
- __init__:
- display_result: ウィジェットを用いてデータフレームを表示する。

