# trading/sbi_trading_logic のクラス仕様書

## history_updater.py

### class HistoryUpdater
- __init__: 
- load_information: 取引情報、買付余力、総入金額を更新せず読み込み
returns:
    pd.DataFrame: 過去の取引履歴
    pd.DataFrame: 信用建余力の履歴
    pd.DataFrame: 入出金の履歴
    float: 直近の損益額
    str: 直近の損益率
- _show_latest_result: 

## price_limit_calculator.py

### class PriceLimitCalculator
東京証券取引所の値幅制限に基づいて上限価格・下限価格を計算するクラス
- __init__: 
- get_price_limit: 基準値段から制限値幅を取得する
- calculate_price_limits: 基準値段から上限価格と下限価格を計算する
- calculate_upper_limit: 基準値段から上限価格を計算する
- calculate_lower_limit: 基準値段から下限価格を計算する
- calculate_for_stocks: 複数銘柄の上限価格と下限価格を計算する

