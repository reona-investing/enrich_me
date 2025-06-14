# trading/sbi/selection/selector のクラス仕様書

## stock_selector.py

### class StockSelector
銘柄選択クラス
- __init__: Args:
    pred_result_df: 予測結果データフレーム
    sector_provider: セクター情報プロバイダー
    price_provider: 価格情報プロバイダー
    trade_limit_provider: 取引制限情報プロバイダー
    sector_analyzer: セクター分析
    weight_calculator: ウェイト計算
    order_allocator: 注文配分
    num_sectors_to_trade: 取引するセクター数
    num_candidate_sectors: 候補セクター数
    top_slope: トップセクターの傾斜
- _get_todays_pred_df: 最新日の予測結果を取得
- _convert_orders_to_dataframe: 注文をデータフレームに変換
- _save_orders_to_csv: 注文をCSVに保存

