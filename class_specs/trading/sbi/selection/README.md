# trading/sbi/selection のクラス仕様書

## one_stop_stock_selector.py

### class OneStopStockSelector
銘柄選択プロセスのためのファサードクラス。
複数のコンポーネントの初期化と連携を簡略化します。
- __init__: Args:
    order_price_df: 価格データ
    pred_result_df: 予測結果データ
    browser_manager: ブラウザマネージャー
    sector_definitions_path: セクター定義ファイルパス (省略時はデフォルト)
    num_sectors_to_trade: 取引するセクター数
    num_candidate_sectors: 候補セクター数
    top_slope: トップセクターの傾斜
- buy_sectors: 選択された買いセクターのリストを取得
- sell_sectors: 選択された売りセクターのリストを取得

