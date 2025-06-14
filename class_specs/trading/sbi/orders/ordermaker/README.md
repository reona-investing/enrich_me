# trading/sbi/orders/ordermaker のクラス仕様書

## batch_order_maker.py

### class BatchOrderMaker
一括注文を行うクラス
- _save_failed_orders: 失敗した注文をCSVに保存する

## order_maker.py

### class OrderMaker
注文組立の基底クラス
- __init__: 
- create_order_request: 注文リクエストを作成する
- _get_margin_trade_section: 信用取引区分を適正化する
- _calculate_short_selling_limit_price: 空売り制限価格を計算する

## position_settler.py

### class PositionSettler
ポジション決済を行うクラス

