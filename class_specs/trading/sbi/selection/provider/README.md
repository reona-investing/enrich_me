# trading/sbi/selection/provider のクラス仕様書

## price_provider.py

### class PriceProvider
価格情報提供クラス
- __init__: Args:
    order_price_df: 注文価格データフレーム
- get_price_data: 価格データを取得
- get_etf_price: 指定したETFの価格を取得

## sector_provider.py

### class SectorProvider
セクター情報提供クラス
- __init__: Args:
    sector_definitions_csv: セクター定義CSVファイルのパス
- get_sector_definitions: セクター定義情報を取得

## trade_limit_provider.py

### class TradeLimitProvider
取引制限情報提供クラス
- __init__: 

