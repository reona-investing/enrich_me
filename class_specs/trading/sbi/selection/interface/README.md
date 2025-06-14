# trading/sbi/selection/interface のクラス仕様書

## analyzer.py

### class ISectorAnalyzer
セクター分析インターフェース
- analyze_sector_predictions: セクター予測を分析
- select_sectors_to_trade: 取引対象セクターを選択

### class IWeightCalculator
ウェイト計算インターフェース
- calculate_weights: 銘柄ウェイトを計算

## dataclasses.py

### class SectorPrediction
セクター予測結果

### class StockWeight
銘柄ウェイト情報

### class OrderUnit
注文単位情報

## portfolio.py

### class IOrderAllocator
注文配分インターフェース
- allocate_orders: 注文を配分

## provider.py

### class ISectorProvider
セクター情報提供インターフェース
- get_sector_definitions: セクター定義情報を取得

### class IPriceProvider
価格情報提供インターフェース
- get_price_data: 価格データを取得
- get_etf_price: 指定したETFの価格を取得

### class ITradeLimitProvider
取引制限情報提供インターフェース

## selector.py

### class IStockSelector
銘柄選択インターフェース

