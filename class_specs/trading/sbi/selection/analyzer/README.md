# trading/sbi/selection/analyzer のクラス仕様書

## sector_analyzer.py

### class SectorAnalyzer
セクター分析クラス
- analyze_sector_predictions: セクター予測を分析
- select_sectors_to_trade: 取引対象セクターを選択

## tradability_analyzer.py

### class TradabilityAnalyzer
取引可能性分析クラス
- __init__: Args:
    sector_provider: セクター情報プロバイダー
    trade_limit_provider: 取引制限情報プロバイダー
- _count_tradable_by_sector: セクターごとの取引可能銘柄数を集計
- _count_total_by_sector: セクターごとの合計銘柄数を集計

## weight_calculator.py

### class WeightCalculator
ウェイト計算クラス
- calculate_weights: 銘柄ウェイトを計算
- _calc_index_weight_components: 終値時点での単位株購入コスト、過去5営業日の出来高平均、時価総額を算出
- _merge_sector_and_price_data: セクター情報と価格データを結合
- _calculate_sector_weights: セクター内での銘柄ウェイトを計算

