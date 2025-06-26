# acquisition/features_updater/scrapers/feature_latest_value/factory のクラス仕様書

## factory.py

### class ScraperFactory
各データソースのスクレイパーを生成するファクトリークラス
- create_scraper: 指定されたデータソースのスクレイパーを生成

Args:
    source (Literal): データソース ('Baltic', 'Tradingview', 'ARCA')
    browser_manager (BrowserManager): ブラウザマネージャー
    
Returns:
    BaseLatestValueScraper: 対応するスクレイパーインスタンス
    
Raises:
    ValueError: サポートされていないデータソースが指定された場合

