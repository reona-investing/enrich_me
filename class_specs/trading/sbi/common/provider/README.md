# trading/sbi/common/provider のクラス仕様書

## margin_provider.py

### class SBIMarginProvider
SBI証券から証拠金情報を提供するクラス
- __init__: コンストラクタ

Args:
    browser_manager (SBIBrowserManager): SBI証券のブラウザセッションを管理するオブジェクト

