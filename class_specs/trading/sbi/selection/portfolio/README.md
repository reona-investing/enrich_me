# trading/sbi/selection/portfolio のクラス仕様書

## etf_allocator.py

### class ETFAllocator
ETF配分クラス
- __init__: Args:
    price_provider: 価格情報プロバイダー
- allocate_etfs: ETFを配分

## order_allocator.py

### class OrderAllocator
注文配分クラス
- __init__: Args:
    unit_calculator: 注文単位計算
    etf_allocator: ETF配分
- allocate_orders: 注文を配分
- allocate_balanced_orders: 買い・売りのバランスをとった注文を配分

## unit_calculator.py

### class UnitCalculator
注文単位計算クラス
- __init__: 
- calculate_units: 注文単位数を計算
- _get_ideal_costs: 理想コストを計算
- _draft_portfolio: 仮ポートフォリオを作成
- _reduce_to_max_unit: 最大単位数に制限
- _reduce_units: 予算内に抑える
- _increase_units: 予算を最大限活用
- _calculate_price_limits: 価格上限を計算

