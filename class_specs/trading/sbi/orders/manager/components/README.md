# trading/sbi/orders/manager/components のクラス仕様書

## base.py

### class BaseExecutor
Order executor base providing common utilities
- __init__: 
- _get_selector: 

## canceller.py

### class OrderCancellerMixin
Handles order cancellation
- _extract_order_row_data: 

## fetcher.py

### class OrderInfoFetcherMixin
Fetches active orders and positions
- _parse_positions_table: 

## placer.py

### class OrderPlacerMixin
Handles order placement

## settler.py

### class PositionSettlerMixin
Handles position settlement

