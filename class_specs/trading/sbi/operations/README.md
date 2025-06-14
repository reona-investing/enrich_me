# trading/sbi/operations のクラス仕様書

## history_manager.py

### class HistoryManager
- __init__: 取引履歴管理クラス
- _add_previous_mtext: 
- _get_conract_summary_df: 
- _format_contracts_df: 
- _format_cashflow_transactions_df: 
- _append_spot_to_list: 

## position_manager.py

### class PositionManager
- __init__: ポジション管理クラス
- _load_data: ポジションデータをCSVファイルから読み込む
- _save_data: ポジションデータをCSVファイルに保存
- _prepare_rows_for_csv: CSVに書き込むための行データを準備
- _trade_params_to_dict: TradeParametersオブジェクトを辞書に変換
- add_position: 新しいポジションを追加し、追加したポジションのインデックスを返す

Args:
    order_params (TradeParameters): 注文パラメータ
    
Returns:
    int: 追加したポジションのインデックス
- update_status: 特定のポジションのステータスを更新
- status_type: 'order_status' または 'settlement_status'
- find_unordered_position_by_params: 指定したパラメータのポジションのリスト内での位置を取得
比較対象は'symbol_code', 'trade_type', 'unit'の3つ
- update_order_id: 発注時IDを更新
- update_by_symbol: 指定された銘柄コードに一致するレコードの特定フィールドを更新します。
複数のレコードが一致した場合は、そのすべてに対して更新を行います。

Args:
    symbol_code (str): 更新対象レコードを選択するためのシンボルコード
    update_key (str): 更新対象とするデータのキー
    update_value (str | int | float): 更新対象とするデータの更新後の値

Returns:
    bool: 1件以上のレコードが更新された場合はTrue、更新対象がない場合はFalse
- get_all_positions: 全ポジション情報を取得
- get_pending_positions: 発注待ちのポジションを取得
- order_status: '未発注'
- settlement_status: '未発注'
- get_open_positions: 決済発注待ちのポジションを取得
- order_status: '発注済'
- settlement_status: '未発注'
- remove_waiting_order: 指定した注文IDのデータを削除

## trade_parameters.py

### class TradeParameters
- validate_positive_price: 価格が正の数であることを確認
- validate_period_value: 期間指定時に日付が設定されているかを確認
- validate_order_type_value: order_typeに応じてorder_type_valueの選択肢を絞る
- validate_limit_order_price: order_typeが指値のとき、limit_order_priceを必須入力とする
- validate_stop_order_trigger_price: order_typeが逆指値の時、stop_order_trigger_priceを必須入力とする
- validate_stop_order_price: order_typeが逆指値でstop_order_typeが指値のとき、stop_order_priceを必須入力とする
- validate_period_value_or_index: period_typeが期間指定のとき、period_valueまたはperiod_indexのいずれかを必ず入力する

## trade_possibility_manager.py

### class TradePossibilityManager
- __init__: 取引制限情報の管理クラス
- _remove_files_in_download_folder: 
- _convert_csv_to_df: 

