# utils/notifier のクラス仕様書

## slack_notifier.py

### class SlackNotifier
- __init__: 
- send_message: SLACKにメッセージを送ります。
Args:
    message (str): 送信するメッセージ
    channel (str): 送信先のチャンネルID (NoneのときはGeneralが指定される)
- send_error_log: SLACKにエラーログを送ります。
Args:
    message (str): 送信するメッセージ
- send_result: SLACKに当日の結果を送ります。
Args:
    message (str): 送信するメッセージ
- start: SLACKにプログラム開始メッセージを送ります。
Args:
    message (str): 送信するメッセージ
    should_send_program_name (bool): メッセージにプログラム名を含めるか
- finish: SLACKにプログラム終了メッセージを送ります。
Args:
    message (str): 送信するメッセージ

