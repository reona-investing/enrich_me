import requests
import time

import os
from dotenv import load_dotenv
load_dotenv()

class SlackNotifier:

    def __init__(self, program_name: str = None):
        self.SLACK_GENERAL = os.getenv('SLACK_GENERAL')  # SlackのWebhook URLを環境変数から取得
        self.SLACK_ERROR_LOG = os.getenv('SLACK_ERROR_LOG')
        self.SLACK_RESULT = os.getenv('SLACK_RESULT')
        self.program_name = program_name

    def send_message(self, message: str):
        payload = {'text': message}  # Slack用のペイロード形式
        requests.post(self.SLACK_GENERAL, json=payload)  # JSON形式でPOSTリクエストを送信

    def send_error_log(self, message: str):
        payload = {'text': message}  # Slack用のペイロード形式
        requests.post(self.SLACK_ERROR_LOG, json=payload)  # JSON形式でPOSTリクエストを送信

    def send_result(self, message: str):
        payload = {'text': message}  # Slack用のペイロード形式
        requests.post(self.SLACK_RESULT, json=payload)  # JSON形式でPOSTリクエストを送信

    def start(self, message: str, should_send_program_name: bool = False):
        self.start_time = time.time()
        if self.program_name and should_send_program_name:
            message = f'{self.program_name}\n{message}'
        self.send_message(message)

    def finish(self, message: str):
        exec_time = int(time.time() - self.start_time)
        minutes, seconds = divmod(exec_time, 60)
        message = f'{message}\n実行時間： {minutes}分{seconds}秒'
        self.send_message(message)

if __name__ == '__main__':
    Slack = SlackNotifier(program_name='test')  # クラス名をSlackNotifierに変更
    message = 'プログラムを開始します。'
    Slack.start(message, should_send_program_name=True)