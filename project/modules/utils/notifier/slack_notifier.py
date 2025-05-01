import time
import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv

load_dotenv()

class SlackNotifier:

    def __init__(self, program_name: str = ''):
        # Slack APIのトークンを環境変数から取得
        self.client = WebClient(token=os.getenv('SLACK_TOKEN') or '')
        self.SLACK_GENERAL = os.getenv('SLACK_GENERAL') or ''
        self.SLACK_ERROR_LOG = os.getenv('SLACK_ERROR_LOG') or ''
        self.SLACK_RESULT = os.getenv('SLACK_RESULT') or ''
        self.program_name = program_name

    def send_message(self, message: str, channel: str = ''):
        '''
        SLACKにメッセージを送ります。
        Args:
            message (str): 送信するメッセージ
            channel (str): 送信先のチャンネルID (NoneのときはGeneralが指定される)
        '''
        try:
            if channel == '':
                channel = self.SLACK_GENERAL
            # Slack APIを使用してメッセージを送信
            response = self.client.chat_postMessage(
                channel=channel,
                text=message
            )
            print(f"メッセージが送信されました: {response['message']['text']}")
        except SlackApiError as e:
            print(f"Slack APIエラー: {e.response['error']}")

    def send_error_log(self, message: str):
        '''
        SLACKにエラーログを送ります。
        Args:
            message (str): 送信するメッセージ
        '''
        self.send_message(message, self.SLACK_ERROR_LOG)

    def send_result(self, message: str):
        '''
        SLACKに当日の結果を送ります。
        Args:
            message (str): 送信するメッセージ
        '''
        self.send_message(message, self.SLACK_RESULT)

    def start(self, message: str, should_send_program_name: bool = False):
        '''
        SLACKにプログラム開始メッセージを送ります。
        Args:
            message (str): 送信するメッセージ
            should_send_program_name (bool): メッセージにプログラム名を含めるか
        '''
        self.start_time = time.time()
        if self.program_name and should_send_program_name:
            message = f'{self.program_name}\n{message}'
        self.send_message(message, self.SLACK_GENERAL)

    def finish(self, message: str):
        '''
        SLACKにプログラム終了メッセージを送ります。
        Args:
            message (str): 送信するメッセージ
        '''
        exec_time = int(time.time() - self.start_time)
        minutes, seconds = divmod(exec_time, 60)
        message = f'{message}\n実行時間： {minutes}分{seconds}秒'
        self.send_message(message, self.SLACK_GENERAL)

if __name__ == '__main__':
    Slack = SlackNotifier(program_name='test')
    message = 'プログラムを開始します。'
    Slack.start(message, should_send_program_name=True)
