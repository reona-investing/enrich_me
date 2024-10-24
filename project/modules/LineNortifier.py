import requests
import time

import os
from dotenv import load_dotenv
load_dotenv()

class LINEnotifier:

    def __init__(self, program_name: str = None):
        access_token = os.getenv('LINE_TOKEN')
        self.LINE_URL = "https://notify-api.line.me/api/notify"
        self.program_name = program_name
        self.headers = {'Authorization': 'Bearer ' + access_token}

    def send_message(self, message: str):
        payload = {'message': message}
        requests.post(self.LINE_URL, headers=self.headers, params=payload)

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
    LINE = LINEnotifier(program_name='test')
    message = 'プログラムを開始します。'
    LINE.start(message, should_send_program_name=True)