import os
import time
from typing import Callable, Optional
from pathlib import Path

class FileUtils:
    @staticmethod
    def get_downloaded_csv(download_folder: str, retry: int = 10, wait_func: Optional[Callable] = None):
        """ダウンロードフォルダから最新のCSVファイルを取得"""
        for i in range(retry):
            newest_file, _ = FileUtils.get_newest_two_csvs(download_folder)
            print(newest_file)
            if newest_file and newest_file.__str__().endswith('.csv'):
                return newest_file
            if wait_func:
                wait_func(1)
            else:
                time.sleep(1)

    @staticmethod
    def get_newest_two_csvs(directory: str):
        """ディレクトリ内の最新2つのファイルを取得"""
        files = os.listdir(directory)
        if not files:
            return None, None
        csvs = list(Path(directory).glob("*.csv"))
        newest_file = max(csvs, key=os.path.getmtime)
        second_newest_file = sorted(csvs, key=os.path.getmtime, reverse=True)[1] if len(files) > 1 else None
        return newest_file, second_newest_file

