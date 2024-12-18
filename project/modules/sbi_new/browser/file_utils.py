import os
import time
from typing import Callable, Optional

class FileUtils:
    @staticmethod
    def get_downloaded_csv(download_folder: str, retry: int = 10, wait_func: Optional[Callable] = None):
        """ダウンロードフォルダから最新のCSVファイルを取得"""
        for i in range(retry):
            newest_file, _ = FileUtils.get_newest_two_files(download_folder)
            if newest_file and newest_file.endswith('.csv'):
                return newest_file
            if wait_func:
                wait_func(1)
            else:
                time.sleep(1)

    @staticmethod
    def get_newest_two_files(directory: str):
        """ディレクトリ内の最新2つのファイルを取得"""
        files = os.listdir(directory)
        if not files:
            return None, None
        full_paths = [os.path.join(directory, f) for f in files]
        newest_file = max(full_paths, key=os.path.getmtime)
        second_newest_file = sorted(full_paths, key=os.path.getmtime, reverse=True)[1] if len(files) > 1 else None
        return newest_file, second_newest_file

