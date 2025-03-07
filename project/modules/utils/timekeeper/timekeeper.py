import time
import asyncio
from functools import wraps
from contextlib import contextmanager


def timekeeper(func):
    """
    関数の実行時間を測定するデコレーター（同期/非同期関数両対応）。
    
    :param func: 実行時間を測定する対象の関数（同期/非同期の両方に対応）。
    :return: 実行時間を出力するラップされた関数。
    """
    if asyncio.iscoroutinefunction(func):  # 関数が async なら非同期処理として扱う
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)  # async 関数は await で実行
            end_time = time.time()
            print(f"Execution time for {func.__name__}: {end_time - start_time:.4f} seconds")
            return result
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)  # 通常の関数はそのまま実行
            end_time = time.time()
            print(f"Execution time for {func.__name__}: {end_time - start_time:.4f} seconds")
            return result
        return sync_wrapper

@contextmanager
def measure_block_time(block_name="コードブロック"):
    """
    任意のコードブロックの実行時間を測定するためのコンテキストマネージャ。
    
    パラメータ:
        block_name (str): 測定対象のブロックの名前。

    戻り値:
        None
    """
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"Execution time for {block_name}: {end_time - start_time:.4f} seconds")

class Timer:
    """
    実行時間を手動で測定するためのシンプルなタイマークラス。
    """
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """タイマーを開始する。"""
        self.start_time = time.time()

    def stop(self):
        """タイマーを停止する。"""
        if self.start_time is None:
            raise ValueError("タイマーが開始されていません。")
        self.end_time = time.time()
        return self.elapsed_time()

    def elapsed_time(self):
        """経過時間を計算する。"""
        if self.start_time is None:
            raise ValueError("タイマーが開始されていません。")
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def reset(self):
        """タイマーをリセットする。"""
        self.start_time = None
        self.end_time = None
