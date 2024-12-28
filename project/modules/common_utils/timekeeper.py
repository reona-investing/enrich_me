import time
from contextlib import contextmanager

def timekeeper(func):
    """
    関数の実行時間を測定するデコレーター。
    
    パラメータ:
        func (callable): 実行時間を測定する対象の関数。

    戻り値:
        callable: 実行時間をログ出力するラップされた関数。
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time for {func.__name__}: {end_time - start_time:.4f} seconds")
        return result

    return wrapper

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
