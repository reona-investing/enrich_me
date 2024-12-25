import threading

class SingletonMeta(type):
    """シングルトン実現用メタクラス（スレッドセーフ）"""
    _instances = {}
    _lock = threading.Lock()  # スレッドセーフを保証

    def __call__(cls, *args, **kwargs):
        with cls._lock:  # ロックを使って競合を防ぐ
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]