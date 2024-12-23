import traceback

def retry(max_attempts: int = 3, delay: float = 3.0):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    tb = traceback.format_exc()
                    print(f"エラーが発生しました: {e}. リトライ中... (試行回数: {attempts})")
                    print(tb)
            print(f"{func.__name__}は最大試行回数に達しました。")
        return wrapper
    return decorator
