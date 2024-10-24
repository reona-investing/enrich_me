#これを継承することで、クラスがシングルトン化する。
class Singleton:
    def __new__(cls, *args):
        if not hasattr(cls, "_self"):
            cls._self = super().__new__(cls)
        return cls._self

class _FlagManager(Singleton):
    def __init__(self):
        '''定義されていないフラグを初期化'''
        self.flags = [
            'should_take_positions',
            'should_process_stock_price',
            'should_update_historical_data',
            'should_settle_positions',
            'should_fetch_invest_result',
            'should_learn',
            ] 
        for flag in self.flags:
            if not hasattr(self, flag):
                setattr(self, flag, False)
                
    def shows_all_flags(self):
        '''すべてのインスタンス変数とその値を表示'''
        for var, value in self.__dict__.items():
            print(f"{var}: {value}")
    
def launch() -> object:
    FlagManager = _FlagManager()
    return FlagManager