import pandas as pd
import yfinance as yf  # Yahoo Financeからデータを取得する場合
import datetime

class PriceLimitCalculator:
    """東京証券取引所の値幅制限に基づいて上限価格・下限価格を計算するクラス"""
    
    def __init__(self):
        # 値幅制限テーブルを定義（基準値段: 制限値幅）
        self.price_limits = [
            (100, 30),
            (200, 50),
            (500, 80),
            (700, 100),
            (1000, 150),
            (1500, 300),
            (2000, 400),
            (3000, 500),
            (5000, 700),
            (7000, 1000),
            (10000, 1500),
            (15000, 3000),
            (20000, 4000),
            (30000, 5000),
            (50000, 7000),
            (70000, 10000),
            (100000, 15000),
            (150000, 30000),
            (200000, 40000),
            (300000, 50000),
            (500000, 70000),
            (700000, 100000),
            (1000000, 150000),
            (1500000, 300000),
            (2000000, 400000),
            (3000000, 500000),
            (5000000, 700000),
            (7000000, 1000000),
            (10000000, 1500000),
            (15000000, 3000000),
            (20000000, 4000000),
            (30000000, 5000000),
            (50000000, 7000000),
            (float('inf'), 10000000)  # 50,000,000円以上
        ]
    
    def get_price_limit(self, base_price):
        """基準値段から制限値幅を取得する"""
        for price_threshold, limit in self.price_limits:
            if base_price < price_threshold:
                return limit
        return self.price_limits[-1][1]  # 最大値（通常は到達しない）
    
    def calculate_price_limits(self, base_price):
        """基準値段から上限価格と下限価格を計算する"""
        limit = self.get_price_limit(base_price)
        upper_limit = base_price + limit
        lower_limit = max(base_price - limit, 0)  # 下限は0円未満にならないようにする
        
        return upper_limit, lower_limit

    def calculate_upper_limit(self, base_price):
        """基準値段から上限価格を計算する"""
        limit = self.get_price_limit(base_price)
        upper_limit = base_price + limit
        
        return upper_limit

    def calculate_lower_limit(self, base_price):
        """基準値段から下限価格を計算する"""
        limit = self.get_price_limit(base_price)
        lower_limit = max(base_price - limit, 0)  # 下限は0円未満にならないようにする
        
        return lower_limit
    
    def calculate_for_stocks(self, stock_data):
        """複数銘柄の上限価格と下限価格を計算する"""
        result = stock_data.copy()
        
        # 上限価格と下限価格を計算
        limits = [self.calculate_price_limits(price) for price in result['base_price']]
        result['upper_limit'] = [limit[0] for limit in limits]
        result['lower_limit'] = [limit[1] for limit in limits]
        result['price_limit'] = [self.get_price_limit(price) for price in result['base_price']]
        
        return result

if __name__ == '__main__': 
    plc = PriceLimitCalculator()
    a = plc.calculate_price_limits(1000)
    print(a)
