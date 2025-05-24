from datetime import datetime

class MergerConfig:
    """企業合併設定"""
    
    # 銘柄コード置換辞書
    CODE_REPLACEMENT = {
        '1334': '1333',
        '8815': '3289',
        '8880': '3291',
        '4555': '4887',
        '5007': '5021',
        '1824': '5076',
        '8385': '5830',
        '8355': '5831',
        '8382': '5832',
        '8369': '5844',
        '4723': '6028',
        '8335': '7167',
        '8394': '7180',
        '8332': '7186',
        '8379': '7337',
        '9062': '9147',
        '9477': '9468',
    }
    
    # 手動調整リスト
    MANUAL_ADJUSTMENTS = [
        {'Code': '1333', 'Date': datetime(2014, 4, 1), 'Rate': 0.1},
        {'Code': '5021', 'Date': datetime(2015, 10, 1), 'Rate': 0.1}
    ]
    
    # 合併情報
    MERGER_INFO = {
        '7167': {
            'Code1': '7167',  # 存続会社
            'Code2': '8333',  # 消滅会社
            'MergerDate': datetime(2016, 12, 19),
            'ExchangeRate': 1.17
        }
    }
    
    # 合併時に加算する財務項目
    ADDITIVE_COLUMNS_FOR_MERGER = {
        'NetSales',
        'OperatingProfit',
        'OrdinaryProfit',
        'Profit',
        'TotalAssets',
        'Equity',
        'CashFlowsFromOperatingActivities',
        'CashFlowsFromInvestingActivities',
        'CashFlowsFromFinancingActivities',
        'CashAndEquivalents',
        'ForecastNetSales',
        'ForecastOperatingProfit',
        'ForecastOrdinaryProfit',
        'ForecastProfit',
        'NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock',
        'NumberOfTreasuryStockAtTheEndOfFiscalYear'
    }