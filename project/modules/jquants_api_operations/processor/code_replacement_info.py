import pandas as pd
from datetime import datetime
import paths

# 設定値や定数
codes_to_replace_dict = {
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

manual_adjustment_dict_list = [
    {'Code': '1333', 'Date': datetime(2014, 4, 1), 'Rate': 0.1},
    {'Code': '5021', 'Date': datetime(2015, 10, 1), 'Rate': 0.1}
]
codes_to_merge_dict = {
    '7167': {'Code1': '7167', 'Code2': '8333', 'MergerDate': datetime(2016, 12, 19), 'ExchangeRate': 1.17}
}