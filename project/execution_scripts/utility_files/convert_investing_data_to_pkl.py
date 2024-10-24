#%% パスを通し、モジュールをインポートする
from pathlib import Path
import sys
PROJECT_FOLDER = str(Path(__file__).parents[2])
ORIGINAL_MODULES = PROJECT_FOLDER + '/modules'
sys.path.append(ORIGINAL_MODULES)

import os
import pandas as pd
import data_pickler

# %%関数の定義
def convert(file_to_convert: str):
    '''
    investingからダウンロードしてきたcsvファイルをpklにコンバートする関数。
    file_to_convert: 変換したいcsvファイルのフルパスを指定します。
    '''
    # ここに処理を追加
    a = pd.read_csv(file_to_convert)
    a['Date'] = pd.to_datetime(a['Date'])
    new_file_name = file_to_convert.replace('.csv', '.pkl')
    data_pickler.dump_as_records(a, new_file_name)
    # ここに処理が終わった後、元のCSVファイルを削除するコードを追加
    os.remove(file_to_convert)


if __name__ == "__main__":
    file_to_convert = "H:/マイドライブ/enrich_me/scraped_data/commodity/raw_Nikkei_Rubber_price.csv"
    convert(file_to_convert)