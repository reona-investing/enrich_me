#%% パスを通し、モジュールをインポートする
from pathlib import Path
import sys
PROJECT_FOLDER = str(Path(__file__).parents[2])
ORIGINAL_MODULES = PROJECT_FOLDER + '/modules'
sys.path.append(ORIGINAL_MODULES)

import os
import pandas as pd

# %%関数の定義
def convert(file_to_convert: str):
    '''
    investingからダウンロードしてきたcsvファイルをpklにコンバートする関数。
    file_to_convert: 変換したいcsvファイルのフルパスを指定します。
    '''
    # ここに処理を追加
    a = pd.read_csv(file_to_convert)
    a['Date'] = pd.to_datetime(a['Date'])
    new_file_name = file_to_convert.replace('.csv', '.parquet')
    a.to_parquet(new_file_name, index=False)
    # ここに処理が終わった後、元のCSVファイルを削除するコードを追加
    os.remove(file_to_convert)


if __name__ == "__main__":
    for x in ['VDE']:
        file_to_convert = f"C:/Users/ryosh/enrich_me/project/scraped_data/us_sector_etf/{x}_historical.csv"
        convert(file_to_convert)