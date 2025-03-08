#%% パスを通し、モジュールをインポートする
from pathlib import Path
import sys
PROJECT_FOLDER = str(Path(__file__).parents[2])
ORIGINAL_MODULES = PROJECT_FOLDER + '/modules'
sys.path.append(ORIGINAL_MODULES)

import os
import pandas as pd

# %%関数の定義

def format_investing_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.iloc[:, :5].copy()
    df.columns = ['Date', 'Close', 'Open', 'High', 'Low']
    df['Date'] = pd.to_datetime(df['Date'])
    
    for col in ["Open", "Close", "High", "Low"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "").astype(float)  # カンマを削除しfloat型に変換

    return df[['Date', 'Open', 'Close', 'High', 'Low']].sort_values(by = 'Date', ascending = True).reset_index(drop=True)


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
    import pandas as pd
    import glob
    import os

    # 指定ディレクトリ
    directory = r"C:\Users\ryosh\Downloads\test"

    # CSVファイルのパスを取得
    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    # 確認
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df = format_investing_df(df)
        df.to_csv(csv_file, index=False)
        print(df.tail(2))
        convert(csv_file)
