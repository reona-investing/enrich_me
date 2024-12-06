#%% パスを通し、モジュールをインポートする
from pathlib import Path
import sys
PROJECT_FOLDER = str(Path(__file__).parents[2])
ORIGINAL_MODULES = PROJECT_FOLDER + '/modules'
sys.path.append(ORIGINAL_MODULES)

import os
import pandas as pd


folder = 'C:/Users/ryosh/OneDrive/デスクトップ/enrich-me/project/'

all_files = []
for root, dirs, files in os.walk(folder):
    for file in files:
        # ファイルのフルパスを取得
        full_path = os.path.join(root, file)
        all_files.append(full_path)

for file_path in all_files:
    if file_path.endswith('.pkl') or file_path.endswith('.pkl.gz'):
        print(file_path)
        file = data_pickler.load_from_records(file_path)
        file_path = file_path.replace('.pkl.gz','.parquet').replace('.pkl','.parquet')
        file.to_parquet(file_path)