#%% モジュールのインポート
# パスを通す
if __name__ == '__main__':
    from pathlib import Path
    import sys
    PROJECT_FOLDER = str(Path(__file__).parents[2])
    ORIGINAL_MODULES = PROJECT_FOLDER + '/modules'
    sys.path.append(ORIGINAL_MODULES)
# 使用するモジュール
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, norm, t, chi2
import quantstats as qs
from IPython.display import HTML, display

import paths
import data_pickler
import MLDataset
import evaluate_model
import calculate_stats

#%% サブ関数
def calculate_return_topix(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    '''コントロールとしてTOPIXのリターンを算出'''
    #データ読み込み
    df = pd.read_csv(paths.FEATURES_TO_SCRAPE_CSV)
    topix_folder = df.loc[df['Name']=='TOPIX', 'Group'].values[0]
    topix_pkl = df.loc[df['Name']=='TOPIX', 'Path'].values[0]
    topix_path = paths.SCRAPED_DATA_FOLDER + '/' + topix_folder + '/' + topix_pkl
    topix_df = pd.read_parquet(topix_path)
    #リターン算出
    return_topix = pd.DataFrame()
    return_topix['Date'] = pd.to_datetime(topix_df['Date'])
    return_topix['TOPIX_daytime'] = topix_df['Close'] / topix_df['Open'] - 1
    return_topix['TOPIX_hold'] = topix_df['Close'] / topix_df['Open'].iat[0]
    return_topix['TOPIX_hold'] = return_topix['TOPIX_hold'].pct_change(1)
    return_topix = return_topix[(return_topix['Date'] >= start_date) & (return_topix['Date'] <= end_date)]
    return return_topix.set_index('Date')


#%% メイン関数
def summarize_ls_model(strategy1_name: str, strategy2_name: str, apply_benchmark: bool,  
                    model1_path: str, model2_path: str, 
                    start_date: datetime, end_date: datetime):
    '''
    モデルのテスト結果を要約してhtmlとして出力
    strategy1_name: メインストラテジーの名前
    strategy2_name: 比較対象のストラテジーの名前
    apply_benchmark: メインストラテジーとベンチマークとの比較を行うかどうか
    model1_path: メインストラテジーのモデルのパス
    model2_path: 比較対象のストラテジーのモデルのパス
    start_date: 開始日
    end_date: 終了日
    '''
    model1 = MLDataset.MLDataset(model1_path)
    model2 = MLDataset.MLDataset(model2_path)

    model1_obj = evaluate_model.LongShortModel(model_name = strategy1_name, pred_result_df = model1.pred_result_df, raw_target_df = model1.raw_target_df,
                                start_date = start_date, end_date = end_date, bin_num = 5)
    model2_obj = evaluate_model.LongShortModel(model_name = strategy2_name, pred_result_df = model2.pred_result_df, raw_target_df = model1.raw_target_df,
                                start_date = start_date, end_date = end_date, bin_num = 5)

    #storategy1とstorategy2の比較
    title = f'{strategy1_name} vs {strategy2_name}'
    html_strategy1_vs_strategy2 = f'{paths.SUMMARY_REPORTS_FOLDER}/Summary_{strategy1_name}_vs_{strategy2_name}.html'
    qs.reports.html(model1_obj.dataframes_dict['日次成績']['LS'], 
                    benchmark=model2_obj.dataframes_dict['日次成績']['LS'], 
                    output=html_strategy1_vs_strategy2,
                    title=title)

    if apply_benchmark:
        benchmark_name = 'TOPIX_hold'
        return_topix = calculate_return_topix(start_date=start_date, end_date=end_date)

        #storategy1とベンチマーク (TOPIX)の比較
        title = f'{strategy1_name} vs {benchmark_name}'
        html_strategy1_vs_benchmark = f'{paths.SUMMARY_REPORTS_FOLDER}/Summary_{strategy1_name}_vs_{benchmark_name}.html'
        qs.reports.html(model1_obj.dataframes_dict['日次成績']['LS'], 
                        benchmark=return_topix['TOPIX_hold'], 
                        output=html_strategy1_vs_benchmark,
                        title=title)
        
        return html_strategy1_vs_strategy2, html_strategy1_vs_benchmark

    return html_strategy1_vs_strategy2

#%% パラメータの設定
if __name__ == '__main__':
    #ストラテジーの名前
    strategy1_name = '48Sectors_Ensembled'
    strategy2_name = '48Sectors'
    #ベンチマークの採否
    apply_benchmark = True
    #モデルのデータセットのパス
    model1_path = f'{paths.ML_DATASETS_FOLDER}/LGBM_New48sectors_Ensembled'
    model2_path = f'{paths.ML_DATASETS_FOLDER}/New48sectors'
    #テストの開始日と終了日
    start_date = datetime(2022,1,1)
    end_date = datetime.today()

#%% htmlの出力
if __name__ == '__main__':
    html_strategy1_vs_strategy2, html_strategy1_vs_benchmark = \
        summarize_ls_model(strategy1_name, strategy2_name, apply_benchmark,  
                           model1_path, model2_path, 
                           start_date, end_date)

#%% ストラテジー1と2の比較
if __name__ == '__main__':
    display(HTML(html_strategy1_vs_strategy2))

#%% ストラテジー1とベンチマークの比較
if __name__ == '__main__':
    if html_strategy1_vs_benchmark is not None:
        display(HTML(html_strategy1_vs_benchmark))

