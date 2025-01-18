
import pandas as pd
from models.dataset import MLDataset

def by_rank(ml_datasets: list[MLDataset], ensemble_rates: list[float]) -> pd.Series:
    '''
    2つ以上のモデルの結果をアンサンブルする（予測順位ベース）
    ml_datasets: アンサンブルしたいモデルのMLDatasetをリストに格納
    ensemble_rates: 各モデルの予測結果を合成する際の重みづけ
    '''
    assert len(ml_datasets) == len(ensemble_rates), "ml_datasetsとensemble_ratesには同じ個数のデータをセットしてください。"
    for i in range(len(ml_datasets)):
        if i == 0:
            ensembled_rank = ml_datasets[i].pred_result_df.groupby('Date')['Pred'].rank(ascending=False) * ensemble_rates[i]
        else:
            ensembled_rank += ml_datasets[i].pred_result_df.groupby('Date')['Pred'].rank(ascending=False) * ensemble_rates[i]
    
    ensembled_rank = pd.DataFrame(ensembled_rank, index=ml_datasets[len(ml_datasets) - 1].pred_result_df.index, columns=['Pred'])

    return ensembled_rank.groupby('Date')[['Pred']].rank(ascending=False)