
import pandas as pd
from models.dataset import MLDataset

def by_rank(datasets: list[MLDataset], ensemble_rates: list[float]) -> pd.Series:
    '''
    2つ以上のモデルの結果をアンサンブルする（予測順位ベース）
    ml_datasets: アンサンブルしたいモデルのMLDatasetをリストに格納
    ensemble_rates: 各モデルの予測結果を合成する際の重みづけ
    '''
    assert len(datasets) == len(ensemble_rates), "ml_datasetsとensemble_ratesには同じ個数のデータをセットしてください。"
    assert len(datasets) != 0, 'datasetsとensemble_ratesには1つ以上の要素を指定してください。'
    for i in range(len(datasets)):
        if i == 0:
            ensembled_rank = datasets[i].evaluation_materials.pred_result_df.groupby('Date')['Pred'].rank(ascending=False) * ensemble_rates[i]
        else:
            ensembled_rank += datasets[i].evaluation_materials.pred_result_df.groupby('Date')['Pred'].rank(ascending=False) * ensemble_rates[i]
    
    ensembled_rank = pd.DataFrame(ensembled_rank, index=datasets[len(datasets) - 1].evaluation_materials.pred_result_df.index, columns=['Pred'])

    return ensembled_rank.groupby('Date')[['Pred']].rank(ascending=False)