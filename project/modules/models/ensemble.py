
import pandas as pd

def by_rank(inputs: list[tuple[pd.DataFrame, float]]) -> pd.DataFrame:
    '''
    2つ以上のモデルの結果を予測順位ベースでアンサンブルする。
    Args:
        inputs (list): （予測結果データフレーム, 重み)のタプルを格納したリスト
    Returns:
        pd.DataFrame: アンサンブル後の予測順位を格納したデータフレーム
    '''
    assert len(inputs) > 0, 'inputsには1つ以上の要素を指定してください。'

    ensembled_rank = None
    for pred_result_df, weight in inputs:
        rank = pred_result_df.groupby('Date')['Pred'].rank(ascending=False) * weight
        ensembled_rank = rank if ensembled_rank is None else ensembled_rank + rank

    ensembled_rank_df = pd.DataFrame(ensembled_rank, index=inputs[0][0].index, columns=['Pred'])

    return ensembled_rank_df.groupby('Date')[['Pred']].rank(ascending=False)