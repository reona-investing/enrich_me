import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.stats import norm


def by_rank(inputs: List[Tuple[pd.DataFrame, float]]) -> pd.DataFrame:
    """
    複数のモデルの結果を予測順位ベースでアンサンブルする
    
    Args:
        inputs: (予測結果データフレーム, 重み)のタプルを格納したリスト
        
    Returns:
        アンサンブル後の予測結果のデータフレーム
    """
    assert len(inputs) > 0, 'inputsには1つ以上の要素を指定してください。'

    ensembled_rank = None
    for pred_result_df, weight in inputs:
        # 日付ごとに順位付け
        rank = pred_result_df.groupby('Date')['Pred'].rank(ascending=False) * weight
        ensembled_rank = rank if ensembled_rank is None else ensembled_rank + rank

    # 結果をデータフレーム化
    ensembled_rank_df = pd.DataFrame(ensembled_rank, index=inputs[0][0].index, columns=['Pred'])
    
    # 日付ごとに再ランク付け
    return ensembled_rank_df.groupby('Date')[['Pred']].rank(ascending=False)


def weighted_average(inputs: List[Tuple[pd.DataFrame, float]]) -> pd.DataFrame:
    """
    複数のモデルの結果を重み付き平均でアンサンブルする
    
    Args:
        inputs: (予測結果データフレーム, 重み)のタプルを格納したリスト
        
    Returns:
        アンサンブル後の予測結果のデータフレーム
    """
    assert len(inputs) > 0, 'inputsには1つ以上の要素を指定してください。'
    
    weighted_sum = None
    total_weight = 0
    
    for pred_result_df, weight in inputs:
        pred_weighted = pred_result_df['Pred'] * weight
        weighted_sum = pred_weighted if weighted_sum is None else weighted_sum + pred_weighted
        total_weight += weight
    
    # 重みの合計で割って正規化
    if total_weight > 0:
        weighted_sum = weighted_sum / total_weight
    
    # 結果をデータフレーム化
    return pd.DataFrame(weighted_sum, index=inputs[0][0].index, columns=['Pred'])


def stacking(base_predictions: List[pd.DataFrame], 
            meta_model,
            target_df: pd.DataFrame,
            train_indices: List,
            test_indices: List) -> pd.DataFrame:
    """
    スタッキングによるアンサンブル
    
    Args:
        base_predictions: ベースモデルの予測結果のリスト
        meta_model: メタモデル（学習済み）
        target_df: 目的変数のデータフレーム
        train_indices: 学習に使用するインデックス
        test_indices: 予測に使用するインデックス
        
    Returns:
        スタッキングによるアンサンブル結果
    """
    # ベースモデルの予測結果を結合
    stacked_features = pd.concat([pred['Pred'] for pred in base_predictions], axis=1)
    stacked_features.columns = [f'model_{i}' for i in range(len(base_predictions))]
    
    # メタモデルの学習
    meta_model.fit(stacked_features.iloc[train_indices], target_df.iloc[train_indices])
    
    # メタモデルによる予測
    meta_predictions = meta_model.predict(stacked_features.iloc[test_indices])
    
    # 結果をデータフレーム化
    result_df = pd.DataFrame(meta_predictions, index=target_df.index[test_indices], columns=['Pred'])
    
    return result_df


def numerai_transform(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    予測結果をNumerai形式に変換する
    
    Args:
        pred_df: 予測結果のデータフレーム
        
    Returns:
        Numerai形式に変換された予測結果
    """
    transformed_df = pred_df.copy()
    
    # 各日付ごとに処理
    for date, group in pred_df.groupby('Date'):
        ranks = group['Pred'].rank()
        n = len(ranks)
        
        # ランクを[0,1]の範囲に正規化
        normalized_ranks = (ranks - 0.5) / n
        
        # ガウス変換
        gaussian_ranks = pd.Series(np.array([norm.ppf(r) for r in normalized_ranks]), index=ranks.index)
        
        # 1.5乗の変換
        transformed = np.sign(gaussian_ranks) * np.abs(gaussian_ranks) ** 1.5
        
        # 変換結果を格納
        transformed_df.loc[transformed.index, 'Pred'] = transformed
    
    return transformed_df