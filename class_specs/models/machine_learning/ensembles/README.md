# models/machine_learning/ensembles のクラス仕様書

## base_ensemble_method.py

### class BaseEnsembleMethod
アンサンブル手法の抽象基底クラス
- ensemble: アンサンブルを実行する抽象メソッド

Args:
    inputs (List[Tuple[pd.DataFrame, float]]): (予測結果データフレーム, 重み)のタプルのリスト
    
Returns:
    pd.DataFrame: アンサンブル後の予測結果を格納したデータフレーム
- _validate_inputs: 入力の妥当性をチェック
- _normalize_weights: 重みを正規化（合計を1.0にする）

## by_predict_value.py

### class ByPredictValueMethod
予測値ベースでのアンサンブル手法
- ensemble: 予測値の重み付き平均でアンサンブルを実行

Args:
    inputs (List[Tuple[pd.DataFrame, float]]): (予測結果データフレーム, 重み)のタプルのリスト
    
Returns:
    pd.DataFrame: アンサンブル後の予測値を格納したデータフレーム

## by_rank.py

### class ByRankMethod
予測順位ベースでのアンサンブル手法
- ensemble: 予測順位ベースでアンサンブルを実行

Args:
    inputs (List[Tuple[pd.DataFrame, float]]): (予測結果データフレーム, 重み)のタプルのリスト
    
Returns:
    pd.DataFrame: アンサンブル後の予測順位を格納したデータフレーム

## by_voting.py

### class ByVotingMethod
投票ベースでのアンサンブル手法
- __init__: Args:
    top_n (int): 各モデルの上位何位までを考慮するか
- ensemble: 各モデルの上位予測に投票を行いアンサンブルを実行

Args:
    inputs (List[Tuple[pd.DataFrame, float]]): (予測結果データフレーム, 重み)のタプルのリスト
    
Returns:
    pd.DataFrame: アンサンブル後の投票スコアを格納したデータフレーム

## factory.py

### class EnsembleMethodFactory
アンサンブル手法のインスタンスを生成・管理するためのファクトリークラス。

事前に登録されたアンサンブル手法を名前で取得したり、
新たな手法を登録するための機能を提供する。

デフォルトで利用可能なアンサンブル手法は以下の通り:
    - 'by_rank': 予測値のランクに基づく手法
    - 'by_predict_value': 予測値の大小に基づく手法
    - 'by_voting': クラスごとの投票に基づく手法
- create_method: 指定された名前のアンサンブル手法を作成

Args:
    method_name (str): アンサンブル手法の名前
    **kwargs: アンサンブル手法固有のパラメータ
    
Returns:
    BaseEnsembleMethod: アンサンブル手法のインスタンス
    
Raises:
    ValueError: 存在しない手法名が指定された場合
- get_available_methods: 利用可能なアンサンブル手法の一覧を取得
- register_method: 新しいアンサンブル手法を登録

