import pandas as pd
from dataclasses import dataclass
from typing import Optional

@dataclass
class MLModel:
    '''
    モデルとスケーラーを対応させて管理するデータクラス
    name (str): インスタンスの名称
    model (any): 機械学習モデル
    scaler (Optional[any]): 機械学習スケーラー（任意）
    '''
    name: str
    model: any  # 実際の機械学習モデル
    scaler: Optional[any]  # スケーラーのインスタンスを想定

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        このアセットのモデルとスケーラーを使って予測を実行します。
        """
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # モデルのpredictメソッドを呼び出す（MLModel抽象クラスの規約に依存）
        predictions = self.model.predict(X_scaled)
        
        # 予測結果をDataFrameとして返す（列名などを考慮）
        if isinstance(predictions, pd.DataFrame):
            return predictions
        elif isinstance(predictions, pd.Series):
            return predictions.to_frame()
        else:
            # numpy arrayなどの場合を想定し、適切なDataFrameに変換
            return pd.DataFrame(predictions, index=X.index, columns=['Pred'])

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        このアセットのモデルとスケーラーを使って学習を実行します。
        """
        if self.scaler:
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
        else:
            X_train_scaled = X_train
        
        # モデルのtrainメソッドを呼び出す（MLModel抽象クラスの規約に依存）
        self.model.train(X_train_scaled, y_train)