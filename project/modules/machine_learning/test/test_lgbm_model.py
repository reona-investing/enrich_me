import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import pickle

from machine_learning.models.lgbm_model import LgbmModel
from machine_learning.params.hyperparams import LgbmParams


class TestLgbmModel(unittest.TestCase):
    """LgbmModelクラスのテスト"""

    def setUp(self):
        """テスト用のダミーデータを作成"""
        # テスト用のデータフレームを作成
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        # 特徴量行列を作成
        self.X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature{i}' for i in range(n_features)]
        )
        
        # カテゴリ変数も追加
        self.X['category1'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
        self.X['category2'] = np.random.choice(['X', 'Y', 'Z'], size=n_samples)
        
        # One-hot encoding
        self.X_encoded = pd.get_dummies(self.X, columns=['category1', 'category2'], drop_first=False)
        
        # 目的変数を作成
        self.y = pd.Series(
            3 * self.X['feature0'] - 2 * self.X['feature2'] + 
            0.5 * (self.X['category1'] == 'A').astype(int) + 
            np.random.normal(0, 0.1, n_samples)
        )
        
        # 欠損値を含むデータセットも作成
        self.X_with_nan = self.X.copy()
        self.X_with_nan.iloc[0:10, 1] = np.nan  # 最初の10行のfeature1に欠損値を入れる
        
        # テスト用のパラメータ
        self.params = LgbmParams(
            objective='regression',
            learning_rate=0.1,
            num_leaves=7,
            num_boost_round=100,  # テスト用に少ない回数に設定
            early_stopping_rounds=5,
            categorical_features=['category1', 'category2']
        )

    def test_initialization(self):
        """初期化が正しく行われるかテスト"""
        model = LgbmModel()
        self.assertIsNone(model.model)
        self.assertIsNone(model.feature_importance)

    def test_train(self):
        """モデルの学習が正しく行われるかテスト"""
        model = LgbmModel()
        # カテゴリカル変数をそのまま使用して学習
        model.train(self.X, self.y, self.params)
        
        # モデルが正しく学習されたことを確認
        self.assertIsNotNone(model.model)
        
        # 特徴量重要度が計算されているか確認
        self.assertIsNotNone(model.feature_importance)
        
        # 特徴量重要度のフォーマットを確認
        self.assertIn('Feature', model.feature_importance.columns)
        self.assertIn('Importance', model.feature_importance.columns)

    def test_train_with_encoded_features(self):
        """エンコード済み特徴量でモデルの学習が正しく行われるかテスト"""
        model = LgbmModel()
        # エンコード済み特徴量を使用して学習（カテゴリ指定なし）
        params = LgbmParams(
            objective='regression',
            learning_rate=0.1,
            num_leaves=7,
            num_boost_round=100,
            early_stopping_rounds=5
        )
        model.train(self.X_encoded, self.y, params)
        
        # モデルが正しく学習されたことを確認
        self.assertIsNotNone(model.model)

    def test_train_with_kwargs(self):
        """パラメータをkwargsとして渡す場合のテスト"""
        model = LgbmModel()
        model.train(
            self.X, self.y,
            objective='regression', 
            learning_rate=0.1,
            num_leaves=7,
            num_boost_round=100,
            categorical_features=['category1', 'category2']
        )
        
        # モデルが正しく学習されたことを確認
        self.assertIsNotNone(model.model)

    def test_predict(self):
        """予測が正しく行われるかテスト"""
        model = LgbmModel()
        model.train(self.X, self.y, self.params)
        
        # 予測を実行
        predictions = model.predict(self.X)
        
        # 予測結果の形状を確認
        self.assertEqual(len(predictions), len(self.X))
        
        # 予測値が数値であることを確認
        self.assertTrue(np.issubdtype(predictions.dtype, np.number))

    def test_predict_with_nan(self):
        """欠損値を含むデータで予測が正しく行われるかテスト"""
        model = LgbmModel()
        model.train(self.X, self.y, self.params)
        
        # 欠損値を含むデータで予測を実行
        predictions = model.predict(self.X_with_nan)
        
        # 予測結果の形状を確認
        self.assertEqual(len(predictions), len(self.X_with_nan))
        
        # 欠損値がある行はNaNになっていることを確認
        self.assertTrue(np.isnan(predictions[:10]).all())
        
        # 欠損値がない行は数値が予測されていることを確認
        self.assertTrue(np.isfinite(predictions[10:]).all())

    def test_predict_without_training(self):
        """学習なしで予測を呼び出した場合にエラーが発生するかテスト"""
        model = LgbmModel()
        with self.assertRaises(ValueError):
            model.predict(self.X)

    def test_feature_importance(self):
        """特徴量重要度が正しく計算されるかテスト"""
        model = LgbmModel()
        model.train(self.X, self.y, self.params)
        
        # 特徴量重要度が計算されていることを確認
        importance_df = model.feature_importance
        self.assertIsNotNone(importance_df)
        
        # 全ての特徴量が含まれていることを確認
        feature_names = set(self.X.columns)
        importance_features = set(importance_df['Feature'])
        self.assertTrue(importance_features.issubset(feature_names))
        
        # 重要度が降順でソートされていることを確認
        importance_values = importance_df['Importance'].values
        self.assertTrue(all(importance_values[i] >= importance_values[i+1] 
                           for i in range(len(importance_values)-1)))

    def test_custom_eval_metric(self):
        """カスタム評価指標を使用できるかテスト"""
        # Numerai相関を使用するパラメータ
        params = LgbmParams(
            objective='regression',
            metric='numerai_corr',  # カスタム評価指標
            learning_rate=0.1,
            num_leaves=7,
            num_boost_round=100,
            early_stopping_rounds=5
        )
        
        model = LgbmModel()
        
        # 日付カラムを追加（Numerai相関のテスト用）
        X_with_date = self.X.copy()
        X_with_date['date'] = np.random.choice(['2023-01-01', '2023-01-02', '2023-01-03'], size=len(self.X))
        
        # 学習を実行（カスタム評価指標を使用）
        # 注意：このテストは実際のNumerai相関の計算が正しいかは検証せず、実行できるかのみ確認
        try:
            model.train(X_with_date, self.y, params)
            self.assertIsNotNone(model.model)
        except Exception as e:
            self.fail(f"カスタム評価指標を使用した学習が失敗しました: {e}")


if __name__ == '__main__':
    unittest.main()