import unittest
import os
import pandas as pd
import tempfile


from machine_learning.factory.collection_factory import ModelCollectionFactory
from machine_learning.collection.model_collection import ModelCollection
from machine_learning.params.hyperparams import LassoParams, LgbmParams


class TestModelCollectionFactory(unittest.TestCase):
    """ModelCollectionFactoryクラスのテスト"""

    def setUp(self):
        """テスト用のダミーデータを作成"""
        # テスト用のセクターデータを作成
        self.sector_data = {
            'sector1': {
                'X': pd.DataFrame({
                    'feature1': [1, 2, 3, 4, 5],
                    'feature2': [5, 4, 3, 2, 1]
                }),
                'y': pd.Series([10, 20, 30, 40, 50])
            },
            'sector2': {
                'X': pd.DataFrame({
                    'feature1': [6, 7, 8, 9, 10],
                    'feature2': [10, 9, 8, 7, 6]
                }),
                'y': pd.Series([60, 70, 80, 90, 100])
            }
        }
        
        # テスト用のパラメータを作成
        self.lasso_params = LassoParams(alpha=0.1, max_iter=2000)
        self.lgbm_params = LgbmParams(learning_rate=0.01, num_leaves=31)

    def test_create_collection(self):
        """新しいコレクションが正しく作成されるかテスト"""
        collection = ModelCollectionFactory.create_collection('lasso')
        self.assertIsInstance(collection, ModelCollection)
        self.assertEqual(collection._model_type, 'lasso')
        
        collection = ModelCollectionFactory.create_collection('lgbm')
        self.assertIsInstance(collection, ModelCollection)
        self.assertEqual(collection._model_type, 'lgbm')

    def test_create_collection_default_type(self):
        """デフォルトのモデルタイプでコレクションが作成されるかテスト"""
        collection = ModelCollectionFactory.create_collection()
        self.assertIsInstance(collection, ModelCollection)
        self.assertEqual(collection._model_type, 'lasso')

    def test_create_from_data(self):
        """データからコレクションが正しく作成されるかテスト"""
        collection = ModelCollectionFactory.create_from_data(
            self.sector_data, 'lasso', self.lasso_params
        )
        
        self.assertIsInstance(collection, ModelCollection)
        self.assertEqual(collection._model_type, 'lasso')
        
        # 全てのセクターのモデルが作成されていることを確認
        self.assertEqual(set(collection.sectors), {'sector1', 'sector2'})
        
    def test_create_from_data_without_training(self):
        """学習なしでコレクションが作成されるかテスト"""
        collection = ModelCollectionFactory.create_from_data(
            self.sector_data, 'lasso', params=None
        )
        
        self.assertIsInstance(collection, ModelCollection)
        self.assertEqual(collection._model_type, 'lasso')
        
        # 全てのセクターのモデルが作成されていることを確認
        self.assertEqual(set(collection.sectors), {'sector1', 'sector2'})

    def test_load_collection(self):
        """保存されたコレクションが正しく読み込まれるかテスト"""
        # テスト用のコレクションを作成して保存
        collection = ModelCollectionFactory.create_collection('lasso')
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            filepath = temp_file.name
            collection.save(filepath)
        
        try:
            # 保存したコレクションを読み込む
            loaded_collection = ModelCollectionFactory.load_collection(filepath)
            
            # 読み込まれたコレクションが正しいことを確認
            self.assertIsInstance(loaded_collection, ModelCollection)
            self.assertEqual(loaded_collection._model_type, 'lasso')
        finally:
            # テスト終了後に一時ファイルを削除
            if os.path.exists(filepath):
                os.remove(filepath)


if __name__ == '__main__':
    unittest.main()