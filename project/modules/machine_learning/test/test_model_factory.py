import unittest

from machine_learning.factory.model_factory import ModelFactory
from machine_learning.models.lasso_model import LassoModel
from machine_learning.models.lgbm_model import LgbmModel


class TestModelFactory(unittest.TestCase):
    """ModelFactoryクラスのテスト"""

    def test_create_lasso_model(self):
        """Lassoモデルが正しく作成されるかテスト"""
        model = ModelFactory.create_model('lasso')
        self.assertIsInstance(model, LassoModel)

    def test_create_lgbm_model(self):
        """LightGBMモデルが正しく作成されるかテスト"""
        model = ModelFactory.create_model('lgbm')
        self.assertIsInstance(model, LgbmModel)
        
    def test_create_model_case_insensitive(self):
        """モデルタイプの大文字小文字を区別しないことをテスト"""
        model1 = ModelFactory.create_model('lasso')
        model2 = ModelFactory.create_model('LASSO')
        model3 = ModelFactory.create_model('Lasso')
        
        self.assertIsInstance(model1, LassoModel)
        self.assertIsInstance(model2, LassoModel)
        self.assertIsInstance(model3, LassoModel)

    def test_create_unknown_model_type(self):
        """未知のモデルタイプでValueErrorが発生するかテスト"""
        with self.assertRaises(ValueError):
            ModelFactory.create_model('unknown_model')

    def test_create_model_default_type(self):
        """デフォルトのモデルタイプ（引数なし）でLassoモデルが作成されるかテスト"""
        model = ModelFactory.create_model()
        self.assertIsInstance(model, LassoModel)


if __name__ == '__main__':
    unittest.main()