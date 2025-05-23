import pandas as pd
from datetime import datetime
from typing import Optional, List

from machine_learning.strategies.base_strategy import Strategy
from machine_learning.core.collection import ModelCollection
from machine_learning.core.registry import ModelRegistry


class SingleLgbmStrategy(Strategy):
    """単一のLightGBMモデルを全データに適用する戦略"""
    
    def __init__(self, name: str = "single_lgbm", save_path: Optional[str] = None):
        """
        Args:
            name: 戦略名
            save_path: 保存先パス（省略可）
        """
        super().__init__(name, save_path)
    
    def prepare_data(self, 
                   train_start_date: datetime,
                   train_end_date: datetime,
                   test_start_date: Optional[datetime] = None,
                   test_end_date: Optional[datetime] = None,
                   outlier_threshold: float = 3.0,
                   no_shift_features: List[str] = None,
                   reuse_features_df: bool = False) -> None:
        """
        データの前処理を行う
        
        Args:
            train_start_date: 学習データの開始日
            train_end_date: 学習データの終了日
            test_start_date: テストデータの開始日（省略時はtrain_end_date）
            test_end_date: テストデータの終了日（省略時はデータの最終日）
            outlier_threshold: 外れ値除去の閾値（±何σ）
            no_shift_features: シフトしない特徴量のリスト
            reuse_features_df: 特徴量を他の業種から再利用するか
        """
        if self.dataset is None:
            raise ValueError("データがセットされていません。load_data()を先に実行してください。")
        
        # パラメータのデフォルト値設定
        test_start_date = test_start_date or train_end_date
        test_end_date = test_end_date or self.dataset.target_df.index.get_level_values('Date').max()
        no_shift_features = no_shift_features or []
        
        # データセットを分割
        self.dataset.split_train_test(
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
            outlier_threshold=outlier_threshold,
            no_shift_features=no_shift_features,
            reuse_features_df=reuse_features_df
        )
        
        # モデルコレクションの初期化
        self.collection = ModelCollection(name=self.name, path=self.save_path)
        
        # 単一のLightGBMモデルを生成
        model = ModelRegistry.create_model('lightgbm', 'LGBM')
        
        # データをセット
        model.load_dataset(
            target_df=self.dataset.target_df,
            features_df=self.dataset.features_df,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
            outlier_threshold=outlier_threshold,
            no_shift_features=no_shift_features,
            reuse_features_df=reuse_features_df
        )
        
        # 生の目的変数と発注価格をセット
        if self.dataset.raw_target_df is not None:
            model.set_raw_target(self.dataset.raw_target_df)
        if self.dataset.order_price_df is not None:
            model.set_order_price(self.dataset.order_price_df)
        
        # モデルをコレクションに追加
        self.collection.add_model(model)
    
    def train(self) -> None:
        """モデルの学習を行う"""
        if self.collection is None:
            raise ValueError("データが準備されていません。prepare_data()を先に実行してください。")
        
        # カテゴリカル特徴量の設定
        if 'Sector_cat' in self.dataset.features_train_df.columns:
            lgbm_model = self.collection.get_model('LGBM')
            lgbm_model.categorical_features = ['Sector_cat']
        
        # すべてのモデルを学習
        self.collection.train_all()
        self.trained = True
    
    def predict(self) -> pd.DataFrame:
        """予測を実行する"""
        if not self.trained or self.collection is None:
            raise ValueError("モデルが学習されていません。train()を先に実行してください。")
        
        # すべてのモデルで予測を実行
        self.collection.predict_all()
        
        # 予測結果を取得
        return self.collection.get_result_df()
    
    @classmethod
    def run(cls, 
           path: str,
           target_df: pd.DataFrame, 
           features_df: pd.DataFrame,
           raw_target_df: pd.DataFrame, 
           order_price_df: pd.DataFrame,
           train_start_date: datetime, 
           train_end_date: datetime,
           test_start_date: Optional[datetime] = None, 
           test_end_date: Optional[datetime] = None,
           train: bool = True,
           **kwargs) -> ModelCollection:
        """
        単一のLightGBMモデル戦略を実行する
        
        Args:
            path: モデルの保存/読み込み先パス
            target_df: 目的変数のデータフレーム
            features_df: 特徴量のデータフレーム
            raw_target_df: 生の目的変数のデータフレーム
            order_price_df: 発注価格のデータフレーム
            train_start_date: 学習データの開始日
            train_end_date: 学習データの終了日
            test_start_date: テストデータの開始日（省略時はtrain_end_date）
            test_end_date: テストデータの終了日（省略時はデータの最終日）
            train: 学習を行うかどうか
            **kwargs: その他のパラメータ
                - outlier_threshold: 外れ値除去の閾値（デフォルト: 3.0）
                - no_shift_features: シフトしない特徴量のリスト（デフォルト: []）
                - reuse_features_df: 特徴量を他の業種から再利用するか（デフォルト: False）
            
        Returns:
            モデルコレクション
        """
        outlier_threshold = kwargs.get('outlier_threshold', 3.0)
        no_shift_features = kwargs.get('no_shift_features', [])
        reuse_features_df = kwargs.get('reuse_features_df', False)
        
        if train:
            # 学習と予測の両方を行う
            # 戦略インスタンスを作成
            strategy = cls(name="single_lgbm", save_path=path)
            
            # データを読み込み
            strategy.load_data(
                target_df=target_df,
                features_df=features_df,
                raw_target_df=raw_target_df,
                order_price_df=order_price_df
            )
            
            # データを準備
            strategy.prepare_data(
                train_start_date=train_start_date,
                train_end_date=train_end_date,
                test_start_date=test_start_date,
                test_end_date=test_end_date,
                outlier_threshold=outlier_threshold,
                no_shift_features=no_shift_features,
                reuse_features_df=reuse_features_df
            )
            
            # モデルを学習
            strategy.train()
            
            # 予測を実行
            strategy.predict()
            
            # 保存
            strategy.save()
            
            return strategy.collection
        else:
            # 予測のみ実行
            try:
                # 既存のコレクションを読み込み
                collection = ModelCollection.load(path)
                
                # 単一モデルのためセクター分割なし
                collection.set_train_test_data_all(
                    target_df=target_df,
                    features_df=features_df,
                    train_start_date=train_start_date,
                    train_end_date=train_end_date,
                    test_start_date=test_start_date,
                    test_end_date=test_end_date,
                    outlier_threshold=outlier_threshold,
                    no_shift_features=no_shift_features,
                    reuse_features_df=reuse_features_df,
                    separate_by_sector=False
                )
                
                # 生の目的変数と発注価格をセット
                collection.set_raw_target_for_all(raw_target_df, separate_by_sector=False)
                collection.set_order_price_for_all(order_price_df)
                
                # 予測を実行
                collection.predict_all()
                
                # 保存
                collection.save(path)
                
                return collection
            except FileNotFoundError:
                print(f"モデルファイルが見つかりません: {path}")
                print("学習モードで実行します。")
                
                # 学習モードで再実行
                return cls.run(
                    path=path,
                    target_df=target_df,
                    features_df=features_df,
                    raw_target_df=raw_target_df,
                    order_price_df=order_price_df,
                    train_start_date=train_start_date,
                    train_end_date=train_end_date,
                    test_start_date=test_start_date,
                    test_end_date=test_end_date,
                    train=True,
                    **kwargs
                )