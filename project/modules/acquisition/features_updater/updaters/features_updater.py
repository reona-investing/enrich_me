import pandas as pd
import os
from typing import Dict, Any
import logging

from utils.browser.browser_manager import BrowserManager
from utils.paths import Paths
from acquisition.features_updater.scrapers import FeatureScraper
from acquisition.features_updater.mergers import FeatureDataMerger


class FeatureUpdater:
    """特徴量データ更新の一連のフローを統合管理するクラス"""
    
    def __init__(self, browser_manager: BrowserManager, base_data_folder: str = None):
        """
        Args:
            browser_manager (BrowserManager): ブラウザ管理インスタンス
            base_data_folder (str, optional): データ保存基底フォルダ。Noneの場合はPathsを使用
        """
        self.browser_manager = browser_manager
        self.feature_scraper = FeatureScraper(browser_manager)
        self.feature_merger = FeatureDataMerger()
        self.base_data_folder = base_data_folder or Paths.SCRAPED_DATA_FOLDER
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        
    async def update_single_feature(self, 
                                    feature_config: Dict[str, Any],
                                    save_data: bool = True,
                                    validate_data: bool = True) -> pd.DataFrame:
        """
        単一特徴量のデータを更新
        
        Args:
            feature_config (Dict): 特徴量設定
                - name (str): 特徴量名
                - investing_code (str): investing.comのコード
                - additional_scrape (str): 追加データソース ('None', 'Baltic', 'Tradingview', 'ARCA')
                - additional_code (str): 追加データソースのコード
                - category (str): データカテゴリ (e.g., 'currency', 'commodity')
                - file_name (str): 保存ファイル名
            save_data (bool): データを保存するかどうか
            validate_data (bool): データ整合性を検証するかどうか
            
        Returns:
            pd.DataFrame: 更新された特徴量データ
        """
        try:
            feature_name = feature_config['name']
            self.logger.info(f"特徴量 '{feature_name}' の更新を開始")
            
            # 新しいデータをスクレイピング
            self.logger.info(f"'{feature_name}' のスクレイピング開始")
            new_scraped_df = await self.feature_scraper.scrape_feature(
                investing_code=feature_config['investing_code'],
                additional_scrape=feature_config.get('additional_scrape', 'None'),
                additional_code=feature_config.get('additional_code', 'None')
            )
            
            if new_scraped_df.empty:
                self.logger.warning(f"'{feature_name}' のスクレイピングでデータが取得できませんでした")
                return pd.DataFrame()
            
            self.logger.info(f"'{feature_name}' のスクレイピング完了: {len(new_scraped_df)} 行取得")
            
            # 既存データの読み込み
            existing_df = self._load_existing_data(feature_config)
            
            # データの結合
            self.logger.info(f"'{feature_name}' のデータ結合開始")
            merged_df = self.feature_merger.merge_feature_data(existing_df, new_scraped_df)
            
            # データ整合性の検証
            if validate_data:
                if not self.feature_merger.validate_data_integrity(merged_df):
                    raise ValueError(f"'{feature_name}' のデータ整合性検証に失敗")
                self.logger.info(f"'{feature_name}' のデータ整合性検証成功")
            
            # データの保存
            if save_data:
                self._save_updated_data(merged_df, feature_config)
                self.logger.info(f"'{feature_name}' のデータ保存完了")
            
            self.logger.info(f"特徴量 '{feature_name}' の更新完了: 最終データ {len(merged_df)} 行")
            return merged_df
            
        except Exception as e:
            self.logger.error(f"特徴量 '{feature_config.get('name', 'Unknown')}' の更新中にエラー: {str(e)}")
            raise
    
    async def update_multiple_features(self, 
                                       features_config: list[Dict[str, Any]],
                                       save_data: bool = True,
                                       validate_data: bool = True,
                                       continue_on_error: bool = True) -> Dict[str, pd.DataFrame]:
        """
        複数特徴量のデータを一括更新
        
        Args:
            features_config (list): 特徴量設定のリスト
            save_data (bool): データを保存するかどうか
            validate_data (bool): データ整合性を検証するかどうか
            continue_on_error (bool): エラー時に処理を継続するかどうか
            
        Returns:
            Dict[str, pd.DataFrame]: 特徴量名をキーとした更新結果
        """
        results = {}
        errors = {}
        
        self.logger.info(f"{len(features_config)} 個の特徴量の一括更新を開始")
        
        for i, config in enumerate(features_config, 1):
            feature_name = config.get('name', f'Feature_{i}')
            try:
                self.logger.info(f"[{i}/{len(features_config)}] 特徴量 '{feature_name}' を処理中")
                
                result_df = await self.update_single_feature(
                    feature_config=config,
                    save_data=save_data,
                    validate_data=validate_data
                )
                results[feature_name] = result_df
                
            except Exception as e:
                error_msg = f"特徴量 '{feature_name}' の更新失敗: {str(e)}"
                errors[feature_name] = error_msg
                self.logger.error(error_msg)
                
                if not continue_on_error:
                    raise
        
        # 結果サマリーのログ出力
        successful_count = len(results)
        failed_count = len(errors)
        self.logger.info(f"一括更新完了: 成功 {successful_count}, 失敗 {failed_count}")
        
        if errors:
            self.logger.warning(f"失敗した特徴量: {list(errors.keys())}")
        
        return results
    
    def _load_existing_data(self, feature_config: Dict[str, Any]) -> pd.DataFrame:
        """既存データの読み込み"""
        try:
            file_path = self._get_data_file_path(feature_config)
            
            if not os.path.exists(file_path):
                self.logger.info(f"既存データファイルが見つかりません: {file_path}")
                return pd.DataFrame()
            
            # ファイル拡張子に応じて読み込み方法を選択
            if file_path.endswith('.parquet'):
                existing_df = pd.read_parquet(file_path)
            elif file_path.endswith('.csv'):
                existing_df = pd.read_csv(file_path)
            else:
                raise ValueError(f"サポートされていないファイル形式: {file_path}")
            
            self.logger.info(f"既存データ読み込み完了: {len(existing_df)} 行")
            return existing_df
            
        except Exception as e:
            self.logger.warning(f"既存データの読み込みに失敗: {str(e)}")
            return pd.DataFrame()
    
    def _save_updated_data(self, df: pd.DataFrame, feature_config: Dict[str, Any]) -> None:
        """更新されたデータの保存"""
        file_path = self._get_data_file_path(feature_config)
        
        # ディレクトリの作成
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # ファイル拡張子に応じて保存方法を選択
        if file_path.endswith('.parquet'):
            df.to_parquet(file_path, index=False)
        elif file_path.endswith('.csv'):
            df.to_csv(file_path, index=False)
        else:
            # デフォルトはparquet形式
            file_path = file_path.replace('.csv', '.parquet')
            df.to_parquet(file_path, index=False)
        
        self.logger.info(f"データ保存完了: {file_path}")
    
    def _get_data_file_path(self, feature_config: Dict[str, Any]) -> str:
        """データファイルのパスを取得"""
        category = feature_config.get('category', 'misc')
        file_name = feature_config.get('file_name', f"{feature_config['name']}_price.parquet")
        
        return os.path.join(self.base_data_folder, category, file_name)
    
    def get_feature_summary(self, feature_config: Dict[str, Any]) -> Dict[str, Any]:
        """特徴量データのサマリー情報を取得"""
        try:
            existing_df = self._load_existing_data(feature_config)
            
            if existing_df.empty:
                return {
                    'feature_name': feature_config['name'],
                    'data_exists': False,
                    'record_count': 0,
                    'date_range': None,
                    'last_update': None
                }
            
            return {
                'feature_name': feature_config['name'],
                'data_exists': True,
                'record_count': len(existing_df),
                'date_range': {
                    'start': existing_df['Date'].min().strftime('%Y-%m-%d'),
                    'end': existing_df['Date'].max().strftime('%Y-%m-%d')
                },
                'last_update': existing_df['Date'].max().strftime('%Y-%m-%d'),
                'file_path': self._get_data_file_path(feature_config)
            }
            
        except Exception as e:
            return {
                'feature_name': feature_config['name'],
                'error': str(e)
            }
    
    async def __aenter__(self):
        """非同期コンテキストマネージャー（Enter）"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャー（Exit）"""
        # 必要に応じてクリーンアップ処理
        pass


# 使用例とテスト
if __name__ == '__main__':
    import asyncio
    from utils.timekeeper import timekeeper
    
    @timekeeper
    async def main():
        # サンプル設定
        sample_features = [
            {
                'name': 'JPbond10',
                'investing_code': 'rates-bonds/japan-10-year-bond-yield',
                'additional_scrape': 'None',
                'additional_code': 'None',
                'category': 'bond',
                'file_name': 'raw_JPbond10_price.parquet'
            },
            {
                'name': 'JPbond2',
                'investing_code': 'rates-bonds/japan-2-year-bond-yield',
                'additional_scrape': 'None',
                'additional_code': 'None',
                'category': 'bond',
                'file_name': 'raw_JPbond2_price.parquet'
            },
            {
                'name': 'Palladium',
                'investing_code': 'commodities/palladium',
                'additional_scrape': 'None',
                'additional_code': 'None',
                'category': 'commodity',
                'file_name': 'raw_CmdPalladium_price.parquet'
            }
        ]
        
        browser_manager = BrowserManager()
        
        async with FeatureUpdater(browser_manager) as updater:
            # 単一特徴量の更新例
            print("=== 単一特徴量更新例 ===")
            single_result = await updater.update_single_feature(
                feature_config=sample_features[0],
                save_data=False  # テスト用にファイル保存を無効化
            )
            print(f"更新結果: {len(single_result)} 行")
            
            # 特徴量サマリーの取得例
            print("\n=== 特徴量サマリー ===")
            for config in sample_features:
                summary = updater.get_feature_summary(config)
                print(f"{summary['feature_name']}: {summary}")
            
            # 複数特徴量の一括更新例
            print("\n=== 複数特徴量一括更新例 ===")
            batch_results = await updater.update_multiple_features(
                features_config=sample_features,
                save_data=False,  # テスト用にファイル保存を無効化
                continue_on_error=True
            )
            
            for feature_name, result_df in batch_results.items():
                print(f"{feature_name}: {len(result_df)} 行更新")
    
    asyncio.run(main())