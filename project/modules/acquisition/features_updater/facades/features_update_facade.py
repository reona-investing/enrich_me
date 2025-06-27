from utils.paths import Paths
from utils.browser.browser_manager import BrowserManager
from utils.timekeeper import timekeeper
from acquisition.features_updater.updaters import FeatureUpdater
from utils.notifier import SlackNotifier
import pandas as pd
import asyncio
import logging
from typing import List, Dict, Any
import os


class FeaturesUpdateFacade:
    """特徴量データ更新のファサードクラス（リファクタリング版）"""
    
    def __init__(self, max_concurrent_tasks: int = 5):
        """
        Args:
            max_concurrent_tasks (int): 同時実行タスク数の上限
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.logger = logging.getLogger(__name__)
        self.slack = SlackNotifier(program_name=os.path.basename(__file__))
        
        # 進捗管理
        self.completed_count = 0
        self.total_count = 0
    
    @timekeeper
    async def update_all(self, 
                         continue_on_error: bool = True,
                         validate_data: bool = True,
                         dry_run: bool = False) -> Dict[str, Any]:
        """
        全特徴量データの一括更新
        
        Args:
            continue_on_error (bool): エラー時に処理を継続するか
            validate_data (bool): データ整合性を検証するか
            dry_run (bool): 実際の保存を行わない（テスト用）
            
        Returns:
            Dict[str, Any]: 更新結果のサマリー
        """
        self.logger.info("全特徴量データの更新を開始")
        
        # 設定の読み込み
        features_config = self._load_features_config()
        adopted_features = [config for config in features_config if config.get('is_adopted', True)]
        
        self.total_count = len(adopted_features)
        self.completed_count = 0
        
        if not adopted_features:
            self.logger.warning("更新対象の特徴量が見つかりません")
            return {'success': 0, 'failed': 0, 'skipped': self.total_count}
        
        self.logger.info(f"{self.total_count} 個の特徴量を更新対象として検出")
        
        # ブラウザマネージャーとアップデーターの初期化
        browser_manager = BrowserManager()
        
        try:
            async with FeatureUpdater(browser_manager) as updater:
                # セマフォによる同時実行数制御
                semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
                
                # 各特徴量の更新タスクを作成
                tasks = [
                    self._process_single_feature_with_semaphore(
                        updater, semaphore, config, dry_run, validate_data
                    ) 
                    for config in adopted_features
                ]
                
                # 全タスクの並列実行
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 結果の集計
                summary = self._summarize_results(results, adopted_features)
                
        except Exception as e:
            self.logger.error(f"全体処理中にエラーが発生: {str(e)}")
            raise
        finally:
            # ブラウザセッションのクリーンアップ
            try:
                await browser_manager.reset_session()
            except:
                pass
        
        # 結果の表示
        self._display_final_summary(summary)
        
        return summary
    
    def _load_features_config(self) -> List[Dict[str, Any]]:
        """設定ファイルから特徴量設定を読み込み"""
        try:
            config_df = pd.read_csv(Paths.FEATURES_TO_SCRAPE_CSV)
            
            features_config = []
            for _, row in config_df.iterrows():
                # パスの構築
                file_path = os.path.join(
                    Paths.SCRAPED_DATA_FOLDER,
                    row['Group'],
                    row['Path']
                )
                
                # NaN値の適切な処理
                additional_scrape = row.get('AdditionalScrape', 'None')
                additional_code = row.get('AdditionalCode', 'None')
                
                # pandas NaN値を'None'に変換
                if pd.isna(additional_scrape) or str(additional_scrape).lower() == 'nan':
                    additional_scrape = 'None'
                if pd.isna(additional_code) or str(additional_code).lower() == 'nan':
                    additional_code = 'None'
                
                config = {
                    'name': row['Name'],
                    'investing_code': row['URL'],
                    'additional_scrape': additional_scrape,
                    'additional_code': additional_code,
                    'category': row['Group'],
                    'file_name': row['Path'],
                    'is_adopted': row.get('is_adopted', True),
                    'full_path': file_path
                }
                features_config.append(config)
            
            return features_config
            
        except Exception as e:
            self.logger.error(f"設定ファイルの読み込みに失敗: {str(e)}")
            raise
    
    async def _process_single_feature_with_semaphore(self,
                                                     updater: FeatureUpdater,
                                                     semaphore: asyncio.Semaphore,
                                                     config: Dict[str, Any],
                                                     dry_run: bool,
                                                     validate_data: bool) -> Dict[str, Any]:
        """セマフォ制御付きの単一特徴量処理"""
        async with semaphore:
            # 処理間隔の調整
            await asyncio.sleep(1)
            
            try:
                result = await self._process_single_feature(
                    updater, config, dry_run, validate_data
                )
                
                self.completed_count += 1
                self._display_progress(config['name'], result)
                
                return {
                    'feature_name': config['name'],
                    'status': 'success',
                    'record_count': len(result) if isinstance(result, pd.DataFrame) else 0,
                    'result': result
                }
                
            except Exception as e:
                self.completed_count += 1
                error_msg = f"{config['name']}: {str(e)}"
                self.logger.error(error_msg)
                
                return {
                    'feature_name': config['name'],
                    'status': 'failed',
                    'error': str(e)
                }
    
    async def _process_single_feature(self,
                                      updater: FeatureUpdater,
                                      config: Dict[str, Any],
                                      dry_run: bool,
                                      validate_data: bool) -> pd.DataFrame:
        """単一特徴量の処理"""
        try:
            # FeatureUpdaterを使用してデータ更新
            result_df = await updater.update_single_feature(
                feature_config=config,
                save_data=not dry_run,  # dry_runの場合は保存しない
                validate_data=validate_data
            )
            
            return result_df
            
        except Exception as e:
            # より詳細なエラー情報をログに記録
            self.logger.error(f"特徴量 '{config['name']}' の処理中にエラー: {str(e)}")
            raise
    
    def _display_progress(self, feature_name: str, result_df: pd.DataFrame):
        """進捗表示"""
        progress_msg = f"{self.completed_count}/{self.total_count}: {feature_name}"
        
        if isinstance(result_df, pd.DataFrame) and not result_df.empty:
            print(progress_msg)
            print("最新データ:")
            print(result_df.tail(2))
        else:
            print(f"{progress_msg} (データなし)")
        
        print('---------------------------------------')
    
    def _summarize_results(self, 
                          results: List[Any], 
                          configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """結果の集計"""
        successful = []
        failed = []
        
        for i, result in enumerate(results):
            config = configs[i]
            
            if isinstance(result, Exception):
                failed.append({
                    'feature_name': config['name'],
                    'error': str(result)
                })
            elif isinstance(result, dict):
                if result['status'] == 'success':
                    successful.append(result)
                else:
                    failed.append(result)
            else:
                # 予期しない結果形式
                failed.append({
                    'feature_name': config['name'],
                    'error': f'予期しない結果形式: {type(result)}'
                })
        
        return {
            'total': len(configs),
            'successful': len(successful),
            'failed': len(failed),
            'failure_details': failed
        }
    
    def _display_final_summary(self, summary: Dict[str, Any]):
        """最終結果の表示"""
        lines = [
            '=' * 50,
            '全データのスクレイピングが完了しました。',
            f"総数: {summary['total']}",
            f"成功: {summary['successful']}",
            f"失敗: {summary['failed']}",
        ]

        if summary['failed'] > 0:
            lines.append('')
            lines.append('失敗した特徴量:')
            for failure in summary['failure_details']:
                lines.append(f"  - {failure['feature_name']}: {failure.get('error', '不明なエラー')}")

        lines.append('=' * 50)
        message = '\n'.join(lines)

        print(message)
        self.slack.send_message(f"\n{message}")
    
    async def update_specific_features(self, 
                                       feature_names: List[str],
                                       continue_on_error: bool = True,
                                       validate_data: bool = True,
                                       dry_run: bool = False) -> Dict[str, Any]:
        """
        指定された特徴量のみを更新
        
        Args:
            feature_names (List[str]): 更新対象の特徴量名リスト
            continue_on_error (bool): エラー時に処理を継続するか
            validate_data (bool): データ整合性を検証するか
            dry_run (bool): 実際の保存を行わない（テスト用）
            
        Returns:
            Dict[str, Any]: 更新結果のサマリー
        """
        self.logger.info(f"指定特徴量の更新開始: {feature_names}")
        
        # 全設定から指定された特徴量のみを抽出
        all_configs = self._load_features_config()
        target_configs = [
            config for config in all_configs 
            if config['name'] in feature_names
        ]
        
        if not target_configs:
            self.logger.warning("指定された特徴量が見つかりません")
            return {'success': 0, 'failed': 0, 'skipped': len(feature_names)}
        
        # 見つからなかった特徴量をログに記録
        found_names = {config['name'] for config in target_configs}
        missing_names = set(feature_names) - found_names
        if missing_names:
            self.logger.warning(f"見つからなかった特徴量: {list(missing_names)}")
        
        # 更新処理（update_allと同じロジック）
        self.total_count = len(target_configs)
        self.completed_count = 0
        
        browser_manager = BrowserManager()
        
        try:
            async with FeatureUpdater(browser_manager) as updater:
                semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
                
                tasks = [
                    self._process_single_feature_with_semaphore(
                        updater, semaphore, config, dry_run, validate_data
                    ) 
                    for config in target_configs
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                summary = self._summarize_results(results, target_configs)
                
        finally:
            try:
                await browser_manager.reset_session()
            except:
                pass
        
        self._display_final_summary(summary)
        return summary
    
    def get_features_status(self) -> pd.DataFrame:
        """全特徴量の状況を取得"""
        configs = self._load_features_config()
        
        status_data = []
        for config in configs:
            try:
                file_path = config['full_path']
                if os.path.exists(file_path):
                    if file_path.endswith('.parquet'):
                        df = pd.read_parquet(file_path)
                    else:
                        df = pd.read_csv(file_path)
                    
                    status_data.append({
                        'Name': config['name'],
                        'Group': config['category'],
                        'IsAdopted': config.get('is_adopted', True),
                        'RecordCount': len(df),
                        'LastDate': df['Date'].max() if 'Date' in df.columns else None,
                        'FilePath': file_path,
                        'Status': 'データあり'
                    })
                else:
                    status_data.append({
                        'Name': config['name'],
                        'Group': config['category'],
                        'IsAdopted': config.get('is_adopted', True),
                        'RecordCount': 0,
                        'LastDate': None,
                        'FilePath': file_path,
                        'Status': 'ファイルなし'
                    })
            except Exception as e:
                status_data.append({
                    'Name': config['name'],
                    'Group': config['category'],
                    'IsAdopted': config.get('is_adopted', True),
                    'RecordCount': 0,
                    'LastDate': None,
                    'FilePath': config.get('full_path', ''),
                    'Status': f'エラー: {str(e)}'
                })
        
        return pd.DataFrame(status_data)


# 使用例
if __name__ == '__main__':
    async def main():
        facade = FeaturesUpdateFacade(max_concurrent_tasks=5)
        
        # 全特徴量の状況確認
        print("=== 特徴量状況確認 ===")
        status_df = facade.get_features_status()
        print(status_df)
        
        # 全特徴量更新（ドライラン）
        
        print("\n=== 全特徴量更新（ドライラン） ===")
        summary = await facade.update_all(dry_run=True)
        print(f"ドライラン結果: {summary}")
        
        '''
        # 特定特徴量のみ更新
        print("\n=== 特定特徴量更新 ===")
        specific_summary = await facade.update_specific_features(
            feature_names=['RoughRice'],
            dry_run=True
        )
        print(f"特定更新結果: {specific_summary}")
        '''
    
    asyncio.get_event_loop().run_until_complete(main())