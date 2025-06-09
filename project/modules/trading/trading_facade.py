import os
from utils.paths import Paths
from utils.notifier import SlackNotifier
from trading.sbi.browser import SBIBrowserManager
# 銘柄選択
from trading.sbi.selection import OneStopStockSelector
# 発注関係
from trading.sbi.orders.interface.order_executor import IOrderExecutor
from trading.sbi.common.interface import IMarginProvider
from trading.sbi.common.provider import SBIMarginProvider
from trading.sbi.orders.manager.order_executor import SBIOrderExecutor
from trading.sbi.orders.ordermaker.batch_order_maker import BatchOrderMaker
from trading.sbi.orders.ordermaker.position_settler import PositionSettler
from trading.sbi import HistoryManager
from trading.sbi_trading_logic import HistoryUpdater
import pandas as pd

class TradingFacade:
    def __init__(self):
        """SBI取引操作を統括するファサード"""
        self.slack = SlackNotifier(program_name=os.path.basename(__file__))
        
        # ブラウザマネージャーの初期化
        self.browser_manager = SBIBrowserManager()
        
        # 新しいインターフェース実装の初期化
        self.margin_provider: IMarginProvider = SBIMarginProvider(self.browser_manager)
        self.order_executor: IOrderExecutor = SBIOrderExecutor(self.browser_manager)
        
        # その他のマネージャー群の初期化
        self.history_manager = HistoryManager(self.browser_manager)
        
        # 注文関連のクラスの初期化
        self.batch_order_maker = BatchOrderMaker(self.order_executor, self.margin_provider)
        self.position_settler = PositionSettler(self.order_executor, self.margin_provider)
        
        self.Paths = Paths

    async def take_positions(self, 
                             order_price_df: pd.DataFrame,
                             pred_result_df: pd.DataFrame,
                             SECTOR_REDEFINITIONS_CSV: str, 
                             num_sectors_to_trade: int = 3, 
                             num_candidate_sectors: int = 5, 
                             top_slope: float = 1.0):
        '''
        信用新規建を行います。
        Args:
            ml_dataset (MLDataset): 機械学習のデータセットを指定します。
            NEW_SECTOR_LIST_CSV (str): セクターを定義したCSVファイルのパスを指定します。
            num_sectors_to_trade (int): 予測の上位・下位何業種を取引対象とするか指定します。
            num_candidate_sectors (int): 取引不可の業種がある場合に、上位・下位何業種まで取引対象を広げるか指定します。
                ※ num_sectors_to_tradeの数以上を指定しないとエラーになります。
            top_slope (float): 最上位・最下位業種に傾斜をつける場合指定します。
                ※ 1.0のとき傾斜なし
        '''
        
        # 銘柄選択処理
        stock_selector = OneStopStockSelector(order_price_df = order_price_df,
                                              pred_result_df = pred_result_df,
                                              browser_manager = self.browser_manager,
                                              sector_definitions_path = SECTOR_REDEFINITIONS_CSV,
                                              num_sectors_to_trade = num_sectors_to_trade,
                                              num_candidate_sectors = num_candidate_sectors,
                                              top_slope = top_slope)
        
        orders_df, _ = await stock_selector.select_stocks()
        
        # 注文一括発注
        results = await self.batch_order_maker.place_batch_orders(orders_df)
        
        # 結果の処理
        success_orders = [result for result in results if result.success]
        failed_orders = [result for result in results if not result.success]
        
        # 結果の通知
        self.slack.send_message(f'発注が完了しました。\n買：{stock_selector.buy_sectors}\n売：{stock_selector.sell_sectors}')
        
        if failed_orders:
            failed_messages = "\n".join([f"{order.message}" for order in failed_orders])
            self.slack.send_message(f'以下の注文の発注に失敗しました。\n{failed_messages}')

    async def take_additionals(self):
        '''
        take_positionsで失敗した注文について、追加で信用新規建を行います。
        '''
        # 失敗注文の読み込み
        import pandas as pd
        failed_orders_df = pd.read_csv(Paths.FAILED_ORDERS_CSV)
        
        if failed_orders_df.empty:
            self.slack.send_message('追加発注対象の注文はありません。')
            return
            
        # 再発注
        results = await self.batch_order_maker.place_batch_orders(failed_orders_df)
        
        # 結果の処理
        success_orders = [result for result in results if result.success]
        failed_orders = [result for result in results if not result.success]
        
        # 結果の通知
        self.slack.send_message('追加発注が完了しました。')
        
        if failed_orders:
            failed_messages = "\n".join([f"{order.message}" for order in failed_orders])
            self.slack.send_message(f'以下の注文の発注に失敗しました。\n{failed_messages}')

    async def take_additionals_until_completed(
        self,
        interval: int = 60,
        max_retries: int = 30,
    ):
        """追加発注リストを、約定するまで継続的に発注する

        Args:
            interval (int): 繰り返し実行の待機時間（秒）
            max_retries (int): ループの最大実行回数

        補足:
            追加発注がすべて失敗し、その原因が「信用建余力」不足では
            ない場合、処理を中断します。
        """
        import pandas as pd
        import asyncio
        from pathlib import Path

        path = Path(Paths.FAILED_ORDERS_CSV)
        if not path.exists():
            self.slack.send_message('追加発注対象の注文はありません。')
            return

        retry_count = 0

        while retry_count < max_retries:
            if path.exists():
                add_df = pd.read_csv(path)
            else:
                add_df = pd.DataFrame()

            await self.margin_provider.refresh()
            margin = await self.margin_provider.get_available_margin()
            active_orders = await self.order_executor.get_active_orders()
            active_cnt = len(active_orders)

            print(f'現在の信用建余力: {margin}円 / 未約定注文: {active_cnt}件')

            if add_df.empty:
                if active_cnt == 0:
                    break
                await asyncio.sleep(interval)
                continue

            if (add_df['UpperLimitTotal'] <= margin).any():
                results = await self.batch_order_maker.place_batch_orders(add_df)
                failed_orders = [r for r in results if not r.success]
                if failed_orders:
                    # すべての追加発注が失敗したかを確認
                    if len(failed_orders) == len(results) and all('信用建余力' not in r.message for r in failed_orders):
                        self.slack.send_message('追加発注が全て失敗しました。信用建余力不足以外が原因のため処理を終了します。')
                        break
                    failed_messages = "\n".join([f"{o.message}" for o in failed_orders])
                    self.slack.send_message(f'以下の注文の発注に失敗しました。1分後に再発注を試みます。\n{failed_messages}')
                else:
                    self.slack.send_message('追加発注が完了しました。')
                    break
            elif active_cnt == 0:
                self.slack.send_message('信用建余力が不足しており、追加発注可能な銘柄がありません。')
                break

            await asyncio.sleep(interval)
            retry_count += 1

        if retry_count >= max_retries:
            self.slack.send_message(
                f'最大リトライ回数({max_retries})に達したため処理を終了します。'
            )

    async def settle_positions(self):
        '''
        信用ポジションの決済注文を発注します。
        '''
        # 全ポジションの決済
        results = await self.position_settler.settle_all_positions()
        
        # 結果の処理
        success_settlements = [result for result in results if result.success]
        failed_settlements = [result for result in results if not result.success]
        
        # 結果の通知
        if not failed_settlements:
            self.slack.send_message('全銘柄の決済注文が完了しました。')
        else:
            failed_messages = "\n".join([f"{order.message}" for order in failed_settlements])
            self.slack.send_message(f'以下の銘柄の決済注文に失敗しました。\n{failed_messages}')

    async def fetch_invest_result(self, SECTOR_REDEFINITIONS_CSV):
        '''
        当日の取引履歴・入出金履歴・買付余力を取得します。
        '''
        history_updater = HistoryUpdater(self.history_manager, self.margin_provider, SECTOR_REDEFINITIONS_CSV)
        trade_history, _, _, _, amount = await history_updater.update_information()
        self.slack.send_result(f'取引履歴等の更新が完了しました。\n{trade_history["日付"].iloc[-1].strftime("%Y-%m-%d")}の取引結果：{amount}円')


if __name__ == '__main__':
    async def main():
        from models.machine_learning import SingleMLDataset
        SINGLE_ML_DATASET_PATH = f'{Paths.ML_DATASETS_FOLDER}/48sectors_Ensembled_learned_in_250603'
        SECTOR_REDEFINITIONS_CSV = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv'
        ml_dataset = SingleMLDataset(SINGLE_ML_DATASET_PATH, 'Ensembled')
        order_price_df = ml_dataset.stock_selection_materials.order_price_df
        pred_result_df = ml_dataset.stock_selection_materials.pred_result_df
        trade_facade = TradingFacade()
      
        '''
        await trade_facade.take_positions(
            order_price_df = order_price_df,
            pred_result_df = pred_result_df,
            SECTOR_REDEFINITIONS_CSV = SECTOR_REDEFINITIONS_CSV,
            num_sectors_to_trade = 3,
            num_candidate_sectors = 5,
            top_slope = 1)
        '''
        await trade_facade.take_additionals()
        

      

        
    
    import asyncio

    asyncio.get_event_loop().run_until_complete(main())