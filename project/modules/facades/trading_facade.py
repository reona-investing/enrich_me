import os
from utils.paths import Paths
from utils.notifier import SlackNotifier
from trading.sbi.browser import SBIBrowserManager
from trading.sbi.orders.interface import IOrderExecutor, IMarginProvider
from trading.sbi.orders.manager.margin_provider import SBIMarginProvider
from trading.sbi.orders.manager.order_executor import SBIOrderExecutor
from trading.sbi.orders.ordermaker.batch_order_maker import BatchOrderMaker
from trading.sbi.orders.ordermaker.position_settler import PositionSettler
from trading.sbi import HistoryManager, TradePossibilityManager
from trading.sbi_trading_logic import StockSelector, HistoryUpdater
from models import MLDataset

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
        self.trade_possibility_manager = TradePossibilityManager(self.browser_manager)
        
        # 注文関連のクラスの初期化
        self.batch_order_maker = BatchOrderMaker(self.order_executor, self.margin_provider)
        self.position_settler = PositionSettler(self.order_executor, self.margin_provider)
        
        self.Paths = Paths

    async def take_positions(self, 
                             ml_dataset: MLDataset, 
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
        # 利用可能な証拠金を取得
        await self.margin_provider.refresh()
        available_margin = await self.margin_provider.get_available_margin()
        
        # 銘柄選択処理
        materials = ml_dataset.stock_selection_materials
        stock_selector = StockSelector(materials.order_price_df, materials.pred_result_df,
                                       self.trade_possibility_manager, self.margin_provider,
                                       SECTOR_REDEFINITIONS_CSV, num_sectors_to_trade, 
                                       num_candidate_sectors, top_slope)
        
        orders_df, _ = await stock_selector.select(available_margin)
        
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
        from models.dataset import MLDataset
        ML_DATASET_PATH = f'{Paths.ML_DATASETS_FOLDER}/48sectors_LASSO_learned_in_250308'
        SECTOR_REDEFINITIONS_CSV = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv'
        ml_dataset = MLDataset(ML_DATASET_PATH)
        trade_facade = TradingFacade()

        '''
        await trade_facade.take_positions(
            ml_dataset= ml_dataset,
            SECTOR_REDEFINITIONS_CSV = SECTOR_REDEFINITIONS_CSV,
            num_sectors_to_trade = 3,
            num_candidate_sectors = 5,
            top_slope = 1)
        '''
        await trade_facade.take_additionals()
        
    
    import asyncio

    asyncio.get_event_loop().run_until_complete(main())