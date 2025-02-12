import os
from utils.paths import Paths
from utils.notifier import SlackNotifier
from trading.sbi import OrderManager, HistoryManager, MarginManager, LoginHandler, TradePossibilityManager
from trading.sbi_trading_logic import StockSelector, NewOrderMaker, AdditionalOrderMaker, PositionSettler, HistoryUpdater
from models import MLDataset

class TradingFacade:
    def __init__(self):
        """SBI取引操作を統括するファサード"""
        self.login_handler = LoginHandler()
        self.slack = SlackNotifier(program_name=os.path.basename(__file__))
        self.history_manager = HistoryManager(self.login_handler)
        self.margin_manager = MarginManager(self.login_handler)
        self.order_manager = OrderManager(self.login_handler)
        self.trade_possibility_manager = TradePossibilityManager(self.login_handler)
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
        materials = ml_dataset.stock_selection_materials
        stock_selector = StockSelector(materials.order_price_df, materials.pred_result_df,
                                       self.trade_possibility_manager, self.margin_manager,
                                       SECTOR_REDEFINITIONS_CSV, num_sectors_to_trade, num_candidate_sectors, top_slope)
        long_orders, short_orders, _ = await stock_selector.select(self.margin_manager.buying_power)
        order_maker = NewOrderMaker(long_orders, short_orders, self.order_manager)
        failed_order_list = await order_maker.run_new_orders()
        self.slack.send_message(f'発注が完了しました。\n買：{stock_selector.buy_sectors}\n売：{stock_selector.sell_sectors}')
        if failed_order_list:
            self.slack.send_message(f'以下の注文の発注に失敗しました。\n{failed_order_list}')

    async def take_additionals(self):
        '''
        take_positionsで失敗した注文について、追加で信用新規建を行います。
        '''
        order_maker = AdditionalOrderMaker(self.order_manager)
        failed_order_list = await order_maker.run_additional_orders()
        self.slack.send_message('追加発注が完了しました。')
        if failed_order_list:
            self.slack.send_message(f'以下の注文の発注に失敗しました。\n{failed_order_list}')

    async def settle_positions(self):
        '''
        信用ポジションの決済注文を発注します。
        '''
        position_settler = PositionSettler(self.order_manager)
        await position_settler.settle_all_margins()
        if not self.order_manager.error_tickers:
            self.slack.send_message('全銘柄の決済注文が完了しました。')
        else:
            self.slack.send_message(f'銘柄コード{self.order_manager.error_tickers}の決済注文に失敗しました。')

    async def fetch_invest_result(self, SECTOR_REDEFINITIONS_CSV):
        '''
        当日の取引履歴・入出金履歴・買付余力を取得します。
        '''
        history_updater = HistoryUpdater(self.history_manager, self.margin_manager, SECTOR_REDEFINITIONS_CSV)
        trade_history, _, _, _, amount = await history_updater.update_information()
        self.slack.send_result(f'取引履歴等の更新が完了しました。\n{trade_history["日付"].iloc[-1].strftime("%Y-%m-%d")}の取引結果：{amount}円')
