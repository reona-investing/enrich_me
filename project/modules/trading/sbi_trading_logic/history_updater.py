#%% モジュールのインポート
import shutil
import pandas as pd
from typing import Tuple
from datetime import datetime
from IPython.display import display
from trading.sbi import MarginManager, HistoryManager
from utils import Paths
#from sbi_trading_logic.order_maker import NewOrderMaker


class HistoryUpdater:
    def __init__(self, history_manager:HistoryManager, margin_manager: MarginManager,
                 sector_list_path: str, 
                 trade_history_path: str = Paths.TRADE_HISTORY_CSV, 
                 buying_power_history_path: str = Paths.BUYING_POWER_HISTORY_CSV, 
                 deposit_history_path: str = Paths.DEPOSIT_HISTORY_CSV):
        self.history_manager = history_manager
        self.margin_manager = margin_manager
        self.sector_list_path = sector_list_path
        self.trade_history_path = trade_history_path
        self.buying_power_history_path = buying_power_history_path
        self.deposit_history_path = deposit_history_path

    async def update_information(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, str]:
        '''
        sbi証券からスクレイピングして、取引情報、買付余力、総入金額を更新
        returns:
            pd.DataFrame: 過去の取引履歴
            pd.DataFrame: 信用建余力の履歴
            pd.DataFrame: 入出金の履歴
            float: 直近の損益額
            str: 直近の損益率
        '''
        trade_history = await self._update_trade_history(self.trade_history_path, self.sector_list_path, self.history_manager)
        buying_power_history = await self._update_buying_power_history(self.buying_power_history_path, trade_history, self.history_manager, self.margin_manager)
        deposit_history = await self._update_deposit_history(self.deposit_history_path, buying_power_history, self.history_manager)

        amount, rate = self._show_latest_result(trade_history)

        return trade_history, buying_power_history, deposit_history, rate, amount


    def load_information(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
        '''
        取引情報、買付余力、総入金額を更新せず読み込み
        returns:
            pd.DataFrame: 過去の取引履歴
            pd.DataFrame: 信用建余力の履歴
            pd.DataFrame: 入出金の履歴
            float: 直近の損益額
            str: 直近の損益率
        '''
        trade_history = pd.read_csv(self.trade_history_path, index_col=None)
        trade_history['日付'] = pd.to_datetime(trade_history['日付'])
        buying_power_history = pd.read_csv(self.buying_power_history_path, index_col=['日付'])
        buying_power_history.index = pd.to_datetime(buying_power_history.index)
        deposit_history = pd.read_csv(self.deposit_history_path, index_col=['日付'])
        deposit_history.index = pd.to_datetime(deposit_history.index)
        rate, amount = self._show_latest_result(trade_history)
        return trade_history, buying_power_history, deposit_history, rate, amount

    async def _update_trade_history(self, trade_history_path: str, sector_list_path: str, 
                                    history_manager: HistoryManager) -> pd.DataFrame:
        '''
        当日の取引結果を取得し、trade_historyのファイルを更新します。
        trade_history_path：過去の取引履歴を記録したdfのファイルパス
        sector_list：銘柄とセクターとの対応を記録したdf
        '''
        print('取引履歴を更新します。')
        # 取引履歴の更新
        trade_history = pd.read_csv(trade_history_path)
        sector_list = pd.read_csv(sector_list_path)
        trade_history['日付'] = pd.to_datetime(trade_history['日付']).dt.date # 後で同じ変換をするが、この処理いる？
        await history_manager.fetch_today_margin_trades(sector_list)
        trade_history = pd.concat([trade_history, history_manager.today_margin_trades_df], axis=0).reset_index(drop=True)
        trade_history['日付'] = pd.to_datetime(trade_history['日付']).dt.date
        trade_history = trade_history.sort_values(['日付', '売or買', '業種', '銘柄コード']).reset_index(drop=True)

        trade_history.to_csv(trade_history_path, index=False)
        trade_history = pd.read_csv(trade_history_path).drop_duplicates(keep='last') # なぜか読み込み直さないとdrop_duplicatesが効かない（データ型の問題？要検討）
        trade_history['日付'] = pd.to_datetime(trade_history['日付']).dt.date
        trade_history.to_csv(trade_history_path, index=False)
        shutil.copy(trade_history_path, Paths.TRADE_HISTORY_BACKUP)

        return trade_history


    async def _update_buying_power_history(self,
                                        buying_power_history_path: str, trade_history: pd.DataFrame, 
                                        history_manager: HistoryManager, margin_manager: MarginManager) -> pd.DataFrame:
        '''
        実行時点での買付余力を取得し、dfに反映します。
        buying_power_history_path：買付余力の推移を記録したdfのファイルパス
        '''
        buying_power_history = pd.read_csv(buying_power_history_path, index_col=None)
        buying_power_history['日付'] = pd.to_datetime(buying_power_history['日付']).dt.date

        # 買付余力の取得
        await margin_manager.fetch()
        # 今日の日付の行がなければ追加、あれば更新
        today = datetime.today().date()
        if buying_power_history[buying_power_history['日付'] == today].empty:
            new_row = pd.DataFrame([[today, margin_manager.buying_power]], columns=['日付', '買付余力'])
            buying_power_history = pd.concat([buying_power_history, new_row], axis=0).reset_index(drop=True)
        else:
            buying_power_history.loc[buying_power_history['日付']==today, '買付余力'] = margin_manager.buying_power

        # 取引がなかった日の行を除去する。
        days_traded = trade_history['日付'].unique()
        did_trade = buying_power_history['日付'].isin(days_traded)
        buying_power_history = buying_power_history[did_trade]
        buying_power_history = buying_power_history.set_index('日付', drop=True)

        print('買付余力の履歴')
        display(buying_power_history.tail(5))
        buying_power_history.to_csv(buying_power_history_path)
        shutil.copy(buying_power_history_path, Paths.BUYING_POWER_HISTORY_BACKUP)

        return buying_power_history


    async def _update_deposit_history(self, 
                                    deposit_history_df_path: str, buying_power_history: pd.DataFrame, 
                                    history_manager: HistoryManager) -> pd.DataFrame:
        '''
        総入金額を算出します。
        '''
        #入出金明細と現物の売買をスクレイピングして、当日以降の「譲渡益税源泉徴収金」と「譲渡益税還付金」を除いた入出金分を加減する。
        deposit_history_df = pd.read_csv(deposit_history_df_path)
        deposit_history_df['日付'] = pd.to_datetime(deposit_history_df['日付']).dt.date
        deposit_history_df = deposit_history_df.set_index('日付', drop=True)
        # 当日の入出金履歴をとる
        await history_manager.fetch_cashflow_transactions()
        cashflow_transactions_df = history_manager.cashflow_transactions_df
        if cashflow_transactions_df.empty:
            capital_diff = 0
        else:
            cashflow_transactions_df = cashflow_transactions_df[(cashflow_transactions_df['日付']>buying_power_history.index[-2])&(cashflow_transactions_df['日付']<=buying_power_history.index[-1])]
            capital_diff = cashflow_transactions_df['入出金額'].sum()
        # 現物の売買による資金の増減をとる
        await history_manager.fetch_today_stock_trades() #現物の売買
        spots_df = history_manager.today_stock_trades_df
        if len(spots_df) == 0:
            spots_diff = 0
        else:
            spots_diff = spots_df['買付余力増減'].sum()

        # 最新日のデータがすでに存在する場合は置換、存在しない場合は追加
        if deposit_history_df.index[-1] != buying_power_history.index[-1]:
            deposit_history_df.loc[buying_power_history.index[-1], '総入金額'] = deposit_history_df.loc[deposit_history_df.index[-1], '総入金額'] + capital_diff + spots_diff
        else:
            deposit_history_df.loc[buying_power_history.index[-1], '総入金額'] = deposit_history_df.loc[deposit_history_df.index[-2], '総入金額'] + capital_diff + spots_diff
        deposit_history_df = deposit_history_df.astype(int)
        print('総入金額の履歴')
        display(deposit_history_df.tail(5))
        deposit_history_df.to_csv(deposit_history_df_path)
        shutil.copy(deposit_history_df_path, Paths.DEPOSIT_HISTORY_BACKUP)

        return deposit_history_df


    def _show_latest_result(self, trade_history: pd.DataFrame) -> tuple[str, float]:
        date = trade_history['日付'].iloc[-1]
        amount = trade_history.loc[trade_history['日付'] == date, '利益（税引前）'].sum()
        rate = amount / trade_history.loc[trade_history['日付'] == date, '取得価格'].sum()
        amount = "{:,.0f}".format(amount)
        print(f'{date.strftime("%Y-%m-%d")}： 利益（税引前）{amount}円（{round(rate * 100, 3)}%）')
        return amount, rate
    

if __name__ == '__main__':
    from trading.sbi import LoginHandler
    import asyncio
    sector_path = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv'
    lh = LoginHandler()
    hm = HistoryManager(lh)
    mm = MarginManager(lh)
    hu = HistoryUpdater(hm, mm, sector_path)
    _, _, _, _, _ = asyncio.run(hu.update_information())