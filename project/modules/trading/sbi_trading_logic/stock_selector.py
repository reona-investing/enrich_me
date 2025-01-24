import shutil
import pandas as pd
from typing import Tuple
from IPython.display import display
from utils.jquants_api_utils import cli
from trading.sbi import MarginManager, TradePossibilityManager
from utils.paths import Paths

class StockSelector:
    def __init__(self, 
                 order_price_df: pd.DataFrame,
                 pred_result_df: pd.DataFrame,
                 trade_possibility_manager: TradePossibilityManager,
                 margin_manager: MarginManager,
                 sector_definisions_csv: str,
                 num_sectors_to_trade: int = 3, 
                 num_candidate_sectors: int = 5, 
                 top_slope: float = 1.0, 
                 ):
        self.trade_possibility_manager = trade_possibility_manager
        self.margin_manager = margin_manager
        self.order_price_df = order_price_df
        self.pred_result_df = pred_result_df
        self.new_sectors_df = pd.read_csv(sector_definisions_csv)
        self.num_sectors_to_trade = num_sectors_to_trade
        self.num_candidate_sectors = num_candidate_sectors
        self.top_slope = top_slope
        self.buy_sectors = []
        self.sell_sectors = []

    async def select(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        当日の売買銘柄を選択する。
        Returns:
            pd.DataFrame: Longの選択銘柄
            pd.DataFrame: Shortの選択銘柄
            pd.DataFrame: 今日の予測業種
        '''
        weight_df = self._get_weight_df(self.order_price_df, self.new_sectors_df)
        todays_pred_df = self._get_todays_pred_df(self.pred_result_df)
        buyable_symbol_codes_df, sellable_symbol_codes_df = await self._get_tradable_dfs(self.new_sectors_df, self.trade_possibility_manager)
        # その業種の半分以上の銘柄が売買可能であれば、売買可能業種に設定
        todays_pred_df = self._append_tradability(todays_pred_df, self.new_sectors_df, buyable_symbol_codes_df, sellable_symbol_codes_df)
        # 売買対象の業種をdfとして抽出
        buy_sectors_df, sell_sectors_df = self._determine_sectors_to_trade(todays_pred_df, self.num_sectors_to_trade, self.num_candidate_sectors)
        self.buy_sectors = buy_sectors_df['Sector'].unique().tolist()
        self.sell_sectors = sell_sectors_df['Sector'].unique().tolist()
        long_orders, short_orders = await self._get_ls_orders_df(weight_df, buy_sectors_df, buyable_symbol_codes_df,
                                                           sell_sectors_df, sellable_symbol_codes_df, self.top_slope)
        long_orders, short_orders = self._calc_cumcost_by_ls(long_orders, short_orders)
        # Long, Shortの選択銘柄をCSVとして出力しておく
        long_orders.to_csv(Paths.LONG_ORDERS_CSV, index=False)
        short_orders.to_csv(Paths.SHORT_ORDERS_CSV, index=False)
        # Google Drive上にバックアップ
        shutil.copy(Paths.LONG_ORDERS_CSV, Paths.LONG_ORDERS_BACKUP)
        shutil.copy(Paths.SHORT_ORDERS_CSV, Paths.SHORT_ORDERS_BACKUP)

        return long_orders, short_orders, todays_pred_df

    def _get_weight_df(self, order_price_df: pd.DataFrame, new_sectors_df: pd.DataFrame) -> pd.DataFrame:
        '''
        銘柄ごとのインデックスウエイトを算出したデータフレームを作成する。
        Args:
            order_price_df (pd.DataFrame): 株価データを含むデータフレーム（列：Code, Close, Volume, Date, etc.）。
            new_sectors_df (pd.DataFrame): 銘柄ごとのセクター情報を含むデータフレーム（列：Code, CompanyName, Sector）。
        Returns:
            pd.DataFrame: 各銘柄のセクターごとのウエイトを含むデータフレーム。
        '''
        cost_and_volume_df = self._calc_index_weight_components(order_price_df)
        return self._calc_weight_df(new_sectors_df, cost_and_volume_df)

    def _calc_index_weight_components(self, price_for_order_df: pd.DataFrame) -> pd.DataFrame:
        '''
        終値時点での単位株購入コスト、過去5営業日の出来高平均、時価総額を算出する。
        Args:
            price_for_order_df (pd.DataFrame): 株価データを含むデータフレーム（列：Code, Close, Volume, Date, etc.）。
        Returns:
            pd.DataFrame: 列に銘柄コード（Code）、単位株購入コスト（EstimatedCost）、 
                        過去5営業日の出来高平均（Past5daysVolume）、時価総額（MarketCapClose）を含むデータフレーム。
        '''
        price_for_order_df['Past5daysVolume'] = price_for_order_df.groupby('Code')['Volume'].rolling(5).mean().reset_index(level=0, drop=True)
        price_for_order_df['EstimatedCost'] = price_for_order_df['Close'] * 100
        cost_and_volume_df = price_for_order_df.loc[price_for_order_df['Date']==price_for_order_df['Date'].iloc[-1], :]
        return cost_and_volume_df[['Code', 'EstimatedCost', 'Past5daysVolume', 'MarketCapClose']].reset_index(drop=True)

    def _calc_weight_df(self, sector_definisions_df: pd.DataFrame, cost_and_volume_df: pd.DataFrame) -> pd.DataFrame:
        '''
        各銘柄の時価総額に基づいて、セクターごとのインデックスウエイトを算出する。
        Args:
            sector_definisions_df (pd.DataFrame): 銘柄ごとのセクター情報を含むデータフレーム（列：Code, CompanyName, Sector）。
            cost_and_volume_df (pd.DataFrame): 銘柄ごとの購入コスト、出来高平均、時価総額を含むデータフレーム。
        Returns:
            pd.DataFrame: 各銘柄のセクターごとのインデックスウエイト（Weight）を含むデータフレーム。
        '''
        sector_definisions_df['Code'] = sector_definisions_df['Code'].astype(str)
        weight_df = pd.merge(sector_definisions_df[['Code', 'CompanyName', 'Sector']], cost_and_volume_df, how='right', on='Code')
        weight_df['Code'] = weight_df['Code'].astype(str)
        weight_df = weight_df[weight_df['Sector'].notnull()]
        sum_values = weight_df.groupby('Sector')['MarketCapClose'].sum()
        weight_df['Weight'] = weight_df.apply(lambda row: row['MarketCapClose'] / sum_values[row['Sector']], axis=1)
        weight_df['Weight'] = weight_df['Weight'].fillna(0)
        return weight_df

    def _get_todays_pred_df(self, target_test_df:pd.DataFrame) -> pd.DataFrame:
        '''最新日の予測結果dfを抽出'''
        target_test_df = target_test_df.reset_index().set_index('Date', drop=True)
        todays_pred_df = target_test_df[target_test_df.index==target_test_df.index[-1]].sort_values('Pred')
        todays_pred_df['Rank'] = todays_pred_df['Pred'].rank(ascending=False).astype(int)
        return todays_pred_df

    async def _get_tradable_dfs(self, sector_definisions_df:pd.DataFrame, trade_possibility_manager:TradePossibilityManager) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        売買可能な銘柄リストをそれぞれ算出する関数。
        Args:
            sector_definisions_df (pd.DataFrame): 銘柄コードを含むセクターリストのDataFrame。
            trade_possibility_manager (TradePossibilityManager): SBI証券で売買可能性を管理するオブジェクト。
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]
                - buyable_symbol_codes_df: 買い可能な銘柄のDataFrame
                - sellable_symbol_codes_df: 売り可能な銘柄のDataFrame
        """
        await trade_possibility_manager.fetch()
        buyable_symbol_codes_df = sector_definisions_df[sector_definisions_df['Code'].isin(trade_possibility_manager.data_dict['buyable_limits'].keys())].copy()
        sellable_symbol_codes_df = sector_definisions_df[sector_definisions_df['Code'].isin(trade_possibility_manager.data_dict['sellable_limits'].keys())].copy()
        buyable_symbol_codes_df['Code'] = buyable_symbol_codes_df['Code'].astype(str)
        sellable_symbol_codes_df['Code'] = sellable_symbol_codes_df['Code'].astype(str)
        sellable_symbol_codes_df['MaxUnit'] = sellable_symbol_codes_df['Code'].map(trade_possibility_manager.data_dict['sellable_limits'])
        sellable_symbol_codes_df['isBorrowingStock'] = sellable_symbol_codes_df['Code'].map(trade_possibility_manager.data_dict['borrowing_stocks'])
        return buyable_symbol_codes_df, sellable_symbol_codes_df

    def _append_tradability(self, todays_pred_df:pd.DataFrame, sector_definisions_df:pd.DataFrame,
                        buyable_symbol_codes_df:pd.DataFrame, sellable_symbol_codes_df:pd.DataFrame) -> pd.DataFrame:
        '''売買それぞれについて、可能な業種一覧を追加'''
        #売買可能業種数と、全体の業種数を算出
        buy = buyable_symbol_codes_df.groupby('Sector')[['Code']].count().rename(columns={'Code':'Buy'}).reset_index()
        sell = sellable_symbol_codes_df.groupby('Sector')[['Code']].count().rename(columns={'Code':'Sell'}).fillna(0).astype(int).reset_index()
        total = sector_definisions_df.groupby('Sector')[['Code']].count().rename(columns={'Code':'Total'}).fillna(0).astype(int).reset_index()
        #上記の内容をdfに落とし込む
        todays_pred_df = pd.merge(todays_pred_df, buy, how='left', on='Sector')
        todays_pred_df = pd.merge(todays_pred_df, sell, how='left', on='Sector')
        todays_pred_df = pd.merge(todays_pred_df, total, how='left', on='Sector')
        todays_pred_df[['Buy', 'Sell']] = todays_pred_df[['Buy', 'Sell']].fillna(0).astype(int)
        #半分以上の銘柄が売買可能な時，売買可能業種とする．
        todays_pred_df['Buyable'] = (todays_pred_df['Buy'] / todays_pred_df['Total'] >= 0.5).astype(int)
        todays_pred_df['Sellable'] = (todays_pred_df['Sell'] / todays_pred_df['Total'] >= 0.5).astype(int)
        return todays_pred_df

    def _determine_sectors_to_trade(self,
                                    todays_pred_df:pd.DataFrame,
                                    num_sectors_to_trade:int=3,
                                    num_candidate_sectors:int=5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        売買対象の業種を決定
        todays_pred_df：直近の日の予測結果df
        num_sectors_to_trade：上位・下位いくつの業種を最終的な売買対象とするか？
        num_sectors_to_trade：上位・下位いくつの業種を売買"候補"とするか？
        '''
        #売買可能な上位・下位3業種を抽出
        sell_sectors_df = todays_pred_df.iloc[:num_candidate_sectors].copy()
        sell_sectors_df = sell_sectors_df[sell_sectors_df['Sellable']==1].iloc[:num_sectors_to_trade]
        buy_sectors_df = todays_pred_df.iloc[-num_candidate_sectors:].copy()
        buy_sectors_df = buy_sectors_df[buy_sectors_df['Buyable']==1].iloc[-num_sectors_to_trade:]

        print('本日のロング予想業種')
        display(buy_sectors_df)
        print('--------------------------------------------')
        print('本日のショート予想業種')
        display(sell_sectors_df)
        print('--------------------------------------------')
        return buy_sectors_df, sell_sectors_df

    async def _get_ls_orders_df(self, weight_df: pd.DataFrame,
                                buy_sectors_df: pd.DataFrame, buyable_symbol_codes_df: pd.DataFrame,
                                sell_sectors_df: pd.DataFrame, sellable_symbol_codes_df: pd.DataFrame,
                                top_slope: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # 売買対象銘柄をdfとして抽出
        long_df = self._get_symbol_codes_to_trade_df(self.new_sectors_df, buy_sectors_df, buyable_symbol_codes_df, weight_df)
        short_df = self._get_symbol_codes_to_trade_df(self.new_sectors_df, sell_sectors_df, sellable_symbol_codes_df, weight_df)
        # 対象業種数が売買で異なる場合、何業種分をETFで補うかを算出
        sectors_to_trade_num, buy_adjuster_num, sell_adjuster_num = self._determine_whether_ordering_ETF(buy_sectors_df, sell_sectors_df)
        # ETFの必要注文数を算出
        long_df, short_df = self._calculate_ETF_orders(long_df, short_df, buy_adjuster_num, sell_adjuster_num)
        long_df['LorS'] = 'Long'
        short_df['LorS'] = 'Short'
        # 注文する銘柄と注文単位数を算出
        return await self._determine_orders(long_df, short_df, sectors_to_trade_num, top_slope, self.margin_manager)

    def _get_symbol_codes_to_trade_df(self,
                                      sector_definisions_df:pd.DataFrame,
                                      sectors_to_trade_df:pd.DataFrame,
                                      tradable_symbol_codes_df:pd.DataFrame,
                                      weight_df:pd.DataFrame) -> pd.DataFrame:
        '''売買それぞれの対象業種を抽出する'''
        symbol_codes_to_trade_df = sector_definisions_df[sector_definisions_df['Sector'].isin(sectors_to_trade_df['Sector'])].copy()
        symbol_codes_to_trade_df = pd.merge(symbol_codes_to_trade_df, sectors_to_trade_df[['Sector', 'Rank']])
        symbol_codes_to_trade_df['Code'] = symbol_codes_to_trade_df['Code'].astype(str)
        symbol_codes_to_trade_df = pd.merge(symbol_codes_to_trade_df, weight_df[['Code', 'Weight', 'EstimatedCost']], on='Code', how='left')
        symbol_codes_to_trade_df = symbol_codes_to_trade_df[symbol_codes_to_trade_df['Code'].isin(tradable_symbol_codes_df['Code'])]
        if 'MaxUnit' in tradable_symbol_codes_df.columns:
            symbol_codes_to_trade_df = pd.merge(symbol_codes_to_trade_df, tradable_symbol_codes_df[['Code', 'MaxUnit', 'isBorrowingStock']], how='left', on='Code')
        symbol_codes_to_trade_df['Weight'] = symbol_codes_to_trade_df['Weight'] / symbol_codes_to_trade_df.groupby('Sector')['Weight'].transform('sum')
        return symbol_codes_to_trade_df

    def _determine_whether_ordering_ETF(self, buy_sectors_df:pd.DataFrame, sell_sectors_df:pd.DataFrame) -> Tuple[int, int, int]:
        '''
        売買する業種数を判定し、不足分をETFで補う
        sectors_to_trade_num：売買する業種数
        buy_adjuster, sell_adjuster：何業種分をETFで置き換えるか？
        '''
        #buy_sectors_df, sell_sectors_dfのうち大きいほうの値をsectorsに格納
        sectors_to_trade_num = max(len(buy_sectors_df), len(sell_sectors_df))

        #buy_sector、sell_sectorがともに0の場合、何もしない
        if sectors_to_trade_num == 0:
            print('売買できる業種がありませんでした。')
        else:
            should_set_buy_adjuster = len(sell_sectors_df) > len(buy_sectors_df)
            if should_set_buy_adjuster:
                buy_adjuster_num = len(sell_sectors_df) - len(buy_sectors_df)
            else:
                buy_adjuster_num = 0
            should_set_sell_adjuster = len(buy_sectors_df) > len(sell_sectors_df)
            if should_set_sell_adjuster:
                sell_adjuster_num = len(buy_sectors_df) - len(sell_sectors_df)
            else:
                sell_adjuster_num = 0

        return sectors_to_trade_num, buy_adjuster_num, sell_adjuster_num

    def _calculate_ETF_orders(self, long_df:pd.DataFrame, short_df:pd.DataFrame,
                            buy_adjuster_num:int, sell_adjuster_num:int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if max(buy_adjuster_num, sell_adjuster_num):
            if buy_adjuster_num != 0:
                bull_list = cli.get_list(code='1308') #上場インデックスファンドTOPIX
                bull_price = cli.get_prices_daily_quotes(code='1308')
                for i in range (0, buy_adjuster_num):
                    bull_list['Code'] = bull_list['Code'].astype(str)[:4]
                    bull_list = bull_list[['Code', 'CompanyName', 'Sector33CodeName']]
                    bull_list['Sector'] = f'ETF{i}'
                    bull_list['Weight'] = 1
                    bull_list['EstimatedCost'] = bull_price['Close'].iat[-1] * 10
                    long_df = pd.concat([long_df, bull_list], axis=0)
            if sell_adjuster_num != 0:
                bear_list = cli.get_list(code='1356') #TOPIXベア2倍上場投信
                bear_price = cli.get_prices_daily_quotes(code='1356')
                for i in range (0, sell_adjuster_num):
                    bear_list['Code'] = bear_list['Code'].astype(str)[:4]
                    bear_list = bear_list[['Code', 'CompanyName', 'Sector33CodeName']]
                    bear_list['Sector'] = f'ETF{i}'
                    bear_list['Weight'] = 0.5 #ベア2倍のため、購入単位を半分にする
                    bear_list['EstimatedCost'] = bear_price['Close'].iat[-1] * 10
                    short_df = pd.concat([short_df, bear_list], axis=0)
        long_df = long_df.dropna(subset=['Weight', 'EstimatedCost'])
        short_df = short_df.dropna(subset=['Weight', 'EstimatedCost'])
        return long_df, short_df

    async def _determine_orders(self, long_df:pd.DataFrame, short_df:pd.DataFrame, 
                                sectors_to_trade_num:int, top_slope: float, 
                                margin_manager: MarginManager) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''発注銘柄と発注単位の算出'''
        #業種ごとの発注限度額の算出
        await margin_manager.fetch()
        maxcost_per_sector = margin_manager.margin_power / (sectors_to_trade_num * 2)
        #トップ業種に傾斜をつけない場合
        if top_slope == 1:
            print(f'業種ごとの発注限度額：{maxcost_per_sector}円')
            long_df = long_df.groupby('Sector').apply(self._get_unit, maxcost=maxcost_per_sector).reset_index(drop=True)
            short_df = short_df.groupby('Sector').apply(self._get_unit, maxcost=maxcost_per_sector).reset_index(drop=True)
        #トップ業種に傾斜をつける場合
        else:
            maxcost_top_sector = maxcost_per_sector * top_slope
            maxcost_except_top = maxcost_per_sector * (1 - (top_slope - 1) / (sectors_to_trade_num - 1))
            print('業種ごとの発注限度額')
            print(f'トップ業種{maxcost_top_sector}円')
            print(f'トップ以外{maxcost_except_top}円')
            long_top = long_df[long_df['Rank']==long_df['Rank'].min()]
            long_except_top = long_df[long_df['Rank']!=long_df['Rank'].min()]
            long_top = self._get_unit(long_top, maxcost=maxcost_top_sector)
            long_except_top = long_except_top.groupby('Sector').apply(self._get_unit, maxcost=maxcost_except_top).reset_index(drop=True)
            long_df = pd.concat([long_top, long_except_top], axis=0).reset_index(drop=True)
            short_top = short_df[short_df['Rank']==short_df['Rank'].max()]
            short_except_top = short_df[short_df['Rank']!=short_df['Rank'].max()]
            short_top = short_top.groupby('Sector').apply(self._get_unit, maxcost=maxcost_top_sector).reset_index(drop=True)
            short_except_top = short_except_top.groupby('Sector').apply(self._get_unit, maxcost=maxcost_except_top).reset_index(drop=True)
            short_df = pd.concat([short_top, short_except_top], axis=0).reset_index(drop=True)

        #発注量の算出
        long_orders = long_df[long_df['Unit']!=0].reset_index(drop=True) #買いの銘柄一覧
        short_orders = short_df[short_df['Unit']!=0].reset_index(drop=True) #売りの銘柄一覧
        #算出結果の表示
        pd.set_option('display.max_rows', None)
        print('ロング予想銘柄一覧')
        print(f'予想額：{long_orders["TotalCost"].sum()}円')
        display(long_orders)
        print('---------------------------')
        print('ショート予想銘柄一覧')
        print(f'予想額：{short_orders["TotalCost"].sum()}円')
        display(short_orders)
        pd.set_option('display.max_rows', 10)
        return long_orders, short_orders

    def _get_unit(self, df:pd.DataFrame, maxcost:int) -> pd.DataFrame:
        '''
        売買銘柄とその株数を算出する関数
        df：売買対象の銘柄を格納したデータフレーム
        maxcost：最大コスト
        '''
        df = self._get_ideal_costs(df, maxcost) #各種コストを算出
        df = self._draft_portfolio(df) #仮ポートフォリオ（最もインデックスに近い購入単位数）を作成
        df = self._reduce_to_max_unit(df)
        df = self._reduce_units(df, maxcost)
        df = self._increase_units(df, maxcost)
        return df.drop(['MaxUnitWithinIdeal', 'MaxCostWithinIdeal', 'MinCostExceedingIdeal', 'ReductionRate'], axis=1)

    def _get_ideal_costs(self, df:pd.DataFrame, maxcost:int) -> pd.DataFrame:
        '''
        以下の3つのコストを算出。
        IdealCost：完全にインデックス通りの銘柄配分となる理想コストを算出
        MaxCostWithinIdeal：理想コスト以下の最大コスト
        MinCostExceedingIdeal：理想コスト以上の最小コスト
        '''
        #完全にインデックス通りの銘柄配分となる理想コストを算出。
        df = df.sort_values('Weight', ascending=False).reset_index(drop=True).copy()
        df['IdealCost'] = (df['Weight'] * maxcost).astype(int)
        #理想コスト以下の最大コストと、理想コスト以上の最小コストを算出。
        df['MaxUnitWithinIdeal'] = df['IdealCost'] // df['EstimatedCost']
        df['MaxCostWithinIdeal'] = df['EstimatedCost'] * df['MaxUnitWithinIdeal']
        df['MinCostExceedingIdeal'] = df['MaxCostWithinIdeal'] + df['EstimatedCost']
        return df

    def _draft_portfolio(self, df:pd.DataFrame) -> pd.DataFrame:
        '''
        予算を無視して、理想コストに最も近い購入単位数を算出。
        '''
        #WithinとOverで、どちらが理想コストに近いかを算出。
        df['ReductionRate'] = abs(df['IdealCost'] - df['MaxCostWithinIdeal']) / abs(df['IdealCost'] - df['MinCostExceedingIdeal'])

        #各行について、最適な単位数（単位：100株）を割り出す。
        for index, row in df.iterrows():
            #理想コストに近いほうをトータルコストとしていったん格納。
            if row['ReductionRate'] <= 1:
                df.loc[index, 'TotalCost'] = row['MaxCostWithinIdeal']
            else:
                df.loc[index, 'TotalCost'] = row['MinCostExceedingIdeal']
            #理想コストから購入単位数を逆算
            df.loc[index, 'Unit'] = df.loc[index, 'TotalCost'] / row['EstimatedCost']
        return df

    def _reduce_to_max_unit(self, df):
        for i in range(len(df)):
            if 'MaxUnit' in df.columns:
                index_to_reduce = df.index[i]
                if df.loc[index_to_reduce, 'Unit'] >= df.loc[index_to_reduce, 'MaxUnit']:
                    df.loc[index_to_reduce, 'Unit'] = df.loc[index_to_reduce, 'MaxUnit']
                    df.loc[index_to_reduce, 'isMaxUnit'] = True
        return df

    def _reduce_units(self, df:pd.DataFrame, maxcost:int) -> pd.DataFrame:
        '''
        金額がオーバーしているとき、発注数量を減らす。
        発注対象銘柄の購入コストが全て理想コスト以内，かつトータルコストが予算内に収まるまで繰り返し。
        ReductionRate：大きいほど優先的に発注数量を減らす。
        '''
        #候補銘柄の抜き取り
        candidate_symbol_codes_df = \
            df.loc[(df['ReductionRate'] >= 1) & (df['Unit'] != 0), :].sort_values('ReductionRate', ascending=False)
        # 予算オーバーの場合、予算内に収まるまで購入数量を減らす。
        for i in range(len(candidate_symbol_codes_df)):
            if df['TotalCost'].sum() <= maxcost:
                break # トータルコストが予算内に収まったら繰り返し終了
            #ReductionRateの値が最も大きい銘柄について、発注数量を1単位減らす。
            index_to_reduce = candidate_symbol_codes_df.index[i]
            df.loc[index_to_reduce, 'TotalCost'] -= df.loc[index_to_reduce, 'EstimatedCost']
            df.loc[index_to_reduce, 'Unit'] -= 1
        return df[[x for x in df.columns if x != 'isMaxUnit']] 

    def _increase_units(self, df:pd.DataFrame, maxcost:int) -> pd.DataFrame:
        '''
        maxcostギリギリまで発注数量を増やす。
        diff_within < diff_overかつ、差が最小になる場合を判定したい。
        '''
        while True:
            free_cost = maxcost - df['TotalCost'].sum() #余裕コスト（最大これだけ増やせる）
            #発注数量を増やす銘柄を選定
            if 'isMaxUnit' in df.columns:
                filtered_df = df[(df['EstimatedCost'] <= free_cost) & (df['isMaxUnit'] == False)]
            else:
                filtered_df = df[df['EstimatedCost'] <= free_cost]
            if filtered_df.empty:
                break #発注数量を増やせる銘柄がなくなったら繰り返し終了。
            min_rate_row = filtered_df['ReductionRate'].idxmin()
            #発注数量とコスト、ReductionRateを更新
            df.loc[min_rate_row, ['TotalCost', 'MaxCostWithinIdeal', 'MinCostExceedingIdeal']] += df.loc[min_rate_row, 'EstimatedCost']
            df.loc[min_rate_row, 'Unit'] += 1
            df.loc[min_rate_row, 'ReductionRate'] = \
            abs(df.loc[min_rate_row, 'IdealCost'] - df.loc[min_rate_row, 'MaxCostWithinIdeal']) / \
                abs(df.loc[min_rate_row, 'IdealCost'] - df.loc[min_rate_row, 'MinCostExceedingIdeal'])
            if 'MaxUnit' in df.columns:
                if df.loc[min_rate_row, 'Unit'] == df.loc[min_rate_row, 'MaxUnit']:
                    df.loc[min_rate_row, 'isMaxUnit'] = True
        return df

    def _calc_cumcost_by_ls(self, long_orders: pd.DataFrame, short_orders: pd.DataFrame):
        '''Long, Shortそれぞれの累積コストを算出します。'''
        long_orders = long_orders.sort_values(by=['Rank', 'TotalCost'], ascending=[True, False])
        long_orders['CumCost_byLS'] = long_orders['TotalCost'].cumsum()
        short_orders = short_orders.sort_values(by=['Rank', 'TotalCost'], ascending=[False, False])
        short_orders['CumCost_byLS'] = short_orders['TotalCost'].cumsum()
        return long_orders, short_orders
    

if __name__  == '__main__':
    async def main():
        from trading.sbi.session import LoginHandler
        from models import MLDataset
        ml = MLDataset(f'{Paths.ML_DATASETS_FOLDER}/New48sectors')
        materials = ml.get_materials_for_stock_selection()
        lh = LoginHandler()
        tpm = TradePossibilityManager(lh)
        mm = MarginManager(lh)
        sd = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv'
        ss = StockSelector(materials.order_price_df, materials.pred_result_df, tpm, mm, sd)
        long, short, today = await ss.select()
        print(short)
    
    import asyncio
    asyncio.run(main())