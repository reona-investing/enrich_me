#%%モジュールのインポート
from utils.paths import Paths

import os
from datetime import datetime
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

#%%
class TradeHistoryAnalyzer:
    def __init__(self, start_date: datetime, end_date: datetime):
        #データフレームを格納する辞書を初期化
        self.dataframes_dict = {}
        # 取引履歴の読み込み
        trade_history = pd.read_csv(Paths.TRADE_HISTORY_CSV)
        trade_history['日付'] = pd.to_datetime(trade_history['日付'])
        trade_history = trade_history[(trade_history['日付'] >= start_date) & (trade_history['日付'] <= end_date)]
        self.trade_history  = trade_history

        # 総入金額の読み込み
        deposit_history = pd.read_csv(Paths.DEPOSIT_HISTORY_CSV)
        deposit_history['日付'] = pd.to_datetime(deposit_history['日付'])
        deposit_history = deposit_history[(deposit_history['日付'] >= start_date) & (deposit_history['日付'] <= end_date)]
        deposit_history = deposit_history.set_index('日付', drop=True)
        deposit_history['平均入金額'] = deposit_history['総入金額'].expanding().mean().astype(int)
        self.deposit_history = deposit_history
        
        # 総入金額の読み込み
        buying_power_history = pd.read_csv(Paths.BUYING_POWER_HISTORY_CSV)
        buying_power_history['日付'] = pd.to_datetime(buying_power_history['日付'])
        buying_power_history = buying_power_history[(buying_power_history['日付'] >= start_date) & (buying_power_history['日付'] <= end_date)]
        buying_power_history = buying_power_history.set_index('日付', drop=True)
        self.buying_power_history = buying_power_history

        # 実行
        self.dataframes_dict['日次リターン'], \
        self.dataframes_dict['日次リターン（集計）'] \
            = self.calculate_daily_profit(self.trade_history)
        self.dataframes_dict['日次リターン（業種別）'] \
            = self.calculate_daily_profit_by_sector()
        self.calculate_daily_leveraged()
        self.calculate_daily_cumulative()
        self.calculate_daily_leveraged_cumurative()
        self.calculate_monthly_profit()
        self.calculate_monthly_profit_by_sector()
        self.calculate_monthly_leveraged()
        self.calculate_total_profit()
        self.calculate_total_profit_by_sector()
        self.calculate_total_leveraged()


    def calculate_daily_profit(self, trade_history = pd.DataFrame):
        '''
        日次の利益を算出する。
        '''
        daily_profit = trade_history.groupby('日付')[['取得価格', '利益（税引前）']].sum()
        daily_profit['利率（税引前）'] = daily_profit['利益（税引前）'] / daily_profit['取得価格']

        # 当日税額を算出する。
        daily_profit['年間利益（税引前）'] = daily_profit.resample('YE')['利益（税引前）'].cumsum()
        daily_profit['年間税額'] = (daily_profit['年間利益（税引前）'] * 0.20315).astype(int)
        daily_profit.loc[daily_profit['年間税額'] < 0, '年間税額'] = 0
        daily_profit['日付'] = daily_profit.index
        daily_profit['当日税額'] = daily_profit.groupby(daily_profit['日付'].dt.year)['年間税額'].diff(1)
        # daily_profit.loc[daily_profit['当日税額'].isna(), '当日税額'] = daily_profit.loc[daily_profit['当日税額'].isna(), '年間税額']
        daily_profit['当日税額'] = daily_profit['当日税額'].where(~daily_profit['当日税額'].isna(), daily_profit['年間税額'])
        daily_profit['当日税額'] = daily_profit['当日税額'].astype(int)

        # 税引後利益・利率を算出する。
        daily_profit['利益（税引後）'] = daily_profit['利益（税引前）'] -  daily_profit['当日税額']
        daily_profit['利率（税引後）'] = daily_profit['利益（税引後）'] / daily_profit['取得価格']

        # 必要列を抜き出す。
        daily_profit = daily_profit[['取得価格', '利益（税引前）', '利率（税引前）', '利益（税引後）', '利率（税引後）']]
        daily_profit_agg = daily_profit[['利率（税引前）', '利率（税引後）']].describe()
        daily_profit_agg.loc['SR'] = \
          daily_profit_agg.loc['mean'] / daily_profit_agg.loc['std']
        
        return daily_profit, daily_profit_agg

    def calculate_daily_profit_by_sector(self):
        '''
        日次の業種別利益を算出する。
        '''
        daily_profit_by_sector = self.trade_history.groupby('業種').apply(lambda x: self.calculate_daily_profit(x)[0], include_groups=False).reset_index(drop=False)
        daily_profit_by_sector = pd.merge(self.trade_history[['日付', '売or買', '業種']], daily_profit_by_sector, how='right', on=['日付', '業種'])
        daily_profit_by_sector = daily_profit_by_sector.drop_duplicates().set_index(['日付', '売or買', '業種']).sort_index()
        
        return daily_profit_by_sector

    def calculate_daily_leveraged(self):
        daily_leveraged = self.deposit_history[['総入金額']]
        daily_leveraged[['利益（税引前）', '利益（税引後）']] = self.dataframes_dict['日次リターン'][['利益（税引前）', '利益（税引後）']]
        daily_leveraged['利率（税引前）'] = daily_leveraged['利益（税引前）'] / daily_leveraged['総入金額']
        daily_leveraged['利率（税引後）'] = daily_leveraged['利益（税引後）'] / daily_leveraged['総入金額']
        self.dataframes_dict['日次リターン（レバ込み）'] = daily_leveraged
        self.dataframes_dict['日次リターン（レバ込み）（集計）'] = daily_leveraged[['利率（税引前）', '利率（税引後）']].describe()
        self.dataframes_dict['日次リターン（レバ込み）（集計）'].loc['SR'] = \
          self.dataframes_dict['日次リターン（レバ込み）（集計）'].loc['mean'] / self.dataframes_dict['日次リターン（レバ込み）（集計）'].loc['std']


    def calculate_daily_cumulative(self):
        daily_cumprod = pd.DataFrame()
        daily_cumprod['累積平均取得価格'] = self.dataframes_dict['日次リターン']['取得価格'].expanding().mean()
        daily_cumprod['通算利益（税引前）'] = self.dataframes_dict['日次リターン']['利益（税引前）'].cumsum()
        daily_cumprod['通算利率（税引前）'] = (self.dataframes_dict['日次リターン'][['利率（税引前）']] + 1).cumprod() - 1
        daily_cumprod['通算利益（税引後）'] = self.dataframes_dict['日次リターン']['利益（税引後）'].cumsum()
        daily_cumprod['通算利率（税引後）'] = (self.dataframes_dict['日次リターン'][['利率（税引後）']] + 1).cumprod() - 1
        self.dataframes_dict['日次リターン（累積）'] = daily_cumprod


    def calculate_daily_leveraged_cumurative(self):
        daily_cumprod = pd.DataFrame()
        daily_cumprod['平均入金額'] = self.deposit_history['平均入金額']
        daily_cumprod['通算利益（税引前）'] = self.dataframes_dict['日次リターン（レバ込み）']['利益（税引前）'].expanding().mean()
        daily_cumprod['通算利益（税引前）'] = self.dataframes_dict['日次リターン（レバ込み）']['利益（税引前）'].cumsum()
        daily_cumprod['通算利率（税引前）'] = (self.dataframes_dict['日次リターン（レバ込み）'][['利率（税引前）']] + 1).cumprod() - 1
        daily_cumprod['通算利益（税引後）'] = self.dataframes_dict['日次リターン（レバ込み）']['利益（税引後）'].cumsum()
        daily_cumprod['通算利率（税引後）'] = (self.dataframes_dict['日次リターン（レバ込み）'][['利率（税引後）']] + 1).cumprod() - 1
        self.dataframes_dict['日次リターン（レバ込み）（累積）'] = daily_cumprod


    def calculate_monthly_profit(self):
        monthly_profit = self.dataframes_dict['日次リターン'].resample('ME')[['取得価格']].mean()
        monthly_profit.columns = ['平均取得価格']
        monthly_profit['日数'] = self.dataframes_dict['日次リターン'].resample('ME')['取得価格'].count()
        monthly_profit[['利益（税引前）', '利益（税引後）']] = self.dataframes_dict['日次リターン'].resample('ME')[['利益（税引前）', '利益（税引後）']].sum()
        monthly_profit['利率（税引前）'] = monthly_profit['利益（税引前）'] / monthly_profit['平均取得価格']
        monthly_profit['利率（税引後）'] = monthly_profit['利益（税引後）'] / monthly_profit['平均取得価格']
        self.dataframes_dict['月次リターン'] = monthly_profit

    def calculate_monthly_profit_by_sector(self):
        df_grouped = self.dataframes_dict['日次リターン（業種別）'].reset_index().groupby(['売or買', '業種'])
        monthly_profit = df_grouped.resample('ME', on='日付')[['取得価格']].mean()
        monthly_profit.columns = ['平均取得価格']
        monthly_profit['日数'] = df_grouped.resample('ME', on='日付')['取得価格'].count()
        monthly_profit[['利益（税引前）', '利益（税引後）']] = df_grouped.resample('ME', on='日付')[['利益（税引前）', '利益（税引後）']].sum()
        monthly_profit['利率（税引前）'] = monthly_profit['利益（税引前）'] / monthly_profit['平均取得価格']
        monthly_profit['利率（税引後）'] = monthly_profit['利益（税引後）'] / monthly_profit['平均取得価格']
        self.dataframes_dict['月次リターン（業種別）'] = monthly_profit.swaplevel(1, 2).swaplevel(0, 1).sort_index()

    def calculate_monthly_leveraged(self):
        monthly_profit = pd.DataFrame()
        monthly_profit['平均入金額'] = self.buying_power_history.resample('ME')['買付余力'].first() + \
                                      self.deposit_history.resample('ME')['総入金額'].mean() - \
                                      self.deposit_history.resample('ME')['総入金額'].first()
        monthly_profit.columns = ['平均入金額']
        monthly_profit['日数'] = self.deposit_history.resample('ME')['総入金額'].count()
        monthly_profit[['利益（税引前）', '利益（税引後）']] = self.dataframes_dict['日次リターン（レバ込み）'].resample('ME')[['利益（税引前）', '利益（税引後）']].sum()
        monthly_profit['利率（税引前）'] = monthly_profit['利益（税引前）'] / monthly_profit['平均入金額']
        monthly_profit['利率（税引後）'] = monthly_profit['利益（税引後）'] / monthly_profit['平均入金額']
        self.dataframes_dict['月次リターン（レバ込み）'] = monthly_profit

    def calculate_total_profit(self):
        total_profit = self.dataframes_dict['日次リターン'][['取得価格']].mean()
        total_profit['日数'] =self.dataframes_dict['日次リターン']['取得価格'].count()
        total_profit['利益（税引前）'] = self.dataframes_dict['日次リターン']['利益（税引前）'].sum()
        total_profit['利益（税引後）'] = self.dataframes_dict['日次リターン']['利益（税引後）'].sum()
        total_profit['利率（税引前）'] = total_profit.loc['利益（税引前）'] / total_profit.loc['取得価格']
        total_profit['利率（税引後）'] = total_profit.loc['利益（税引後）'] / total_profit.loc['取得価格']
        total_profit = pd.DataFrame(total_profit[['取得価格', '日数', '利益（税引前）', '利益（税引後）', '利率（税引前）', '利率（税引後）']], columns=['通算']).T
        total_profit[['取得価格', '利益（税引前）', '利益（税引後）']] = total_profit[['取得価格', '利益（税引前）', '利益（税引後）']].astype(int)
        total_profit = total_profit.rename(columns={'利率（税引前）': '平均リターン（税引前）', '利率（税引後）': '平均リターン（税引後）'})
        self.dataframes_dict['通算損益'] = total_profit

    def calculate_total_profit_by_sector(self):
        df_grouped = self.dataframes_dict['日次リターン（業種別）'].reset_index().groupby(['売or買', '業種'])
        total_profit = df_grouped[['取得価格']].mean()
        total_profit['日数'] =df_grouped['取得価格'].count()
        total_profit['利益（税引前）'] = df_grouped['利益（税引前）'].sum()
        total_profit['利益（税引後）'] = df_grouped['利益（税引後）'].sum()
        total_profit['平均リターン（税引前）'] = total_profit['利益（税引前）'] / total_profit['取得価格']
        total_profit['平均リターン（税引後）'] = total_profit['利益（税引後）'] / total_profit['取得価格']
        self.dataframes_dict['通算損益（業種別）'] = total_profit     

    def calculate_total_leveraged(self):
        total_profit = self.deposit_history[['総入金額']].mean()
        total_profit.index = ['平均入金額']
        total_profit['日数'] =self.deposit_history['総入金額'].count()
        total_profit['利益（税引前）'] = self.dataframes_dict['日次リターン（レバ込み）']['利益（税引前）'].sum()
        total_profit['利益（税引後）'] = self.dataframes_dict['日次リターン（レバ込み）']['利益（税引後）'].sum()
        total_profit['利率（税引前）'] = total_profit.loc['利益（税引前）'] / total_profit.loc['平均入金額']
        total_profit['利率（税引後）'] = total_profit.loc['利益（税引後）'] / total_profit.loc['平均入金額']
        total_profit = pd.DataFrame(total_profit[['平均入金額', '日数', '利益（税引前）', '利益（税引後）', '利率（税引前）', '利率（税引後）']], columns=['通算']).T
        total_profit[['平均入金額', '利益（税引前）', '利益（税引後）']] = total_profit[['平均入金額', '利益（税引前）', '利益（税引後）']].astype(int)
        total_profit = total_profit.rename(columns={'利率（税引前）': 'トータルリターン（税引前）', '利率（税引後）': 'トータルリターン（税引後）'})
        self.dataframes_dict['通算損益（レバ込み）'] = total_profit

    def display_result(self):
        '''任意のデータフレームを動的に描画(改善版)'''
        # データフレーム選択用ドロップダウン
        df_dropdown = widgets.Dropdown(
            options=self.dataframes_dict.keys(),
            description='データフレーム:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )

        # 出力形式選択用ラジオボタン（テーブル or グラフ）
        output_mode = widgets.RadioButtons(
            options=['テーブル', 'グラフ'],
            description='表示形式:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )

        # スタイリングチェックボックス（テーブル表示時のみ有効）
        styling_checkbox = widgets.Checkbox(
            value=False,
            description='カラースケールスタイリング（数値列）',
            layout=widgets.Layout(width='300px')
        )

        # ダウンロードボタン
        download_button = widgets.Button(
            description='CSVダウンロード',
            tooltip='現在表示中のDataFrameをCSVとしてダウンロード',
            icon='download'
        )

        # 出力エリア
        output_area = widgets.Output()

        # グラフに関するオプション（Y列選択）
        y_columns_dropdown = widgets.SelectMultiple(
            description='グラフ表示列:',
            options=[],
            value=(),
            layout=widgets.Layout(width='300px', height='120px'),
            style={'description_width': 'initial'},
            disabled=True
        )

        # イベントハンドラ
        def on_df_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                selected_df_name = df_dropdown.value
                selected_df = self.dataframes_dict[selected_df_name]
                # グラフ用Y列選択を更新(数値列のみ)
                numeric_cols = selected_df.select_dtypes(include='number').columns.tolist()
                y_columns_dropdown.options = numeric_cols
                if len(numeric_cols) > 0:
                    y_columns_dropdown.disabled = False
                    y_columns_dropdown.value = tuple(numeric_cols[:2]) if len(numeric_cols) >= 2 else tuple(numeric_cols)
                else:
                    y_columns_dropdown.disabled = True
                    y_columns_dropdown.value = ()

        def on_output_mode_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                # テーブル・グラフ表示選択に応じて適宜オプションを有効/無効化
                if output_mode.value == 'テーブル':
                    styling_checkbox.disabled = False
                    y_columns_dropdown.disabled = True
                else:
                    styling_checkbox.disabled = True
                    selected_df = self.dataframes_dict[df_dropdown.value]
                    numeric_cols = selected_df.select_dtypes(include='number').columns.tolist()
                    if len(numeric_cols) > 0:
                        y_columns_dropdown.disabled = False
                    else:
                        y_columns_dropdown.disabled = True

        def on_show_button_click(_):
            with output_area:
                output_area.clear_output()
                selected_df_name = df_dropdown.value
                selected_df = self.dataframes_dict[selected_df_name]

                if output_mode.value == 'テーブル':
                    # テーブル表示
                    styled_df = selected_df
                    if styling_checkbox.value:
                        # 数値列をカラースケールでスタイリング
                        numeric_cols = styled_df.select_dtypes(include='number').columns
                        cmap = sns.diverging_palette(5, 250, as_cmap=True)
                        styled_df = styled_df.style.background_gradient(cmap=cmap, subset=numeric_cols, axis=0)
                    pd.set_option('display.max_rows', None)
                    pd.set_option('display.max_columns', None)
                    display(styled_df)
                    pd.reset_option('display.max_rows')
                    pd.reset_option('display.max_columns')
                else:
                    # グラフ表示
                    if len(y_columns_dropdown.value) == 0:
                        print("グラフ表示する列が選択されていません。")
                        return
                    fig = go.Figure()
                    for col in y_columns_dropdown.value:
                        fig.add_trace(go.Scatter(x=selected_df.index, y=selected_df[col], mode='lines', name=col))
                    fig.update_layout(
                        title=f"{selected_df_name} のグラフ表示",
                        xaxis_title='日付(インデックス)',
                        yaxis_title='値',
                        hovermode='x unified'
                    )
                    fig.show()

            # 「表示・更新」クリック時に「出力結果」タブへ切り替え
            tab.selected_index = 1

        def on_download_button_click(_):
            # ダウンロードボタン
            selected_df_name = df_dropdown.value
            selected_df = self.dataframes_dict[selected_df_name]
            csv_data = selected_df.to_csv().encode('utf-8')
            # ダウンロードリンク生成
            from IPython.display import HTML
            html = HTML(
                f'<a download="{selected_df_name}.csv" href="data:text/csv;base64,{csv_data.decode("utf-8").encode("ascii","xmlcharrefreplace").decode("ascii")}">こちらをクリックしてCSVをダウンロード</a>')
            with output_area:
                output_area.clear_output()
                display(html)

        # 各種イベント登録
        df_dropdown.observe(on_df_change, names='value')
        output_mode.observe(on_output_mode_change, names='value')
        download_button.on_click(on_download_button_click)

        # 表示ボタン
        show_button = widgets.Button(description='表示/更新', icon='refresh')
        show_button.on_click(on_show_button_click)

        # 初期表示時に列選択などを設定
        on_df_change({'type':'change','name':'value','new':df_dropdown.value})

        # レイアウト構成（タブを使用して整理）
        tab = widgets.Tab()
        
        # "表示設定"タブ
        config_box = widgets.VBox([
            widgets.HTML("<h3>表示設定</h3>"),
            df_dropdown,
            output_mode,
            styling_checkbox,
            y_columns_dropdown,
            show_button,
            download_button
        ])

        # "出力結果"タブ
        result_box = widgets.VBox([
            widgets.HTML("<h3>出力結果</h3>"),
            output_area
        ])

        tab.children = [config_box, result_box]
        tab.set_title(0, '表示設定')
        tab.set_title(1, '出力結果')

        display(tab)


    def plot_graphs(self):
        '''任意のグラフを動的に描画'''
        selected_df = self.dataframes_dict['日次リターン（累積）']
        fig = px.line(selected_df, y=['通算利益（税引前）', '通算利益（税引後）'], title='通算利益推移')


if __name__ == '__main__':
    TradeHistoryAnalyzer(start_date=datetime(2023,3,1), end_date=datetime.today())