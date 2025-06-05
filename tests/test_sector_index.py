import pandas as pd
from calculation import SectorIndex

def test_sector_index_by_dict_simple():
    si = SectorIndex()
    price_col = si.price_cols
    fin_col = si.fin_cols
    sector_col = si.sector_cols

    stock_price = pd.DataFrame({
        price_col['日付']: pd.to_datetime(['2020-01-01', '2020-01-02']),
        price_col['銘柄コード']: ['1111', '1111'],
        price_col['始値']: [100, 110],
        price_col['高値']: [110, 115],
        price_col['安値']: [90, 105],
        price_col['終値']: [105, 108],
        price_col['調整係数']: [1, 1],
        price_col['取引高']: [1000, 1100],
    })

    stock_fin = pd.DataFrame({
        fin_col['銘柄コード']: ['1111'],
        fin_col['日付']: pd.to_datetime(['2019-12-31']),
        fin_col['発行済み株式数']: [100000],
        fin_col['当会計期間終了日']: pd.to_datetime(['2019-12-31']),
    })

    marketcap_df = si.calc_marketcap(stock_price, stock_fin)
    sector_index = si.calc_sector_index_by_dict({'Test': ['1111']}, marketcap_df)

    assert not sector_index.empty
    assert sector_col['終値'] in sector_index.columns
