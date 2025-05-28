from dataclasses import dataclass
import pandas as pd

@dataclass
class StockSelectionMaterials:
    order_price_df: pd.DataFrame
    pred_result_df: pd.DataFrame