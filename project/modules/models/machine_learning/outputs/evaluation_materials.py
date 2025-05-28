from dataclasses import dataclass
import pandas as pd

@dataclass
class EvaluationMaterials:
    pred_result_df: pd.DataFrame
    raw_target_df: pd.DataFrame