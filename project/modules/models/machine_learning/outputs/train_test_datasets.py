from dataclasses import dataclass
import pandas as pd

@dataclass
class TrainTestDatasets:
    target_train_df: pd.DataFrame
    target_test_df: pd.DataFrame
    features_train_df: pd.DataFrame
    features_test_df: pd.DataFrame