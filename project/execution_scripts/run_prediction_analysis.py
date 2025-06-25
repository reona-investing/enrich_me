from __future__ import annotations

import numpy as np
import pandas as pd

from modules.performance import PredictionReturnExecutor
from modules.return_metrics_tool import TradeDailyReturnGenerator


def main() -> None:
    dates = pd.date_range("2024-01-01", periods=30)
    predicted = pd.Series(np.random.randn(len(dates)) / 100, index=dates)
    actual = pd.Series(np.random.randn(len(dates)) / 100, index=dates)

    executor = PredictionReturnExecutor(predicted, actual)
    result = executor.execute()
    print(result)


if __name__ == "__main__":
    main()
