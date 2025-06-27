import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from ml_core.data_bundle import DataBundle
from ml_core.unified_mlobject import UnifiedMLObject
from ml_core.multi_mlobject import MultiMLObject
from ml_core.ml_package import MLPackage


def _make_dummy_bundle() -> DataBundle:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    sectors = ["A", "B"]
    index = pd.MultiIndex.from_product([dates, sectors], names=["Date", "Sector"])
    features = pd.DataFrame({"f1": np.random.randn(len(index)), "f2": np.random.randn(len(index))}, index=index)
    targets = pd.DataFrame({"y": np.random.randn(len(index))}, index=index)
    order_price = pd.DataFrame({"price": np.random.rand(len(index))}, index=index)
    bundle = DataBundle(features, targets, order_price, {"outlier_threshold": 3.0})
    return bundle


def test_unified_mlobject_cycle():
    bundle = _make_dummy_bundle()
    ml = UnifiedMLObject()
    pkg = MLPackage(ml, bundle)
    pkg.train()
    preds = pkg.predict()
    assert not preds.empty
    with tempfile.TemporaryDirectory() as d:
        pkg.save(Path(d))
        loaded = MLPackage.load(Path(d))
        preds2 = loaded.predict()
        assert preds2.shape == preds.shape


def test_multi_mlobject_cycle():
    bundle = _make_dummy_bundle()
    ml = MultiMLObject()
    pkg = MLPackage(ml, bundle)
    pkg.train()
    preds = pkg.predict()
    assert not preds.empty
    with tempfile.TemporaryDirectory() as d:
        pkg.save(Path(d))
        loaded = MLPackage.load(Path(d))
        preds2 = loaded.predict()
        assert preds2.shape == preds.shape

