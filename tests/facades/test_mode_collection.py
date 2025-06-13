import builtins
import pytest

from project.modules.facades.mode_setting import (
    ModeCollection,
    OrderExecutionMode,
    MachineLearningMode,
    DataUpdateMode,
)


def test_adjust_machine_learning_mode(capsys):
    mc = ModeCollection(
        order_execution_mode=OrderExecutionMode.NEW,
        machine_learning_mode=MachineLearningMode.NONE,
    )
    captured = capsys.readouterr()
    assert mc.machine_learning_mode is MachineLearningMode.LOAD_ONLY
    assert "machine_learning_modeを'load_only'に変更" in captured.out


def test_adjust_data_update_mode(capsys):
    mc = ModeCollection(
        machine_learning_mode=MachineLearningMode.TRAIN_AND_PREDICT,
        data_update_mode=DataUpdateMode.NONE,
    )
    captured = capsys.readouterr()
    assert mc.data_update_mode is DataUpdateMode.LOAD_ONLY
    assert "data_update_modeを'load_only'に変更" in captured.out

