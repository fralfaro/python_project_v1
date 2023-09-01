import numpy as np
import pytest
from pandas import DataFrame
from pandas._testing import assert_frame_equal

from python_project.metrics.standard import StandardMetrics


@pytest.mark.parametrize(
    "y, pred", [(np.array([1, 2, 3, 4, 5]), np.array([2, 3, 4, 4, 4]))]
)
@pytest.mark.parametrize(
    "func,expected",
    [
        (StandardMetrics.mae, 0.8),
        (StandardMetrics.mse, 0.8),
        (StandardMetrics.rmse, 0.8944),
        (StandardMetrics.mape, 0.4067),
        (StandardMetrics.maape, 0.3536),
        (StandardMetrics.wmape, 0.2667),
        (StandardMetrics.mmape, 0.25),
        (StandardMetrics.smape, 0.3149),
    ],
)
def test_single_metrics(y, pred, func, expected):
    result = func(y, pred)
    assert result == expected


@pytest.mark.parametrize(
    "y, pred",
    [
        (
            np.array([0, 0]),
            np.array([0, 0]),
        )
    ],
)
@pytest.mark.parametrize(
    "func,expected",
    [
        (StandardMetrics.mae, 0),
        (StandardMetrics.mse, 0),
        (StandardMetrics.rmse, 0),
        (StandardMetrics.mape, np.inf),
        (StandardMetrics.maape, np.inf),
        (StandardMetrics.wmape, np.inf),
        (StandardMetrics.mmape, 0),
        (StandardMetrics.smape, np.inf),
    ],
)
def test_single_metrics_2(y, pred, func, expected):
    result = func(y, pred)
    assert result == expected


@pytest.mark.parametrize(
    "y,yhat, expected",
    [
        (
            np.array([1, 0, 3, 4, 5]),
            np.array([0, 3, 4, 4, 4]),
            DataFrame(
                {
                    "mae": [1.2],
                    "mse": [2.4],
                    "rmse": [1.55],
                    "mape": [np.inf],
                    "maape": [0.58],
                    "wmape": [0.46],
                    "mmape": [0.78],
                    "smape": [0.90],
                }
            ),
        )
    ],
)
def test_summary_metrics(y, yhat, expected):
    result = StandardMetrics().calculate(y, yhat)
    assert_frame_equal(result, expected)
