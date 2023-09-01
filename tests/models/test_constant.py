from datetime import datetime, date

import pytest
import pandas as pd
import numpy as np

from python_project.data import ForecastParams, ForecastData, ForecastProblem
from python_project.models.constant import ConstantParams, ConstantModel


def test_should_raise_error_on_none():
    with pytest.raises(TypeError):
        ConstantParams(None)


@pytest.mark.parametrize(
    "value,expected",
    [(1, "CONSTANT_1.0"), (2, "CONSTANT_2.0")],
)
def test_name_model(value, expected):
    result = ConstantParams(value=value)
    assert result.name_model == expected


@pytest.mark.parametrize(
    "value,df,ds,expected",
    [
        (
            1,
            pd.DataFrame(
                {
                    "y": [1.0 for x in range(1, 17 + 1)],
                    "ds": pd.date_range(
                        datetime.strptime("2020-01-06", "%Y-%m-%d"),
                        periods=17,
                        freq="7d",
                    ),
                }
            ),
            date(2021, 1, 2),
            np.ones(15),
        )
    ],
)
def test_Constant_do_forecast(value, df, ds, expected):
    cons_params = ConstantParams(value=value)

    params = ForecastParams(
        forecast_start=ds,
    )
    data = ForecastData(df=df)
    forecast_problem = ForecastProblem(params=params, data=data)
    result = ConstantModel(cons_params).do_forecast(forecast_problem)
    np.testing.assert_array_equal(result, expected)
