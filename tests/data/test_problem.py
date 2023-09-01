from datetime import datetime

import pytest
from pandas import DataFrame, date_range
from pydantic import ValidationError

from python_project.data import ForecastParams, ForecastData
from python_project.data.problem import ForecastProblem


def test_should_raise_error_on_none():
    with pytest.raises(TypeError):
        ForecastProblem(None)


@pytest.mark.parametrize(
    "df,ds",
    [
        (
            DataFrame(
                {
                    "ds": date_range(
                        datetime.strptime("2020-01-06", "%Y-%m-%d"),
                        periods=2,
                        freq="7d",
                    ),
                    "y": [1, 1],
                }
            ),
            datetime.strptime("2019-02-17", "%Y-%m-%d"),
        )
    ],
)
def test_empty_data_before_forecast_start(df, ds):
    params = ForecastParams(forecast_start=ds)
    data = ForecastData(df=df)
    with pytest.raises(ValidationError):
        ForecastProblem(params=params, data=data)
