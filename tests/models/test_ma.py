# -*- coding: utf-8 -*-
from datetime import datetime, date

import pytest
import pandas as pd
from pydantic import ValidationError
import numpy as np

from python_project.data import ForecastParams, ForecastData, ForecastProblem
from python_project.models.ma import SMAParams, SMAModel


def test_should_raise_error_on_none():
    with pytest.raises(TypeError):
        SMAParams(None)


@pytest.mark.parametrize(
    "periods",
    [-1, -2, -3, -4, -5],
)
def test_periods_positive(periods):
    with pytest.raises(ValidationError):
        SMAParams(periods=periods)


@pytest.mark.parametrize(
    "periods,expected",
    [(1, "SMA_1"), (2, "SMA_2")],
)
def test_name_model(periods, expected):
    result = SMAParams(periods=periods)
    assert result.name_model == expected


@pytest.mark.parametrize(
    "periods,df,ds,expected",
    [
        (
            12,
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
def test_SMA_do_forecast(periods, df, ds, expected):
    sma_params = SMAParams(periods=periods)

    params = ForecastParams(
        forecast_start=ds,
    )
    data = ForecastData(df=df)
    forecast_problem = ForecastProblem(params=params, data=data)
    result = SMAModel(sma_params).do_forecast(forecast_problem)
    np.testing.assert_array_equal(result, expected)
