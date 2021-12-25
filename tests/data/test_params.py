# -*- coding: utf-8 -*-
from datetime import datetime, date

import pytest
from pydantic import ValidationError

from python_project.data import ForecastParams


def test_should_raise_error_on_none():
    with pytest.raises(TypeError):
        ForecastParams(None)


@pytest.mark.parametrize(
    "params", [{"forecast_start": datetime.strptime("2020-02-17", "%Y-%m-%d")}]
)
def test_params(params):
    ForecastParams(**params)


@pytest.mark.parametrize(
    "params",
    [
        {
            "forecast_start": datetime.strptime("2020-02-17", "%Y-%m-%d"),
            "forecast_weeks": -1,
        }
    ],
)
def test_negative_forecast_weeks(params):
    with pytest.raises(ValidationError):
        ForecastParams(**params)


@pytest.mark.parametrize(
    "params",
    [
        {
            "forecast_start": datetime.strptime("2020-02-17", "%Y-%m-%d"),
            "test_weeks": -1,
        }
    ],
)
def test_negative_test_weeks(params):
    with pytest.raises(ValidationError):
        ForecastParams(**params)


@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {
                "forecast_start": datetime.strptime("2019-01-01", "%Y-%m-%d"),
                "test_weeks": 2,
                "forecast_weeks": 10,
            },
            13,
        )
    ],
)
def test_forecast_periods(params, expected):
    params = ForecastParams(**params)
    assert expected == params.forecast_periods


@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {
                "forecast_start": datetime.strptime("2019-01-01", "%Y-%m-%d"),
                "test_weeks": 2,
                "forecast_weeks": 10,
            },
            date(2018, 12, 18),
        )
    ],
)
def test_test_start(params, expected):
    params = ForecastParams(**params)
    assert expected == params.test_start


@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {"forecast_start": datetime.strptime("2019-01-01", "%Y-%m-%d")},
            "201902",
        )
    ],
)
def test_target_date(params, expected):
    params = ForecastParams(**params)
    assert expected == params.target_date
