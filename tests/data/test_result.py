# -*- coding: utf-8 -*-
import pytest
from pandas import DataFrame
from pydantic import ValidationError

from python_project.data.result import ForecastResult


def test_should_raise_error_on_none():
    with pytest.raises(TypeError):
        ForecastResult(None)


def test_forecast_empty():
    with pytest.raises(ValidationError):
        ForecastResult(forecast=DataFrame(), metrics=DataFrame({"y": [1]}))


def test_metrics_empty():
    with pytest.raises(ValidationError):
        ForecastResult(forecast=DataFrame({"y": [1]}), metrics=DataFrame())
