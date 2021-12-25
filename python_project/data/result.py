# -*- coding: utf-8 -*-
from pandas import DataFrame
from pydantic import BaseModel, validator


class ForecastResult(BaseModel):
    """
    Validate output data for forecast process
    """

    forecast: DataFrame
    metrics: DataFrame

    class Config:
        arbitrary_types_allowed = True

    @validator("forecast")
    def forecast_no_empty(cls, forecast):
        assert not forecast.empty, "without data forecast"
        return forecast

    @validator("metrics")
    def metrics_no_empty(cls, metrics):
        assert not metrics.empty, "without data metrics"
        return metrics
