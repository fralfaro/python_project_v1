# -*- coding: utf-8 -*-
from pydantic import validator

from .base import ForecastSolver, SolverParams
import pandas as pd

from ..data.problem import ForecastProblem
import numpy as np


class SMAParams(SolverParams):
    """
    Parametros modelo media movil
    """

    periods: int = 12

    @property
    def name_model(self) -> str:
        return "SMA_" + str(self.periods)

    @validator("periods")
    def periods_positive(cls, periods):
        assert periods > 0, "periods no positivo"
        return periods


class SMAModel(ForecastSolver):
    """
    Modelo Media movil
    """

    def do_forecast(self, forecast_problem: ForecastProblem) -> np.ndarray:
        """
        Modelo Media movil
        """

        obs = (
            forecast_problem.data.df["y"]
            .tail(self.models_params.periods)
            .tolist()
        )
        predictions = []

        for i in range(1, forecast_problem.params.forecast_periods + 1):
            value = pd.Series(obs).tail(self.models_params.periods).mean()
            obs.append(value)
            predictions.append(value)

        return np.array(predictions)
