import abc
import datetime as dt
from typing import Tuple
import numpy as np
import pandas as pd


from pydantic import BaseModel

from python_project.data import ForecastProblem, ForecastResult
from python_project.metrics.standard import StandardMetrics


class SolverParams(BaseModel, abc.ABC):
    """
    Model parameters interface
    """

    @property
    @abc.abstractmethod
    def name_model(self) -> str:
        """
        Name model

        :return: name model
        """
        pass


class ForecastSolver(abc.ABC):
    """
    Model algorithm interface
    """

    def __init__(self, models_params: SolverParams):
        self.models_params = models_params

    @abc.abstractmethod
    def do_forecast(self, forecast_problem: ForecastProblem) -> np.ndarray:
        """
        Apply forecast

        :param forecast_problem: objeto ForecastProblem
        :return: el resultado del forecast
        """
        pass

    def forecaster(self, forecast_problem: ForecastProblem) -> ForecastResult:
        """
        Resultado final del forecast, con sus respectivas metricas

        :param df: dataframe con las fechas y las ventas (ds, y)
        :return: forecast, metricas
        """
        train, test = self.test_train(forecast_problem)
        result_forecast = self.run_forecast(forecast_problem, train)
        result_metrics = self.metrics(
            test, result_forecast, self.models_params.name_model
        )

        result = ForecastResult(
            forecast=result_forecast, metrics=result_metrics
        )

        return result

    @staticmethod
    def test_train(
        forecast_problem: ForecastProblem,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Conjunto de entrenamiento  y de testeo

        :param df: dataframe con las fechas y las ventas (ds, y)
        :param params:  fecha para realizar el split
        :return: train set  y test set
        """
        v1 = forecast_problem.params.forecast_start
        v2 = dt.timedelta(weeks=forecast_problem.params.test_weeks)

        start = pd.Timestamp(v1 - v2)
        end = pd.Timestamp(v1)

        train = forecast_problem.data.df.loc[lambda x: x.index < start]
        test = forecast_problem.data.df.loc[
            lambda x: (start <= x.index) & (x.index < end)
        ]

        return train, test

    @staticmethod
    def metrics(
        test: pd.DataFrame, forecast: pd.DataFrame, name_model: str
    ) -> pd.DataFrame:
        """
        Retorna las metricas del forecast

        :param test: test set
        :param forecast: forecast
        :return: metricas
        """
        df = test.merge(forecast, on=["ds"], how="inner")
        result = (
            StandardMetrics()
            .calculate(df["y"], df["yhat"])
            .assign(model=name_model)
        )
        return result

    def run_forecast(
        self, forecast_problem: ForecastProblem, train: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Seejecuta el forecast y se le da el respectivo formato

        :param df: dataframe con las fechas y las ventas (ds, y)
        :return: forecast (formato estandarizado)
        """
        # formato
        start_date = max(train.index) + dt.timedelta(weeks=1)
        end_date = max(train.index) + dt.timedelta(
            weeks=forecast_problem.params.forecast_periods
        )

        ds_range = pd.date_range(start_date, end_date, freq="7d")

        df_predict = pd.DataFrame({"yhat": self.do_forecast(forecast_problem)})

        df_predict.index = ds_range
        df_predict.index.name = "ds"

        return df_predict
