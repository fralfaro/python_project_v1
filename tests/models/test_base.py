from datetime import date, datetime
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from python_project.data import (
    ForecastResult,
    ForecastParams,
    ForecastData,
    ForecastProblem,
)
from python_project.models.base import ForecastSolver


@pytest.mark.parametrize(
    "df,expected",
    [
        (
            pd.DataFrame({"y": [1]}),
            ForecastResult(
                forecast=pd.DataFrame({"y": [1]}),
                metrics=pd.DataFrame({"y": [1]}),
            ),
        )
    ],
)
def test_ForecastSolver_forecaster(mocker, df, expected):
    class_forecast = mocker.MagicMock()
    forecast_problem = mocker.MagicMock()

    train = class_forecast.test_train
    train.return_value = ("train", "test")

    result_forecast = class_forecast.run_forecast
    result_forecast.return_value = df

    result_metrics = class_forecast.metrics
    result_metrics.return_value = df

    result = ForecastSolver.forecaster(class_forecast, forecast_problem)

    class_forecast.test_train.assert_called_once()
    class_forecast.run_forecast.assert_called_once()
    class_forecast.metrics.assert_called_once()

    assert_frame_equal(result.forecast, expected.forecast)
    assert_frame_equal(result.metrics, expected.metrics)


@pytest.mark.parametrize(
    "df,expected",
    [
        (
            pd.DataFrame(
                columns=["ds", "y"],
                data=[
                    ["2020-01-06", 10],
                    ["2020-01-13", 20],
                    ["2020-01-20", 30],
                ],
            )
            .assign(ds=lambda x: pd.to_datetime(x.ds))
            .set_index("ds"),
            pd.DataFrame(
                {
                    "yhat": range(1, 17 + 1),
                    "ds": pd.date_range(
                        datetime(2020, 1, 27, 0, 0, 0), periods=17, freq="7d"
                    ),
                }
            )
            .set_index("ds")
            .asfreq("7d"),
        )
    ],
)
def test_ForecastSolver_run_forecast(df, expected, mocker):
    class_forecast = mocker.MagicMock()
    forecast_problem = mocker.MagicMock()

    n_value = 12 + 4 + 1
    forecast_problem.params.forecast_periods = n_value

    yhat = class_forecast.do_forecast
    yhat.return_value = [x for x in range(1, n_value + 1)]

    class_forecast.name_model = "model"

    df_predict = ForecastSolver.run_forecast(
        class_forecast, forecast_problem, df
    )

    class_forecast.do_forecast.assert_called_once()

    assert_frame_equal(df_predict, expected)


@pytest.mark.parametrize(
    "df,ds,expected",
    [
        (
            pd.DataFrame(
                columns=["ds", "y"],
                data=[["2017-12-04", 10], ["2018-01-01", 14]],
            ).assign(ds=lambda x: pd.to_datetime(x.ds)),
            date(2018, 1, 2),
            (
                pd.DataFrame(
                    columns=["ds", "y"],
                    data=[[datetime(2017, 12, 4, 0, 0, 0), 10]],
                ).set_index("ds"),
                pd.DataFrame(
                    columns=["ds", "y"],
                    data=[[datetime(2018, 1, 1, 0, 0, 0), 14]],
                ).set_index("ds"),
            ),
        )
    ],
)
def test_forecast_test_train(df, ds, expected):
    params = ForecastParams(
        forecast_start=ds,
    )
    data = ForecastData(df=df)
    forecast_problem = ForecastProblem(params=params, data=data)
    train, test = ForecastSolver.test_train(forecast_problem)
    assert_frame_equal(train, expected[0])
    assert_frame_equal(test, expected[1])


@pytest.mark.parametrize(
    "test, forecast,error",
    [
        (
            (
                pd.DataFrame(
                    columns=["ds", "y"],
                    data=[
                        ["2019-01-01", 10],
                        ["2019-01-02", 20],
                        ["2019-01-03", 30],
                    ],
                )
                .assign(ds=lambda x: pd.to_datetime(x.ds))
                .set_index("ds")
            ),
            (
                pd.DataFrame(
                    columns=["ds", "yhat"],
                    data=[
                        ["2019-01-01", 11],
                        ["2019-01-02", 18],
                        ["2019-01-03", 33],
                    ],
                )
                .assign(ds=lambda x: pd.to_datetime(x.ds))
                .set_index("ds")
            ),
            pd.DataFrame(
                columns=[
                    "mae",
                    "mse",
                    "rmse",
                    "mape",
                    "maape",
                    "wmape",
                    "mmape",
                    "smape",
                    "model",
                ],
                data=[[2.00, 4.67, 2.16, 0.1, 0.1, 0.1, 0.09, 0.1, "model"]],
            ),
        )
    ],
)
def test_forecast_metrics(
    test,
    forecast,
    error,
):
    result = ForecastSolver.metrics(test, forecast, "model")
    assert_frame_equal(result, error)
