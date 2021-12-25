# -*- coding: utf-8 -*-
from datetime import datetime

import pytest
from pandas import DataFrame
from pydantic import ValidationError

from pandas._testing import assert_frame_equal

from python_project.data import ForecastData


def test_should_raise_error_on_none():
    with pytest.raises(TypeError):
        ForecastData(None)


def test_data_empty():
    with pytest.raises(ValidationError):
        ForecastData(df=DataFrame())


@pytest.mark.parametrize("df", [DataFrame({"dss": [1, 2], "y": [1, 2]})])
def test_wrong_cols(df):
    with pytest.raises(ValidationError):
        ForecastData(df=df)


@pytest.mark.parametrize(
    "df",
    [
        DataFrame(
            {
                "ds": [
                    datetime.strptime("2020-02-17", "%Y-%m-%d"),
                    datetime.strptime("2020-02-17", "%Y-%m-%d"),
                ],
                "y": [1, 10],
            }
        )
    ],
)
def test_data_duplicated(df):
    with pytest.raises(ValidationError):
        ForecastData(df=df)


def test_data_from_file(data_01_file_csv, data_01_obj):
    result = ForecastData.from_file(data_01_file_csv)
    assert_frame_equal(result.df, data_01_obj.df)
