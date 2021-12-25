# -*- coding: utf-8 -*-
import pytest
import pandas as pd
from pathlib import Path

from python_project.data import ForecastData


@pytest.fixture(name="data_01_file_csv")
def data_file():
    path = Path("./tests/data/data.csv")
    assert path.exists()
    return path


@pytest.fixture(name="data_01_obj")
def data_objs():
    ds = pd.date_range(start="2020-01-01", periods=2, freq="W")
    y = [1, 2]

    df = pd.DataFrame({"ds": ds, "y": y}).assign(
        ds=lambda df: pd.to_datetime(df["ds"], format="%Y-%m")
    )

    return ForecastData(df=df)
