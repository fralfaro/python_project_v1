# -*- coding: utf-8 -*-
import os
from pydantic import BaseModel, validator
import pandas as pd


class ForecastData(BaseModel):
    """
    Validate forecast data
    """

    df: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def from_file(path_layout: os.PathLike) -> "ForecastData":
        """
        Read data from ".csv"

        :param path_layout: path
        :return: "ForecastData" object
        """
        with open(path_layout, "rb") as csv_file:
            # datos
            result = pd.read_csv(csv_file, sep=",").assign(
                ds=lambda df: pd.to_datetime(df["ds"], format="%Y-%m")
            )

            return ForecastData(df=result)

    @validator("df")
    def data_no_empty(cls, df):
        assert not df.empty, "without data"
        return df

    @validator("df")
    def data_correct_cols(cls, df):
        REQUIRED_COLS = [
            "ds",
            "y",
        ]
        assert set(df.columns.tolist()) == set(
            REQUIRED_COLS
        ), "columns not valid"
        return df.sort_values("ds")

    @validator("df")
    def data_no_duplicates(cls, df):
        should_be_index = [
            "ds",
        ]

        assert (
            df.duplicated(should_be_index, False).sum() == 0
        ), "duplicate data"
        return df
