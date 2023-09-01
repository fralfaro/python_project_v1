from pandas._libs.tslibs.timestamps import Timestamp
from pydantic import BaseModel, validator

from .data import ForecastData
from .params import ForecastParams


class ForecastProblem(BaseModel):
    """
    Necessary information to resolve forecast problem
    """

    params: ForecastParams
    data: ForecastData

    @validator("data")
    def validator_data_start(cls, data, values):
        start = Timestamp(values["params"].forecast_start)
        result = data.df.copy().loc[lambda x: x.ds <= start]
        assert not result.empty, f"without data before to {start}"

        data.df = data.df.set_index("ds")
        return data
