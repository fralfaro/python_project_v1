from datetime import timedelta, date
from pydantic import BaseModel, validator


class ForecastParams(BaseModel):
    """
    Initial params for forecast
    """

    forecast_start: date
    forecast_weeks: int = 12
    test_weeks: int = 2

    @property
    def forecast_periods(self):
        """
        Number of periods to forecast

        :return: Number of periods (test + forecast)
        """
        return self.forecast_weeks + self.test_weeks + 1

    @property
    def test_start(self):
        """
        Date for the testing process

        :return: Target date for the forecast process
        """
        return self.forecast_start - timedelta(weeks=self.test_weeks)

    @property
    def target_date(self):
        """
        Target date for the forecast process

        :return: Target date for the forecast process
        """
        target = self.forecast_start + timedelta(weeks=1)
        return str(target.isocalendar()[0] * 100 + target.isocalendar()[1])

    @validator("forecast_weeks")
    def forecast_weeks_positive(cls, forecast_weeks):
        assert forecast_weeks > 0, "forecast_weeks not positive"
        return forecast_weeks

    @validator("test_weeks")
    def test_weeks_positive(cls, test_weeks):
        assert test_weeks > 0, "test_weeks not positive"
        return test_weeks
