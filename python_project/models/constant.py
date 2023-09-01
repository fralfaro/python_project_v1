import numpy as np

from python_project.data import ForecastProblem
from python_project.models.base import ForecastSolver, SolverParams


class ConstantParams(SolverParams):
    """
    Parametros modelo constante
    """

    value: float = 0

    @property
    def name_model(self):
        return "CONSTANT_" + str(self.value)


class ConstantModel(ForecastSolver):
    """
    Modelo constante
    """

    def do_forecast(self, forecast_problem: ForecastProblem) -> np.ndarray:
        """
        Modelo constante
        """

        predictions = [
            self.models_params.value
        ] * forecast_problem.params.forecast_periods

        return np.array(predictions)
