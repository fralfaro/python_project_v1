# -*- coding: utf-8 -*-
from .base import MetricCalculator
import pandas as pd
import numpy as np

np.seterr(divide="ignore", invalid="ignore")


class StandardMetrics(MetricCalculator):

    # metrics
    # a) Scale-dependent errors
    @staticmethod
    def mae(targets: np.ndarray, predictions: np.ndarray) -> float:
        """
        Mean absolute error (MAE):

        .. math::
            MAE =\\dfrac{\\sum _{i=1}^{n}|y_{i}-\hat{y}_{i}|}{n}

        :param targets: real value
        :param predictions: predicted value
        :return: metrics value
        """
        error = predictions - targets
        return round(np.abs(error).mean(), 4)

    @staticmethod
    def mse(targets: np.ndarray, predictions: np.ndarray) -> float:
        """
        Mean squared error (MSE):

        .. math::
            MSE =\\dfrac{\\sum _{i=1}^{n}(y_{i}-\hat{y}_{i})^2}{n}

        :param targets: real value
        :param predictions: predicted value
        :return: metrics value
        """
        error = predictions - targets
        return round((error ** 2).mean(), 4)

    @staticmethod
    def rmse(targets: np.ndarray, predictions: np.ndarray) -> float:
        """
        Root mean squared error (RMSE)

        .. math::
            RMSE =\\sqrt{\\dfrac{\\sum _{i=1}^{n}(y_{i}-\hat{y}_{i})^2}{n}}

        :param targets: real value
        :param predictions: predicted value
        :return: metrics value
        """
        error = predictions - targets
        return round(np.sqrt((error ** 2).mean()), 4)

    # b) Percentage errors
    @staticmethod
    def mape(targets: np.ndarray, predictions: np.ndarray) -> float:
        """
        Mean absolute percentage error (MAPE):

        .. math::
            MAPE =\\dfrac{100}{n}\sum _{i=1}^{n} \
            \\left| \\dfrac{y_{i}-\hat{y}_{i}}{y_{i}} \\right|


        :param targets: real value
        :param predictions: predicted value
        :return: metrics value
        """
        error = predictions - targets

        if any(x == 0 for x in targets):
            return np.inf
        else:
            return round(np.abs(error / targets).mean(), 4)

    @staticmethod
    def maape(targets: np.ndarray, predictions: np.ndarray) -> float:
        """
        Mean arctangent percentage error (MAAPE):

        .. math::
            MAAPE =\\dfrac{100}{n}\sum _{i=1}^{n} \
            arctan(\\left| \\dfrac{y_{i}-\hat{y}_{i}}{y_{i}} \\right|)

        :param targets: real value
        :param predictions: predicted value
        :return: metrics value
        """

        error = predictions - targets

        if any((x, y) == (0, 0) for x, y in zip(predictions, targets)):
            return np.inf

        else:
            return round(np.arctan(np.abs(error / targets)).mean(), 4)

    @staticmethod
    def wmape(targets: np.ndarray, predictions: np.ndarray) -> float:
        """
        Weighted mean absolute percentage error (WMAPE):

        .. math::
            WMAPE = \\dfrac{\\sum _{i=1}^{n}\
            |y_{i}-\hat{y}_{i}|}{\\sum _{i=1}^{n}|y_{i}|}

        :param targets: real value
        :param predictions: predicted value
        :return: metrics value
        """
        error = predictions - targets
        sum_values = np.sum(targets)

        if sum_values == 0:
            return np.inf
        else:
            return round(np.abs(error).sum() / sum_values, 4)

    @staticmethod
    def mmape(targets: np.ndarray, predictions: np.ndarray) -> float:
        """
        Modified mean absolute percentage error (MMAPE):

        .. math::
            MMAPE = \\dfrac{100}{n}\sum _{i=1}^{n} \
             \\dfrac{|y_{i}-\hat{y}_{i}|}{|y_{i}| + 1}

        :param targets: real value
        :param predictions: predicted value
        :return: metrics value
        """
        error = np.abs(predictions - targets)
        denom = 1 + np.abs(targets)

        return round(np.mean(error / denom), 4)

    @staticmethod
    def smape(targets: np.ndarray, predictions: np.ndarray) -> float:
        """
        Symmetric mean absolute percentage error (SMAPE):

        .. math::
            SMAPE =\\dfrac{100}{n}\sum _{i=1}^{n} \
            \\dfrac{|y_{i}-\hat{y}_{i}|}{(|y_{i}| + |\hat{y}_{i}|)/2}

        :param targets: real value
        :param predictions: predicted value
        :return: metrics value
        """
        error = predictions - targets
        sum_values = np.abs(predictions) + np.abs(targets)

        if any(x == 0 for x in sum_values):
            return np.inf

        else:
            return round(2 * np.mean(np.abs(error) / sum_values), 4)

    def calculate(
        self, targets: np.ndarray, predictions: np.ndarray
    ) -> pd.DataFrame:
        """
        All standard metrics:

         - mae
         - mse
         - rmse
         - mape
         - maape
         - wmape
         - mmape
         - smape

        :param targets: real value
        :param predictions: predicted value
        :return: All standard metrics
        """
        df_result = pd.DataFrame()

        df_result["mae"] = [round(self.mae(targets, predictions), 2)]
        df_result["mse"] = [round(self.mse(targets, predictions), 2)]
        df_result["rmse"] = [round(self.rmse(targets, predictions), 2)]
        df_result["mape"] = [round(self.mape(targets, predictions), 2)]
        df_result["maape"] = [round(self.maape(targets, predictions), 2)]
        df_result["wmape"] = [round(self.wmape(targets, predictions), 2)]
        df_result["mmape"] = [round(self.mmape(targets, predictions), 2)]
        df_result["smape"] = [round(self.smape(targets, predictions), 2)]

        return df_result
