# -*- coding: utf-8 -*-
import abc
from typing import Any


class MetricCalculator(abc.ABC):
    """
    Time series metrics interface
    """

    def __call__(self, targets: Any, predictions: Any) -> Any:
        return self.calculate(targets, predictions)

    @abc.abstractmethod
    def calculate(self, targets: Any, predictions: Any) -> Any:
        """
        Calculate Time series metrics

        :param targets: real value
        :param predictions: predicted value
        :return: Time series metrics
        """
        pass
