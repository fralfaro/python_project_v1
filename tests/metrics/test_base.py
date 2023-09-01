from typing import Any
import logging

from python_project.metrics.base import MetricCalculator

logger = logging.getLogger(__name__)


class DummyCalculator(MetricCalculator):
    def calculate(self, targets, predictions) -> Any:
        logger.info("Calling DummyCalculator.calculate")
        return 1


def test_call_order(mocker, caplog):
    caplog.set_level(level=logging.INFO)
    some_result = mocker.MagicMock()
    calc = DummyCalculator()
    result = calc(some_result, some_result)
    assert result == 1
    assert "Calling DummyCalculator.calculate" in caplog.text
