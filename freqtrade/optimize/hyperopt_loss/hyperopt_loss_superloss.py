"""
CalmarHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""
from datetime import datetime

from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.data.metrics import calculate_calmar
from freqtrade.optimize.hyperopt import IHyperOptLoss
from freqtrade.plugins.perfcheck_renderers import calculate_ratios


class Supreme(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation uses the Calmar Ratio calculation.
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               config: Config, *args, **kwargs) -> float:
        """
        Objective function, returns smaller number for more optimal results.

        Uses Calmar Ratio calculation.
        """
        performance_minutes = config['perfcheck_config'].get('update_performance_minutes', 15)
        maximal_deliminator = 1 / (results["locked"].max() / results["total"].iloc[-1])
        print(performance_minutes, "minutes per interval")
        try: # Should have at least three hours worth of changing balance, not too much to ask for
            ratios = calculate_ratios(results["total"], timeframe=f"{performance_minutes}m")
        except Exception as e:
            print(e)
            ratios = {"sharpe_ratio": 0, "calmar_ratio": 0}
        optimal_trades = 250
        optimal_data_points = 36
        multiplier = min(min(trade_count / optimal_trades, results.shape[0] / optimal_data_points), 1) ** 10
        rating = ratios["sharpe_ratio"] * multiplier * max(ratios["calmar_ratio"], 0) * maximal_deliminator
        return -rating
