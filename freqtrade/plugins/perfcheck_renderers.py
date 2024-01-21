import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel

from freqtrade.persistence import Trade

multiplier = 15

def __render_many_graphs_mapped(dfds, name, ppl, remap=False, render_informative=True):
    fig = make_subplots(rows=1, cols=2)
    analyzed_tickers = []
    checked_tickers = []
    leverages = [1]

    def transform_ratio_values(short_name, ratio_value):
        if short_name == "calmar":
            ratio_value = ratio_value / 5
        elif short_name == "omega":
            ratio_value = (ratio_value - 1) * 100 / 2
        return ratio_value

    max_return_scalar = 0

    def generate_mapped_returns(price_series, initial_investement):
        if remap:
            scalar_return = price_series.ffill().pct_change().dropna().mean()
            ma = scalar_return if scalar_return > max_return_scalar else max_return_scalar
            current_multiplier = ma / scalar_return
            print(f"Current multiplier: {current_multiplier}")
        else:
            current_multiplier = 1
        cum_change = 1 + price_series.fillna(method="ffill").pct_change().dropna() * current_multiplier
        change_series = cum_change.cumprod() * initial_investement
        return [change_series, current_multiplier]

    for dfd in dfds:
        if render_informative:
            for col in dfd.columns.tolist():
                splits = col.split("-")
                if len(splits) > 1 and splits[1] not in checked_tickers:
                    informative_name = splits[1]
                    scalar_return = dfd[col].ffill().pct_change().dropna().mean()
                    max_return_scalar = scalar_return if scalar_return > max_return_scalar else max_return_scalar
                    checked_tickers.append(splits[1])
        scalar_return = dfd["total"].ffill().pct_change().dropna().mean()
        max_return_scalar = scalar_return if scalar_return > max_return_scalar else max_return_scalar

    for ix, dfd in enumerate(dfds):
        if render_informative:
            for col in dfd.columns.tolist():
                splits = col.split("-")

                if len(splits) > 1 and splits[1] not in analyzed_tickers:
                    informative_name = splits[1]
                    change_series = generate_mapped_returns(dfd[col], dfd['total'][dfd.index[0]])
                    metrics = calculate_ratios(change_series[0])
                    for key, metric in metrics.items():
                        metrics[key] = transform_ratio_values(key.split("_")[0], metric)
                    ratio_names = [ip.split("_")[0] for ip in list(metrics.keys())]
                    ratio_values = list(metrics.values())
                    fig.add_trace(go.Scatter(x=dfd.index, y=change_series[0], opacity=.4,
                                             mode='lines',
                                             name=f"{informative_name} at {round(change_series[1], 2)}x"))
                    fig.add_trace(go.Bar(
                        x=ratio_values,
                        y=ratio_names,
                        orientation='h', name=f"{informative_name}", text=ratio_values,
                        textposition="auto"), row=1, col=2)
                    analyzed_tickers.append(splits[1])

        ratio_names = []
        ratio_values = []

        change_series = generate_mapped_returns(dfd["total"], dfd['total'][dfd.index[0]])
        metrics = calculate_ratios(change_series[0])
        for key, value in metrics.items():
            short_name = key.split("_")[0]
            ratio_names.append(short_name)
            ratio_values.append(transform_ratio_values(short_name, value))

        fig.add_trace(go.Bar(
            x=ratio_values,
            y=ratio_names,
            orientation='h', name=ppl[ix], text=ratio_values, textposition="auto"), row=1, col=2)
        fig.add_trace(go.Scatter(x=dfd.index, y=change_series[0].tolist(),
                                 mode='lines+text',
                                 name=f"Balance of {ppl[ix]} at {round(change_series[1], 2)}x",

                                 text=[""],
                                 textposition="top right",
                                 textfont=dict(
                                     family="sans serif",
                                     size=18,
                                     color="crimson"
                                 )))
    fig.update_layout(title=f'Plot of {name}',
                      xaxis_title='Date',
                      yaxis_title='Balance(USDT)')
    return fig


def render_graph_by_files(file_names, remap=False, render_informative=False):
    dfds = []
    names = []
    for name in file_names:
        file_name = name
        if os.path.exists(file_name):
            dfd = pd.read_pickle(file_name)
            if 'date' in dfd.columns.tolist() and not dfd["date"].isna().any():
                dfd.set_index("date", inplace=True)
            dfd.sort_index(inplace=True)
            dfds.append(dfd)
            names.append(name)
        else:
            return None

    artwork_name = ''.join([i.split('/')[-1].split(".")[0] for i in
                            file_names])  # TODO: could be generated later outside of this class's scope

    # Set the title and axis labels
    fig = __render_many_graphs_mapped(dfds, artwork_name, names, remap, render_informative)

    return fig


def render_graph(dataframe, perfconfig):
    remap = perfconfig.get("remap", False)
    name = perfconfig["name"]
    render_informative = perfconfig.get("render_informative", False)
    dfds = [dataframe]
    names = [name]
    artwork_name = name
    fig = __render_many_graphs_mapped(dfds, artwork_name, names, remap, render_informative)
    return fig


def return_results(fig, file_name=None, resolution_x=1200, resolution_y=800):
    if file_name is not None:
        image_file = f"{file_name.replace('.pkl', '')}_graph.png"
        fig.write_image(image_file, format="png", width=resolution_x, height=resolution_y)
        return image_file
    else:
        fig.show(renderer='browser')
        return


def create_folder_if_does_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Path folder created", path)
    return path


def calculate_ratios(account_values, append="", custom_multiplier=None):
    global multiplier
    if custom_multiplier is not None:
        multiplier = custom_multiplier

    multmax = 365 * (60 / multiplier) * 24
    try:
        # Convert the list of account values to a pandas series
        account_series = account_values
        # Calculate the minute by minute returns
        returns = account_series.pct_change().dropna()
        risk_free = 0
        # Calculate the sharpe ratio
        # sharpe_ratio = (mean_return - risk_free) / std_return
        # Calculate the maximum drawdown
        max_drawdown = (account_series / account_series.cummax()).min() - 1

        log_returns = np.log(1 + returns)
        # Calculate the logarithmic mean return
        log_mean_return = log_returns.mean() * multmax
        # print(log_mean_return)
        # Calculate the standard deviation of logarithmic returns
        log_std_return = log_returns.std() * np.sqrt(multmax)
        # Calculate the variability-weighted return
        variability_weighted_return = (log_mean_return - risk_free) / log_std_return
        sharpe_ratio = variability_weighted_return
        calmar_ratio = (account_series.iloc[-1] / account_series.iloc[0] - 1) / abs(max_drawdown)

        if np.isnan(sharpe_ratio) or np.isnan(calmar_ratio):
            return {'sharpe_ratio' + append: 0, 'calmar_ratio' + append: 0,
                    'sortino_ratio' + append: 0, 'vwr_ratio' + append: 0}

        # Return the ratios as a dictionary
        return {'sharpe_ratio' + append: sharpe_ratio, 'calmar_ratio' + append: calmar_ratio,
                'sortino_ratio' + append: 0, 'vwr_ratio' + append: variability_weighted_return}
    except Exception as e:
        print("Exception occurred while calculating ratios", e)
        return {'sharpe_ratio' + append: 0, 'calmar_ratio' + append: 0,
                'sortino_ratio' + append: 0, 'vwr_ratio' + append: 0}


class PerformanceMeteredStrategy():
    curdir = os.path.join(os.getcwd(), "user_data")
    perfcheck_folder = create_folder_if_does_not_exist(os.path.join(curdir, "perfchecks"))
    metric_columns = ['sharpe_ratio', 'calmar_ratio', 'omega_ratio', 'vwr_ratio']
    balance_list = pd.DataFrame(columns=["date", "total", "used", "free"] + metric_columns)
    balance_filez = None
    backtest_filez = None
    live_perfcheck = Layout(name="no perfcheck yet")
    live_console = None
    perfcheck_config = None
    initialized = False
    informative_strategies = {}
    last_checked_experimental = {}
    current_profits = {}

    def __init__(self, perfcheck_config, bot_name, runmode):
        global multiplier
        self.perfcheck_config = perfcheck_config  # self.config.get("perfcheck_config")
        self.perfcheck_config["name"] = bot_name  # self.config.get("bot_name")
        self.runmode = runmode
        multiplier = self.perfcheck_config['update_performance_minutes']
        # if "ask_name" not in self.perfcheck_config or not self.perfcheck_config["ask_name"]:
        #     self.perfcheck_config["name"] = self.config.get("bot_name")
        # else:
        #     self.perfcheck_config["name"] = Prompt.ask("Perfcheck name?")
        if "default_exit_callback" not in self.perfcheck_config:
            self.perfcheck_config["default_exit_callback"] = None
        # self.load_informative_strategies(self.perfcheck_config["informative_strategies"])
        self.balance_file_setter()

    def post_trade(self, current_time, trade: Trade, current_rate):
        self.balance_list.to_pickle(self.balance_filez)
        return

    def bot_loop_start_callback(self, current_time, wallets):
        self.wallets = wallets
        self.render_difference(current_time)

    def balance_file_setter(self):
        if self.balance_filez is None:
            self.balance_filez = os.path.join(self.perfcheck_folder,
                                              self.perfcheck_config["name"] + f".{self.runmode}.pkl")
            if self.runmode in ('hyperopt'):  # self.config['runmode'].value in ('hyperopt'):
                # self.balance_list = pd.DataFrame(columns=["date", "total", "used", "free"] + self.metric_columns)
                print("removing the balance list file", self.balance_list)
                if os.path.exists(self.balance_filez):
                    os.remove(self.balance_filez)
            if self.runmode in ('live', 'dry_run') and os.path.exists(self.balance_filez):
                self.balance_list = pd.read_pickle(self.balance_filez)
                print("loading again")
        return self.balance_filez

    def load_informative_strategies(self, strategies):
        for strategy in strategies:
            strategy_file_path = os.path.join(self.perfcheck_folder,
                                              strategy if strategy.endswith('.pkl') else strategy + ".pkl")
            self.informative_strategies[strategy] = pd.read_pickle(strategy_file_path)

    def render_difference(self, current_time: datetime):
        date_now = pd.to_datetime(current_time).floor(f"{self.perfcheck_config['update_performance_minutes']}min")
        informative_columns = []
        pair_prices = {}
        current_balance = self.calculate_total_profits()
        if current_balance is not None:
            # self.last_checked = date_now
            used_usdt = self.wallets.get_used('USDT')
            usdt_free = self.wallets.get_free('USDT')
            new_element = pd.DataFrame(
                [{'date': date_now, 'total': current_balance, 'free': usdt_free, 'used': used_usdt} | pair_prices])

            new_element.set_index('date', inplace=True)
            self.balance_list = self.balance_list.combine_first(new_element)
        else:
            print("error - last balance is none")

    def render_performance(self, current_time: datetime):
        date_now = pd.to_datetime(current_time).floor(f"{self.perfcheck_config['update_performance_minutes']}min")
        metrics = calculate_ratios(self.balance_list["total"])
        metrics["date"] = date_now
        infmetrics = {}
        informative_columns = []
        for infcolname in informative_columns:
            # print(infcolname)
            metrics_informative = calculate_ratios(self.balance_list[infcolname],
                                                   "-" + infcolname.split('-')[1])
            infmetrics = infmetrics | metrics_informative
        metrics = pd.DataFrame([metrics])
        metrics.set_index('date', inplace=True)
        self.balance_list = self.balance_list.combine_first(metrics)
        if "date" in self.balance_list.columns:
            self.balance_list.drop("date", axis=1, inplace=True)
        self.balance_list.loc[:, self.metric_columns] = self.balance_list.loc[:, self.metric_columns].fillna(
            method='ffill')
        # print(self.balance_list)
        if not self.runmode in ("hyperopt"):
            self.print_performance_stat(self.balance_list.iloc[-1].squeeze().to_dict(), date_now, infmetrics)
        else:
            print(self.balance_list)

    def custom_exit_callback(self, feed_pair, current_time, current_profit, trade, wallets):
        self.initialized = True
        # print(self.initialized)
        self.wallets = wallets
        date_now = pd.to_datetime(current_time).floor(f"{self.perfcheck_config['update_performance_minutes']}min")
        if feed_pair not in self.last_checked_experimental:
            self.last_checked_experimental[feed_pair] = date_now - timedelta(
                minutes=self.perfcheck_config['update_performance_minutes'])
        if self.last_checked_experimental[feed_pair] < date_now:
            self.current_profits[feed_pair] = trade.stake_amount * (1 + current_profit)
            self.last_checked_experimental[feed_pair] = date_now
            self.try_repaint(current_time)

    def try_repaint(self, current_time):
        date_now = pd.to_datetime(current_time).floor(f"{self.perfcheck_config['update_performance_minutes']}min")
        if self.perfcheck_config["default_exit_callback"] is None or self.perfcheck_config["default_exit_callback"]:
            self.rerender_current_closed(self.wallets.get_all_positions())
        if all([last_checked == date_now for pair, last_checked in self.last_checked_experimental.items()]):
            self.render_difference(current_time)

            # print("not new enough", date_now, self.last_checked_experimental)

    def rerender_current_closed(self, open_positions):
        open_tickers = list(open_positions.keys())
        position_pairs_prev = list(self.current_profits.keys())
        for pair in position_pairs_prev:
            if pair not in open_tickers:
                print(f"Removing {pair} from open positions")
                self.try_remove_pair(pair)

    def try_remove_pair(self, pair):
        if pair in self.current_profits:
            del self.current_profits[pair]
            del self.last_checked_experimental[pair]

    def calculate_total_profits(self):
        usdt = self.wallets.get_free('USDT')

        open_positions = self.wallets.get_all_positions()
        open_tickers = list(open_positions.keys())

        position_pairs_prev = list(self.current_profits.keys())
        self.rerender_current_closed(open_positions)

        loaded_all = len(open_tickers) == len(position_pairs_prev)

        if loaded_all:
            for pair, stake in self.current_profits.items():
                # print(pair, stake)
                usdt += stake
            return usdt
        else:
            return None

    def confirm_trade_exit_callback(self, pair, trade, rate, current_time, wallets):
        self.wallets = wallets
        if self.perfcheck_config["default_exit_callback"] is None:
            self.perfcheck_config["default_exit_callback"] = False
        self.try_remove_pair(pair)
        self.post_trade(current_time, trade, rate)
        self.render_difference(current_time)
        self.render_performance(current_time)

    def print_performance_stat(self, last_row_init, date, informative_metrics):
        print(date)
        last_row = last_row_init | informative_metrics
        for name, informative_strategy_frame in self.informative_strategies.items():
            try:
                metrics_informative = informative_strategy_frame.loc[date, :].squeeze().to_dict()
                for key, value in metrics_informative.items():
                    last_row[f"{key}-{name}"] = value
            except Exception as e:
                print("could not process informative strategy", e)
        date_str = date.strftime("%Y-%m-%d %H:%M:%S")
        if self.live_console is None:
            self.live_console = Live(self.live_perfcheck, refresh_per_second=4)
            self.live_console.start()
        self.live_perfcheck = Layout(name="perfcheck")
        strategy_name = self.perfcheck_config["name"]
        vals = {strategy_name: {}}
        for key, value in last_row.items():
            metric_name = key.split('-')
            is_informative = len(metric_name) > 1
            if is_informative:
                if metric_name[1] not in vals:
                    vals[metric_name[1]] = {}
                vals[metric_name[1]][metric_name[0]] = value
            else:
                vals[strategy_name][metric_name[0]] = value
        layouts = []
        print(date_str)
        for k, v in vals.items():
            stri = ""
            for key, value in v.items():
                is_metric = key.endswith('_ratio')
                name = key
                color = "green" if value > 1 else "red"
                color_metric = "purple" if is_metric else 'yellow'
                stri += f"[{color_metric}]{name}[/{color_metric}]: [{color}]{value}[/{color}]\n"
            layouts.append(Layout(Panel(stri, title=k, subtitle=date_str, height=30), name=k))
        self.live_perfcheck.split_row(
            *layouts
        )
        self.live_console.update(self.live_perfcheck)
