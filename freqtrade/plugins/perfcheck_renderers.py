import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt

from freqtrade.persistence import Trade


# multiplier = 15
def calculate_ratios(account_values: pd.Series, benchmark_returns = None, append=""):
    # Calculate dynamic multiplier based on average timedelta of the index
    if not isinstance(account_values.index, pd.DatetimeIndex):
        raise ValueError("Account series index must be a DatetimeIndex for dynamic multiplier calculation.")

    time_deltas = account_values.index.to_series().diff().dropna()
    average_delta = time_deltas.mean()
    multiplier = average_delta.total_seconds() / 60
    # print("MULTS", multiplier)

    try:
        multmax = 365 * 24 * (60 / multiplier)
        account_series = account_values
        returns = account_series.pct_change().dropna()
        risk_free = 0
        max_drawdown = (account_series / account_series.cummax()).min() - 1

        log_returns = np.log1p(returns)
        log_mean_return = log_returns.mean() * multmax
        log_std_return = log_returns.std() * np.sqrt(multmax)
        variability_weighted_return = log_mean_return / log_std_return
        sharpe_ratio = variability_weighted_return
        calmar_ratio = (account_series.iloc[-1] / account_series.iloc[0] - 1) / abs(max_drawdown)
        m2_ratio = None
        if benchmark_returns is not None:
            # benchmark_returns = benchmark_returns * 2
            benchmark_returns = np.log1p(benchmark_returns.pct_change().dropna())
            benchmark_std_log_returns = benchmark_returns.std() * np.sqrt(multmax)
            m2_ratio = sharpe_ratio * benchmark_std_log_returns
            # print("m2 ratio", m2_ratio)

        return {
            f'sharpe_ratio{append}': sharpe_ratio,
            f'calmar_ratio{append}': calmar_ratio,
            f'sortino_ratio{append}': 0,
            f'vwr_ratio{append}': variability_weighted_return,
            f'm2_ratio{append}': m2_ratio
        }
    except Exception as e:
        return {
            f'sharpe_ratio{append}': 0,
            f'calmar_ratio{append}': 0,
            f'sortino_ratio{append}': 0,
            f'vwr_ratio{append}': 0,
            f'm2_ratio{append}': 0
        }

def generate_benchmark(dp, trades, first_date, initial_investment=10000, timeframe="1m"):
    pair_trade_counts = trades.groupby('pair')['is_short'].apply(lambda x: (x == False).sum() - (x == True).sum()).to_dict()
    total_trade_count = sum(abs(count) for count in pair_trade_counts.values())
    pair_trade_percentages = {pair: (count / total_trade_count) for pair, count in pair_trade_counts.items()}
    # print(pair_trade_counts, pair_trade_percentages)
    # Generate a DataFrame consisting of total balance by each datapoint in the data column
    balance_by_date = pd.DataFrame()

    for pair, percentage in pair_trade_percentages.items():
        pair_df = dp.historic_ohlcv(pair, timeframe)
        pair_df = pair_df[pair_df['date'] >= first_date]
        print(pair_df)
        allocated_balance = initial_investment * abs(percentage)  # Assuming initial_balance is available in dp
        pair_df['allocated_balance'] = allocated_balance
        change_series = pair_df['close'] / pair_df['close'].iloc[0] if percentage > 0 else pair_df['close'].iloc[0] / pair_df['close']
        pair_short = pair.split('/')[0]
        pair_df[f'total_value_{pair_short}'] = pair_df['allocated_balance'] * change_series

        pair_df.set_index('date', inplace=True)
        pair_df = pair_df[[f'total_value_{pair_short}']]
        print(f"Profit for {pair_short} is {pair_df[f'total_value_{pair_short}'].iloc[-1]}, initial investment is {pair_df[f'total_value_{pair_short}'].iloc[0]}")
        balance_by_date = balance_by_date.combine_first(pair_df)

    balance_by_date = balance_by_date.ffill().bfill()

    # print(balance_by_date)

    balance_by_date["benchmark"] = (balance_by_date.sum(axis=1))
    return balance_by_date

def generate_profit_single_pair(dp, pair, first_date, initial_investment=10000, timeframe="1m"):
    pair_df = dp.historic_ohlcv(pair, timeframe)
    pair_df = pair_df[pair_df['date'] >= first_date]
    print(pair_df)
    pair_df['allocated_balance'] = initial_investment
    pair_df['benchmark'] = pair_df['close'] / pair_df['close'].iloc[0] * pair_df['allocated_balance']
    pair_df.set_index("date", inplace=True)
    print(pair_df)
    return pair_df

def render_perfcheck_simple(balance_df: pd.DataFrame, dp, perfconfig, trades: pd.DataFrame=None, initial_investment=None, timeframe="1m"):
    initial_investment = next(item for item in [initial_investment, balance_df['closed_total'].bfill().iloc[0], 10000] if item is not None)
    # Generate benchmark using the dataprovider and the initial investment
    first_date = balance_df.index[0]
    if trades is None or trades.shape[0] == 0:
        benchmark_df = generate_profit_single_pair(dp, "BTC/USDT:USDT", first_date, initial_investment, timeframe)
    else:
        try:
            benchmark_df = generate_benchmark(dp, trades, first_date, initial_investment, timeframe)
        except Exception as e:
            benchmark_df = generate_profit_single_pair(dp, "BTC/USDT:USDT", first_date, initial_investment, timeframe)
            print("Error generating benchmark:", e)

    print(benchmark_df)

    fig = make_subplots(rows=1, cols=2)

    # Add the account balance line
    fig.add_trace(
        go.Scatter(
            x=balance_df.index,
            y=balance_df['total'],
            mode='lines',
            name='Account Balance'
        )
    )
    # Add the account closed balance line
    fig.add_trace(
        go.Scatter(
            x=balance_df.index,
            y=balance_df['closed_total'],
            mode='lines',
            name='Account Balance (Closed)'
        )
    )

    # Add the benchmark line
    fig.add_trace(
        go.Scatter(
            x=benchmark_df.index,
            y=benchmark_df['benchmark'],
            mode='lines',
            name='Benchmark'
        )
    )

    # Add performance metrics to the second column of the fig
    ratios = calculate_ratios(balance_df['total'], benchmark_df['benchmark'])
    ratio_names = list(ratios.keys())
    ratio_values = list(ratios.values())

    # Add a bar chart with the performance ratios
    fig.add_trace(
        go.Bar(
            x=ratio_values,
            y=ratio_names,
            orientation='h',
            name='Performance Ratios'
        ),
        row=1, col=2
    )

    ratios_benchmark = calculate_ratios(benchmark_df['benchmark'], balance_df['total'])
    ratio_names_benchmark = list(ratios_benchmark.keys())
    ratio_values_benchmark = list(ratios_benchmark.values())
    # Add a bar chart with the performance ratios for the benchmark to the second column of the fig
    fig.add_trace(
        go.Bar(
            x=ratio_values_benchmark,
            y=ratio_names_benchmark,
            orientation='h',
            name='Benchmark Ratios'
        ),
        row=1, col=2
    )

        # Update layout for the second column
    fig.update_yaxes(title_text='Ratios', row=1, col=2)
    fig.update_xaxes(title_text='Values', row=1, col=2)
    # Update layout to add titles and axis labels
    fig.update_layout(
        title='Account Balance and Benchmark Comparison',
        xaxis_title='Date',
        yaxis_title='Value'
    )

    return fig





def __render_many_graphs_mapped(dfds, name, ppl, remap=False, render_informative=True):
    render_informative = True
    remap = True
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
            current_multiplier = min(max(ma / scalar_return, 1), 10)
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
                    print(informative_name)
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
                    print(informative_name)
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
        change_series_closed = generate_mapped_returns(dfd["closed_total"], dfd['total'][dfd.index[0]])
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
                                 name=f"Balance {round(change_series[1], 2)}x",

                                 text=[""],
                                 textposition="top right",
                                 textfont=dict(
                                     family="sans serif",
                                     size=18,
                                     color="crimson"
                                 )))
        fig.add_trace(go.Scatter(x=dfd.index, y=change_series_closed[0].tolist(),
                                 mode='lines',
                                 name=f"Balance closed {round(change_series[1], 2)}x"))
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


# def render_graph(dataframe, perfconfig):
#     remap = perfconfig.get("remap", False)
#     name = perfconfig["name"]
#     render_informative = perfconfig.get("render_informative", False)
#     dfds = [dataframe]
#     names = [name]
#     artwork_name = name
#     fig = __render_many_graphs_mapped(dfds, artwork_name, names, remap, render_informative)
#     return fig

def render_graph(dataframe, perfconfig, dp, trades, timeframe="1m"):
    fig = render_perfcheck_simple(dataframe, dp, perfconfig, trades, timeframe=timeframe)
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
    arbitrage_id = None

    def __init__(self, perfcheck_config, bot_name, runmode, timeframe):
        self.timeframe = timeframe
        global multiplier
        self.perfcheck_config = perfcheck_config  # self.config.get("perfcheck_config")
        self.perfcheck_config["name"] = bot_name# self.config.get("bot_name")
        self.runmode = runmode
        if "arbitrage-id" in self.perfcheck_config:
            if "arbitrage-sum" not in self.perfcheck_config:
                raise Exception("ID provided but no complimentary strategies")
            self.perfcheck_config["name"] += f"_{self.perfcheck_config['arbitrage-id']}"
            self.arbitrage_sum = self.perfcheck_config["arbitrage-sum"]
            self.arbitrage_id = self.perfcheck_config["arbitrage-id"]
        elif self.runmode in ("backtest"):
            prepend = Prompt.ask("Name of config to prepend?").strip()
            if len(prepend) > 0:
                self.perfcheck_config["name"] = prepend + self.perfcheck_config["name"]
        multiplier = self.perfcheck_config['update_performance_minutes']

        # if "ask_name" not in self.perfcheck_config or not self.perfcheck_config["ask_name"]:
        #     self.perfcheck_config["name"] = self.config.get("bot_name")
        # else:
        #     self.perfcheck_config["name"] = Prompt.ask("Perfcheck name?")
        if "default_exit_callback" not in self.perfcheck_config:
            self.perfcheck_config["default_exit_callback"] = None
        # self.load_informative_strategies(self.perfcheck_config["informative_strategies"])
        self.balance_file_setter()

    def post_trade(self):
        self.balance_list.to_pickle(self.balance_filez)
        if self.arbitrage_id is not None:
            blist = self.balance_list.copy()#.rename(columns={"total": "total_" + self.arbitrage_id,
                                                             # "used": "used_" + self.arbitrage_id,
                                                             # "free": "free_" + self.arbitrage_id})
            for sumr in self.arbitrage_sum:
                fpath = self.balance_filez.replace(self.arbitrage_id, sumr)
                while not os.path.exists(fpath):
                    print("waiting for balance file", fpath)
                    time.sleep(1)
                df = pd.read_pickle(fpath).rename(columns={"total": "total_" + sumr,
                                                             "used": "used_" + sumr,
                                                             "free": "free_" + sumr})
                blist = pd.merge(left_index=True, right_index=True, left=blist, right=df, how="left")
                blist["total"] = blist["total"] + blist["total_" + sumr]
            self.balance_list = blist
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
        if pd.to_datetime(current_time) is not None:
            date_now = pd.to_datetime(current_time).floor(f"{self.perfcheck_config['update_performance_minutes']}min")
            informative_columns = []
            pair_prices = {}
            current_balance = self.calculate_total_profits()
            if current_balance is not None:
                # self.last_checked = date_now
                used_usdt = self.wallets.get_used('USDT')
                usdt_free = self.wallets.get_free('USDT')
                usdt_total = self.wallets.get_total('USDT')
                new_element = pd.DataFrame(
                    [{'date': date_now, 'total': current_balance, 'free': usdt_free, 'used': used_usdt, 'closed_total': usdt_total} | pair_prices])

                new_element.set_index('date', inplace=True)
                self.balance_list = self.balance_list.combine_first(new_element)
            else:
                print("error - last balance is none", current_time)
        else:
            print("Error - current time is none", current_time)

    def render_performance(self, current_time: datetime):
        if pd.to_datetime(current_time) is not None:
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
        else:
            print("Error - current time is none", current_time)

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
        self.render_difference(current_time)
        if self.runmode not in ('hyperopt', "backtest"):
            self.post_trade()
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
