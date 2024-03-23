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
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.exchange import timeframe_to_minutes
# multiplier = 15
def calculate_ratios(account_values: pd.Series, benchmark_returns = None, timeframe="1m"):
    # Calculate dynamic multiplier based on average timedelta of the index
    multiplier = timeframe_to_minutes(timeframe)
    print("Timeframe multiplier", multiplier)
    sharpe_ratio = None
    calmar_ratio = None
    m2_ratio = None
    try:
        multmax = 365 * 24 * (60 / multiplier)
        account_series = account_values
        returns = account_series.pct_change().dropna()
        risk_free = 0
        max_drawdown = (account_series / account_series.cummax()).min() - 1

        log_returns = np.log1p(returns)
        log_mean_return = log_returns.mean() * multmax
        log_std_return = log_returns.std() * np.sqrt(multmax)
        sharpe_ratio = log_mean_return / log_std_return
        calmar_ratio = (account_series.iloc[-1] / account_series.iloc[0] - 1) / abs(max_drawdown)
        m2_ratio = None
        if benchmark_returns is not None:
            benchmark_returns = np.log1p(benchmark_returns.pct_change().dropna())
            benchmark_std_log_returns = benchmark_returns.std() * np.sqrt(multmax)
            m2_ratio = sharpe_ratio * benchmark_std_log_returns

    except Exception as e:
        print(e)
    finally:
        return {
            f'sharpe_ratio': sharpe_ratio,
            f'calmar_ratio': calmar_ratio,
            f'm2_ratio': m2_ratio
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
    ratios = calculate_ratios(balance_df['total'], benchmark_df['benchmark'], timeframe)
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

    ratios_benchmark = calculate_ratios(benchmark_df['benchmark'], balance_df['total'], timeframe)
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




def render_graph(dataframe, perfconfig, dp, trades, timeframe="1m"):
    print(dataframe, "GRAPHDATA")
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
