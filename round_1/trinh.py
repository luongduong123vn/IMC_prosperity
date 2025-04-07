import sys
sys.path.append("..")
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
from numpy.linalg import inv as inv_matrix
import pandas as pd
import math

import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class Trader:
    def __init__(self):
        self.bid_dict = {}
        self.ask_dict = {}
        self.bid_vol_dict = {}
        self.ask_vol_dict = {}
        self.weights_dict = {}

        # Alpha settings-------------------------------------------------
        self.truncation = 50
        self.booksize = 150
        self.decay = 0
        self.long_only = 0
            # 0 = None; 1 = Long only; 2 = Short only
        self.neutralize_market = True

    def run(self, state: TradingState):
        # Chon nums_days cho toi uu, lâsy nhiều quá thì chạy chậm
        self.get_all_data(state,nums_days=30)

        # Tao alpha o day ------------------------------------------------
        alpha_id = -ts_returns(self.mid_price,1)
        # alpha2 =-(vec_sum(self.bid_vol)+vec_sum(self.ask_vol))

        # alpha_id = if_else(abs(self.volume_imbalance)>0.5,alpha1,alpha2)
        # -----------------------------------------------------------------
        result = self.calculate_alpha_weights(alpha_id)

        traderData = "SAMPLE" 
        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    

        
    def get_all_data(self, state: TradingState,nums_days=30):
        self.orders = []
        for symbol in state.order_depths:
            if symbol not in self.bid_dict:
                self.bid_dict[symbol] = {}
                self.ask_dict[symbol] = {}
                self.bid_vol_dict[symbol] = {}
                self.ask_vol_dict[symbol] = {}
                self.weights_dict[symbol] = {}
            order_depth: OrderDepth = state.order_depths[symbol]

            if len(order_depth.buy_orders) != 0:
                bid = list(order_depth.buy_orders.keys())
                bid_amount = list(order_depth.buy_orders.values())
            else:
                bid = []
                bid_amount = []
            if len(order_depth.sell_orders) != 0:
                ask = list(order_depth.sell_orders.keys())
                ask_amount = list(order_depth.sell_orders.values())
            else:
                ask = []
                ask_amount = []

            self.bid_dict[symbol].update({state.timestamp:np.array(bid)})
            self.ask_dict[symbol].update({state.timestamp:np.array(ask)})
            self.bid_vol_dict[symbol].update({state.timestamp:np.array(bid_amount)})
            self.ask_vol_dict[symbol].update({state.timestamp:np.array(ask_amount)})
            self.weights_dict[symbol].update({state.timestamp:np.array(state.position.get(symbol, 0))})
            
            self.bid = pd.DataFrame(self.bid_dict).tail(nums_days)
            self.ask = pd.DataFrame(self.ask_dict).tail(nums_days)
            self.bid_vol = pd.DataFrame(self.bid_vol_dict).tail(nums_days)
            self.ask_vol = pd.DataFrame(self.ask_vol_dict).tail(nums_days)
            self.weights = pd.DataFrame(self.weights_dict).tail(nums_days)

            # self.best_ask = vec_min(self.ask)
            # self.best_bid = vec_max(self.bid)
            self.total_vol_bid = vec_sum(self.bid_vol)
            self.total_vol_ask = vec_sum(self.ask_vol)
            # self.volume = abs(self.total_vol_bid)+abs(self.total_vol_ask)
            self.mid_price = (vec_max(self.bid)+vec_min(self.ask))/2
            # self.returns = ts_returns(self.mid_price,1)
            self.volume_imbalance = purify((self.total_vol_ask+self.total_vol_bid)/(abs(self.total_vol_ask)+self.total_vol_bid)).ffill().bfill()
            # self.spread = self.best_ask - self.best_bid #bid ask spread implementation
            # self.vwap=ts_sum(self.mid_price*self.volume,22)/ts_sum(self.volume,22)
            # self.market_returns = group_mean(self.returns,pd.DataFrame(1,columns=self.returns.columns,index=self.returns.index))
            # self.adv20 = ts_mean(self.volume,20)

            # self.rsk68_beta = ts_beta(self.returns,self.market_returns,60)
            # self.rsk68_residual_return = regression_neut(self.returns,self.market_returns)
            # self.rsk68_weight_edadv = ts_decay_exp(self.adv20,5,2)
            # self.rsk68_weight_volatility_short = ts_std(self.returns,20)
            
            # self.historical_volatility_10 = historical_volatility_d(self.mid_price,10)
            # self.parkinson_volatility_10 = parkinson_volatility_d(self.best_ask, self.best_bid,10)
            self.limit_weights = pd.DataFrame(self.booksize,columns=self.weights.columns,index=self.weights.index).tail(nums_days)

    
    def calculate_alpha_weights(self, alpha_id:pd.DataFrame):
        result = {}  
        # cal price  --------------------------------------------------
        price=self.pricing_asset()
        # price=self.mid_price.iloc[-1]

        # cal weights  --------------------------------------------------
        if self.neutralize_market:
            alpha_id = group_neutralize(alpha_id, pd.DataFrame(1,columns=alpha_id.columns,index=alpha_id.index))
        if self.decay>1:
            alpha_id = ts_decay_linear(alpha_id,self.decay)
        
        if self.long_only == 1:
            alpha_id = if_else(alpha_id>0, alpha_id, 0)
        elif self.long_only == 2:
            alpha_id = if_else(alpha_id<0, alpha_id, 0)

        alpha = (alpha_id.div(alpha_id.abs().sum(axis=1), axis=0)*self.limit_weights).ffill().bfill().fillna(0)
        alpha = alpha.clip(lower=-self.truncation, upper=self.truncation)
        today_weights = (alpha-self.weights).iloc[-1]

        # execute   --------------------------------------------------
        for symbol in today_weights.index:
            if int(today_weights[symbol])!=0:
                orders=self.execute_order(symbol,price[symbol],today_weights[symbol])
                result[symbol] = [orders]
        return result
    
    def pricing_asset(self):
        return ((vec_mean(self.bid)+vec_mean(self.ask))/2).iloc[-1]
    
    def execute_order(self, symbol: str, price: int, quantity: int) -> None:
        return Order(symbol, int(price), int(quantity))

def ts_delay(x: pd.DataFrame, 
             d: int = 1, 
             **kwargs) -> pd.DataFrame:
    
    return x.shift(d)


def ts_delta(x: pd.DataFrame, 
             d: int = 1, 
             **kwargs) -> pd.DataFrame:
    return x.diff(d)

def days_from_last_change(x: pd.DataFrame, 
                          **kwargs) -> pd.DataFrame:
    def last_change(l):
        temp = []
        dem = 1
        for i in range(len(l)):
            if l[i] == 1:
                dem += 1
                temp.append(dem)
            else:
                dem = 1
                temp.append(1)
        return temp
    
    t_diff = if_else(x.diff() == 0, pd.DataFrame(1, columns=x.columns, index=x.index), pd.DataFrame(0, columns=x.columns, index=x.index))
    for col in t_diff.columns:
        t_diff[col] = last_change(t_diff[col].to_list())
    return t_diff



def ts_weighted_delay(x: pd.DataFrame, 
                      k: float = 0.5, 
                      **kwargs) -> pd.DataFrame:
    return k * x + (1 - k) * ts_delay(x, 1)


def hump_decay(x: pd.DataFrame, 
               p: float = 0, 
               **kwargs) -> pd.DataFrame:
    return if_else(abs(ts_delta(x, 1)) > p * abs(x + ts_delay(x, 1)), x, ts_delay(x, 1))



def inst_tvr(x: pd.DataFrame, 
             d: int = 1, 
             **kwargs) -> pd.DataFrame:
    weights_t = x
    weights_t1 = ts_delay(x,1)
    
    return ts_mean(abs(weights_t - weights_t1),d)



def jump_decay(x: pd.DataFrame, 
                d: int = 1, 
                sensitivity: float = 0.5, 
                force: float = 0.1, 
                **kwargs) -> pd.DataFrame:
    
    return if_else(abs(x - ts_delay(x, 1)) > sensitivity * ts_std(x, d), ts_delay(x, 1) + ts_delta(x, 1) * force, x)



def kth_element(x: pd.DataFrame, 
                d: int, 
                k: int = 1, 
                ignore: str = "NAN 0", 
                **kwargs) -> pd.DataFrame:
    ignore_list = ignore.split()
    for val in ignore_list:
        if val == "NAN":
            x = x.replace(np.nan, np.inf)
        else:
            x = x.replace(float(val), np.inf)
    return x.rolling(d).apply(lambda x: sorted(x)[k - 1], raw=True)



def last_diff_value(x: pd.DataFrame, 
                    d: int = 1, 
                    **kwargs) -> pd.DataFrame:
    days_change = days_from_last_change(x)
    for i in range(2, d):
        x = if_else(days_change == i, ts_delay(x, i), x)
    return x



def ts_arg_max(x: pd.DataFrame, 
                d: int = 1, 
                **kwargs) -> pd.DataFrame:
    return x.rolling(d).agg(np.argmax)



def ts_arg_min(x: pd.DataFrame, 
               d: int = 1, 
               **kwargs) -> pd.DataFrame:
    return x.rolling(d).agg(np.argmin)



def ts_av_diff(x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    return x - ts_mean(x, d)



def ts_backfill(x: pd.DataFrame, 
                lookback: int, 
                k: int = 1, 
                ignore: str = "NAN", 
                **kwargs) -> pd.DataFrame:
    ignore_list = ignore.split()
    for val in ignore_list:
        if val == "NAN":
            x = x.replace(np.nan, np.inf)
        else:
            x = x.replace(float(val), np.inf)
    return x.rolling(lookback).apply(lambda x: sorted(x)[k - 1], raw=True).replace(np.inf, np.nan)



def ts_co_kurtosis(
    y: pd.DataFrame,
    x: pd.DataFrame,
    d: int, 
    **kwargs) -> pd.DataFrame:
    tmp = ts_mean(((y - ts_mean(y,d)) * (x - ts_mean(x,d))**3))
    new_df = tmp / (ts_std(y,d) * ts_std(y,d)**3)
    return new_df


def ts_acf(
    df: pd.DataFrame,
    lag: int=1,
    d: int=100, 
    **kwargs) -> pd.DataFrame:
    return ts_cov(df, ts_delay(df, lag), d) / ts_var(df, d)

def ts_corr(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    d: int, 
    **kwargs) -> pd.DataFrame:
    return df1.fillna(0).rolling(d).corr(df2.fillna(0))



def ts_co_skewness(
    y: pd.DataFrame,
    x: pd.DataFrame,
    d: int, 
    **kwargs) -> pd.DataFrame:
    tmp = ts_mean(((x - ts_mean(x,d)) * (y - ts_mean(y,d))**2))
    new_df = tmp / (ts_std(x,d) * ts_std(y,d)**2)
    return new_df



def ts_count_nans(x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    temp = is_nan(x)
    return temp.rolling(d).sum()



def ts_count(x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    return x.rolling(d).count()

def ts_cov(df1: pd.DataFrame, df2: pd.DataFrame, d: int, **kwargs) -> pd.DataFrame:
    return df1.rolling(d).cov(df2)


def ts_var(df1: pd.DataFrame, d: int, **kwargs) -> pd.DataFrame:
    return df1.rolling(d).var()



def ts_decay_exp(x: pd.DataFrame, d: int = 1, f: float = 1, **kwargs) -> pd.DataFrame:
    if d >= 2:
        return add(*[ts_delay(x, i) * (f ** (d - i)) for i in range(d)]) / sum([(f ** (d - i)) for i in range(d + 1)])
    else:
        return x



def ts_decay_linear(x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    if d >= 2:
        return add(*[ts_delay(x, i) * (d - i) for i in range(d)]) / sum(range(d + 1))
    else:
        return x

def ts_ir(x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    return ts_mean(x, d) / ts_std(x, d)

def ts_kurtosis(x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    return x.rolling(d).kurt()



def ts_max(x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    return x.rolling(d).max()



def ts_max_diff(x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    return x - ts_max(x, d)



def ts_mean(x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    return x.rolling(d).mean()



def ts_median(x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    return x.rolling(d).median()



def ts_min(x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    return x.rolling(d).min()



def ts_min_diff(x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    return x - ts_min(x, d)



def ts_min_max_cps(x: pd.DataFrame, d: int = 1, f: float = 2, **kwargs) -> pd.DataFrame:
    return ts_min(x, d) + ts_max(x, d) - f * x



def ts_min_max_diff(x: pd.DataFrame, d: int = 1, f: float = 0.5, **kwargs) -> pd.DataFrame:
    return x - f * (ts_min(x, d) + ts_max(x, d))



def ts_moment(x: pd.DataFrame, d: int = 1, k: int = 0,standardized=False, **kwargs) -> pd.DataFrame:
    x_mean = ts_mean(x, d)
    res = pd.DataFrame(index=x.index, columns=x.columns)
    
    if standardized:
        return ts_mean((x-x_mean)**k,30).fillna(0)/(ts_moment(x,d,2)**(k/2))
    else:
        return ts_mean((x-x_mean)**k,30).fillna(0)



def ts_partial_corr(x: pd.DataFrame, y: pd.DataFrame, z: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    new_df = (ts_corr(x, y, d) - ts_corr(x, z, d) * ts_corr(z, y, d)) / np.sqrt((1 - ts_corr(x, z, d)**2) * (1 - ts_corr(y, z, d)**2))
    return new_df



def ts_percentage(x: pd.DataFrame, d: int = 1, percentage: float = 0.5, **kwargs) -> pd.DataFrame:
    return x.rolling(d).apply(np.prod)



def ts_rank(x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    return x.rolling(d).rank(pct=True)
def merge_matrix(*args):
    if not args:
        raise ValueError("At least one DataFrame must be provided to merge_matrix.")
    
    # Validate that all DataFrames have the same index and columns
    base_index = args[0].index
    base_columns = args[0].columns
    for idx, df in enumerate(args[1:], start=2):
        if not df.index.equals(base_index):
            raise ValueError(f"DataFrame {idx} has a different index.")
        if not df.columns.equals(base_columns):
            raise ValueError(f"DataFrame {idx} has different columns.")
    
    # Construct a dictionary where each key is a column name and each value is a list of CustomList objects
    merged_columns = {
        column: [list(values) for values in zip(*(df[column] for df in args))]
        for column in base_columns
    }
    
    # Create the merged DataFrame in one operation and make a copy to de-fragment
    merged_df = pd.DataFrame(merged_columns, index=base_index).copy()
    
    return merged_df
def ts_regression(
    y: pd.DataFrame,
    *args,
    d: int = 20,
    rettype: int = 0,
    **kwargs
) -> pd.DataFrame:
    """
    Perform rolling OLS for each ticker in y against multiple DataFrames in *args,
    where each DataFrame also has the same (or overlapping) tickers as columns, index as date.

    The last argument in *args may be an int/float to override 'd'.

    Args:
        y (pd.DataFrame): Dependent variable(s). Columns = tickers, index = date.
        *args: zero or more DataFrames with the same structure (columns=tickers, index=date).
               Optionally the last one can be an int/float to override 'd'.
        d (int): Rolling window size if not overridden by *args.
        rettype (int): Determines which statistic to return:
            0 -> Residual of last obs in the window
            1 -> Intercept
            2 -> Slopes
            3 -> Fitted last obs
            4 -> SSR
            5 -> SST
            6 -> R-squared

    Returns:
        pd.DataFrame or pd.MultiIndex DataFrame: Rolling regression results per ticker.
    """
    # ------------------------------------------------------
    # 1) Parse *args
    #    If the last arg is numeric => override d
    # ------------------------------------------------------
    args_list = list(args)
    if len(args_list) > 0 and isinstance(args_list[-1], (int, float)):
        d = int(args_list[-1])
        args_list = args_list[:-1]  # remove last

    # The remaining items in args_list must be DataFrames
    X_dfs = args_list  # each is a pd.DataFrame(index=date, columns=tickers)

    # Some sanity check
    if len(X_dfs) == 0:
        raise ValueError("No DataFrames in *args to use as regressors.")

    # ------------------------------------------------------
    # 2) Helper: rolling OLS (matrix formula) for one ticker
    # ------------------------------------------------------
    def _rolling_ols_single(
        y_series: pd.Series,
        X_comb: pd.DataFrame,
        window: int
    ) -> pd.DataFrame:
        """
        For one ticker's y and a multi-col X_comb, compute rolling OLS betas.
        Return a DataFrame of shape (time, #features).
        The last column in X_comb will be the intercept (1.0).
        """
        n = len(y_series)
        nfeat = X_comb.shape[1]
        idx = y_series.index

        X_vals = X_comb.values
        y_vals = y_series.values

        betas = np.full((n, nfeat), np.nan)

        for i in range(n):
            if i < window - 1:
                continue
            start = i - window + 1
            end = i + 1  # slice is exclusive at end
            Xw = X_vals[start:end, :]
            yw = y_vals[start:end]

            # Drop any rows with NaNs
            row_mask = (
                ~np.isnan(Xw).any(axis=1) &
                ~np.isnan(yw)
            )
            Xw = Xw[row_mask]
            yw = yw[row_mask]

            # Must have at least nfeat points to invert (X'X)
            if len(yw) < nfeat:
                continue

            XtX = Xw.T @ Xw
            try:
                inv_XtX = np.linalg.inv(XtX)
            except np.linalg.LinAlgError:
                # singular => skip
                continue
            Xty = Xw.T @ yw
            beta = inv_XtX @ Xty
            betas[i, :] = beta

        beta_df = pd.DataFrame(
            betas,
            index=idx,
            columns=X_comb.columns
        )
        return beta_df

    # ------------------------------------------------------
    # 3) For each ticker in y, gather the matching columns
    #    from each DataFrame in X_dfs and combine them
    # ------------------------------------------------------
    results_list = []
    tickers_y = y.columns

    for tkr in tickers_y:
        # Gather that ticker's column from each DF in X_dfs
        # So if we have 2 DataFrames, each has columns=[AAPL,GOOG,MSFT],
        # we do X_df1[tkr] and X_df2[tkr]
        # If tkr not in X_df.columns, we skip or fill with NaN.
        X_cols_for_ticker = []
        for X_df in X_dfs:
            if tkr in X_df.columns:
                X_cols_for_ticker.append(X_df[[tkr]])
            else:
                # Possibly skip if the DF doesn't have this ticker
                pass

        if len(X_cols_for_ticker) == 0:
            # No regressors found for this ticker => skip or fill
            # We'll create an empty placeholder
            # or raise an error
            continue

        # Now combine horizontally => multiple regressors
        # Example: if X_cols_for_ticker has [df1[[tkr]], df2[[tkr]]] => 2 columns
        # Named 'tkr' in each => rename them to avoid duplicates
        # We'll rename each to something distinct
        renamed_list = []
        for i, df_part in enumerate(X_cols_for_ticker):
            # Suppose each df_part has 1 column: tkr
            # rename it to f"{tkr}_var{i}"
            new_name = f"{tkr}_var{i+1}"
            tmp = df_part.copy()
            tmp.columns = [new_name]
            renamed_list.append(tmp)

        # Concat horizontally
        X_comb = pd.concat(renamed_list, axis=1)

        # Add intercept as last column
        X_comb["Intercept__"] = 1.0

        # Align X_comb with y[tkr] in time
        X_comb, y_tkr = X_comb.align(y[[tkr]], join="inner", axis=0)

        # We'll reduce y_tkr from DataFrame to Series
        y_series = y_tkr[tkr]

        # ------------------------------------------------------
        # 4) Compute rolling betas for this ticker
        # ------------------------------------------------------
        betas_df = _rolling_ols_single(y_series, X_comb, d)

        # ------------------------------------------------------
        # 5) From betas, compute the user-requested statistic
        #    for each date
        # ------------------------------------------------------
        out_df = pd.DataFrame(index=betas_df.index, columns=[tkr], dtype=float)

        if rettype in [0, 3, 4, 5, 6]:
            # We need to compute SSR, SST, residual, fitted, etc.
            for i, date in enumerate(betas_df.index):
                if not np.isfinite(betas_df.iloc[i, :]).all():
                    out_df.iloc[i, 0] = np.nan
                    continue
                # betas for this date
                b_row = betas_df.iloc[i, :].values
                # Let's reconstruct the rolling window
                if i < d - 1:
                    out_df.iloc[i, 0] = np.nan
                    continue
                start = i - d + 1
                end = i + 1
                Xw = X_comb.values[start:end, :]
                yw = y_series.values[start:end]
                # drop NaNs
                mask = (
                    ~np.isnan(Xw).any(axis=1) &
                    ~np.isnan(yw)
                )
                Xw = Xw[mask]
                yw = yw[mask]

                if len(Xw) == 0:
                    out_df.iloc[i, 0] = np.nan
                    continue

                # Resid/fitted for last obs
                X_last = X_comb.values[i, :]  # shape (#features, )
                y_last = y_series.values[i]
                yhat_last = np.dot(X_last, b_row)
                resid_last = y_last - yhat_last

                # SSR
                yhat_w = Xw @ b_row
                resid_w = yw - yhat_w
                SSR = np.sum(resid_w**2)

                # SST
                mean_yw = np.mean(yw)
                SST = np.sum((yw - mean_yw)**2)
                if np.isclose(SST, 0.0):
                    r2 = np.nan
                else:
                    r2 = 1.0 - SSR / SST

                if rettype == 0:
                    # residual last obs
                    out_df.iloc[i, 0] = resid_last
                elif rettype == 3:
                    out_df.iloc[i, 0] = yhat_last
                elif rettype == 4:
                    out_df.iloc[i, 0] = SSR
                elif rettype == 5:
                    out_df.iloc[i, 0] = SST
                elif rettype == 6:
                    out_df.iloc[i, 0] = r2

            results_list.append(out_df)

        elif rettype == 1:
            # Return intercept
            # betas_df["Intercept__"] is the column
            # shape => same index, 1 column => ticker
            intercept = betas_df["Intercept__"].to_frame(name=tkr)
            results_list.append(intercept)

        elif rettype == 2:
            # Return slopes => one or more columns
            # We'll create a MultiIndex: (ticker, varname)
            slope_df = betas_df.drop(columns=["Intercept__"], errors="ignore")
            # rename columns => (ticker, regressor_name)
            # new_cols = pd.MultiIndex.from_tuples(
            #     [(tkr, c) for c in slope_df.columns],
            #     names=["Ticker", "Regressor"]
            # )
            slope_df.columns = [tkr for c in slope_df.columns]
            tmp=merge_matrix(*[slope_df.iloc[:,[c]] for c in range(len(slope_df.columns))])
            results_list.append(tmp)
        else:
            raise ValueError(f"Invalid rettype: {rettype}")

    # ------------------------------------------------------
    # 6) Combine all ticker results
    # ------------------------------------------------------
    if len(results_list) == 0:
        # Possibly nothing?
        return pd.DataFrame()

    if rettype == 2:
        # We have a list of DataFrames each with MultiIndex columns
        # => concat horizontally
        final_result = pd.concat(results_list, axis=1)
        return final_result
    else:
        # For rettype in [0,1,3,4,5,6], each is a DataFrame with 1 column per ticker
        # => concat horizontally
        final_result = pd.concat(results_list, axis=1)

    return final_result

def ts_returns(x: pd.DataFrame, d: int = 1, mode: int = 1, **kwargs) -> pd.DataFrame:
    if mode == 1:
        return (x - ts_delay(x, d)) / ts_delay(x, d)
    elif mode == 2:
        return (x - ts_delay(x, d)) / ((x + ts_delay(x, d)) / 2)



def ts_scale(x: pd.DataFrame, d: int = 1, constant: float = 0, **kwargs) -> pd.DataFrame:
    return (x - ts_min(x, d)) / (ts_max(x, d) - ts_min(x, d)) + constant

def ts_skewness(x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    return x.rolling(d).skew()

def ts_std(x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    return x.rolling(d).std()



def ts_sum(x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    return x.rolling(d).sum()



def ts_theilsen(x: pd.DataFrame, y: pd.DataFrame, d: int = 1, lag: int = 0, rettype: int = 0, **kwargs) -> pd.DataFrame:
    from scipy.stats import theilslopes
    
    x = x.shift(lag).fillna(0)
    y = y.fillna(0)

    def get_value(z, t, name, rettype, lag=0):
        X = z[name]
        y = t[name]
        slope, intercept, low_slope, high_slope = theilslopes(y, X)

        if rettype == 0:
            return slope
        elif rettype == 1:
            return intercept
        elif rettype == 2:
            return low_slope
        elif rettype == 3:
            return high_slope

    c = list(x.columns)
    df = pd.DataFrame(columns=c, index=x.index)
    for j in range(d, x.shape[0]):
        for i in c:
            df.iloc[j, c.index(i)] = get_value(x.iloc[j-d:j, :], y.iloc[j-d:j, :], i, rettype, lag)
    df = purify(df)
    return df



def ts_triple_corr(x: pd.DataFrame, y: pd.DataFrame, z: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:

    return ts_triple_cov(x,y,z,d)/(ts_std(x,d)*ts_std(y,d)*ts_std(z,d))

  

def ts_triple_cov(x: pd.DataFrame, y: pd.DataFrame, z: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    return ts_mean(((x - ts_mean(x, d)) * (y - ts_mean(y, d)) * (z - ts_mean(z, d))))



def ts_zscore(x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    return (x - ts_mean(x, d)) / ts_std(x, d)



def ts_entropy(x: pd.DataFrame, d: int=1, buckets: int = 10, **kwargs) -> pd.DataFrame:
    def entropy(hist):
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist))

    def rolling_entropy(x):
        hist, _ = np.histogram(x, bins=buckets)
        hist = hist / d
        return np.log(d) - entropy(hist) / d

    return x.rolling(d).apply(rolling_entropy, raw=True)



def ts_vector_neut(x: pd.DataFrame,y: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    return x - ts_vector_proj(x, y, d=d)



def ts_vector_proj(x: pd.DataFrame, y: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    res = y*ts_sum(x*y,d)/ts_sum(y*y,d)
    return res

def vector_proj(x: pd.DataFrame, y: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    market = pd.DataFrame(1,columns=x.columns,index=x.index)
    res = y*group_sum(x*y,market)/group_sum(y*y,market)
    return res


def vector_neut(x: pd.DataFrame, y: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
    return x - vector_proj(x, y)

def ts_rank_gmean_amean_diff(*args: pd.DataFrame, d: int = 10, **kwargs) -> pd.DataFrame:
    convert_rank = [rank(arg) for arg in args]
    l = len(convert_rank[0])
    temp = pd.DataFrame(columns=convert_rank[0].columns, index=convert_rank[0].index)
    for i in range(d, l):
        temp_df = [arg.iloc[i-d:i, :] for arg in convert_rank]
        temp.iloc[i, :] = rank_gmean_amean_diff(*temp_df)[-1]
    return temp



def ts_delta_limit(x, y, limit_volume=0.1,d=1):
    s1=ts_delta(x,d)
    limit_value=abs(y)*limit_volume
    return max_(-limit_value,min_(s1, limit_value))


def ts_mul_model(x, d, rettype = 0):
    time = (x==x).fillna(1).cumsum()
    x1 = ts_mean(x,d)
    x2 = ts_mean(x1,3)
    M = x/x2
    ME = ts_mean(M,22)
    SIN = M/ME
    TCI = x/SIN
    TC = ts_regression(TCI,time,252,rettype=3)
    if rettype==0:
        return TC
    elif rettype==1:
        TCS = TC*SIN
        return TCS
    else:
        I = x/TCS
        return I    

def ts_add_model(x, d, rettype = 0):
    time = (x==x).fillna(1).cumsum()
    x1 = ts_mean(x,d)
    x2 = ts_mean(x1,3)
    SI = x - x2
    NS = ts_mean(SI,22)
    NS_mean = ts_mean(NS,22)
    S= NS - NS_mean
    TCI = x-S
    TC = ts_regression(TCI,time,252,rettype=3)
    if rettype==0:
        return TC
    elif rettype==1:
        TCS = TC+S
        return TCS
    else:
        I = x-TC-S
        return I    

def ts_sarima(x, forecast_horizon=100, order=(1,1,1), seasonal_order=(2,0,0,63)):

    p, d, q = order    
    P, D, Q, S = seasonal_order
    
    # Store original series for later undifferencing
    original_x = x.copy()
    
    # Apply seasonal differencing if D > 0
    if D > 0:
        for i in range(D):
            x = ts_delta(x, S)  # Seasonal differencing
    
    # Apply non-seasonal differencing if d > 0
    if d > 0:
        for i in range(d):
            x = ts_delta(x, 1)  # Non-seasonal differencing
    
    # Compute moving average estimate (short-term trend)
    xhat = ts_mean(x, 22)  # 22-period moving average for trend
    error = x - xhat  # Compute error (residuals)

    # Prepare lag terms for regression
    ar_terms = [ts_delay(x, i) for i in range(1, p + 1)]       # Non-seasonal AR terms
    ma_terms = [ts_delay(error, i) for i in range(1, q + 1)]   # Non-seasonal MA terms
    sar_terms = [ts_delay(x, S * i) for i in range(1, P + 1)]  # Seasonal AR terms
    sma_terms = [ts_delay(error, S * i) for i in range(1, Q + 1)]  # Seasonal MA terms

    # Combine all terms for regression
    all_terms = ar_terms + ma_terms + sar_terms + sma_terms
    
    # Compute the regression model
    model_fit = ts_regression(
        xhat,
        *all_terms,
        d=forecast_horizon,
        rettype=3
    )
    
    # Add error component back to the result
    result = model_fit + error
    
    # For proper forecasting, we would need to undo the differencing
    # This would require integration code here
    
    # Forward fill missing values
    result = result.ffill()
    
    return result

def ts_arima(x, day=100, order=(1,1,1)):
    p, d, q = order
    
    # Store original series for later undifferencing
    original_x = x.copy()
    
    # Apply differencing if needed
    if d > 0:
        differenced_x = x.copy()
        for i in range(d):
            differenced_x = ts_delta(differenced_x, 1)
    else:
        differenced_x = x.copy()
    
    # Compute the trend component - using moving average as in original
    trend = ts_mean(differenced_x, 22)
    
    # Compute residuals (stationary component)
    residuals = differenced_x - trend
    
    # Build predictor arrays for regression
    ar_terms = [ts_delay(differenced_x, i) for i in range(1, p + 1)]  # AR terms
    ma_terms = [ts_delay(residuals, i) for i in range(1, q + 1)]      # MA terms
    
    # Fit the model using regression
    model_fit = ts_regression(
        differenced_x,  # Target is the differenced series itself
        *ar_terms,      # Include AR terms
        *ma_terms,      # Include MA terms
        d=day,
        rettype=3
    )
    
    # For forecasting, we need to reintegrate if we applied differencing
    if d > 0:
        # Undo the differencing to get back to the original scale
        result = model_fit.copy()
        # for i in range(d):
        #     # Use cumulative sum to undo differencing
        #     # This part would need more context on ts_delta implementation
        #     # This is a placeholder for the undifferencing logic
        #     result = result.cumsum() + original_x.iloc[0]  # Assuming x is a pandas Series
    else:
        result = model_fit
    
    # Forward fill any NaN values as in the original
    result = (result+residuals).ffill()
    
    return result

def ts_mul_model(x, d, rettype=0):

    time = (x == x).fillna(1).cumsum()  # Create a cumulative time index for missing values
    x1 = ts_mean(x, d)  # Compute d-period moving average
    x2 = ts_mean(x1, 3)  # Compute 3-period moving average of x1
    M = x / x2  # Scale x by the moving average
    ME = ts_mean(M, 22)  # 22-period moving average of M
    SIN = M / ME  # Seasonal component
    TCI = x / SIN  # Trend-cyclical index
    TC = ts_regression(TCI, time, 252, rettype=3)  # Trend-cyclical regression model
    if rettype == 0:
        return TC
    elif rettype == 1:
        TCS = TC * SIN  # Apply seasonal component
        return TCS
    else:
        I = x / TCS  # Inverse of TCS
        return I

def ts_add_model(x, d, rettype=0):
    time = (x == x).fillna(1).cumsum()  # Create a cumulative time index for missing values
    x1 = ts_mean(x, d)  # Compute d-period moving average
    x2 = ts_mean(x1, 3)  # Compute 3-period moving average of x1
    SI = x - x2  # Seasonal index (subtract moving average from x)
    NS = ts_mean(SI, 22)  # Non-seasonal component
    NS_mean = ts_mean(NS, 22)  # Mean of non-seasonal component
    S = NS - NS_mean  # Seasonal component
    TCI = x - S  # Trend-cyclical index
    TC = ts_regression(TCI, time, 252, rettype=3)  # Trend-cyclical regression model
    TCS = TC + S  # Apply seasonal component
    I = x - TC - S  # Inverse of trend and seasonal component
    if rettype == 0:
        return TC
    elif rettype == 1:
        
        return S
    elif rettype == 2:
        
        return I
    else: 
        return  TC, S, I



def ts_mae(x,d=100):
    return ts_mean(abs(x-ts_mean(x,d)),d)


def ts_mse(x,d=100):
    return ts_mean((x-ts_mean(x,d))**2,d)


def ts_mape(x,d=100):
    return ts_mean(abs((x-ts_mean(x,d))/x),d)


def ts_smape(x,d=100):
    xhat = ts_mean(x,d)
    return ts_mean(2*abs(x-xhat)/(abs(x)+abs(xhat)),d)


def ts_derivative(y,x,d=1):
    return ts_delta(y,d)/ts_delta(x,d)
def vec_mean(df:pd.DataFrame):
    new_df = df.map(lambda x: np.mean(x))
    return new_df
def vec_choose(df:pd.DataFrame,order):
    def choose(x):
        try:
            return x[order]
        except:
            return 0
    new_df = df.map(choose)
    return new_df
def vec_len(df:pd.DataFrame):
    new_df = df.map(lambda x: len(x))
    return new_df
def vec_count(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    new_df = df.map(lambda x: np.count_nonzero(~np.isnan(x)))
    return new_df
def vec_max(df:pd.DataFrame):
    new_df = df.map(lambda x: np.max(x))
    return new_df
def vec_argmax(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    new_df = df.map(lambda x: np.argmax(x))
    return new_df

def vec_min(df:pd.DataFrame):
    new_df = df.map(lambda x: np.min(x))
    return new_df

def vec_argmin(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    new_df = df.map(lambda x: np.argmin(x))
    return new_df

def vec_norm(df: pd.DataFrame, l: int = 1, **kwargs) -> pd.DataFrame:
    new_df = df.map(lambda x: (np.sum(np.power(np.abs(x), l))) ** (1 / l))
    return new_df

def vec_percentage(df: pd.DataFrame, percentage: float = 0.5, **kwargs) -> pd.DataFrame:
    new_df = df.map(lambda x: np.quantile(x, percentage))
    return new_df

def vec_powersum(df: pd.DataFrame, constant: int = 2, **kwargs) -> pd.DataFrame:
    new_df = df.map(lambda x: np.sum(np.power(np.abs(x), constant)))
    return new_df

def vec_range(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return vec_max(x) - vec_min(x)

def vec_kurtosis(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    from scipy.stats import kurtosis
    new_df = df.map(lambda x: kurtosis(x))
    return new_df

def vec_skewness(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    from scipy.stats import skew
    new_df = df.map(lambda x: skew(x))
    return new_df

def vec_stddev(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    new_df = df.map(lambda x: np.std(x))
    return new_df
def vec_sum(df:pd.DataFrame):
    new_df = df.map(lambda x: np.sum(x))
    return new_df

def add(*args: pd.DataFrame, filter: bool = False, **kwargs) -> pd.DataFrame:
    if filter:
        args = [arg.fillna(0) for arg in args]
    total = sum(args)
    return total


def abs(x, **kwargs) -> pd.DataFrame:
    return sign(x)*x

def mean(*args: pd.DataFrame, filter: bool = False, **kwargs) -> pd.DataFrame:
    if filter:
        args = [arg.fillna(0) for arg in args]
    total = sum(args)
    return total / len(args)


def cummean(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return np.cumsum(df)/np.cumsum(df==df)

def ceil(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return np.ceil(x)


def divide(x: pd.DataFrame, y:pd.DataFrame, **kwargs) -> pd.DataFrame:
    return x / y


def exp(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return np.exp(x)


def floor(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return np.floor(x)


def fraction(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return sign(x) * (abs(x) - np.floor(abs(x)))


def inverse(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return 1 / x


def log(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return np.log(x)


def log_diff(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return ts_delta(np.log(x), 1)

def max_(*args: pd.DataFrame, **kwargs) -> pd.DataFrame:
    try:
        return np.maximum(*args)
    except:
        return np.max(args)

def min_(*args: pd.DataFrame, **kwargs) -> pd.DataFrame:
    try:
        return np.minimum(*args)
    except:
        return np.min(args)
    

def densify(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    x = x.T
    for column in x.columns:
        unique_values = x[column].unique()
        mapping = {value: i for i, value in enumerate(unique_values)}
        x[column] = x[column].map(mapping)
    return x.T

def multiply(*args: pd.DataFrame, filter: bool = False, **kwargs) -> pd.DataFrame:
    if filter:
        args = [arg.fillna(0) for arg in args]
    product = np.prod(args)
    return product


def nan_mask(x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return if_else(y < 0, np.nan, x)


def nan_out(x: pd.DataFrame, lower: float = 0, upper: float = 0, **kwargs) -> pd.DataFrame:
    return if_else((x < lower) | (x > upper), np.nan, x)


def power(x: pd.DataFrame, y: float, **kwargs) -> pd.DataFrame:
    return np.power(x, y)

def purify(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    return x


def replace(x: pd.DataFrame, target: str = "", dest: str = "", **kwargs) -> pd.DataFrame:
    tar = list(map(float, target.split(' ')))
    de = list(map(float, dest.split(' ')))
    if len(tar) == len(de):
        for i in range(len(de)):
            t = pd.DataFrame(de[i], columns=x.columns, index=x.index)
            x = np.where(x == tar[i], t, x)
    elif len(de) == 1:
        x = np.where(x.isin(tar), pd.DataFrame(de[0], columns=x.columns, index=x.index), x)
    return x


def reverse(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return -x

def sigmoid(x: pd.DataFrame,*args,**kwargs) -> pd.DataFrame :
    return 1/(1+exp(-x))
    

def round(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return np.round(x)


def round_down(df: pd.DataFrame, f: int, **kwargs) -> pd.DataFrame:
    return (df // f) * f


def sign(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return np.sign(x)

def signed_power(x: pd.DataFrame, y: float, **kwargs) -> pd.DataFrame:
    return sign(x) * power(abs(x), y)


def s_log_1p(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return sign(x) * log(1+abs(x))


def sqrt(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return np.sqrt(x)


def subtract(x: pd.DataFrame, y: pd.DataFrame, filter: bool = False, **kwargs) -> pd.DataFrame:
    if filter:
        x = x.fillna(0)
        y = y.fillna(0)
    return x - y


def to_nan(x: pd.DataFrame, value: float = 0, reverse: bool = False, **kwargs) -> pd.DataFrame:
    if reverse:
        return if_else(x.isna(), value, x)
    else:
        return if_else(x == value, np.nan, x)

def range_(*args: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return max_(*args) - min_(*args)

def if_else(condition, x, y, **kwargs) -> pd.DataFrame:
    # Determine the base DataFrame (the one to use for index and columns)
    base = None
    if isinstance(condition, pd.DataFrame):
        base = condition
    elif isinstance(x, pd.DataFrame):
        base = x
    elif isinstance(y, pd.DataFrame):
        base = y
    else:
        raise ValueError("At least one of the inputs must be a DataFrame.")
    
    # Convert scalars to DataFrames matching the base's structure
    if np.isscalar(condition):
        condition = pd.DataFrame(condition, index=base.index, columns=base.columns)
    if np.isscalar(x):
        x = pd.DataFrame(x, index=base.index, columns=base.columns)
    if np.isscalar(y):
        y = pd.DataFrame(y, index=base.index, columns=base.columns)
    
    # Perform element-wise selection based on the condition
    return pd.DataFrame(np.where(condition, x, y), index=base.index, columns=base.columns)

def is_nan(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return if_else(x.isna(), pd.DataFrame(1, columns=x.columns, index=x.index), pd.DataFrame(0, columns=x.columns, index=x.index))


# 5. Cross Sectional Operators

def normalize(x: pd.DataFrame, useStd: bool = False, **kwargs) -> pd.DataFrame:
    if useStd:
        return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)
    else:
        return x.sub(x.mean(axis=1), axis=0)


def one_side(x: pd.DataFrame, side: str = 'long', **kwargs) -> pd.DataFrame:
    if side == 'long':
        return x.where(x > 0, 0)
    else:
        return x.where(x < 0, 0)


def rank(x: pd.DataFrame,pct=True, **kwargs) -> pd.DataFrame:
    return x.rank(axis=1, ascending=True, pct=pct)


def get_top_n(df: pd.DataFrame, top_n: int, **kwargs) -> pd.DataFrame:
    return df.rank(axis=1, ascending=False) <= top_n


def signed_rank(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return sign(x)*rank(x)


def rank_by_side(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    x[x > 0] = rank(x[x > 0])
    x[x <= 0] = rank(x[x <= 0])
    return x


def generalized_rank(df: pd.DataFrame, m: int = 1, **kwargs) -> pd.DataFrame:
    N = len(df.columns)
    rank_df = pd.DataFrame(index=df.index, columns=df.columns)
    for i in df.columns:
        rank_i = 0
        for j in df.columns:
            rank_i += np.abs(df[i] - df[j]) ** m * np.sign(df[i] - df[j])
        rank_df[i] = rank_i / N
    return rank_df

def regression_neut(x: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    return x - regression_proj(x, *args)

# 
def regression_proj(x: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    import statsmodels.api as sm
    
    x = x.fillna(0).T
    args = [purify(arg.fillna(0)).T for arg in args]
    y = merge_matrix(*args)
    
    def get_value(z, t, name):
        y = pd.DataFrame(np.array(z[name].values.tolist())).fillna(0)
        X = pd.DataFrame(np.array(t[name].values.tolist())).fillna(0)
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        return model.predict()
    
    df = pd.DataFrame(columns=x.columns, index=x.index)
    for col in x.columns:
        df[col] = get_value(x, y, col)
    return purify(df).T

def scale(alpha: pd.DataFrame, scale: float = 1, longscale: float = 1, shortscale: float = 1, **kwargs) -> pd.DataFrame:
    if len(alpha.columns) == 1:
        return alpha * 0
    else:
        alpha = alpha.sub(alpha.mean(axis=1), axis=0)
        positive_alpha = alpha[alpha > 0]
        negative_alpha = alpha[alpha < 0]
        positive_sum = positive_alpha.sum(axis=1)
        negative_sum = negative_alpha.abs().sum(axis=1)
        positive_alpha = positive_alpha.div(positive_sum, axis=0)
        negative_alpha = negative_alpha.div(negative_sum, axis=0)
        alpha[alpha > 0] = positive_alpha * (longscale / (longscale + shortscale)) * scale
        alpha[alpha < 0] = negative_alpha * (shortscale / (longscale + shortscale)) * scale
        return alpha


def scale_down(p: pd.DataFrame, constant: float = 0, **kwargs) -> pd.DataFrame:
    return p.sub(p.min(axis=1), axis=0).div(p.max(axis=1) - p.min(axis=1), axis=0) - constant




def zscore(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)


def rank_gmean_amean_diff(*args: pd.DataFrame, **kwargs) -> pd.DataFrame:
    ranks = [rank(df) for df in args]
    geometric_mean = np.prod(ranks, axis=0)**(1/len(args))
    arithmetic_mean = np.mean(ranks, axis=0)
    return geometric_mean - arithmetic_mean



def arc_cos(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return np.arccos(x)


def arc_sin(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return np.arcsin(x)


def arc_tan(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return np.arctan(x)


def cos(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return np.cos(x)


def sin(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return np.sin(x)


def tan(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return np.tan(x)


def bucket(df: pd.DataFrame, buckets: str = "", rang: str = "", **kwargs) -> pd.DataFrame:
    def temp(x, buckets="", rang=""):
        if buckets:
            bucket_values = list(map(float, buckets.split(',')))
            for i in range(len(bucket_values)):
                if x > bucket_values[i]:
                    if i == len(bucket_values) - 1:
                        return i + 1
                    else:
                        continue
                else:
                    return i
        if rang:
            start, end, step = map(float, rang.split(','))
            bucket_values = np.arange(start, end, step)
            for i in range(len(bucket_values)):
                if x > bucket_values[i]:
                    if i == len(bucket_values) - 1:
                        return i + 1
                    else:
                        continue
                else:
                    return i
        return None

    df_temp = pd.DataFrame(index=df.index)
    for c in df.columns:
        df_temp[c] = df[c].apply(temp, buckets=buckets, rang=rang)
    return df_temp


def clamp(x: pd.DataFrame, lower: float = 0, upper: float = 0, inverse: bool = False, mask: float = np.nan, **kwargs) -> pd.DataFrame:
    q = if_else(x < lower, lower, x)
    u = if_else(q > upper, upper, q)
    temp = pd.DataFrame(mask, columns=x.columns, index=x.index)
    inv = pd.DataFrame(inverse, columns=x.columns, index=x.index)
    v = if_else((x > lower) & (x < upper), temp, x)
    return if_else(inv, v, u)


def filter(x: pd.DataFrame, h: str = '1,2,3,4', t: str = '0.5,0.05,0.005', **kwargs) -> pd.DataFrame:
    h1 = h.split(',')
    t1 = t.split(',')

    p1 = 0
    for i in range(1, len(h1) + 1):
        p1 += float(h1[i - 1]) * ts_delay(x, i)
    p2 = 0
    for j in range(1, len(t1) + 1):
        p2 += float(t1[j - 1]) * ts_delay(filter(x, h, t), j)
    return p1 + p2


def keep(x: pd.DataFrame, f: pd.DataFrame, period: int = 5, **kwargs) -> pd.DataFrame:
    D = days_from_last_change(f)
    u = trade_when(D < period, x, D > period)
    return u


def left_tail(x: pd.DataFrame, maximum: float = 0, **kwargs) -> pd.DataFrame:
    return if_else(x > maximum, np.nan, x)


def right_tail(x: pd.DataFrame, minimum: float = 0, **kwargs) -> pd.DataFrame:
    return if_else(x < minimum, np.nan, x)


def tail(x: pd.DataFrame, lower: float = 0, upper: float = 0, newval: float = 0, **kwargs) -> pd.DataFrame:
    return if_else((lower < x) & (x < upper), newval, x)

def sigmoid(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return 1 / (1 + np.exp(-x))


def tanh(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return x.transform(np.tanh)


def trade_when(trigger_trade: pd.DataFrame, alpha: pd.DataFrame, trigger_exit: pd.DataFrame, **kwargs) -> pd.DataFrame:
    temp = pd.DataFrame(np.nan, columns=alpha.columns, index=alpha.index)
    alpha_res = if_else(trigger_exit > 0, temp, if_else(trigger_trade > 0, alpha, ts_delay(alpha, 1)))
    return alpha_res

def sl_td(event: pd.DataFrame, signal: pd.DataFrame, close: pd.DataFrame, percent=0.1, **kwargs) -> pd.DataFrame:
    # Capture the close price at the time of the event
    close_at_event = trade_when(event, close, -1)

    # Determine stop-loss condition
    stop_loss_condition = abs(close - close_at_event) / close > percent

    # Generate stop-loss signals
    alpha = trade_when(event, signal, stop_loss_condition)

    return alpha

def group_count(x: pd.DataFrame, group, **kwargs) -> pd.DataFrame:
    df = pd.DataFrame()
    g = group.map(str).reindex(x.index).values
    v = x.values
    va = []
    for i in range(len(g)):
        df.index = g[i]
        df["v"] = v[i]
        df["v"] = df.groupby(df.index).agg("count")
        va.append(df["v"].tolist())
    df = pd.DataFrame(va, index=x.index, columns=x.columns)
    return df



def group_max(x: pd.DataFrame, group, **kwargs) -> pd.DataFrame:
    df = pd.DataFrame()
    g = group.map(str).reindex(x.index).values
    v = x.values
    va = []
    for i in range(len(g)):
        df.index = g[i]
        df["v"] = v[i]
        df["v"] = df.groupby(df.index).agg("max")
        va.append(df["v"].tolist())
    df = pd.DataFrame(va, index=x.index, columns=x.columns)
    return df



def group_mean(x: pd.DataFrame, group, **kwargs) -> pd.DataFrame:
    df = pd.DataFrame()
    g = group.map(str).reindex(x.index).values
    v = x.values
    va = []
    for i in range(len(g)):
        df.index = g[i]
        df["v"] = v[i]
        df["v"] = df.groupby(df.index).agg("mean")
        va.append(df["v"].tolist())
    df = pd.DataFrame(va, index=x.index, columns=x.columns)
    return df

def group_var(x: pd.DataFrame, group, **kwargs) -> pd.DataFrame:
    

    df = pd.DataFrame()
    g = group.map(str).reindex(x.index).values
    v = x.values
    va = []
    for i in range(len(g)):
        df.index = g[i]
        df["v"] = v[i]
        df["v"] = df.groupby(df.index).agg("var")
        va.append(df["v"].tolist())
    df = pd.DataFrame(va, index=x.index, columns=x.columns)
    return df

def group_skew(x: pd.DataFrame, group, **kwargs) -> pd.DataFrame:
    return group_moment(x,group,3,True)

def group_kurt(x: pd.DataFrame, group, **kwargs) -> pd.DataFrame:
    return group_moment(x,group,4,True)



def group_moment(x: pd.DataFrame, group,k=2,standardized=False, **kwargs) -> pd.DataFrame:
    if standardized:
        return group_mean((x-group_mean(x,group))**k,group)/(group_moment(x,group,2)**(k/2))
    else:
        return group_mean((x-group_mean(x,group))**k,group)


def group_median(x: pd.DataFrame, group, **kwargs) -> pd.DataFrame:
    df = pd.DataFrame()
    g = group.map(str).reindex(x.index).values
    v = x.values
    va = []
    for i in range(len(g)):
        df.index = g[i]
        df["v"] = v[i]
        df["v"] = df.groupby(df.index).agg("median")
        va.append(df["v"].tolist())
    df = pd.DataFrame(va, index=x.index, columns=x.columns)
    return df

def group_min(x: pd.DataFrame, group, **kwargs) -> pd.DataFrame:
    df = pd.DataFrame()
    g = group.map(str).reindex(x.index).values
    v = x.values
    va = []
    for i in range(len(g)):
        df.index = g[i]
        df["v"] = v[i]
        df["v"] = df.groupby(df.index).agg("min")
        va.append(df["v"].tolist())
    df = pd.DataFrame(va, index=x.index, columns=x.columns)
    return df

def group_neutralize(x: pd.DataFrame, group, **kwargs) -> pd.DataFrame:
    return x - group_mean(x, group)

def group_normalize(x: pd.DataFrame, group, scale: float = 1, **kwargs) -> pd.DataFrame:
    return x * scale / group_sum(abs(x), group)


def group_percentage(
    x: pd.DataFrame, group, percentage: float = 0.5
, **kwargs) -> pd.DataFrame:
    df = pd.DataFrame()
    g = group.map(str).reindex(x.index).values
    v = x.values
    va = []
    for i in range(len(g)):
        df.index = g[i]
        df["v"] = v[i]
        df["v"] = df.groupby(df.index).quantile(percentage)
        va.append(df["v"].tolist())
    df = pd.DataFrame(va, index=x.index, columns=x.columns)
    return df


def group_rank(x: pd.DataFrame, group, **kwargs) -> pd.DataFrame:
    df = pd.DataFrame()
    g = group.map(str).reindex(x.index).values
    v = x.values
    va = []
    for i in range(len(g)):
        df.index = g[i]
        df["v"] = v[i]
        df["v"] = df.groupby(df.index).rank()
        va.append(df["v"].tolist())
    df = pd.DataFrame(va, index=x.index, columns=x.columns)
    return df


def group_scale(x: pd.DataFrame, group, **kwargs) -> pd.DataFrame:
    return (
        x - group_min(x, group)
    ) / (group_max(x, group) - group_min(x, group))


def group_sum(x: pd.DataFrame, group, **kwargs) -> pd.DataFrame:
    df = pd.DataFrame()
    g = group.map(str).reindex(x.index).values
    v = x.values
    va = []
    for i in range(len(g)):
        df.index = g[i]
        df["v"] = v[i]
        df["v"] = df.groupby(df.index).agg("sum")
        va.append(df["v"].tolist())
    df = pd.DataFrame(va, index=x.index, columns=x.columns)
    return df

def group_std(x: pd.DataFrame, group, **kwargs) -> pd.DataFrame:
    return sqrt(group_var(x,group))

def group_zscore(x: pd.DataFrame, group, **kwargs) -> pd.DataFrame:
    return group_neutralize(x, group) / group_std(x, group)



def ts_beta(returns,d):
    market_returns=group_mean(returns,'market')
    return ts_cov(returns,market_returns,d)/ts_var(market_returns,d)

def ts_correlation_port(returns,d):
    market_returns=group_mean(returns,'market')
    return ts_corr(returns,market_returns,d)

def ts_unsystematic_risk(returns,d):
    market_returns=group_mean(returns,'market')
    return ts_var(returns,d)-ts_beta(returns,d)**2*ts_var(market_returns,d)

def ts_systematic_risk(returns,d):
    market_returns=group_mean(returns,'market')
    return ts_var(market_returns,d)


def historical_volatility_d(close,d):
    return ts_std(log(close/ts_delay(close,1)),d)

def parkinson_volatility_d(high, low,d):
    log_high_low_ratio = log(high / low) ** 2
    return (ts_sum(log_high_low_ratio,d)/(4*d*np.log(2)))**0.5