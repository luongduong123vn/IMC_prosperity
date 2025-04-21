from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math
import pandas as pd
import statistics

import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation
from typing import List, Tuple, Union, Dict

from math import log, sqrt, exp
from statistics import NormalDist


class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

class MarketData:
    end_pos: Dict[str, int] = {}
    buy_sum: Dict[str, int] = {}
    sell_sum: Dict[str, int] = {}
    bid_prices: Dict[str, List[float]] = {}
    bid_volumes: Dict[str, List[int]] = {}
    ask_prices: Dict[str, List[float]] = {}
    ask_volumes: Dict[str, List[int]] = {}
    fair: Dict[str, float] = {}

class Product:
    AMETHYSTS = "RAINFOREST_RESIN"
    STARFRUIT = "KELP"
    SQUID_INK = "SQUID_INK"
    PICNIC_BASKET1 = "PICNIC_BASKET1"  
    PICNIC_BASKET2 = "PICNIC_BASKET2" 
    ORCHIDS = "MAGNIFICENT_MACARONS"    
    JAMS = "JAMS"                          # was CHOCOLATE
    CROISSANTS = "CROISSANTS"              # was STRAWBERRIES
    DJEMBES = "DJEMBES" 
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"
    SPREAD1 = "SPREAD1"
    SPREAD2 = "SPREAD2"
    ARTIFICAL1 = "ARTIFICAL1"
    ARTIFICAL2 = "ARTIFICAL2"
    COCONUT = "VOLCANIC_ROCK"
    COCONUT_COUPON = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

PARAMS = {
    Product.AMETHYSTS: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 50,
        "min_spread": 0
    },
    Product.STARFRUIT: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 19,
        "reversion_beta": -0.26,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        'ret_vol': 0.00035,
    },
    Product.SPREAD: {
        "default_spread_mean": 379.50439988484239,
        # "default_spread_std" is not directly used since we compute the sample std,
        "spread_std_window": 45,
        "zscore_threshold": 7,
        "target_position": 58,
    },
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 19,
        "reversion_beta": -0.1,
        "reversion_alpha": -0.065,
        "reversion_gamma": -0.05,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        'ret_vol': 0.001,
        "drift": 0,
        "manage_position": False,
        "soft_position_limit": 30,
        "spread_adjustment": 1
    },
    Product.SPREAD1: {
        "default_spread_mean": 48.777856,
        "default_spread_std": 85.119723,
        "spread_window": 55,
        "zscore_threshold": 4,
        "target_position": 100,
    },
    Product.SPREAD2: {
        "default_spread_mean": 30.2336,
        "default_spread_std": 59.8536,
        "spread_window": 59,
        "zscore_threshold": 6,
        "target_position": 100,
    },
    Product.PICNIC_BASKET1: {
        "b2_adjustment_factor": 0.05
    },
    Product.COCONUT_COUPON: {
        "mean_volatility": 0.8621,
        "threshold": 0.0845123235721419,
        "strike": 9750,
        "time_to_expiry": 5 / 252,
        "std_window": 50,
        "zscore_threshold": 1.5,
        "zscore_threshold_low": -1.5,
        "zscore_threshold_high": 1,
        "zscore_threshold_low_exit": -1.2,
        "zscore_threshold_high_exit": 0.75,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "mean_volatility": 0.15959997370608378,
        "threshold": 0.00163,
        "strike": 9500,
        "time_to_expiry": 4 / 252,
        "std_window": 6,
        "zscore_threshold": 21,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 0.15959997370608378,
        "threshold": 0.00163,
        "strike": 9750,
        "time_to_expiry": 4 / 252,
        "std_window": 6,
        "zscore_threshold": 21,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 16,
        "threshold_high": 14.3, #17.178
        "threshold_low": 0.877,
        "threshold_exit": 8.31,
        "strike": 10000,
        "time_to_expiry": 4 / 252,
        "std_window": 6,
        "zscore_threshold": 21,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "mean_volatility": 16,
        "threshold_high": 3.04, #3.975
        "threshold_low": 0.249,
        "threshold_exit": 1.41,
        "strike": 10250,
        "time_to_expiry": 4 / 252,
        "std_window": 6,
        "zscore_threshold": 21,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "mean_volatility": 16,
        "threshold_high": 7.54, #9.80
        "threshold_low": 1.55,
        "threshold_exit": 3.47,
        "strike": 10500,
        "time_to_expiry": 4 / 252,
        "std_window": 6,
        "zscore_threshold": 21,
    },
    Product.ORCHIDS: {
        "make_edge": 1,
        "make_probability": 0.8,
        "conversion_limit": 10,
        "periods_below_max": 10,
        "periods_above_min": 20,
        "hard_level": 50,
    },
}

PICNIC1_WEIGHTS = {
    Product.DJEMBES: 1,
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
}
PICNIC2_WEIGHTS = {
    Product.CROISSANTS: 4,
    Product.JAMS: 2}

BASKET_WEIGHTS = {
    Product.JAMS: 3,
    Product.CROISSANTS: 6,
    Product.DJEMBES: 1,
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.AMETHYSTS: 50,
            Product.STARFRUIT: 50,
            Product.SQUID_INK: 50,
            Product.PICNIC_BASKET1: 60,
            Product.JAMS: 250,
            Product.CROISSANTS: 350,
            Product.DJEMBES: 60,
            Product.COCONUT: 400,
            Product.COCONUT_COUPON: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
            Product.ORCHIDS: 75
        }
        self.history = {}
        self.risk_adjustment = 0.8
        self.max_spread = 5

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        bid_volume: int,
        ask_volume: int,
    ) -> (int, int):
        if product == "SQUID_INK":
            buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order(product, round(bid), buy_quantity))  # Buy order
        else:
            buy_quantity = 1*(self.LIMIT[product] - (position + buy_order_volume))
            if buy_quantity > 0:
                orders.append(Order(product, round(bid), buy_quantity))  # Buy order
        if product == "SQUID_INK":
            sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        else:
            sell_quantity = 1*(self.LIMIT[product] + (position - sell_order_volume))
            if sell_quantity > 0:
                orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    
    def calculate_time_series_slope_n(self, values: Union[List[float], np.ndarray, pd.Series]) -> float:
        """
        Calculate the slope of the line of best fit through a time series of values.
        """
        if isinstance(values, (list, pd.Series)):
            values = np.array(values)
        
        n = len(values)
        if n < 2:
            return 0
        
        # For n consecutive timestamps (0 to n-1), the mean is (n-1)/2
        t_mean = (n - 1) / 2
        v_mean = np.mean(values)
        
        # Create timestamp deviations from mean
        t_deviations = np.arange(n) - t_mean
        
        # Calculate slope (beta)
        numerator = np.sum(t_deviations * (values - v_mean))
        denominator = np.sum(t_deviations ** 2)
        
        return numerator / denominator if denominator != 0 else 0


    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def starfruit_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.STARFRUIT]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.STARFRUIT]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("starfruit_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["starfruit_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            z = np.random.normal(0, 1)
            
            # Calculate volatility based on historical returns if we have enough data
            if len(traderObject.get('starfruit_price_history', [])) >= 10:
                # Get the last 10 prices
                prices = traderObject['starfruit_price_history'][-10:]
                # Calculate returns
                returns = np.diff(prices) / prices[:-1]
                # Calculate volatility
                ret_vol = float(np.std(returns))
            else:
                # Use default volatility from params
                ret_vol = self.params[Product.STARFRUIT]["ret_vol"]

            traderObject['starfruit_price_history'].append(mmmid_price)
            # Keep only the last 10 prices
            traderObject['starfruit_price_history'] = traderObject['starfruit_price_history'][-50:]
            starfruit_fv_history = traderObject['starfruit_price_history'][-15:]
                
            if traderObject.get("starfruit_last_price", None) != None:
                last_price = traderObject["starfruit_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = ret_vol * z
                #fair = round(sum(starfruit_fv_history)/len(starfruit_fv_history))
                fair = mmmid_price * (1 + last_returns*self.params[Product.STARFRUIT]["reversion_beta"])
            else:
                fair = mmmid_price
            traderObject["starfruit_last_price"] = mmmid_price
             # Update price history
                # Add new price
            
            return fair
        return None
    
    def ink_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("ink_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["ink_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            z = np.random.normal(0, 1)
            
            # Calculate volatility based on historical returns if we have enough data
            if len(traderObject.get('ink_price_history', [])) >= 10:
                # Get the last 10 prices
                prices = traderObject['ink_price_history'][-10:]
                # Calculate returns
                returns = np.diff(prices) / prices[:-1]
                # Calculate volatility
                ret_vol = float(np.std(returns))
            else:
                # Use default volatility from params
                ret_vol = self.params[Product.SQUID_INK]["ret_vol"]

            traderObject['ink_price_history'].append(mmmid_price)
            # Keep only the last 10 prices
            traderObject['ink_price_history'] = traderObject['ink_price_history'][-50:]
            ink_fv_history = traderObject['ink_price_history'][-5:]
            starfruit_price_history = traderObject['ink_price_history']
            prices = starfruit_price_history
            returns = np.diff(prices) / prices[:-1]
            if len(returns) >= len(starfruit_price_history)-1:
                n = len(returns)
                slope = self.calculate_time_series_slope_n(prices)
                # recent_mean = np.mean(returns) if n >= 3 else np.mean(returns)
                # next_return = recent_mean + slope * (n - (n-1)/2)
                # recent_mean = np.mean(prices[-3:]) if n >= 3 else np.mean(prices)
                # next_prices = recent_mean + slope * (n - (n-1)/2)
            else:
                slope = 0
            beta = self.params[Product.SQUID_INK]["reversion_beta"]
            alpha = self.params[Product.SQUID_INK]["reversion_alpha"]
            gamma = self.params[Product.SQUID_INK]["reversion_gamma"]
            if traderObject.get("ink_last_price", None) != None:
                last_price = traderObject["ink_last_price"]
                if len(traderObject["ink_price_history"]) >= 10:
                    last_10_price = traderObject["ink_price_history"][-10]
                else:
                    last_10_price = last_price

                last_returns = returns[-1]  
                last_returns_3 = returns[-3] if len(returns) >= 3 else 0
                last_returns_11 = returns[-11] if len(returns) >= 11 else 0

                last_10_returns = (mmmid_price - last_10_price) / last_10_price
                pred_returns = self.params[Product.SQUID_INK]["drift"] + ret_vol * z
                #fair = round(sum(ink_fv_history)/len(ink_fv_history),2)
                fair = mmmid_price * (1 + last_returns_3*beta + last_returns_11*alpha)
                #fair = mmmid_price * (1 + last_returns*gamma)
                #fair = mmmid_price * (1 + last_returns)
                #fair = mmmid_price * (1 + next_return)
                #fair = next_prices
                #fair = mmmid_price
            else:
                fair = mmmid_price
            traderObject["ink_last_price"] = mmmid_price
             # Update price history
                # Add new price
            
            return fair
        return None
    
    def product_fair_value(self, order_depth: OrderDepth, traderObject, product: Product, adverse_volume: int) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= adverse_volume
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= adverse_volume
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get(f"{product}_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject[f"{product}_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            z = np.random.normal(0, 1)
            
            # Calculate volatility based on historical returns if we have enough data
            if len(traderObject.get(f'{product}_price_history', [])) >= 10:
                # Get the last 10 prices
                prices = traderObject[f'{product}_price_history'][-10:]
                # Calculate returns
                returns = np.diff(prices) / prices[:-1]
                # Calculate volatility
                ret_vol = float(np.std(returns))
            else:
                # Use default volatility from params
                ret_vol = self.params[product]["ret_vol"]

            traderObject[f'{product}_price_history'].append(mmmid_price)
            # Keep only the last 10 prices
            traderObject[f'{product}_price_history'] = traderObject[f'{product}_price_history'][-50:]
            fv_history = traderObject[f'{product}_price_history'][-5:]
            price_history = traderObject[f'{product}_price_history']
            prices = price_history
            returns = np.diff(prices) / prices[:-1]
            if len(returns) >= len(price_history)-1:
                n = len(returns)
                slope = self.calculate_time_series_slope_n(prices)
                # recent_mean = np.mean(returns) if n >= 3 else np.mean(returns)
                # next_return = recent_mean + slope * (n - (n-1)/2)
                # recent_mean = np.mean(prices[-3:]) if n >= 3 else np.mean(prices)
                # next_prices = recent_mean + slope * (n - (n-1)/2)
            else:
                slope = 0
            if traderObject.get(f"{product}_last_price", None) != None:
                last_price = traderObject[f"{product}_last_price"]
                if len(traderObject[f"{product}_price_history"]) >= 10:
                    last_10_price = traderObject[f"{product}_price_history"][-10]
                else:
                    last_10_price = last_price
                last_returns = (mmmid_price - last_price) / last_price
                last_10_returns = (mmmid_price - last_10_price) / last_10_price
                pred_returns = self.params[product]["drift"] + ret_vol * z
                #fair = round(sum(ink_fv_history)/len(ink_fv_history),2)
                fair = mmmid_price * (1 + last_returns*self.params[product]["reversion_beta"])
                #fair = mmmid_price * (1 + last_returns)
                #fair = mmmid_price * (1 + next_return)
                #fair = next_prices
                #fair = mmmid_price
            else:
                fair = mmmid_price
            traderObject[f"{product}_last_price"] = mmmid_price
             # Update price history
                # Add new price
            
            return fair
        return None

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        asks_volume_above_fair = [
            volume
            for price, volume in order_depth.sell_orders.items()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]
        bids_volume_below_fair = [
            volume
            for price, volume in order_depth.buy_orders.items()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None
        best_ask_volume = order_depth.sell_orders[best_ask_above_fair] if best_ask_above_fair != None else 0
        best_bid_volume = order_depth.buy_orders[best_bid_below_fair] if best_bid_below_fair != None else 0
        #spread = best_ask_above_fair - best_bid_below_fair

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
            best_bid_volume,
            best_ask_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def make_order_starfruit(
        self,
        product,
        traderObject,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        asks_volume_above_fair = [
            volume
            for price, volume in order_depth.sell_orders.items()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]
        bids_volume_below_fair = [
            volume
            for price, volume in order_depth.buy_orders.items()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None
        best_ask_volume = order_depth.sell_orders[best_ask_above_fair] if best_ask_above_fair != None else 0
        best_bid_volume = order_depth.buy_orders[best_bid_below_fair] if best_bid_below_fair != None else 0
        ask_volume = sum(order_depth.sell_orders.values())
        bid_volume = sum(order_depth.buy_orders.values())
        ask_edge = default_edge
        bid_edge = default_edge
        edge = default_edge
        if len(traderObject.get('starfruit_price_history', [])) >= 10:
            # Get the last 10 prices
            starfruit_vol_history = traderObject['starfruit_price_history'][-10:]
            prices = starfruit_vol_history
            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            # Calculate volatility
            realized_vol = float(np.std(returns))
            
            edge = default_edge
        else:
            # Use default volatility from params
            edge = default_edge
        ask = round(fair_value + edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                #ask = best_ask_above_fair - 1  # penny
                ask = best_ask_above_fair - edge

        bid = round(fair_value - edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                #bid = best_bid_below_fair + 1
                bid = best_bid_below_fair + edge

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1
        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
            bid_volume,
            ask_volume,
            )

        return orders, buy_order_volume, sell_order_volume
    
    def make_order_new(
        self,
        product,
        traderObject,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        asks_volume_above_fair = [
            volume
            for price, volume in order_depth.sell_orders.items()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]
        bids_volume_below_fair = [
            volume
            for price, volume in order_depth.buy_orders.items()
            if price < fair_value - disregard_edge
        ]


        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None
        best_ask_volume = order_depth.sell_orders[best_ask_above_fair] if best_ask_above_fair != None else 0
        best_bid_volume = order_depth.buy_orders[best_bid_below_fair] if best_bid_below_fair != None else 0
        ask_volume = sum(order_depth.sell_orders.values())
        bid_volume = sum(order_depth.buy_orders.values())
        if len(traderObject.get(f'{product}_price_history', [])) >= 10:
            # Get the last 10 prices
            starfruit_vol_history = traderObject[f'{product}_price_history'][-10:]
            prices = starfruit_vol_history
            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            # Calculate volatility
            realized_vol = float(np.std(returns))
            # if realized_vol / self.params[product]["ret_vol"] >= 3:
            #     #edge = -max(round((realized_vol / self.params[Product.SQUID_INK]["ret_vol"]) * default_edge  * 0.7), default_edge)   
            #     edge = 0
            # elif realized_vol / self.params[product]["ret_vol"] <= 0.4:
            #     #edge = min(round((realized_vol / self.params[Product.SQUID_INK]["ret_vol"]) * default_edge  * 1.5), default_edge) 
            #     edge = -1
            # else:
            #     edge = 1

            edge = default_edge
        else:
            # Use default volatility from params
            edge = default_edge
        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                #ask = best_ask_above_fair - 1  # penny
                ask = best_ask_above_fair - edge

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                #bid = best_bid_below_fair + 1
                bid = best_bid_below_fair + edge
        spread_factor = self.params[product]["spread_adjustment"]
        if manage_position:
            if position > soft_position_limit:
                bid = best_bid_below_fair - spread_factor * edge if best_bid_below_fair != None else fair_value - spread_factor*edge
                if ask >= fair_value and ask <= fair_value + 1:
                    ask = fair_value
                elif ask > fair_value + 1:
                    ask = ask - 1
            elif position < -1 * soft_position_limit:
                ask = best_ask_above_fair + spread_factor * edge if best_ask_above_fair != None else fair_value + spread_factor*edge
                if bid <= fair_value and bid >= fair_value - 1:
                    bid = fair_value
                elif bid < fair_value - 1:
                    bid = bid + 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
            bid_volume,
            ask_volume,
        )

        return orders, buy_order_volume, sell_order_volume
    
    def make_order_ink(
        self,
        product,
        traderObject,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        asks_volume_above_fair = [
            volume
            for price, volume in order_depth.sell_orders.items()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]
        bids_volume_below_fair = [
            volume
            for price, volume in order_depth.buy_orders.items()
            if price < fair_value - disregard_edge
        ]


        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None
        best_ask_volume = order_depth.sell_orders[best_ask_above_fair] if best_ask_above_fair != None else 0
        best_bid_volume = order_depth.buy_orders[best_bid_below_fair] if best_bid_below_fair != None else 0
        ask_volume = sum(order_depth.sell_orders.values())
        bid_volume = sum(order_depth.buy_orders.values())
        if len(traderObject.get('ink_price_history', [])) >= 10:
            # Get the last 10 prices
            starfruit_vol_history = traderObject['ink_price_history'][-10:]
            prices = starfruit_vol_history
            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            # Calculate volatility
            realized_vol = float(np.std(returns))
            if realized_vol / self.params[Product.SQUID_INK]["ret_vol"] >= 3:
                #edge = -max(round((realized_vol / self.params[Product.SQUID_INK]["ret_vol"]) * default_edge  * 0.7), default_edge)   
                edge = 0
            elif realized_vol / self.params[Product.SQUID_INK]["ret_vol"] <= 0.4:
                #edge = min(round((realized_vol / self.params[Product.SQUID_INK]["ret_vol"]) * default_edge  * 1.5), default_edge) 
                edge = -1
            else:
                edge = 1

            #edge = default_edge
        else:
            # Use default volatility from params
            edge = default_edge
        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                #ask = best_ask_above_fair - 1  # penny
                ask = best_ask_above_fair - edge

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                #bid = best_bid_below_fair + 1
                bid = best_bid_below_fair + edge
        spread_factor = self.params[Product.SQUID_INK]["spread_adjustment"]
        if manage_position:
            if position > soft_position_limit:
                bid = best_bid_below_fair - spread_factor * edge if best_bid_below_fair != None else fair_value - spread_factor*edge
                if ask >= fair_value and ask <= fair_value + 1:
                    ask = fair_value
                elif ask > fair_value + 1:
                    ask = ask - 1
            elif position < -1 * soft_position_limit:
                ask = best_ask_above_fair + spread_factor * edge if best_ask_above_fair != None else fair_value + spread_factor*edge
                if bid <= fair_value and bid >= fair_value - 1:
                    bid = fair_value
                elif bid < fair_value - 1:
                    bid = bid + 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
            bid_volume,
            ask_volume,
        )

        return orders, buy_order_volume, sell_order_volume
    
    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def get_coconut_coupon_mid_price(
        self, coconut_coupon_order_depth: OrderDepth, traderData: Dict[str, Any]
    ):
        if (
            len(coconut_coupon_order_depth.buy_orders) > 0
            and len(coconut_coupon_order_depth.sell_orders) > 0
        ):
            best_bid = max(coconut_coupon_order_depth.buy_orders.keys())
            best_ask = min(coconut_coupon_order_depth.sell_orders.keys())
            traderData["prev_coupon_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return traderData["prev_coupon_price"]
        
    def trade_10000(self, state, market_data,traderObject, product, strike):
        if "iv_arb_limit" not in traderObject:
            traderObject["iv_arb_limit"] = 100
        delta_sum = 0
        orders = {}
        for p in ["VOLCANIC_ROCK", product]:
            orders[p] = []
        fair = market_data.fair[product]
        underlying_fair = market_data.fair["VOLCANIC_ROCK"]
        dte = 4 - state.timestamp / 1_000_000
        v_t = self.implied_vol_call(fair, underlying_fair, strike, dte, 0)
        delta = BlackScholes.delta(underlying_fair, strike, dte, v_t)
        gamma = BlackScholes.gamma(underlying_fair, strike, dte, v_t)
        m_t = np.log(strike / underlying_fair) / np.sqrt(dte / 365)
        # print(f"m_t = {m_t}")

        # , ,
        base_coef = 0.14786181  # 0.13571776890273662 + 0.00229274*dte
        linear_coef = 0.00099561  # -0.03685812200491957 + 0.0072571*dte
        squared_coef = 0.23544086  # 0.16277746617792221 + 0.01096456*dte
        fair_iv = base_coef + linear_coef * m_t + squared_coef * (m_t ** 2)
        # print(fair_iv)

        diff = v_t - fair_iv
        if f"prices_{product}" not in traderObject:
            traderObject[f"prices_{product}"] = [diff]
        else:
            traderObject[f"prices_{product}"].append(diff)
        threshold = 0.0035
        fill_amount = 0
        # print(diff)
        if len(traderObject[f"prices_{product}"]) > 25:
            diff -= np.mean(np.array(traderObject[f"prices_{product}"][-20:]))
            traderObject[f"prices_{product}"].pop(0)
            current_position = market_data.end_pos[f"{product}"]
            if diff > threshold:  # short vol so sell option, buy und
                amount = min(market_data.buy_sum["VOLCANIC_ROCK"], market_data.sell_sum[f"{product}"],traderObject["iv_arb_limit"] - current_position)
                amount = min(amount, -sum(market_data.ask_volumes["VOLCANIC_ROCK"]),
                             sum(market_data.bid_volumes[f"{product}"]))
                option_amount = amount
                rock_amount = amount


                for i in range(0, len(market_data.bid_prices[f"{product}"])):
                    fill = min(market_data.bid_volumes[f"{product}"][i], option_amount)
                    fill_amount = fill
                    delta_sum -= delta * fill
                    if fill != 0:
                        orders[f"{product}"].append(Order(f"{product}",market_data.bid_prices[f"{product}"][i],-fill))
                        market_data.sell_sum[f"{product}"] -= fill
                        market_data.end_pos[f"{product}"] -= fill
                        option_amount -= fill
                # print(rock_amount)
                # for i in range(0, len(market_data.ask_prices["VOLCANIC_ROCK"])):
                #     delta_fill = delta * fill_amount
                #     #print(fill)
                #     if delta_fill != 0:
                #         orders["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", market_data.ask_prices["VOLCANIC_ROCK"][i], delta_fill))
                #         market_data.buy_sum["VOLCANIC_ROCK"] -= delta_fill
                #         market_data.end_pos["VOLCANIC_ROCK"] += delta_fill
                #         rock_amount -= delta_fill
                #         #print(fill)

            elif diff < -threshold:  # long vol
                # print("LONG")
                # print("----")
                amount = min(market_data.buy_sum[f"{product}"], market_data.sell_sum[f"VOLCANIC_ROCK"],traderObject["iv_arb_limit"] + current_position)
                # print(amount)
                amount = min(amount, -sum(market_data.ask_volumes[f"{product}"]),sum(market_data.bid_volumes["VOLCANIC_ROCK"]))
                # print(amount)
                option_amount = amount
                rock_amount = amount
                # print(f"{rock_amount} rocks")
                for i in range(0, len(market_data.ask_prices[f"{product}"])):
                    fill = min(-market_data.ask_volumes[f"{product}"][i], option_amount)
                    fill_amount = fill
                    delta_sum += delta * fill
                    if fill != 0:
                        orders[f"{product}"].append(Order(f"{product}",market_data.ask_prices[f"{product}"][i], fill))
                        market_data.buy_sum[f"{product}"] -= fill
                        market_data.end_pos[f"{product}"] += fill
                        option_amount -= fill

                # for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK"])):
                #     delta_fill = delta * fill_amount
                #     #print(fill)
                #     if delta_fill != 0:
                #         orders["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", market_data.bid_prices["VOLCANIC_ROCK"][i], -delta_fill))
                #         market_data.sell_sum["VOLCANIC_ROCK"] -= delta_fill
                #         market_data.end_pos["VOLCANIC_ROCK"] -= delta_fill
                #         rock_amount -= delta_fill

        return orders[f"{product}"], delta, gamma

    def coconut_hedge_orders(
        self,
        coconut_order_depth: OrderDepth,
        coconut_coupon_order_depth: OrderDepth,
        coconut_coupon_orders: List[Order],
        coconut_position: int,
        coconut_coupon_position: int,
        delta: float,
        gamma: float,
    ) -> List[Order]:
        if coconut_coupon_orders == None or len(coconut_coupon_orders) == 0:
            coconut_coupon_position_after_trade = coconut_coupon_position
        else:
            coconut_coupon_position_after_trade = coconut_coupon_position + sum(
                order.quantity for order in coconut_coupon_orders
            )

        #target_coconut_position = -delta * coconut_coupon_position_after_trade - 0.5 * gamma * coconut_coupon_position_after_trade**2
        target_coconut_position = -delta * coconut_coupon_position_after_trade

        if abs(coconut_position) >= abs(target_coconut_position)*0.5 and abs(coconut_position) <= abs(target_coconut_position)*1.5:
            return None

        target_coconut_quantity = target_coconut_position - coconut_position

        orders: List[Order] = []
        if target_coconut_quantity > 0:
            # Buy COCONUT
            best_ask = min(coconut_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_coconut_quantity),
                self.LIMIT[Product.COCONUT] - coconut_position,
            )
            if quantity > 0:
                orders.append(Order(Product.COCONUT, best_ask, round(quantity)))

        elif target_coconut_quantity < 0:
            # Sell COCONUT
            best_bid = max(coconut_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_coconut_quantity),
                self.LIMIT[Product.COCONUT] + coconut_position,
            )
            if quantity > 0:
                orders.append(Order(Product.COCONUT, best_bid, -round(quantity)))

        return orders
    
    def submit_order(self, product, orders, price, volume):
        orders.append(Order(product, round(price), volume))

    def orchids_implied_bid_ask(
        self,
        observation: ConversionObservation,
        traderObject: dict,
    ) -> (float, float, bool):
        # Calculate base implied bid and ask
        base_implied_bid = observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1
        base_implied_ask = observation.askPrice + observation.importTariff + observation.transportFees + 0.1
        base_implied_mid = (base_implied_bid + base_implied_ask) / 2
        
        # Check if sunlight index exists
        if hasattr(observation, 'sunlightIndex'):
            # Track histories in traderObject
            if 'sunlight_history' not in traderObject:
                traderObject['sunlight_history'] = []
            if 'sugar_price_history' not in traderObject:
                traderObject['sugar_price_history'] = []
            if 'implied_mid_history' not in traderObject:
                traderObject['implied_mid_history'] = []
            
            # Add current values to histories
            traderObject['sunlight_history'].append(observation.sunlightIndex)
            if hasattr(observation, 'sugarPrice'):
                traderObject['sugar_price_history'].append(observation.sugarPrice)
            traderObject['implied_mid_history'].append(base_implied_mid)
            
            # Keep only last 26 periods for sunlight and 6 periods for prices
            periods_below_max = self.params[Product.ORCHIDS]["periods_below_max"]
            periods_above_min = self.params[Product.ORCHIDS]["periods_above_min"]
            total_lookback = max(periods_below_max, periods_above_min) + 6
            hard_level = self.params[Product.ORCHIDS]["hard_level"]
            if len(traderObject['sunlight_history']) > total_lookback:
                traderObject['sunlight_history'].pop(0)
            if len(traderObject['sugar_price_history']) > 6:
                traderObject['sugar_price_history'].pop(0)
            if len(traderObject['implied_mid_history']) > 6:
                traderObject['implied_mid_history'].pop(0)
            
            # Check conditions if we have enough history
            if len(traderObject['sunlight_history']) >= total_lookback and len(traderObject['sugar_price_history']) >= 6 and len(traderObject['implied_mid_history']) >= 6:
                last_10_indices = traderObject['sunlight_history'][-periods_below_max:]
                period_11_index = traderObject['sunlight_history'][-(periods_below_max+1)]
                last_20_indices = traderObject['sunlight_history'][-periods_above_min:]
                period_21_index = traderObject['sunlight_history'][-(periods_above_min+1)]
                current_sunlight = observation.sunlightIndex
                current_sugar_price = observation.sugarPrice
                
                # Check if all last 10 indices are below period 11 and current sunlight is below 50
                if all(idx < period_11_index for idx in last_10_indices):
                    if current_sunlight < hard_level:
                        # Use sugar prices from -6 to -1 (5 periods) and implied mids from -5 to 0 (5 periods)
                        # This creates a lagged relationship where we compare current implied mid with previous sugar price
                        lookback_sugar_prices = traderObject['sugar_price_history'][-6:-1]  # Last 5 sugar prices
                        lookback_implied_mids = traderObject['implied_mid_history'][-5:]  # Current and last 4 implied mids
                        
                        if len(lookback_sugar_prices) > 0 and len(lookback_implied_mids) > 0:
                            # Calculate the ratio for each period, comparing current implied mid with previous sugar price
                            ratios = [mid/price for mid, price in zip(lookback_implied_mids[1:], lookback_sugar_prices)]
                            avg_ratio = sum(ratios) / len(ratios)
                            
                            # Predict next implied mid using current sugar price
                            implied_mid = current_sugar_price * avg_ratio
                            
                            # Adjust bid and ask around the new implied mid while maintaining the same spread
                            spread = 2
                            return implied_mid - spread/2, implied_mid + spread/2, True

                elif all(idx > period_21_index for idx in last_20_indices):
                    if current_sunlight < hard_level:
                        #Use sugar prices from -6 to -1 (5 periods) and implied mids from -5 to 0 (5 periods)
                        # This creates a lagged relationship where we compare current implied mid with previous sugar price
                        lookback_sugar_prices = traderObject['sugar_price_history'][-6:-1]  # Last 5 sugar prices
                        lookback_implied_mids = traderObject['implied_mid_history'][-5:]  # Current and last 4 implied mids
                        
                        if len(lookback_sugar_prices) > 0 and len(lookback_implied_mids) > 0:
                            # Calculate the ratio for each period, comparing current implied mid with previous sugar price
                            ratios = [mid/price for mid, price in zip(lookback_implied_mids[1:], lookback_sugar_prices)]
                            avg_ratio = sum(ratios) / len(ratios)
                            
                            # Predict next implied mid using current sugar price
                            implied_mid = current_sugar_price * avg_ratio
                            
                            # Adjust bid and ask around the new implied mid while maintaining the same spread
                            spread = 2
                            return implied_mid - spread/2, implied_mid + spread/2, True
        
        # Default case - return base calculations
        return base_implied_bid, base_implied_ask, False

    def orchids_arb_take(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        position: int,
        implied_bid: float,
        implied_ask: float,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.ORCHIDS]
        buy_order_volume = 0
        sell_order_volume = 0

        buy_quantity = position_limit - position
        sell_quantity = position_limit + position

        ask = round(observation.askPrice) - 2

        if ask > implied_ask:
            edge = (ask - implied_ask) * self.params[Product.ORCHIDS][
                "make_probability"
            ]
        else:
            edge = 0

        for price in sorted(list(order_depth.sell_orders.keys())):
            if price > implied_bid - edge:
                break

            if price < implied_bid - edge:
                quantity = min(
                    abs(order_depth.sell_orders[price]), buy_quantity
                )  # max amount to buy
                if quantity > 0:
                    orders.append(Order(Product.ORCHIDS, round(price), quantity))
                    buy_order_volume += quantity

        for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
            if price < implied_ask + edge:
                break

            if price > implied_ask + edge:
                quantity = min(
                    abs(order_depth.buy_orders[price]), sell_quantity
                )  # max amount to sell
                if quantity > 0:
                    orders.append(Order(Product.ORCHIDS, round(price), -quantity))
                    sell_order_volume += quantity

        return orders, buy_order_volume, sell_order_volume

    def orchids_arb_clear(self, position: int) -> int:
        conversions = -min(position*int(np.sign(position)), self.params[Product.ORCHIDS]["conversion_limit"])*int(np.sign(position))
        #conversions = -position
        return conversions

    def orchids_arb_make(
        self,
        observation: ConversionObservation,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        implied_bid: float,
        implied_ask: float,
        orchids_implied_used: bool,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.ORCHIDS]

        # Implied Bid = observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1
        # Implied Ask = observation.askPrice + observation.importTariff + observation.transportFees

        aggressive_ask = round(observation.askPrice) - self.params[Product.ORCHIDS]["make_edge"]
        aggressive_bid = round(observation.bidPrice) + self.params[Product.ORCHIDS]["make_edge"]

        if not orchids_implied_used:
            if aggressive_bid < implied_bid:
                bid = aggressive_bid
            else:
                bid = implied_bid - 1
        else:
            bid = implied_bid

        if not orchids_implied_used:
            if aggressive_ask >= implied_ask + 0.5:
                ask = aggressive_ask
            elif aggressive_ask + 1 >= implied_ask + 0.5:
                ask = aggressive_ask
                #ask = aggressive_ask + 1
            else:
                ask = implied_ask + 1
                #ask = implied_ask + 2
        else:
            ask = implied_ask

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.ORCHIDS, round(bid), buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.ORCHIDS, round(ask), -sell_quantity))

        return orders, buy_order_volume, sell_order_volume

    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def get_synthetic_basket_order_depth(self, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        # Use basket weights for JAMS, CROISSANTS, and DJEMBES.
        JAMS_PER_BASKET = BASKET_WEIGHTS[Product.JAMS]
        CROISSANTS_PER_BASKET = BASKET_WEIGHTS[Product.CROISSANTS]
        DJEMBES_PER_BASKET = BASKET_WEIGHTS[Product.DJEMBES]
        synthetic_order_price = OrderDepth()
        jams_best_bid = max(order_depths[Product.JAMS].buy_orders.keys()) if order_depths[Product.JAMS].buy_orders else 0
        jams_best_ask = min(order_depths[Product.JAMS].sell_orders.keys()) if order_depths[Product.JAMS].sell_orders else float('inf')
        croissants_best_bid = max(order_depths[Product.CROISSANTS].buy_orders.keys()) if order_depths[Product.CROISSANTS].buy_orders else 0
        croissants_best_ask = min(order_depths[Product.CROISSANTS].sell_orders.keys()) if order_depths[Product.CROISSANTS].sell_orders else float('inf')
        djembes_best_bid = max(order_depths[Product.DJEMBES].buy_orders.keys()) if order_depths[Product.DJEMBES].buy_orders else 0
        djembes_best_ask = min(order_depths[Product.DJEMBES].sell_orders.keys()) if order_depths[Product.DJEMBES].sell_orders else float('inf')
        implied_bid = (jams_best_bid * JAMS_PER_BASKET +
                       croissants_best_bid * CROISSANTS_PER_BASKET +
                       djembes_best_bid * DJEMBES_PER_BASKET)
        implied_ask = (jams_best_ask * JAMS_PER_BASKET +
                       croissants_best_ask * CROISSANTS_PER_BASKET +
                       djembes_best_ask * DJEMBES_PER_BASKET)
        if implied_bid > 0:
            jams_bid_volume = order_depths[Product.JAMS].buy_orders[jams_best_bid] // JAMS_PER_BASKET
            croissants_bid_volume = order_depths[Product.CROISSANTS].buy_orders[croissants_best_bid] // CROISSANTS_PER_BASKET
            djembes_bid_volume = order_depths[Product.DJEMBES].buy_orders[djembes_best_bid] // DJEMBES_PER_BASKET
            implied_bid_volume = min(jams_bid_volume, croissants_bid_volume, djembes_bid_volume)
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume
        if implied_ask < float('inf'):
            jams_ask_volume = -order_depths[Product.JAMS].sell_orders[jams_best_ask] // JAMS_PER_BASKET
            croissants_ask_volume = -order_depths[Product.CROISSANTS].sell_orders[croissants_best_ask] // CROISSANTS_PER_BASKET
            djembes_ask_volume = -order_depths[Product.DJEMBES].sell_orders[djembes_best_ask] // DJEMBES_PER_BASKET
            implied_ask_volume = min(jams_ask_volume, croissants_ask_volume, djembes_ask_volume)
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume
        return synthetic_order_price

    def convert_synthetic_basket_orders(self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        component_orders = {
            Product.JAMS: [],
            Product.CROISSANTS: [],
            Product.DJEMBES: [],
        }
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        best_bid = max(synthetic_basket_order_depth.buy_orders.keys()) if synthetic_basket_order_depth.buy_orders else 0
        best_ask = min(synthetic_basket_order_depth.sell_orders.keys()) if synthetic_basket_order_depth.sell_orders else float("inf")
        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity
            if quantity > 0 and price >= best_ask:
                jams_price = min(order_depths[Product.JAMS].sell_orders.keys())
                croissants_price = min(order_depths[Product.CROISSANTS].sell_orders.keys())
                djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                jams_price = max(order_depths[Product.JAMS].buy_orders.keys())
                croissants_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                djembes_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                continue
            jams_order = Order(Product.JAMS, jams_price, quantity * BASKET_WEIGHTS[Product.JAMS])
            croissants_order = Order(Product.CROISSANTS, croissants_price, quantity * BASKET_WEIGHTS[Product.CROISSANTS])
            djembes_order = Order(Product.DJEMBES, djembes_price, quantity * BASKET_WEIGHTS[Product.DJEMBES])
            component_orders[Product.JAMS].append(jams_order)
            component_orders[Product.CROISSANTS].append(croissants_order)
            component_orders[Product.DJEMBES].append(djembes_order)
        return component_orders

    # --- New spread functions (changed as requested) ---
    def execute_spread_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth]
    ) -> Any:
        if target_position == basket_position:
            return None
        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        if target_position > basket_position:
            # Need to buy baskets (and sell synthetic)
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])
            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders[synthetic_bid_price])
            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            basket_orders = [Order(Product.PICNIC_BASKET1, basket_ask_price, execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)]
            aggregate_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders
        else:
            # Need to sell baskets (and buy synthetic)
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask_price])
            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            basket_orders = [Order(Product.PICNIC_BASKET1, basket_bid_price, -execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)]
            aggregate_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders

    def spread_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread_data: Dict[str, Any]
    ):
        # Use PICNIC_BASKET1 as our basket product.
        if Product.PICNIC_BASKET1 not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)
        # Keep history length equal to the window.
        if len(spread_data["spread_history"]) < self.params[Product.SPREAD]["spread_std_window"]:
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])
        zscore = (spread - self.params[Product.SPREAD]["default_spread_mean"]) / spread_std

        if zscore >= self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(-self.params[Product.SPREAD]["target_position"],
                                                  basket_position, order_depths)
        if zscore <= -self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(self.params[Product.SPREAD]["target_position"],
                                                  basket_position, order_depths)
        spread_data["prev_zscore"] = zscore
        return None


    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        market_data = MarketData()
        # Initialize price history if it doesn't exist
        if 'starfruit_price_history' not in traderObject:
            traderObject['starfruit_price_history'] = []
        if 'ink_price_history' not in traderObject:
            traderObject['ink_price_history'] = []
        # for product in [Product.CHOCOLATE, Product.STRAWBERRIES, Product.ROSES, Product.GIFT_BASKET, Product.GIFT_BASKET_1]:
        #     if f'{product}_price_history' not in traderObject:
        #         traderObject[f'{product}_price_history'] = []
        
        conversions = 0
        result = {}

        #self.history = traderObject
        for product in ["PICNIC_BASKET1", "PICNIC_BASKET2", "CROISSANTS", "JAMS", "DJEMBES"]:
            order_depth = state.order_depths[product]
            orders = []
            
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
            else:
                mid_price = self._get_last_price(product)

            position = state.position.get(product, 0)
            self._update_history(product, mid_price)

            if product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
                orders = self._basket_arbitrage(product, position)
            if product in ["CROISSANTS", "JAMS", "DJEMBES"]: # give 350k on backtest but bad on website
                orders = self._dynamic_market_make(product, mid_price, position)
            result[product] = orders

        for product in ['RAINFOREST_RESIN']:
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position[product] if state.position.get(product) else 0
            orders: List[Order] = []

            acceptable_price = 10000

            #print("Acceptable price : " + str(acceptable_price))
            #print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

            buy_orders = sorted(order_depth.buy_orders.items(), reverse = True)
            sell_orders = sorted(order_depth.sell_orders.items())

            buy = abs(50 - position)
            sell = abs(50 + position)

            spread_buy_pos = 0
            spread_sell_pos = 0
                # +EV ORDER MATCH:
            for price, qty in sell_orders:
                if price < acceptable_price or (price == acceptable_price and (position <= 0)): # or is better for resin, and is better for kelp
                    buy_qty = min(buy, -qty)
                    self.submit_order(product, orders, price, buy_qty)
                    buy -= buy_qty
                    spread_sell_pos += 1
                    if buy == 0:
                        break

            for price, qty in buy_orders:
                if price > acceptable_price or (price == acceptable_price and (position >= 0)): #  or is better for resin, and is better for kelp
                    sell_qty = min(sell, qty)
                    self.submit_order(product, orders, price, -sell_qty)
                    sell -= sell_qty
                    spread_buy_pos += 1
                    if sell == 0:
                        break

            # MARKET MAKING:
            if len(buy_orders) > spread_buy_pos:
                best_buy = buy_orders[spread_buy_pos][0]
            else:
                best_buy = acceptable_price - 5

            if len(sell_orders) > spread_sell_pos:
                best_sell = sell_orders[spread_sell_pos][0]
            else:
                best_sell = acceptable_price + 5

            if buy != 0 and best_buy <= acceptable_price:
                if abs(best_buy - acceptable_price) <= 1:
                    if position < 0:
                        self.submit_order(product, orders, acceptable_price, -position)
                        buy += position
                    self.submit_order(product, orders, best_buy, buy)
                else:
                    self.submit_order(product, orders, best_buy + 1, buy)

            if sell != 0 and best_sell >= acceptable_price:
                if abs(best_sell - acceptable_price) <= 1:
                    if position > 0:
                        self.submit_order(product, orders, acceptable_price, -position)
                        sell -= position
                    self.submit_order(product, orders, best_sell, -sell)
                else:
                    self.submit_order(product, orders, best_sell - 1, -sell)

            result[product] = orders

        if Product.SPREAD not in traderObject:
            traderObject[Product.SPREAD] = {"spread_history": {}, "prev_zscore": 0}
            # We'll store a spread history for the basket product here.
            traderObject[Product.SPREAD]["spread_history"] = []
        basket_position = state.position[Product.PICNIC_BASKET1] if Product.PICNIC_BASKET1 in state.position else 0
        spread_orders = self.spread_orders(state.order_depths, Product.PICNIC_BASKET1, basket_position, traderObject[Product.SPREAD])
        if spread_orders is not None:
            # In the conversion, the synthetic basket orders are converted into component orders.
            result[Product.JAMS] = spread_orders[Product.JAMS]
            result[Product.CROISSANTS] = spread_orders[Product.CROISSANTS]
            result[Product.DJEMBES] = spread_orders[Product.DJEMBES]
            result[Product.PICNIC_BASKET1] = spread_orders[Product.PICNIC_BASKET1]

        # if Product.AMETHYSTS in self.params and Product.AMETHYSTS in state.order_depths:
        #     amethyst_position = (
        #         state.position[Product.AMETHYSTS]
        #         if Product.AMETHYSTS in state.position
        #         else 0
        #     )
        #     amethyst_take_orders, buy_order_volume, sell_order_volume = (
        #         self.take_orders(
        #             Product.AMETHYSTS,
        #             state.order_depths[Product.AMETHYSTS],
        #             self.params[Product.AMETHYSTS]["fair_value"],
        #             self.params[Product.AMETHYSTS]["take_width"],
        #             amethyst_position,
        #         )
        #     )
        #     amethyst_clear_orders, buy_order_volume, sell_order_volume = (
        #         self.clear_orders(
        #             Product.AMETHYSTS,
        #             state.order_depths[Product.AMETHYSTS],
        #             self.params[Product.AMETHYSTS]["fair_value"],
        #             self.params[Product.AMETHYSTS]["clear_width"],
        #             amethyst_position,
        #             buy_order_volume,
        #             sell_order_volume,
        #         )
        #     )
        #     amethyst_make_orders, _, _ = self.make_orders(
        #         Product.AMETHYSTS,
        #         state.order_depths[Product.AMETHYSTS],
        #         self.params[Product.AMETHYSTS]["fair_value"],
        #         amethyst_position,
        #         buy_order_volume,
        #         sell_order_volume,
        #         self.params[Product.AMETHYSTS]["disregard_edge"],
        #         self.params[Product.AMETHYSTS]["join_edge"],
        #         self.params[Product.AMETHYSTS]["default_edge"],
        #         True,
        #         self.params[Product.AMETHYSTS]["soft_position_limit"],
        #     )
        #     result[Product.AMETHYSTS] = (
        #         amethyst_take_orders + amethyst_clear_orders + amethyst_make_orders
        #     )
        if Product.STARFRUIT in self.params and Product.STARFRUIT in state.order_depths:
            starfruit_position = (
                state.position[Product.STARFRUIT]
                if Product.STARFRUIT in state.position
                else 0
            )
            starfruit_fair_value = self.starfruit_fair_value(
                state.order_depths[Product.STARFRUIT], traderObject
            )
            
            
            starfruit_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.STARFRUIT,
                    state.order_depths[Product.STARFRUIT],
                    starfruit_fair_value,
                    self.params[Product.STARFRUIT]["take_width"],
                    starfruit_position,
                    self.params[Product.STARFRUIT]["prevent_adverse"],
                    self.params[Product.STARFRUIT]["adverse_volume"],
                )
            )
            starfruit_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.STARFRUIT,
                    state.order_depths[Product.STARFRUIT],
                    starfruit_fair_value,
                    self.params[Product.STARFRUIT]["clear_width"],
                    starfruit_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            starfruit_make_orders, _, _ = self.make_order_starfruit(
                Product.STARFRUIT,
                traderObject,
                state.order_depths[Product.STARFRUIT],
                starfruit_fair_value,
                starfruit_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.STARFRUIT]["disregard_edge"],
                self.params[Product.STARFRUIT]["join_edge"],
                self.params[Product.STARFRUIT]["default_edge"],
            )
            result[Product.STARFRUIT] = (
                starfruit_take_orders + starfruit_clear_orders + starfruit_make_orders
            )

        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            ink_position = (
                state.position[Product.SQUID_INK]
                if Product.SQUID_INK in state.position
                else 0
            )
            # tinh fair value trc
            ink_fair_value = self.ink_fair_value(
                state.order_depths[Product.SQUID_INK], traderObject
            )
            
            
            ink_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    ink_fair_value,
                    self.params[Product.SQUID_INK]["take_width"],
                    ink_position,
                    self.params[Product.SQUID_INK]["prevent_adverse"],
                    self.params[Product.SQUID_INK]["adverse_volume"],
                )
            )
            ink_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    ink_fair_value,
                    self.params[Product.SQUID_INK]["clear_width"],
                    ink_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            ink_make_orders, _, _ = self.make_order_ink(
                Product.SQUID_INK,
                traderObject,
                state.order_depths[Product.SQUID_INK],
                ink_fair_value,
                ink_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.SQUID_INK]["disregard_edge"],
                self.params[Product.SQUID_INK]["join_edge"],
                self.params[Product.SQUID_INK]["default_edge"],
                self.params[Product.SQUID_INK]["manage_position"],
                self.params[Product.SQUID_INK]["soft_position_limit"],
            )
            result[Product.SQUID_INK] = (
                ink_take_orders + ink_clear_orders + ink_make_orders
            )
        
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
    
    def call_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate the Black-Scholes delta of a European call option.

        Parameters:
        - S: Current stock price
        - K: Strike price
        - T: Time to maturity (in days)
        - r: Risk-free interest rate
        - sigma: Volatility (annual)

        Returns:
        - delta: Call option delta
        """
        r = 0
        T = T / 365
        if T == 0 or sigma == 0:
            return 1.0 if S > K else 0.0

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return 0.5 * (1 + math.erf(d1 / math.sqrt(2)))

    
    def implied_vol_call(self, market_price, S, K, T_days, r, tol=0.00000000000001, max_iter=250):
        """
        Calculate implied volatility from market call option price using bisection.

        Parameters:
        - market_price: observed market price of the option
        - S: spot price
        - K: strike price
        - T_days: time to maturity in days
        - r: risk-free interest rate
        - tol: convergence tolerance
        - max_iter: maximum number of iterations

        Returns:
        - Implied volatility (sigma)
        """
        # Set reasonable initial bounds
        sigma_low = 0.01
        sigma_high = 0.35

        for _ in range(max_iter):
            sigma_mid = (sigma_low + sigma_high) / 2
            price = self.black_scholes_call(S, K, T_days, r, sigma_mid)

            if abs(price - market_price) < tol:
                return sigma_mid

            if price > market_price:
                sigma_high = sigma_mid
            else:
                sigma_low = sigma_mid

        return (sigma_low + sigma_high) / 2  # Final estimate
    
    def norm_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def _load_history(self, trader_data):
        try:
            self.history = json.loads(trader_data) if trader_data else {}
        except:
            self.history = {}

    def _update_history(self, product, price):
        if product not in self.history:
            self.history[product] = []
        self.history[product].append(price)
        if len(self.history[product]) > 100:
            self.history[product].pop(0)

    def _get_last_price(self, product, rolling_window=10):
        # Use the average of the last `rolling_window` recorded prices.
        # If there's not enough data, just average all we have.
        # If no data exists at all, default to 0 (or do no orders).
        data = self.history.get(product, [])
        if not data:
            return 0
        window_data = data[-rolling_window:]
        return float(np.mean(window_data))

    def _basket_arbitrage(self, basket, position):
        if basket == "PICNIC_BASKET1":
            constituents = [("CROISSANTS", 6), ("JAMS", 3), ("DJEMBES", 1)]
        else:
            constituents = [("CROISSANTS", 4), ("JAMS", 2)]

        basket_price = self._get_last_price(basket)
        theoretical_price = sum(self._get_last_price(prod) * qty for prod, qty in constituents)
        spread = theoretical_price - basket_price

        orders = []
        if spread > 50 and position < POSITION_LIMITS[basket]:
            volume = min(POSITION_LIMITS[basket] - position, 10)
            orders.append(Order(basket, int(basket_price), volume))
        elif spread < -50 and position > -POSITION_LIMITS[basket]:
            volume = min(position + POSITION_LIMITS[basket], 10)
            orders.append(Order(basket, int(basket_price), -volume))

        return orders

    # Dynamic Market Making Strategy for CROISSANTS, JAMS, and DJEMBES
    def _dynamic_market_make(self, product, mid_price, position):
        recent_prices = self.history[product][-10:]
        volatility = np.std(recent_prices) if recent_prices else 1
        spread = min(volatility * 1.5, 10)
        bid_price = int(mid_price - spread / 2)
        ask_price = int(mid_price + spread / 2)

        max_buy = POSITION_LIMITS[product] - position
        max_sell = POSITION_LIMITS[product] + position

        orders = []
        if max_buy > 0:
            orders.append(Order(product, bid_price, max_buy))
        if max_sell > 0:
            orders.append(Order(product, ask_price, -max_sell))

        return orders
    
    def black_scholes_call(self, S: float, K: float, T_days: float, r: float, sigma: float) -> float:
        """Black-Scholes price of a European call option."""
        T = T_days / 365.0
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)

    def spread_arbitrage(self, state, traderObject, product_1, product_2, product_3):
        strike_1 = int(product_1.split("_")[-1])
        strike_2 = int(product_2.split("_")[-1])
        strike_3 = int(product_3.split("_")[-1])
        price_dict = {}
        for product in [product_1, product_2, product_3]:
            # if product not in traderObject:
            #     traderObject[product] = {
            #         "prev_coupon_price": 0,
            #         "past_coupon_vol": [],
            #     }

            if (
                product in self.params
                and product in state.order_depths
            ):
                coconut_coupon_position = (
                    state.position[product]
                    if product in state.position
                    else 0
                )
            coconut_coupon_order_depth = state.order_depths[product]
            best_bid = max(coconut_coupon_order_depth.buy_orders.keys()) if len(coconut_coupon_order_depth.buy_orders) > 0 else 0
            best_ask = min(coconut_coupon_order_depth.sell_orders.keys()) if len(coconut_coupon_order_depth.sell_orders) > 0 else 0
            best_bid_volume = abs(coconut_coupon_order_depth.buy_orders[best_bid]) if len(coconut_coupon_order_depth.buy_orders) > 0 else 0
            best_ask_volume = abs(coconut_coupon_order_depth.sell_orders[best_ask]) if len(coconut_coupon_order_depth.sell_orders) > 0 else 0
            
            # Calculate available volume based on position limits
            max_buy = POSITION_LIMITS[product] - coconut_coupon_position
            max_sell = POSITION_LIMITS[product] + coconut_coupon_position
            
            # Adjust volumes based on position limits
            best_bid_volume = min(best_bid_volume, max_buy)
            best_ask_volume = min(best_ask_volume, max_sell)
            
            price_dict[product] = (best_bid, best_ask, best_bid_volume, best_ask_volume, coconut_coupon_position)
            
        available_volume = min(price_dict[product_1][2], price_dict[product_2][3], price_dict[product_3][2])
        order_10000 = []
        order_10250 = []
        order_10500 = []
        price_3 = price_dict[product_3][0]
        price_2 = price_dict[product_2][1]
        price_1 = price_dict[product_1][0]
        arb_ratio_1 = (strike_2 - strike_1) / (strike_3 - strike_1)
        arb_ratio_2 = (strike_3 - strike_2) / (strike_3 - strike_1)
        expected_profit = 0
        
        if available_volume > 0:
            if price_3 - price_2 > price_2 - price_1 + 30:
                volume_1 = round(available_volume * arb_ratio_1)
                volume_2 = available_volume
                volume_3 = round(available_volume * arb_ratio_2)
                
                # Double check that volumes don't exceed position limits
                volume_1 = min(volume_1, POSITION_LIMITS[product_1] - price_dict[product_1][4])
                volume_2 = min(volume_2, POSITION_LIMITS[product_2] - price_dict[product_2][4])
                volume_3 = min(volume_3, POSITION_LIMITS[product_3] - price_dict[product_3][4])
                
                order_10000 = [Order(product_1, price_1, -volume_1)]
                order_10250 = [Order(product_2, price_2, volume_2)]
                order_10500 = [Order(product_3, price_3, -volume_3)]
                expected_profit = (price_1 * arb_ratio_1 + price_3 * arb_ratio_2 - price_2) * available_volume
                
                # Store arbitrage information when opening position
                if "arbitrage_positions" not in traderObject:
                    traderObject["arbitrage_positions"] = {}
                
                traderObject["arbitrage_positions"][f"{product_1}_{product_2}_{product_3}"] = {
                    "products": [product_1, product_2, product_3],
                    "volumes": [volume_1, volume_2, volume_3]
                }
        
        return order_10000, order_10250, order_10500, expected_profit

    def spread_arbitrage_close(self, state, traderObject, product_1, product_2, product_3):
        # Retrieve stored arbitrage information
        arb_key = f"{product_1}_{product_2}_{product_3}"
        if "arbitrage_positions" not in traderObject or arb_key not in traderObject["arbitrage_positions"]:
            return [], [], []
            
        arb_info = traderObject["arbitrage_positions"][arb_key]
        stored_products = arb_info["products"]
        stored_volumes = arb_info["volumes"]
        
        # Create orders to close the stored volumes
        order_10000 = []
        order_10250 = []
        order_10500 = []
        
        # Get current prices
        price_dict = {}
        for product in [product_1, product_2, product_3]:
            coconut_coupon_order_depth = state.order_depths[product]
            best_bid = max(coconut_coupon_order_depth.buy_orders.keys()) if len(coconut_coupon_order_depth.buy_orders) > 0 else 0
            best_ask = min(coconut_coupon_order_depth.sell_orders.keys()) if len(coconut_coupon_order_depth.sell_orders) > 0 else 0
            price_dict[product] = (best_bid, best_ask)

        price_3 = price_dict[product_3][1]
        price_2 = price_dict[product_2][0]
        price_1 = price_dict[product_1][1]
        
        # Create closing orders with stored volumes
        if price_3 - price_2 <= price_2 - price_1 + 5:
            order_10000 = [Order(stored_products[0], price_1, stored_volumes[0])]
            order_10250 = [Order(stored_products[1], price_2, -stored_volumes[1])]
            order_10500 = [Order(stored_products[2], price_3, stored_volumes[2])]
        
            # Remove the closed arbitrage position
            del traderObject["arbitrage_positions"][arb_key]
        
        return order_10000, order_10250, order_10500

    def _volcanic_rock_voucher_orders(self, spot, mid_price, voucher_product, position, best_bid, best_ask,volatility=0.2178):
        strike = int(voucher_product.split("_")[-1])
        # if len(self.history["VOLCANIC_ROCK"]) > 50:
        #     recent_prices = self.history["VOLCANIC_ROCK"][-50:] 
        #     volatility = ((np.std(recent_prices)/recent_prices[-1])*np.sqrt(252))
        # else:
        #     volatility = 0.2178
        volatility = volatility
        #theoretical_price = self._black_scholes_call(spot, strike, self.params[voucher_product]["time_to_expiry"], volatility)
        theoretical_price = spot-strike
        spread = theoretical_price - mid_price
        orders = []
        spread_threshold = 0
        spread_threshold_exit = 0
        in_the_money = (spot - strike) / spot
        if in_the_money > 0.04:
            spread_threshold = 1
            spread_threshold_exit = 0
        elif in_the_money <= 0.04 and in_the_money >= 0:
            spread_threshold = min(1 / ((in_the_money/0.04)**2), 15)
            spread_threshold_exit = 0
        # elif in_the_money > -0.025 and in_the_money < 0:
        #     spread_threshold = -30
        # elif in_the_money <= -0.025 and in_the_money >= -0.05:
        #     spread_threshold = -50
        # elif in_the_money <= -0.05:
        #     spread_threshold = -70
        if in_the_money > 0.04:
            if spread >= spread_threshold:
                volume = min(POSITION_LIMITS[voucher_product] - position, 10)
                orders.append(Order(voucher_product, best_ask, volume))
            elif position > 0 and spread <= spread_threshold_exit:
                volume = min(position, 10)
                orders.append(Order(voucher_product, best_bid, -volume))
            elif spread <= -spread_threshold:
                volume = min(POSITION_LIMITS[voucher_product] + position, 10)
                orders.append(Order(voucher_product, best_bid, -volume))
            elif position < 0 and spread >= -spread_threshold_exit:
                volume = min(position, 10)
                orders.append(Order(voucher_product, best_ask, volume))
        else:
            if spread >= spread_threshold:
                volume = min(POSITION_LIMITS[voucher_product] - position, 10)
                orders.append(Order(voucher_product, best_ask, volume))
            elif position > 0 and spread <= spread_threshold_exit:
                volume = min(position, 10)
                orders.append(Order(voucher_product, best_bid, -volume))
            elif spread <= -spread_threshold:
                volume = min(POSITION_LIMITS[voucher_product] + position, 10)
                orders.append(Order(voucher_product, best_bid, -volume))
            elif position < 0 and spread >= -spread_threshold_exit:
                volume = min(position, 10)
                orders.append(Order(voucher_product, best_ask, volume))
        return orders

POSITION_LIMITS = {
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
    "SQUID_INK": 50,
    "CROISSANTS": 250,
    "JAMS": 350,
    "DJEMBES": 60,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 100,
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200,
    "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200,
    "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
}
