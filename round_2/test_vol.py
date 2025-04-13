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

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Tuple, Union, Dict


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

class Product:
    AMETHYSTS = "RAINFOREST_RESIN"
    STARFRUIT = "KELP"
    SQUID_INK = "SQUID_INK"
    GIFT_BASKET = "PICNIC_BASKET1"
    GIFT_BASKET_1 = "PICNIC_BASKET2"
    CHOCOLATE = "CROISSANTS"
    STRAWBERRIES = "JAMS"
    ROSES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"
    SPREAD_1 = "SPREAD_1"

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

    Product.SPREAD: {
        "default_spread_mean": 48,
        "default_spread_std": 70,
        "spread_std_window": 100,
        "zscore_threshold_high": 2.4, # 2.5 is better on backtester, 2 is better on website
        "zscore_threshold_low": -2.4,
        "zscore_threshold_exit_high": -2.5,
        "zscore_threshold_exit_low": 2.5,
        "target_position": 58,
    },
    Product.SPREAD_1: {
        "default_spread_mean_high": 60, #60, -40
        "default_spread_mean_low": -40,
        "default_spread_std": 50,
        "spread_std_window": 200,
        "spread_high": 130, #max 150
        "spread_low": -90, #min -150
        "zscore_threshold_high": 15,
        "zscore_threshold_low": -15,
        "target_position": 98,
    },
    Product.STRAWBERRIES: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 0,
        "adverse_volume_low": 50,
        'adverse_volume_high': 100,
        "reversion_beta": -0.055,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        'ret_vol': 0.0001,
        "drift": 0,
        "manage_position": False,
        "soft_position_limit": 30,
        "spread_adjustment": 1
    },
    Product.ROSES: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 0,
        "adverse_volume_low": 50,
        'adverse_volume_high': 100,
        "reversion_beta": 0,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        'ret_vol': 0.001,
        "drift": 0,
        "manage_position": False,
        "soft_position_limit": 30,
        "spread_adjustment": 1
    },
    Product.CHOCOLATE: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 0,
        "adverse_volume_low": 50,
        'adverse_volume_high': 100,
        "reversion_beta": -0.12,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        'ret_vol': 0.0001,
        "drift": 0,
        "manage_position": False,
        "soft_position_limit": 30,
        "spread_adjustment": 1
    },
}

BASKET_WEIGHTS = {
    Product.CHOCOLATE: 6,
    Product.STRAWBERRIES: 3,
    Product.ROSES: 1,
}

BASKET_WEIGHTS_1 = {
    Product.CHOCOLATE: 4,
    Product.STRAWBERRIES: 2,
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
            Product.GIFT_BASKET: 60,
            Product.GIFT_BASKET_1: 100,
            Product.CHOCOLATE: 250,
            Product.STRAWBERRIES: 350,
            Product.ROSES: 60,
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
            traderObject['starfruit_price_history'] = traderObject['starfruit_price_history'][-100:]
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
            traderObject['ink_price_history'] = traderObject['ink_price_history'][-100:]
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
            traderObject[f'{product}_price_history'] = traderObject[f'{product}_price_history'][-100:]
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
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def get_synthetic_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth], spread_product: Product
    ) -> OrderDepth:
        # Constants
        if spread_product == Product.SPREAD:
            CHOCOLATE_PER_BASKET = BASKET_WEIGHTS[Product.CHOCOLATE]
            STRAWBERRIES_PER_BASKET = BASKET_WEIGHTS[Product.STRAWBERRIES]
            ROSES_PER_BASKET = BASKET_WEIGHTS[Product.ROSES]
        else:
            CHOCOLATE_PER_BASKET = BASKET_WEIGHTS_1[Product.CHOCOLATE]
            STRAWBERRIES_PER_BASKET = BASKET_WEIGHTS_1[Product.STRAWBERRIES]
            ROSES_PER_BASKET = 0

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        chocolate_best_bid = (
            max(order_depths[Product.CHOCOLATE].buy_orders.keys())
            if order_depths[Product.CHOCOLATE].buy_orders
            else 0
        )
        chocolate_best_ask = (
            min(order_depths[Product.CHOCOLATE].sell_orders.keys())
            if order_depths[Product.CHOCOLATE].sell_orders
            else float("inf")
        )
        strawberries_best_bid = (
            max(order_depths[Product.STRAWBERRIES].buy_orders.keys())
            if order_depths[Product.STRAWBERRIES].buy_orders
            else 0
        )
        strawberries_best_ask = (
            min(order_depths[Product.STRAWBERRIES].sell_orders.keys())
            if order_depths[Product.STRAWBERRIES].sell_orders
            else float("inf")
        )
        roses_best_bid = (
            max(order_depths[Product.ROSES].buy_orders.keys())
            if order_depths[Product.ROSES].buy_orders
            else 0
        )
        roses_best_ask = (
            min(order_depths[Product.ROSES].sell_orders.keys())
            if order_depths[Product.ROSES].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            chocolate_best_bid * CHOCOLATE_PER_BASKET
            + strawberries_best_bid * STRAWBERRIES_PER_BASKET
            + roses_best_bid * ROSES_PER_BASKET
        )
        implied_ask = (
            chocolate_best_ask * CHOCOLATE_PER_BASKET
            + strawberries_best_ask * STRAWBERRIES_PER_BASKET
            + roses_best_ask * ROSES_PER_BASKET
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            chocolate_bid_volume = (
                order_depths[Product.CHOCOLATE].buy_orders[chocolate_best_bid]
                // CHOCOLATE_PER_BASKET
            )
            strawberries_bid_volume = (
                order_depths[Product.STRAWBERRIES].buy_orders[strawberries_best_bid]
                // STRAWBERRIES_PER_BASKET
            )
            roses_bid_volume = (
                order_depths[Product.ROSES].buy_orders[roses_best_bid]
                // ROSES_PER_BASKET if ROSES_PER_BASKET != 0 else 0
            )
            implied_bid_volume = min(
                chocolate_bid_volume, strawberries_bid_volume, roses_bid_volume
            ) if spread_product == Product.SPREAD else min(chocolate_bid_volume, strawberries_bid_volume)
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            chocolate_ask_volume = (
                -order_depths[Product.CHOCOLATE].sell_orders[chocolate_best_ask]
                // CHOCOLATE_PER_BASKET
            )
            strawberries_ask_volume = (
                -order_depths[Product.STRAWBERRIES].sell_orders[strawberries_best_ask]
                // STRAWBERRIES_PER_BASKET
            )
            roses_ask_volume = (
                -order_depths[Product.ROSES].sell_orders[roses_best_ask]
                // ROSES_PER_BASKET if ROSES_PER_BASKET != 0 else 0
            )
            implied_ask_volume = min(
                chocolate_ask_volume, strawberries_ask_volume, roses_ask_volume
            ) if spread_product == Product.SPREAD else min(chocolate_ask_volume, strawberries_ask_volume)
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def convert_synthetic_basket_orders(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth], spread_product: Product
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CHOCOLATE: [],
            Product.STRAWBERRIES: [],
            Product.ROSES: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(
            order_depths, spread_product
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                chocolate_price = min(
                    order_depths[Product.CHOCOLATE].sell_orders.keys()
                )
                strawberries_price = min(
                    order_depths[Product.STRAWBERRIES].sell_orders.keys()
                )
                roses_price = min(order_depths[Product.ROSES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                chocolate_price = max(order_depths[Product.CHOCOLATE].buy_orders.keys())
                strawberries_price = max(
                    order_depths[Product.STRAWBERRIES].buy_orders.keys()
                )
                roses_price = max(order_depths[Product.ROSES].buy_orders.keys())
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            chocolate_order = Order(
                Product.CHOCOLATE,
                chocolate_price,
                quantity * BASKET_WEIGHTS[Product.CHOCOLATE],
            )
            strawberries_order = Order(
                Product.STRAWBERRIES,
                strawberries_price,
                quantity * BASKET_WEIGHTS[Product.STRAWBERRIES],
            )
            roses_order = Order(
                Product.ROSES, roses_price, quantity * BASKET_WEIGHTS[Product.ROSES]
            )

            # Add the component orders to the respective lists
            component_orders[Product.CHOCOLATE].append(chocolate_order)
            component_orders[Product.STRAWBERRIES].append(strawberries_order)
            component_orders[Product.ROSES].append(roses_order)

        return component_orders

    def execute_spread_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        spread_product: Product,
    ):

        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[product]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths, spread_product)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(product, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths, spread_product
            )
            aggregate_orders[product] = basket_orders
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(product, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths, spread_product
            )
            aggregate_orders[product] = basket_orders
            return aggregate_orders

    def spread_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        spread_product: Product,
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        if product not in order_depths.keys():
            return None

        basket_order_depth = order_depths[product]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths, spread_product)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        if (
            len(spread_data["spread_history"])
            < self.params[spread_product]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[spread_product]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])

        zscore = (
            spread - self.params[spread_product]["default_spread_mean"]
        ) / spread_std

        if zscore >= self.params[spread_product]["zscore_threshold_high"]:
            if basket_position != -self.params[spread_product]["target_position"]:
                return self.execute_spread_orders(
                    -self.params[spread_product]["target_position"],
                    basket_position,
                    order_depths,
                    product,
                    spread_product,
                )

        elif zscore <= self.params[spread_product]["zscore_threshold_low"]:
            if basket_position != self.params[spread_product]["target_position"]:
                return self.execute_spread_orders(
                    self.params[spread_product]["target_position"],
                    basket_position,
                    order_depths,
                    product,
                    spread_product,
                )
            
        # if zscore >= self.params[spread_product]["zscore_threshold_high"]:
        #     if basket_position != -self.params[spread_product]["target_position"]:
        #         return self.execute_spread_orders(
        #             -self.params[spread_product]["target_position"],
        #             basket_position,
        #             order_depths,
        #             product,
        #             spread_product,
        #         )
        # elif zscore <= self.params[spread_product]["zscore_threshold_exit_high"]:
        #     if basket_position < 0:
        #         return self.execute_spread_orders(
        #             self.params[spread_product]["target_position"],
        #             basket_position,
        #             order_depths,
        #             product,
        #             spread_product,
        #         )
        # elif zscore <= self.params[spread_product]["zscore_threshold_low"]:
        #     if basket_position != self.params[spread_product]["target_position"]:
        #         return self.execute_spread_orders(
        #             self.params[spread_product]["target_position"],
        #             basket_position,
        #             order_depths,
        #             product,
        #             spread_product,
        #         )
        # elif zscore >= self.params[spread_product]["zscore_threshold_exit_low"]:
        #     if basket_position > 0:
        #         return self.execute_spread_orders(
        #             -self.params[spread_product]["target_position"],
        #             basket_position,
        #             order_depths,
        #             product,
        #             spread_product,
        #         )

        spread_data["prev_zscore"] = zscore
        return None

    def spread_orders_1(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        spread_product: Product,
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        if product not in order_depths.keys():
            return None

        basket_order_depth = order_depths[product]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths, spread_product)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        if (
            len(spread_data["spread_history"])
            < self.params[spread_product]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[spread_product]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])

        if spread > self.params[spread_product]["spread_high"]:
            if basket_position != -self.params[spread_product]["target_position"]:
                return self.execute_spread_orders(
                    -self.params[spread_product]["target_position"],
                    basket_position,
                    order_depths,
                    product,
                    spread_product,
                )
        elif spread < self.params[spread_product]["default_spread_mean_high"]:
            if basket_position < 0:
                return self.execute_spread_orders(
                    self.params[spread_product]["target_position"],
                    basket_position,
                    order_depths,
                    product,
                    spread_product,
                )
        elif spread < self.params[spread_product]["default_spread_mean_low"]:
            if basket_position != self.params[spread_product]["target_position"]:
                return self.execute_spread_orders(
                    self.params[spread_product]["target_position"],
                    basket_position,
                    order_depths,
                    product,
                    spread_product,
                )
        elif spread > self.params[spread_product]["default_spread_mean_low"]:
            if basket_position > 0:
                return self.execute_spread_orders(
                    -self.params[spread_product]["target_position"],
                    basket_position,
                    order_depths,
                    product,
                    spread_product,
                )

        return None

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
            
        # Initialize price history if it doesn't exist
        if 'starfruit_price_history' not in traderObject:
            traderObject['starfruit_price_history'] = []
        if 'ink_price_history' not in traderObject:
            traderObject['ink_price_history'] = []
        for product in [Product.CHOCOLATE, Product.STRAWBERRIES, Product.ROSES, Product.GIFT_BASKET, Product.GIFT_BASKET_1]:
            if f'{product}_price_history' not in traderObject:
                traderObject[f'{product}_price_history'] = []
            
        result = {}

        # self.history = traderObject
        # for product in ["PICNIC_BASKET1", "PICNIC_BASKET2", "CROISSANTS", "JAMS", "DJEMBES"]:
        #     order_depth = state.order_depths[product]
        #     orders = []
            
        #     if order_depth.buy_orders and order_depth.sell_orders:
        #         best_bid = max(order_depth.buy_orders.keys())
        #         best_ask = min(order_depth.sell_orders.keys())
        #         mid_price = (best_bid + best_ask) / 2
        #     else:
        #         mid_price = self._get_last_price(product)

        #     position = state.position.get(product, 0)
        #     self._update_history(product, mid_price)

        #     if product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
        #         orders = self._basket_arbitrage(product, position)
        #     elif product in ["CROISSANTS", "JAMS", "DJEMBES"]:
        #         orders = self._dynamic_market_make(product, mid_price, position)
        #     result[product] = orders

        if Product.AMETHYSTS in self.params and Product.AMETHYSTS in state.order_depths:
            amethyst_position = (
                state.position[Product.AMETHYSTS]
                if Product.AMETHYSTS in state.position
                else 0
            )
            amethyst_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.AMETHYSTS,
                    state.order_depths[Product.AMETHYSTS],
                    self.params[Product.AMETHYSTS]["fair_value"],
                    self.params[Product.AMETHYSTS]["take_width"],
                    amethyst_position,
                )
            )
            amethyst_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.AMETHYSTS,
                    state.order_depths[Product.AMETHYSTS],
                    self.params[Product.AMETHYSTS]["fair_value"],
                    self.params[Product.AMETHYSTS]["clear_width"],
                    amethyst_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            amethyst_make_orders, _, _ = self.make_orders(
                Product.AMETHYSTS,
                state.order_depths[Product.AMETHYSTS],
                self.params[Product.AMETHYSTS]["fair_value"],
                amethyst_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.AMETHYSTS]["disregard_edge"],
                self.params[Product.AMETHYSTS]["join_edge"],
                self.params[Product.AMETHYSTS]["default_edge"],
                True,
                self.params[Product.AMETHYSTS]["soft_position_limit"],
            )
            result[Product.AMETHYSTS] = (
                amethyst_take_orders + amethyst_clear_orders + amethyst_make_orders
            )
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
                
        if Product.SPREAD not in traderObject:
            traderObject[Product.SPREAD] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket_position = (
            state.position[Product.GIFT_BASKET]
            if Product.GIFT_BASKET in state.position
            else 0
        )
        spread_orders = self.spread_orders(
            state.order_depths,
            Product.GIFT_BASKET,
            Product.SPREAD,
            basket_position,
            traderObject[Product.SPREAD],
        )
        if spread_orders != None:
            result[Product.CHOCOLATE] = spread_orders[Product.CHOCOLATE]
            result[Product.STRAWBERRIES] = spread_orders[Product.STRAWBERRIES]
            result[Product.ROSES] = spread_orders[Product.ROSES]
            result[Product.GIFT_BASKET] = spread_orders[Product.GIFT_BASKET]

        if Product.SPREAD_1 not in traderObject:
            traderObject[Product.SPREAD_1] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket_position = (
            state.position[Product.GIFT_BASKET_1]
            if Product.GIFT_BASKET_1 in state.position
            else 0
        )
        spread_orders = self.spread_orders_1(
            state.order_depths,
            Product.GIFT_BASKET_1,
            Product.SPREAD_1,
            basket_position,
            traderObject[Product.SPREAD_1],
        )
        if spread_orders != None:
            result[Product.CHOCOLATE] = spread_orders[Product.CHOCOLATE]
            result[Product.STRAWBERRIES] = spread_orders[Product.STRAWBERRIES]
            result[Product.ROSES] = spread_orders[Product.ROSES]
            result[Product.GIFT_BASKET_1] = spread_orders[Product.GIFT_BASKET_1]

        for product in [Product.CHOCOLATE, Product.STRAWBERRIES]:
            if product in state.order_depths:
                position = (
                    state.position[product]
                    if product in state.position
                    else 0
                )
                # tinh fair value trc
                fair_value = self.product_fair_value(
                    state.order_depths[product], traderObject, product, self.params[product]["adverse_volume"]
                )
                
                
                take_orders, buy_order_volume, sell_order_volume = (
                    self.take_orders(
                        product,
                        state.order_depths[product],
                        fair_value,
                        self.params[product]["take_width"],
                        position,
                        self.params[product]["prevent_adverse"],
                        self.params[product]["adverse_volume"],
                    )
                )
                clear_orders, buy_order_volume, sell_order_volume = (
                    self.clear_orders(
                        product,
                        state.order_depths[product],
                        fair_value,
                        self.params[product]["clear_width"],
                        position,
                        buy_order_volume,
                        sell_order_volume,
                    )
                )
                make_orders, _, _ = self.make_order_new(
                    product,
                    traderObject,
                    state.order_depths[product],
                    fair_value,
                    position,
                    buy_order_volume,
                    sell_order_volume,
                    self.params[product]["disregard_edge"],
                    self.params[product]["join_edge"],
                    self.params[product]["default_edge"],
                    self.params[product]["manage_position"],
                    self.params[product]["soft_position_limit"],
                )
                result[product] = (
                    take_orders + clear_orders + make_orders
                )
        

        conversions = 0
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
    
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

POSITION_LIMITS = {
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
    "SQUID_INK": 50,
    "CROISSANTS": 250,
    "JAMS": 350,
    "DJEMBES": 60,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 100,
}
