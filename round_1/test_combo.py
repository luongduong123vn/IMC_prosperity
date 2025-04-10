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
from typing import List, Tuple, Union


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


PARAMS = {
    Product.AMETHYSTS: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 0,  # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 50,
        "min_spread": 0
    },
    Product.STARFRUIT: {
        "take_width": 1.5,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 19,
        "reversion_beta": 0,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        'ret_vol': 0.00035,
    },
    Product.SQUID_INK: {
        "take_width": 1.5,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 19,
        "reversion_beta": 0,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        'ret_vol': 0.001,
    },
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.AMETHYSTS: 50, Product.STARFRUIT: 50, Product.SQUID_INK: 50}

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
                fair = mmmid_price
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
            
            # Calculate volatility and return trend if we have enough data
            if len(traderObject.get('ink_price_history', [])) >= 10:
                # Get the last 10 prices
                prices = traderObject['ink_price_history'][-10:]
                # Calculate returns
                returns = np.diff(prices) / prices[:-1]
                # Calculate volatility
                ret_vol = float(np.std(returns))
                # Calculate return trend strength
                return_trend = self.calculate_time_series_slope_n(returns)
                n = len(returns)
                recent_mean = np.mean(returns) if n >= 3 else np.mean(returns)
                next_return = recent_mean + return_trend * (n - (n-1)/2)
                # Adjust prediction based on volatility-return trend correlation
            else:
                # Use default values
                ret_vol = self.params[Product.SQUID_INK]["ret_vol"]
                trend_adjustment = 0
                next_return = 0

            traderObject['ink_price_history'].append(mmmid_price)
            # Keep only the last 100 prices
            traderObject['ink_price_history'] = traderObject['ink_price_history'][-100:]
            
            if traderObject.get("ink_last_price", None) != None:
                last_price = traderObject["ink_last_price"]
                if len(traderObject["ink_price_history"]) >= 10:
                    last_10_price = traderObject["ink_price_history"][-10]
                else:
                    last_10_price = last_price
                last_returns = (mmmid_price - last_price) / last_price
                # fair = mmmid_price * (1 + next_return) if next_return != 0 else mmmid_price
                if ret_vol > 0.0015:
                    trend_adjustment = np.mean(returns[-5:])
                else:
                    trend_adjustment = 0
                #fair = mmmid_price * (1+trend_adjustment)
                fair = mmmid_price
            else:
                fair = mmmid_price
                
            traderObject["ink_last_price"] = mmmid_price
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
        spread = best_ask_above_fair - best_bid_below_fair

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
        if spread > self.params[product]["min_spread"]:
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
            
            if ask_volume <= 35:
                ask_edge = 1
            else:
                ask_edge = 1

            if bid_volume <= 35:
                bid_edge = 1
            else:
                bid_edge = 1
        else:
            # Use default volatility from params
            ask_edge = default_edge
            bid_edge = default_edge
        ask = round(fair_value + edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                #ask = best_ask_above_fair - 1  # penny
                ask = min(best_ask_above_fair - ask_edge,round(fair_value + edge))

        bid = round(fair_value - edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                #bid = best_bid_below_fair + 1
                bid = max(best_bid_below_fair + bid_edge,round(fair_value - edge))

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

    def make_order_ink(
        self,
        product,
        traderObject,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ):
        orders: List[Order] = []
        
        # Get current market state
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else fair_value + default_edge
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else fair_value - default_edge
        ask_volume = sum(order_depth.sell_orders.values())
        bid_volume = sum(order_depth.buy_orders.values())
        total_volume = abs(ask_volume) + abs(bid_volume)
        current_spread = best_ask - best_bid
        
        # Initialize parameters
        position_limit = self.LIMIT[product]
        
        # Parameters for dynamic quote adjustment
        alpha = 0.1  # Volatility sensitivity
        beta = 0.1   # Liquidity sensitivity
        gamma = 0.1  # Inventory sensitivity
        delta = 0.1  # Asymmetry parameter
        max_improvement = 2   # Maximum price improvement ticks
        min_improvement = -3  # Minimum price improvement ticks (negative means wider quotes)
        
        # Default adjustment factors
        volatility_factor = 1.0
        liquidity_factor = 1.0
        inventory_factor = 1.0
        
        # Calculate factors if we have enough data
        if len(traderObject.get('ink_price_history', [])) >= 10:
            prices = traderObject['ink_price_history'][-10:]
            returns = np.diff(prices) / prices[:-1]
            
            # 1. Volatility Factor
            current_volatility = float(np.std(returns))
            baseline_volatility = self.params[Product.SQUID_INK]["ret_vol"]
            volatility_factor = 1 + alpha * (current_volatility / baseline_volatility)
            
            # 2. Liquidity Factor
            if 'normal_volume' in traderObject:
                normal_volume = traderObject['normal_volume']
            else:
                normal_volume = 100  # Default value if not available
                
            liquidity_factor = 1 + beta * (1 - (total_volume / normal_volume))
            liquidity_factor = max(1.0, liquidity_factor)  # Ensure factor is at least 1
            
            # 3. Inventory Factor
            target_position = 10  # Usually market makers aim for neutral position
            inventory_ratio = (position - target_position) / position_limit
            inventory_factor = 1 + gamma * abs(inventory_ratio)**2
            
            # Calculate combined adjustment factor
            combined_factor = volatility_factor * liquidity_factor * inventory_factor
            
            # Calculate dynamic price improvement based on the factors
            # For high volatility/low liquidity, this will be negative (wider quotes)
            # For low volatility/high liquidity, this will be positive (tighter quotes)
            raw_improvement = max_improvement - (combined_factor * (max_improvement - min_improvement))
            
            # Initialize with symmetric adjustments
            bid_improvement = raw_improvement
            ask_improvement = raw_improvement
        
            # Apply inventory-based asymmetric adjustment if position is not zero
            if position != 0:
                # Apply inventory-based asymmetric adjustment
                inventory_adjustment = delta * (position / position_limit)
                
                # When long (positive position), improve ask price to encourage selling
                # When short (negative position), improve bid price to encourage buying
                bid_improvement = raw_improvement * (1 + inventory_adjustment)
                ask_improvement = raw_improvement * (1 - inventory_adjustment)
        else:
            bid_improvement = default_edge
            ask_improvement = default_edge
        
        # Convert continuous improvements to discrete ticks and constrain
        bid_ticks = max(min_improvement, min(max_improvement, round(bid_improvement)))
        ask_ticks = max(min_improvement, min(max_improvement, round(ask_improvement)))
        
        # Calculate final quote prices relative to the best bid/ask
        ask = round(fair_value + default_edge)
        if order_depth.sell_orders:
            if abs(best_ask - fair_value) <= join_edge:
                ask = best_ask  # join
            else:
                # Apply ticks - positive ticks make quotes tighter, negative ticks make quotes wider
                ask = best_ask - ask_ticks

        bid = round(fair_value - default_edge)
        if order_depth.buy_orders:
            if abs(fair_value - best_bid) <= join_edge:
                bid = best_bid
            else:
                # Apply ticks - positive ticks make quotes tighter, negative ticks make quotes wider
                bid = best_bid + bid_ticks
        
        # If the current spread is very wide, consider using fair value as a reference
        max_reasonable_spread = 10  # Maximum spread we consider reasonable
        if current_spread > max_reasonable_spread:
            # When spread is too wide, quote relative to fair value instead
            # Note: for negative improvements (wider quotes), we increase the edge
            bid_adj_factor = 1 + (min(0, bid_improvement) / abs(min_improvement)) if bid_improvement < 0 else (1 - bid_improvement/max_improvement)
            ask_adj_factor = 1 + (min(0, ask_improvement) / abs(min_improvement)) if ask_improvement < 0 else (1 - ask_improvement/max_improvement)
            
            bid = round(fair_value - default_edge * bid_adj_factor)
            ask = round(fair_value + default_edge * ask_adj_factor)
        else:
            # Apply ticks to best bid/ask - can now make quotes wider or tighter
            bid = best_bid + bid_ticks
            ask = best_ask - ask_ticks
                
            # Ensure bid doesn't cross ask
            if bid >= ask:
                midpoint = (best_bid + best_ask) / 2
                bid = math.floor(midpoint)
                ask = math.ceil(midpoint)

        # Execute market making
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
        
        # Optionally: Store current volume for future reference
        if 'normal_volume' not in traderObject:
            traderObject['normal_volume'] = total_volume
        else:
            # Update normal volume with exponential moving average
            traderObject['normal_volume'] = 0.95 * traderObject['normal_volume'] + 0.05 * total_volume
        
        return orders, buy_order_volume, sell_order_volume
    
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
            
        # Initialize price history if it doesn't exist
        if 'starfruit_price_history' not in traderObject:
            traderObject['starfruit_price_history'] = []
        if 'ink_price_history' not in traderObject:
            traderObject['ink_price_history'] = []
            
        result = {}

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
            )
            result[Product.SQUID_INK] = (
                ink_take_orders + ink_clear_orders + ink_make_orders
            )

            # result[Product.STARFRUIT] = (starfruit_clear_orders + starfruit_make_orders
            # )

        conversions = 0
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData