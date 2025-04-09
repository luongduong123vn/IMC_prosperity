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
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 19,
        "reversion_beta": 0,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        'ret_vol': 0.001,
        "drift": 0.00001,
        "manage_position": False,
        "soft_position_limit": 30,
        "spread_adjustment": 1
    },
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.AMETHYSTS: 50, Product.STARFRUIT: 50, Product.SQUID_INK: 50}


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
                # Adjust prediction based on volatility-return trend correlation
                trend_adjustment = 0.5119 * (ret_vol / self.params[Product.SQUID_INK]["ret_vol"]) * return_trend
            else:
                # Use default values
                ret_vol = self.params[Product.SQUID_INK]["ret_vol"]
                trend_adjustment = 0

            traderObject['ink_price_history'].append(mmmid_price)
            # Keep only the last 100 prices
            traderObject['ink_price_history'] = traderObject['ink_price_history'][-100:]
            
            if traderObject.get("ink_last_price", None) != None:
                last_price = traderObject["ink_last_price"]
                pred_returns = self.params[Product.SQUID_INK]["drift"] + ret_vol * z + trend_adjustment
                fair = mmmid_price * (1 + pred_returns)
            else:
                fair = mmmid_price
                
            traderObject["ink_last_price"] = mmmid_price
            return fair
        return None

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

        # Calculate dynamic edges based on volatility and return trend
        if len(traderObject.get('ink_price_history', [])) >= 10:
            prices = traderObject['ink_price_history'][-10:]
            returns = np.diff(prices) / prices[:-1]
            realized_vol = float(np.std(returns))
            return_trend = self.calculate_time_series_slope_n(returns)
            
            # Adjust edges based on volatility and return trend correlation
            vol_ratio = realized_vol / self.params[Product.SQUID_INK]["ret_vol"]
            trend_factor = 0.5119 * return_trend  # Using the correlation coefficient
            
            # Base edge calculation
            if vol_ratio >= 3:
                base_edge = -max(round(vol_ratio * default_edge * 0.7), default_edge)
            elif vol_ratio <= 0.4:
                base_edge = min(round(vol_ratio * default_edge * 1.5), default_edge)
            else:
                base_edge = default_edge
            
            # Separate bid and ask edges based on trend direction
            if trend_factor > 0:  # Positive trend
                # More aggressive on ask side (selling), more conservative on bid side
                ask_edge = base_edge * (1 + abs(trend_factor))
                bid_edge = base_edge * (1 - abs(trend_factor))
            else:  # Negative trend
                # More aggressive on bid side (buying), more conservative on ask side
                bid_edge = base_edge * (1 + abs(trend_factor))
                ask_edge = base_edge * (1 - abs(trend_factor))
        else:
            ask_edge = default_edge
            bid_edge = default_edge

        # Calculate ask price
        ask = round(fair_value + ask_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - ask_edge

        # Calculate bid price
        bid = round(fair_value - bid_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + bid_edge

        spread_factor = self.params[Product.SQUID_INK]["spread_adjustment"]
        if manage_position:
            if position > soft_position_limit:
                bid = best_bid_below_fair - spread_factor * bid_edge if best_bid_below_fair != None else fair_value - spread_factor*bid_edge
                if ask >= fair_value and ask <= fair_value + 1:
                    ask = fair_value
                elif ask > fair_value + 1:
                    ask = ask - 1
            elif position < -1 * soft_position_limit:
                ask = best_ask_above_fair + spread_factor * ask_edge if best_ask_above_fair != None else fair_value + spread_factor*ask_edge
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
        )
        return orders