from datamodel import Listing, Observation, OrderDepth, UserId, TradingState, Order, ConversionObservation, Symbol, Trade, ProsperityEncoder
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
import json
from collections import defaultdict


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    # ORCHIDS = "ORCHIDS"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC1 = "SYNTHETIC1"
    SYNTHETIC2 = "SYNTHETIC2"
    SPREAD1 = "SPREAD1"
    SPREAD2 = "SPREAD2"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0.5,
        "volume_limit": 0,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.26917,
        "kelp_min_edge": 2,
    },
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 2,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.1492,
        "squid_ink_min_edge": 2,
    },
    # # Product.ORCHIDS: {
    #     "picnic_basket1_beta": -5.2917,
    #     "returns_threshold": 0.01,
    #     "clear_threshold": 0,
    #     "make_probability": 0.800,
    # },
    Product.SPREAD1: {
        "default_spread1_mean": 48.7624,#43.9226,#48.76243333, #379.50439988484239,
        "default_spread1_std": 83.5354,#85.2931783102648,#85.11945081,#76.07966,
        "spread1_std_window": 10,
        "zscore_threshold": 2,
        "target_position": 60,
    },
    Product.SPREAD2: {
        "default_spread2_mean": 30.23596667, #379.50439988484239,
        "default_spread2_std": 54.0495,#59.84920022,#76.07966,
        "spread2_std_window": 400,
        "zscore_threshold": 2,
        "target_position": 100,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "mean_volatility": 	0.0155,
        "strike": 9500,
        "starting_time_to_expiry": 245 / 250,
        "std_window": 30,
        "zscore_threshold": 5.1
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 	0.0126,
        "strike": 9750,
        "starting_time_to_expiry": 245 / 250,
        "std_window": 30,
        "zscore_threshold": 5.1
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 	0.0102,
        "strike": 10000,
        "starting_time_to_expiry": 245 / 250,
        "std_window": 30,
        "zscore_threshold": 5.1
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "mean_volatility": 	0.009285,
        "strike": 10250,
        "starting_time_to_expiry": 245 / 250,
        "std_window": 30,
        "zscore_threshold": 5.1
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "mean_volatility": 	0.0095675,
        "strike": 10500,
        "starting_time_to_expiry": 245 / 250,
        "std_window": 30,
        "zscore_threshold": 5.1

}
}

BASKET_WEIGHTS1 = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1,
}
BASKET_WEIGHTS2 = {
    Product.CROISSANTS: 4,
    Product.JAMS: 2,
}

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
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

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
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0 
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


class Trader:
    def __init__(self, params=None):
        self.spread1_data = defaultdict(list)
        self.spread2_data = defaultdict(list)
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.KELP: 50,
            Product.RAINFOREST_RESIN: 50,
            Product.SQUID_INK:50,
            # Product.ORCHIDS: 100,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.VOLCANIC_ROCK: 400,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
        }
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

    def take_best_orders_with_adverse(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        adverse_volume: int,
    ) -> (int, int):

        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if abs(best_ask_amount) <= adverse_volume:
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
            if abs(best_bid_amount) <= adverse_volume:
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
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

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

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def make_rainforest_resin_orders(
        self,
        order_depth: OrderDepth,
        fair_value: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        volume_limit: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        baaf = min(
            [
                price
                for price in order_depth.sell_orders.keys()
                if price > fair_value + 1
            ]
        ) if len([
                price
                for price in order_depth.sell_orders.keys()
                if price > fair_value + 1
            ]) > 0 else None
        bbbf = max(
            [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        ) if len([price for price in order_depth.buy_orders.keys() if price < fair_value - 1]) > 0 else None
        if bbbf != None and baaf != None:
            if baaf <= fair_value + 2:
                if position <= volume_limit:
                    baaf = fair_value + 3  # still want edge 2 if position is not a concern

            if bbbf >= fair_value - 2:
                if position >= -volume_limit:
                    bbbf = fair_value - 3  # still want edge 2 if position is not a concern

            buy_order_volume, sell_order_volume = self.market_make(
                Product.RAINFOREST_RESIN,
                orders,
                bbbf + 1,
                baaf - 1,
                position,
                buy_order_volume,
                sell_order_volume,
            )
        return orders, buy_order_volume, sell_order_volume
    
    def squid_ink_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
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
                if traderObject.get("squid_ink_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["squid_ink_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("squid_ink_last_price", None) != None:
                last_price = traderObject["squid_ink_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.SQUID_INK]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["squid_ink_last_price"] = mmmid_price
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

        if prevent_adverse:
            buy_order_volume, sell_order_volume = self.take_best_orders_with_adverse(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
                adverse_volume,
            )
        else:
            buy_order_volume, sell_order_volume = self.take_best_orders(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
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

    def make_kelp_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        min_edge: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        aaf = [
            price
            for price in order_depth.sell_orders.keys()
            if price >= round(fair_value + min_edge)
        ]
        bbf = [
            price
            for price in order_depth.buy_orders.keys()
            if price <= round(fair_value - min_edge)
        ]
        baaf = min(aaf) if len(aaf) > 0 else round(fair_value + min_edge)
        bbbf = max(bbf) if len(bbf) > 0 else round(fair_value - min_edge)
        buy_order_volume, sell_order_volume = self.market_make(
            Product.KELP,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume
    
    def make_squid_ink_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        min_edge: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        aaf = [
            price
            for price in order_depth.sell_orders.keys()
            if price >= round(fair_value + min_edge)
        ]
        bbf = [
            price
            for price in order_depth.buy_orders.keys()
            if price <= round(fair_value - min_edge)
        ]
        baaf = min(aaf) if len(aaf) > 0 else round(fair_value + min_edge)
        bbbf = max(bbf) if len(bbf) > 0 else round(fair_value - min_edge)
        buy_order_volume, sell_order_volume = self.market_make(
            Product.SQUID_INK,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
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

    def get_synthetic1_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        CROISSANTS_PER_BASKET = BASKET_WEIGHTS1[Product.CROISSANTS]
        JAMS_PER_BASKET = BASKET_WEIGHTS1[Product.JAMS]
        DJEMBES_PER_BASKET = BASKET_WEIGHTS1[Product.DJEMBES]

        # Initialize the synthetic1 basket order depth
        synthetic1_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        croissants_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        croissants_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        jams_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        jams_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )
        djembes_best_bid = (
            max(order_depths[Product.DJEMBES].buy_orders.keys())
            if order_depths[Product.DJEMBES].buy_orders
            else 0
        )
        djembes_best_ask = (
            min(order_depths[Product.DJEMBES].sell_orders.keys())
            if order_depths[Product.DJEMBES].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic1 basket
        implied_bid = (
            croissants_best_bid * CROISSANTS_PER_BASKET
            + jams_best_bid * JAMS_PER_BASKET
            + djembes_best_bid * DJEMBES_PER_BASKET
        )
        implied_ask = (
            croissants_best_ask * CROISSANTS_PER_BASKET
            + jams_best_ask * JAMS_PER_BASKET
            + djembes_best_ask * DJEMBES_PER_BASKET
        )

        # Calculate the maximum number of synthetic1 baskets available at the implied bid and ask
        if implied_bid > 0:
            croissants_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[croissants_best_bid]
                // CROISSANTS_PER_BASKET
            )
            jams_bid_volume = (
                order_depths[Product.JAMS].buy_orders[jams_best_bid]
                // JAMS_PER_BASKET
            )
            djembes_bid_volume = (
                order_depths[Product.DJEMBES].buy_orders[djembes_best_bid]
                // DJEMBES_PER_BASKET
            )
            implied_bid_volume = min(
                croissants_bid_volume, jams_bid_volume, djembes_bid_volume
            )
            synthetic1_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            croissants_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[croissants_best_ask]
                // CROISSANTS_PER_BASKET
            )
            jams_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[jams_best_ask]
                // JAMS_PER_BASKET
            )
            djembes_ask_volume = (
                -order_depths[Product.DJEMBES].sell_orders[djembes_best_ask]
                // DJEMBES_PER_BASKET
            )
            implied_ask_volume = min(
                croissants_ask_volume, jams_ask_volume, djembes_ask_volume
            )
            synthetic1_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic1_order_price
    
    def get_synthetic2_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        CROISSANTS_PER_BASKET = BASKET_WEIGHTS2[Product.CROISSANTS]
        JAMS_PER_BASKET = BASKET_WEIGHTS2[Product.JAMS]

        # Initialize the synthetic2 basket order depth
        synthetic2_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        croissants_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        croissants_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        jams_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        jams_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic2 basket
        implied_bid = (
            croissants_best_bid * CROISSANTS_PER_BASKET
            + jams_best_bid * JAMS_PER_BASKET
            
        )
        implied_ask = (
            croissants_best_ask * CROISSANTS_PER_BASKET
            + jams_best_ask * JAMS_PER_BASKET
            
        )

        # Calculate the maximum number of synthetic2 baskets available at the implied bid and ask
        if implied_bid > 0:
            croissants_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[croissants_best_bid]
                // CROISSANTS_PER_BASKET
            )
            jams_bid_volume = (
                order_depths[Product.JAMS].buy_orders[jams_best_bid]
                // JAMS_PER_BASKET
            )
          
            implied_bid_volume = min(
                croissants_bid_volume, jams_bid_volume,
            )
            synthetic2_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            croissants_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[croissants_best_ask]
                // CROISSANTS_PER_BASKET
            )
            jams_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[jams_best_ask]
                // JAMS_PER_BASKET
            )
            
            implied_ask_volume = min(
                croissants_ask_volume, jams_ask_volume, 
            )
            synthetic2_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic2_order_price

    def convert_synthetic1_basket_orders(
        self, synthetic1_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBES: [],
        }

        # Get the best bid and ask for the synthetic1 basket
        synthetic1_basket_order_depth = self.get_synthetic1_basket_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic1_basket_order_depth.buy_orders.keys())
            if synthetic1_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic1_basket_order_depth.sell_orders.keys())
            if synthetic1_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic1 basket order
        for order in synthetic1_orders:
            # Extract the price and quantity from the synthetic1 basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic1 basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                croissants_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                jams_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
                djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                croissants_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                jams_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
                djembes_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                # The synthetic1 basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            croissants_order = Order(
                Product.CROISSANTS,
                croissants_price,
                quantity * BASKET_WEIGHTS1[Product.CROISSANTS],
            )
            jams_order = Order(
                Product.JAMS,
                jams_price,
                quantity * BASKET_WEIGHTS1[Product.JAMS],
            )
            djembes_order = Order(
                Product.DJEMBES, djembes_price, quantity * BASKET_WEIGHTS1[Product.DJEMBES]
            )

            # Add the component orders to the respective lists
            component_orders[Product.CROISSANTS].append(croissants_order)
            component_orders[Product.JAMS].append(jams_order)
            component_orders[Product.DJEMBES].append(djembes_order)

        return component_orders
    
    def convert_synthetic2_basket_orders(
        self, synthetic2_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
        }

        # Get the best bid and ask for the synthetic2 basket
        synthetic2_basket_order_depth = self.get_synthetic2_basket_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic2_basket_order_depth.buy_orders.keys())
            if synthetic2_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic2_basket_order_depth.sell_orders.keys())
            if synthetic2_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic2 basket order
        for order in synthetic2_orders:
            # Extract the price and quantity from the synthetic2 basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic2 basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                croissants_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                jams_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                croissants_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                jams_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
            else:
                continue

            # Create orders for each component
            croissants_order = Order(
                Product.CROISSANTS,
                croissants_price,
                quantity * BASKET_WEIGHTS2[Product.CROISSANTS],
            )
            jams_order = Order(
                Product.JAMS,
                jams_price,
                quantity * BASKET_WEIGHTS2[Product.JAMS],
            )

            component_orders[Product.CROISSANTS].append(croissants_order)
            component_orders[Product.JAMS].append(jams_order)

        return component_orders

    def execute_spread1_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):

        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic1_order_depth = self.get_synthetic1_basket_order_depth(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic1_bid_price = max(synthetic1_order_depth.buy_orders.keys())
            synthetic1_bid_volume = abs(
                synthetic1_order_depth.buy_orders[synthetic1_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic1_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.PICNIC_BASKET1, basket_ask_price, execute_volume)
            ]
            synthetic1_orders = [
                Order(Product.SYNTHETIC1, synthetic1_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic1_basket_orders(
                synthetic1_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic1_ask_price = min(synthetic1_order_depth.sell_orders.keys())
            synthetic1_ask_volume = abs(
                synthetic1_order_depth.sell_orders[synthetic1_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic1_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.PICNIC_BASKET1, basket_bid_price, -execute_volume)
            ]
            synthetic1_orders = [
                Order(Product.SYNTHETIC1, synthetic1_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic1_basket_orders(
                synthetic1_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders
        
    def execute_spread2_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):

        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
        synthetic2_order_depth = self.get_synthetic1_basket_order_depth(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic2_bid_price = max(synthetic2_order_depth.buy_orders.keys())
            synthetic2_bid_volume = abs(
                synthetic2_order_depth.buy_orders[synthetic2_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic2_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.PICNIC_BASKET2, basket_ask_price, execute_volume)
            ]
            synthetic2_orders = [
                Order(Product.SYNTHETIC2, synthetic2_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic2_basket_orders(
                synthetic2_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET2] = basket_orders
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic2_ask_price = min(synthetic2_order_depth.sell_orders.keys())
            synthetic2_ask_volume = abs(
                synthetic2_order_depth.sell_orders[synthetic2_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic2_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.PICNIC_BASKET2, basket_bid_price, -execute_volume)
            ]
            synthetic2_orders = [
                Order(Product.SYNTHETIC2, synthetic2_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic2_basket_orders(
                synthetic2_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET2] = basket_orders
            return aggregate_orders

    def spread1_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread1_data: Dict[str, Any],
    ):
        if Product.PICNIC_BASKET1 not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic1_order_depth = self.get_synthetic1_basket_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic1_swmid = self.get_swmid(synthetic1_order_depth)
        spread1 = basket_swmid - synthetic1_swmid
        self.spread1_data["spread1_history"].append(spread1)

        if (
            len(self.spread1_data["spread1_history"])
            < self.params[Product.SPREAD1]["spread1_std_window"]
        ):
            return None
        elif (
            len(self.spread1_data["spread1_history"])
            > self.params[Product.SPREAD1]["spread1_std_window"]
        ):
            self.spread1_data["spread1_history"].pop(0)

        spread1_std = np.std(self.spread1_data["spread1_history"])

        zscore = (
            spread1 - self.params[Product.SPREAD1]["default_spread1_mean"]
        ) / spread1_std

        if zscore >= self.params[Product.SPREAD1]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD1]["target_position"]:
                return self.execute_spread1_orders(
                    -self.params[Product.SPREAD1]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD1]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD1]["target_position"]:
                return self.execute_spread1_orders(
                    self.params[Product.SPREAD1]["target_position"],
                    basket_position,
                    order_depths,
                )

        self.spread1_data["prev_zscore"] = zscore
        return None
    
    def spread2_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread2_data: Dict[str, Any],
    ):
        if Product.PICNIC_BASKET2 not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
        synthetic2_order_depth = self.get_synthetic2_basket_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic2_swmid = self.get_swmid(synthetic2_order_depth)
        spread2 = basket_swmid - synthetic2_swmid
        self.spread2_data["spread2_history"].append(spread2)

        if (
            len(self.spread2_data["spread2_history"])
            < self.params[Product.SPREAD2]["spread2_std_window"]
        ):
            return None
        elif (
            len(self.spread2_data["spread2_history"])
            > self.params[Product.SPREAD2]["spread2_std_window"]
        ):
            self.spread2_data["spread2_history"].pop(0)

        spread2_std = np.std(self.spread2_data["spread2_history"])

        zscore = (
            spread2 - self.params[Product.SPREAD2]["default_spread2_mean"]
        ) / spread2_std

        if zscore >= self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread2_orders(
                    -self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread2_orders(
                    self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                )

        self.spread2_data["prev_zscore"] = zscore
        return None

    def get_volcanic_rock_voucher_9500_mid_price(
        self, volcanic_rock_voucher_9500_order_depth: OrderDepth, traderData: Dict[str, Any]
    ):
        if (
            len(volcanic_rock_voucher_9500_order_depth.buy_orders) > 0
            and len(volcanic_rock_voucher_9500_order_depth.sell_orders) > 0
        ):
            best_bid = max(volcanic_rock_voucher_9500_order_depth.buy_orders.keys())
            best_ask = min(volcanic_rock_voucher_9500_order_depth.sell_orders.keys())
            traderData["prev_coupon_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return traderData["prev_coupon_price"]
        
    def get_volcanic_rock_voucher_9750_mid_price(
        self, volcanic_rock_voucher_9750_order_depth: OrderDepth, traderData: Dict[str, Any]
    ):
        if (
            len(volcanic_rock_voucher_9750_order_depth.buy_orders) > 0
            and len(volcanic_rock_voucher_9750_order_depth.sell_orders) > 0
        ):
            best_bid = max(volcanic_rock_voucher_9750_order_depth.buy_orders.keys())
            best_ask = min(volcanic_rock_voucher_9750_order_depth.sell_orders.keys())
            traderData["prev_coupon_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return traderData["prev_coupon_price"]
        
    def get_volcanic_rock_voucher_10000_mid_price(
        self, volcanic_rock_voucher_10000_order_depth: OrderDepth, traderData: Dict[str, Any]
    ):
        if (
            len(volcanic_rock_voucher_10000_order_depth.buy_orders) > 0
            and len(volcanic_rock_voucher_10000_order_depth.sell_orders) > 0
        ):
            best_bid = max(volcanic_rock_voucher_10000_order_depth.buy_orders.keys())
            best_ask = min(volcanic_rock_voucher_10000_order_depth.sell_orders.keys())
            traderData["prev_coupon_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return traderData["prev_coupon_price"]
        
    def get_volcanic_rock_voucher_10250_mid_price(
        self, volcanic_rock_voucher_10250_order_depth: OrderDepth, traderData: Dict[str, Any]
    ):
        if (
            len(volcanic_rock_voucher_10250_order_depth.buy_orders) > 0
            and len(volcanic_rock_voucher_10250_order_depth.sell_orders) > 0
        ):
            best_bid = max(volcanic_rock_voucher_10250_order_depth.buy_orders.keys())
            best_ask = min(volcanic_rock_voucher_10250_order_depth.sell_orders.keys())
            traderData["prev_coupon_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return traderData["prev_coupon_price"]
        
    def get_volcanic_rock_voucher_10500_mid_price(
        self, volcanic_rock_voucher_10500_order_depth: OrderDepth, traderData: Dict[str, Any]
    ):
        if (
            len(volcanic_rock_voucher_10500_order_depth.buy_orders) > 0
            and len(volcanic_rock_voucher_10500_order_depth.sell_orders) > 0
        ):
            best_bid = max(volcanic_rock_voucher_10500_order_depth.buy_orders.keys())
            best_ask = min(volcanic_rock_voucher_10500_order_depth.sell_orders.keys())
            traderData["prev_coupon_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return traderData["prev_coupon_price"]
        
        
    def volcanic_rock_hedge_orders_9500(
        self,
        volcanic_rock_order_depth: OrderDepth,
        volcanic_rock_voucher_9500_order_depth: OrderDepth,
        volcanic_rock_voucher_9500_orders: List[Order],
        volcanic_rock_position: int,
        volcanic_rock_voucher_9500_position: int,
        delta: float
    ) -> List[Order]:
        if volcanic_rock_voucher_9500_orders == None or len(volcanic_rock_voucher_9500_orders) == 0:
            volcanic_rock_voucher_9500_position_after_trade = volcanic_rock_voucher_9500_position
        else:
            volcanic_rock_voucher_9500_position_after_trade = volcanic_rock_voucher_9500_position + sum(order.quantity for order in volcanic_rock_voucher_9500_orders)
        
        target_volcanic_rock_position = -delta * volcanic_rock_voucher_9500_position_after_trade
        
        if target_volcanic_rock_position == volcanic_rock_position:
            return None
        
        target_volcanic_rock_quantity = target_volcanic_rock_position - volcanic_rock_position

        orders: List[Order] = []
        if target_volcanic_rock_quantity > 0:
            # Buy VOLCANIC_ROCK
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity),
                self.LIMIT[Product.VOLCANIC_ROCK] - volcanic_rock_position,
            )
        
        elif target_volcanic_rock_quantity < 0:
            # Sell VOLCANIC_ROCK
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity),
                self.LIMIT[Product.VOLCANIC_ROCK] + volcanic_rock_position,
            )
        
        return orders
    
    def volcanic_rock_hedge_orders_9750(
        self,
        volcanic_rock_order_depth: OrderDepth,
        volcanic_rock_voucher_9750_order_depth: OrderDepth,
        volcanic_rock_voucher_9750_orders: List[Order],
        volcanic_rock_position: int,
        volcanic_rock_voucher_9750_position: int,
        delta: float
    ) -> List[Order]:
        if volcanic_rock_voucher_9750_orders == None or len(volcanic_rock_voucher_9750_orders) == 0:
            volcanic_rock_voucher_9750_position_after_trade = volcanic_rock_voucher_9750_position
        else:
            volcanic_rock_voucher_9750_position_after_trade = volcanic_rock_voucher_9750_position + sum(order.quantity for order in volcanic_rock_voucher_9750_orders)
        
        target_volcanic_rock_position = -delta * volcanic_rock_voucher_9750_position_after_trade
        
        if target_volcanic_rock_position == volcanic_rock_position:
            return None
        
        target_volcanic_rock_quantity = target_volcanic_rock_position - volcanic_rock_position

        orders: List[Order] = []
        if target_volcanic_rock_quantity > 0:
            # Buy VOLCANIC_ROCK
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity),
                self.LIMIT[Product.VOLCANIC_ROCK] - volcanic_rock_position,
            )
        
        elif target_volcanic_rock_quantity < 0:
            # Sell VOLCANIC_ROCK
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity),
                self.LIMIT[Product.VOLCANIC_ROCK] + volcanic_rock_position,
            )
        
        return orders
    
    def volcanic_rock_hedge_orders_10000(
        self,
        volcanic_rock_order_depth: OrderDepth,
        volcanic_rock_voucher_10000_order_depth: OrderDepth,
        volcanic_rock_voucher_10000_orders: List[Order],
        volcanic_rock_position: int,
        volcanic_rock_voucher_10000_position: int,
        delta: float
    ) -> List[Order]:
        if volcanic_rock_voucher_10000_orders == None or len(volcanic_rock_voucher_10000_orders) == 0:
            volcanic_rock_voucher_10000_position_after_trade = volcanic_rock_voucher_10000_position
        else:
            volcanic_rock_voucher_10000_position_after_trade = volcanic_rock_voucher_10000_position + sum(order.quantity for order in volcanic_rock_voucher_10000_orders)
        
        target_volcanic_rock_position = -delta * volcanic_rock_voucher_10000_position_after_trade
        
        if target_volcanic_rock_position == volcanic_rock_position:
            return None
        
        target_volcanic_rock_quantity = target_volcanic_rock_position - volcanic_rock_position

        orders: List[Order] = []
        if target_volcanic_rock_quantity > 0:
            # Buy VOLCANIC_ROCK
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity),
                self.LIMIT[Product.VOLCANIC_ROCK] - volcanic_rock_position,
            )
        
        elif target_volcanic_rock_quantity < 0:
            # Sell VOLCANIC_ROCK
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity),
                self.LIMIT[Product.VOLCANIC_ROCK] + volcanic_rock_position,
            )
        
        return orders
    
    def volcanic_rock_hedge_orders_10250(
        self,
        volcanic_rock_order_depth: OrderDepth,
        volcanic_rock_voucher_10250_order_depth: OrderDepth,
        volcanic_rock_voucher_10250_orders: List[Order],
        volcanic_rock_position: int,
        volcanic_rock_voucher_10250_position: int,
        delta: float
    ) -> List[Order]:
        if volcanic_rock_voucher_10250_orders == None or len(volcanic_rock_voucher_10250_orders) == 0:
            volcanic_rock_voucher_10250_position_after_trade = volcanic_rock_voucher_10250_position
        else:
            volcanic_rock_voucher_10250_position_after_trade = volcanic_rock_voucher_10250_position + sum(order.quantity for order in volcanic_rock_voucher_10250_orders)
        
        target_volcanic_rock_position = -delta * volcanic_rock_voucher_10250_position_after_trade
        
        if target_volcanic_rock_position == volcanic_rock_position:
            return None
        
        target_volcanic_rock_quantity = target_volcanic_rock_position - volcanic_rock_position

        orders: List[Order] = []
        if target_volcanic_rock_quantity > 0:
            # Buy VOLCANIC_ROCK
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity),
                self.LIMIT[Product.VOLCANIC_ROCK] - volcanic_rock_position,
            )
        
        elif target_volcanic_rock_quantity < 0:
            # Sell VOLCANIC_ROCK
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity),
                self.LIMIT[Product.VOLCANIC_ROCK] + volcanic_rock_position,
            )
        
        return orders

    def volcanic_rock_hedge_orders_10500(
        self,
        volcanic_rock_order_depth: OrderDepth,
        volcanic_rock_voucher_10500_order_depth: OrderDepth,
        volcanic_rock_voucher_10500_orders: List[Order],
        volcanic_rock_position: int,
        volcanic_rock_voucher_10500_position: int,
        delta: float
    ) -> List[Order]:
        if volcanic_rock_voucher_10500_orders == None or len(volcanic_rock_voucher_10500_orders) == 0:
            volcanic_rock_voucher_10500_position_after_trade = volcanic_rock_voucher_10500_position
        else:
            volcanic_rock_voucher_10500_position_after_trade = volcanic_rock_voucher_10500_position + sum(order.quantity for order in volcanic_rock_voucher_10500_orders)
        
        target_volcanic_rock_position = -delta * volcanic_rock_voucher_10500_position_after_trade
        
        if target_volcanic_rock_position == volcanic_rock_position:
            return None
        
        target_volcanic_rock_quantity = target_volcanic_rock_position - volcanic_rock_position

        orders: List[Order] = []
        if target_volcanic_rock_quantity > 0:
            # Buy VOLCANIC_ROCK
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity),
                self.LIMIT[Product.VOLCANIC_ROCK] - volcanic_rock_position,
            )
        
        elif target_volcanic_rock_quantity < 0:
            # Sell VOLCANIC_ROCK
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity),
                self.LIMIT[Product.VOLCANIC_ROCK] + volcanic_rock_position,
            )
        
        return orders

    def volcanic_rock_voucher_9500_orders(
        self,
        volcanic_rock_voucher_9500_order_depth: OrderDepth,
        volcanic_rock_voucher_9500_position: int,
        traderData: Dict[str, Any],
        volatility: float,
    ) -> List[Order]:
        traderData['past_coupon_vol'].append(volatility)
        if len(traderData['past_coupon_vol']) < self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]['std_window']:
            return None, None

        if len(traderData['past_coupon_vol']) > self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]['std_window']:
            traderData['past_coupon_vol'].pop(0)
        
        vol_z_score = (volatility - self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]['mean_volatility']) / np.std(traderData['past_coupon_vol'])
        if (
            vol_z_score 
            >= self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]['zscore_threshold']
        ):
            if volcanic_rock_voucher_9500_position != -self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_9500]:
                target_volcanic_rock_voucher_9500_position = -self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_9500]
                if len(volcanic_rock_voucher_9500_order_depth.buy_orders) > 0:
                    best_bid = max(volcanic_rock_voucher_9500_order_depth.buy_orders.keys())
                    target_quantity = abs(target_volcanic_rock_voucher_9500_position - volcanic_rock_voucher_9500_position)
                    quantity = min(
                        target_quantity,
                        abs(volcanic_rock_voucher_9500_order_depth.buy_orders[best_bid]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_9500, best_bid, -quantity)], []
                    else:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_9500, best_bid, -quantity)], [Order(Product.VOLCANIC_ROCK_VOUCHER_9500, best_bid, -quote_quantity)]

        elif (
            vol_z_score
            <= -self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["zscore_threshold"]
        ):
            if volcanic_rock_voucher_9500_position != self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_9500]:
                target_volcanic_rock_voucher_9500_position = self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_9500]
                if len(volcanic_rock_voucher_9500_order_depth.sell_orders) > 0:
                    best_ask = min(volcanic_rock_voucher_9500_order_depth.sell_orders.keys())
                    target_quantity = abs(target_volcanic_rock_voucher_9500_position - volcanic_rock_voucher_9500_position)
                    quantity = min(
                        target_quantity,
                        abs(volcanic_rock_voucher_9500_order_depth.sell_orders[best_ask]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_9500, best_ask, quantity)], []
                    else:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_9500, best_ask, quantity)], [Order(Product.VOLCANIC_ROCK_VOUCHER_9500, best_ask, quote_quantity)]

        return None, None
    
    def volcanic_rock_voucher_9750_orders(
        self,
        volcanic_rock_voucher_9750_order_depth: OrderDepth,
        volcanic_rock_voucher_9750_position: int,
        traderData: Dict[str, Any],
        volatility: float,
    ) -> List[Order]:
        traderData['past_coupon_vol'].append(volatility)
        if len(traderData['past_coupon_vol']) < self.params[Product.VOLCANIC_ROCK_VOUCHER_9750]['std_window']:
            return None, None

        if len(traderData['past_coupon_vol']) > self.params[Product.VOLCANIC_ROCK_VOUCHER_9750]['std_window']:
            traderData['past_coupon_vol'].pop(0)
        
        vol_z_score = (volatility - self.params[Product.VOLCANIC_ROCK_VOUCHER_9750]['mean_volatility']) / np.std(traderData['past_coupon_vol'])
        if (
            vol_z_score 
            >= self.params[Product.VOLCANIC_ROCK_VOUCHER_9750]['zscore_threshold']
        ):
            if volcanic_rock_voucher_9750_position != -self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_9750]:
                target_volcanic_rock_voucher_9750_position = -self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_9750]
                if len(volcanic_rock_voucher_9750_order_depth.buy_orders) > 0:
                    best_bid = max(volcanic_rock_voucher_9750_order_depth.buy_orders.keys())
                    target_quantity = abs(target_volcanic_rock_voucher_9750_position - volcanic_rock_voucher_9750_position)
                    quantity = min(
                        target_quantity,
                        abs(volcanic_rock_voucher_9750_order_depth.buy_orders[best_bid]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_9750, best_bid, -quantity)], []
                    else:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_9750, best_bid, -quantity)], [Order(Product.VOLCANIC_ROCK_VOUCHER_9750, best_bid, -quote_quantity)]

        elif (
            vol_z_score
            <= -self.params[Product.VOLCANIC_ROCK_VOUCHER_9750]["zscore_threshold"]
        ):
            if volcanic_rock_voucher_9750_position != self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_9750]:
                target_volcanic_rock_voucher_9750_position = self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_9750]
                if len(volcanic_rock_voucher_9750_order_depth.sell_orders) > 0:
                    best_ask = min(volcanic_rock_voucher_9750_order_depth.sell_orders.keys())
                    target_quantity = abs(target_volcanic_rock_voucher_9750_position - volcanic_rock_voucher_9750_position)
                    quantity = min(
                        target_quantity,
                        abs(volcanic_rock_voucher_9750_order_depth.sell_orders[best_ask]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_9750, best_ask, quantity)], []
                    else:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_9750, best_ask, quantity)], [Order(Product.VOLCANIC_ROCK_VOUCHER_9750, best_ask, quote_quantity)]

        return None, None
    
    def volcanic_rock_voucher_10000_orders(
        self,
        volcanic_rock_voucher_10000_order_depth: OrderDepth,
        volcanic_rock_voucher_10000_position: int,
        traderData: Dict[str, Any],
        volatility: float,
    ) -> List[Order]:
        traderData['past_coupon_vol'].append(volatility)
        if len(traderData['past_coupon_vol']) < self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]['std_window']:
            return None, None

        if len(traderData['past_coupon_vol']) > self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]['std_window']:
            traderData['past_coupon_vol'].pop(0)
        
        vol_z_score = (volatility - self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]['mean_volatility']) / np.std(traderData['past_coupon_vol'])
        if (
            vol_z_score 
            >= self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]['zscore_threshold']
        ):
            if volcanic_rock_voucher_10000_position != -self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_10000]:
                target_volcanic_rock_voucher_10000_position = -self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_10000]
                if len(volcanic_rock_voucher_10000_order_depth.buy_orders) > 0:
                    best_bid = max(volcanic_rock_voucher_10000_order_depth.buy_orders.keys())
                    target_quantity = abs(target_volcanic_rock_voucher_10000_position - volcanic_rock_voucher_10000_position)
                    quantity = min(
                        target_quantity,
                        abs(volcanic_rock_voucher_10000_order_depth.buy_orders[best_bid]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_10000, best_bid, -quantity)], []
                    else:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_10000, best_bid, -quantity)], [Order(Product.VOLCANIC_ROCK_VOUCHER_10000, best_bid, -quote_quantity)]

        elif (
            vol_z_score
            <= -self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]["zscore_threshold"]
        ):
            if volcanic_rock_voucher_10000_position != self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_10000]:
                target_volcanic_rock_voucher_10000_position = self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_10000]
                if len(volcanic_rock_voucher_10000_order_depth.sell_orders) > 0:
                    best_ask = min(volcanic_rock_voucher_10000_order_depth.sell_orders.keys())
                    target_quantity = abs(target_volcanic_rock_voucher_10000_position - volcanic_rock_voucher_10000_position)
                    quantity = min(
                        target_quantity,
                        abs(volcanic_rock_voucher_10000_order_depth.sell_orders[best_ask]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_10000, best_ask, quantity)], []
                    else:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_10000, best_ask, quantity)], [Order(Product.VOLCANIC_ROCK_VOUCHER_10000, best_ask, quote_quantity)]

        return None, None
    
    def volcanic_rock_voucher_10250_orders(
        self,
        volcanic_rock_voucher_10250_order_depth: OrderDepth,
        volcanic_rock_voucher_10250_position: int,
        traderData: Dict[str, Any],
        volatility: float,
    ) -> List[Order]:
        traderData['past_coupon_vol'].append(volatility)
        if len(traderData['past_coupon_vol']) < self.params[Product.VOLCANIC_ROCK_VOUCHER_10250]['std_window']:
            return None, None

        if len(traderData['past_coupon_vol']) > self.params[Product.VOLCANIC_ROCK_VOUCHER_10250]['std_window']:
            traderData['past_coupon_vol'].pop(0)
        
        vol_z_score = (volatility - self.params[Product.VOLCANIC_ROCK_VOUCHER_10250]['mean_volatility']) / np.std(traderData['past_coupon_vol'])
        if (
            vol_z_score 
            >= self.params[Product.VOLCANIC_ROCK_VOUCHER_10250]['zscore_threshold']
        ):
            if volcanic_rock_voucher_10250_position != -self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_10250]:
                target_volcanic_rock_voucher_10250_position = -self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_10250]
                if len(volcanic_rock_voucher_10250_order_depth.buy_orders) > 0:
                    best_bid = max(volcanic_rock_voucher_10250_order_depth.buy_orders.keys())
                    target_quantity = abs(target_volcanic_rock_voucher_10250_position - volcanic_rock_voucher_10250_position)
                    quantity = min(
                        target_quantity,
                        abs(volcanic_rock_voucher_10250_order_depth.buy_orders[best_bid]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_10250, best_bid, -quantity)], []
                    else:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_10250, best_bid, -quantity)], [Order(Product.VOLCANIC_ROCK_VOUCHER_10250, best_bid, -quote_quantity)]

        elif (
            vol_z_score
            <= -self.params[Product.VOLCANIC_ROCK_VOUCHER_10250]["zscore_threshold"]
        ):
            if volcanic_rock_voucher_10250_position != self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_10250]:
                target_volcanic_rock_voucher_10250_position = self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_10250]
                if len(volcanic_rock_voucher_10250_order_depth.sell_orders) > 0:
                    best_ask = min(volcanic_rock_voucher_10250_order_depth.sell_orders.keys())
                    target_quantity = abs(target_volcanic_rock_voucher_10250_position - volcanic_rock_voucher_10250_position)
                    quantity = min(
                        target_quantity,
                        abs(volcanic_rock_voucher_10250_order_depth.sell_orders[best_ask]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_10250, best_ask, quantity)], []
                    else:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_10250, best_ask, quantity)], [Order(Product.VOLCANIC_ROCK_VOUCHER_10250, best_ask, quote_quantity)]

        return None, None
    
    
    def volcanic_rock_voucher_10500_orders(
        self,
        volcanic_rock_voucher_10500_order_depth: OrderDepth,
        volcanic_rock_voucher_10500_position: int,
        traderData: Dict[str, Any],
        volatility: float,
    ) -> List[Order]:
        traderData['past_coupon_vol'].append(volatility)
        if len(traderData['past_coupon_vol']) < self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]['std_window']:
            return None, None

        if len(traderData['past_coupon_vol']) > self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]['std_window']:
            traderData['past_coupon_vol'].pop(0)
        
        vol_z_score = (volatility - self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]['mean_volatility']) / np.std(traderData['past_coupon_vol'])
        if (
            vol_z_score 
            >= self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]['zscore_threshold']
        ):
            if volcanic_rock_voucher_10500_position != -self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_10500]:
                target_volcanic_rock_voucher_10500_position = -self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_10500]
                if len(volcanic_rock_voucher_10500_order_depth.buy_orders) > 0:
                    best_bid = max(volcanic_rock_voucher_10500_order_depth.buy_orders.keys())
                    target_quantity = abs(target_volcanic_rock_voucher_10500_position - volcanic_rock_voucher_10500_position)
                    quantity = min(
                        target_quantity,
                        abs(volcanic_rock_voucher_10500_order_depth.buy_orders[best_bid]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_10500, best_bid, -quantity)], []
                    else:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_10500, best_bid, -quantity)], [Order(Product.VOLCANIC_ROCK_VOUCHER_10500, best_bid, -quote_quantity)]

        elif (
            vol_z_score
            <= -self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]["zscore_threshold"]
        ):
            if volcanic_rock_voucher_10500_position != self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_10500]:
                target_volcanic_rock_voucher_10500_position = self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_10500]
                if len(volcanic_rock_voucher_10500_order_depth.sell_orders) > 0:
                    best_ask = min(volcanic_rock_voucher_10500_order_depth.sell_orders.keys())
                    target_quantity = abs(target_volcanic_rock_voucher_10500_position - volcanic_rock_voucher_10500_position)
                    quantity = min(
                        target_quantity,
                        abs(volcanic_rock_voucher_10500_order_depth.sell_orders[best_ask]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_10500, best_ask, quantity)], []
                    else:
                        return [Order(Product.VOLCANIC_ROCK_VOUCHER_10500, best_ask, quantity)], [Order(Product.VOLCANIC_ROCK_VOUCHER_10500, best_ask, quote_quantity)]

        return None, None
    
    def get_past_returns(
        self,
        traderObject: Dict[str, Any],
        order_depths: Dict[str, OrderDepth],
        timeframes: Dict[str, int],
    ):
        returns_dict = {}

        for symbol, timeframe in timeframes.items():
            traderObject_key = f"{symbol}_price_history"
            if traderObject_key not in traderObject:
                traderObject[traderObject_key] = []

            price_history = traderObject[traderObject_key]

            if symbol in order_depths:
                order_depth = order_depths[symbol]
                if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                    current_price = (
                        max(order_depth.buy_orders.keys())
                        + min(order_depth.sell_orders.keys())
                    ) / 2
                else:
                    if len(price_history) > 0:
                        current_price = float(price_history[-1])
                    else:
                        returns_dict[symbol] = None
                        continue
            else:
                if len(price_history) > 0:
                    current_price = float(price_history[-1])
                else:
                    returns_dict[symbol] = None
                    continue

            price_history.append(
                f"{current_price:.1f}"
            ) 

            if len(price_history) > timeframe:
                price_history.pop(0)

            if len(price_history) == timeframe:
                past_price = float(price_history[0])  # Convert string back to float
                returns = (current_price - past_price) / past_price
                returns_dict[symbol] = returns
            else:
                returns_dict[symbol] = None

        return returns_dict

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        past_returns_timeframes = {"PICNIC_BASKET1": 500}
        past_returns_dict = self.get_past_returns(
            traderObject, state.order_depths, past_returns_timeframes
        )

        result = {}
        conversions = 0

        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            rainforest_resin_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            rainforest_resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    rainforest_resin_position,
                )
            )
            rainforest_resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    rainforest_resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            rainforest_resin_make_orders, _, _ = self.make_rainforest_resin_orders(
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                rainforest_resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["volume_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                rainforest_resin_take_orders + rainforest_resin_clear_orders + rainforest_resin_make_orders
            )

        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            squid_ink_position = (
                state.position[Product.SQUID_INK]
                if Product.SQUID_INK in state.position
                else 0
            )
            squid_ink_fair_value = self.squid_ink_fair_value(
                state.order_depths[Product.SQUID_INK], traderObject
            )
            squid_ink_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    squid_ink_fair_value,
                    self.params[Product.SQUID_INK]["take_width"],
                    squid_ink_position,
                    self.params[Product.SQUID_INK]["prevent_adverse"],
                    self.params[Product.SQUID_INK]["adverse_volume"],
                )
            )
            squid_ink_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    squid_ink_fair_value,
                    self.params[Product.SQUID_INK]["clear_width"],
                    squid_ink_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            squid_ink_make_orders, _, _ = self.make_squid_ink_orders(
                state.order_depths[Product.SQUID_INK],
                squid_ink_fair_value,
                self.params[Product.SQUID_INK]["squid_ink_min_edge"],
                squid_ink_position,
                buy_order_volume,
                sell_order_volume,
            )
            result[Product.SQUID_INK] = (
                squid_ink_take_orders + squid_ink_clear_orders + squid_ink_make_orders
            )

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            kelp_fair_value = self.kelp_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            kelp_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["take_width"],
                    kelp_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            kelp_make_orders, _, _ = self.make_kelp_orders(
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]["kelp_min_edge"],
                kelp_position,
                buy_order_volume,
                sell_order_volume,
            )
            result[Product.KELP] = (
                kelp_take_orders + kelp_clear_orders + kelp_make_orders
            )

    

        if Product.SPREAD1 not in traderObject:
            traderObject[Product.SPREAD1] = {
                "spread1_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket_position = (
            state.position[Product.PICNIC_BASKET1]
            if Product.PICNIC_BASKET1 in state.position
            else 0
        )
        spread1_orders = self.spread1_orders(
            state.order_depths,
            Product.PICNIC_BASKET1,
            basket_position,
            traderObject[Product.SPREAD1],
        )

        if spread1_orders != None:
            result[Product.CROISSANTS] = spread1_orders[Product.CROISSANTS]
            result[Product.JAMS] = spread1_orders[Product.JAMS]
            result[Product.DJEMBES] = spread1_orders[Product.DJEMBES]

        if Product.SPREAD2 not in traderObject:
            traderObject[Product.SPREAD2] = {
                "spread1_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket_position = (
            state.position[Product.PICNIC_BASKET2]
            if Product.PICNIC_BASKET2 in state.position
            else 0
        )
        spread2_orders = self.spread2_orders(
            state.order_depths,
            Product.PICNIC_BASKET2,
            basket_position,
            traderObject[Product.SPREAD1],
        )

        if spread2_orders != None:
            result[Product.CROISSANTS] = spread2_orders[Product.CROISSANTS]
            result[Product.JAMS] = spread2_orders[Product.JAMS]


        if Product.VOLCANIC_ROCK_VOUCHER_9500 not in traderObject:
            traderObject[Product.VOLCANIC_ROCK_VOUCHER_9500] = {
                "prev_coupon_price": 0,
                "past_coupon_vol": []
            }

        if (
            Product.VOLCANIC_ROCK_VOUCHER_9500 in self.params
            and Product.VOLCANIC_ROCK_VOUCHER_9500 in state.order_depths
        ):
            volcanic_rock_voucher_9500_position = (
                state.position[Product.VOLCANIC_ROCK_VOUCHER_9500]
                if Product.VOLCANIC_ROCK_VOUCHER_9500 in state.position
                else 0
            )

            volcanic_rock_position = (
                state.position[Product.VOLCANIC_ROCK]
                if Product.VOLCANIC_ROCK in state.position
                else 0
            )
            volcanic_rock_order_depth = state.order_depths[Product.VOLCANIC_ROCK]
            volcanic_rock_voucher_9500_order_depth = state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_9500]
            volcanic_rock_mid_price = (
                min(volcanic_rock_order_depth.buy_orders.keys())
                + max(volcanic_rock_order_depth.sell_orders.keys())
            ) / 2
            volcanic_rock_voucher_9500_mid_price = self.get_volcanic_rock_voucher_9500_mid_price(
                volcanic_rock_voucher_9500_order_depth, traderObject[Product.VOLCANIC_ROCK_VOUCHER_9500]
            )
            tte = (
                self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["starting_time_to_expiry"]
                - (state.timestamp) / 1000000 / 250
            )
            volatility = BlackScholes.implied_volatility(
                volcanic_rock_voucher_9500_mid_price,
                volcanic_rock_mid_price,
                self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["strike"],
                tte,
            )
            delta = BlackScholes.delta(
                volcanic_rock_mid_price,
                self.params[Product.VOLCANIC_ROCK_VOUCHER_9500]["strike"],
                tte,
                volatility,
            )
    
            volcanic_rock_voucher_9500_take_orders, volcanic_rock_voucher_9500_make_orders = self.volcanic_rock_voucher_9500_orders(
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_9500],
                volcanic_rock_voucher_9500_position,
                traderObject[Product.VOLCANIC_ROCK_VOUCHER_9500],
                volatility,
            )

            volcanic_rock_orders = self.volcanic_rock_hedge_orders_9500(
                state.order_depths[Product.VOLCANIC_ROCK],
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_9500],
                volcanic_rock_voucher_9500_take_orders,
                volcanic_rock_position,
                volcanic_rock_voucher_9500_position,
                delta
            )

            if volcanic_rock_voucher_9500_take_orders != None or volcanic_rock_voucher_9500_make_orders != None:
                result[Product.VOLCANIC_ROCK_VOUCHER_9500] = volcanic_rock_voucher_9500_take_orders + volcanic_rock_voucher_9500_make_orders

        
        if Product.VOLCANIC_ROCK_VOUCHER_9750 not in traderObject:
            traderObject[Product.VOLCANIC_ROCK_VOUCHER_9750] = {
                "prev_coupon_price": 0,
                "past_coupon_vol": []
            }

        if (
            Product.VOLCANIC_ROCK_VOUCHER_9750 in self.params
            and Product.VOLCANIC_ROCK_VOUCHER_9750 in state.order_depths
        ):
            volcanic_rock_voucher_9750_position = (
                state.position[Product.VOLCANIC_ROCK_VOUCHER_9750]
                if Product.VOLCANIC_ROCK_VOUCHER_9750 in state.position
                else 0
            )

            volcanic_rock_position = (
                state.position[Product.VOLCANIC_ROCK]
                if Product.VOLCANIC_ROCK in state.position
                else 0
            )

            volcanic_rock_order_depth = state.order_depths[Product.VOLCANIC_ROCK]
            volcanic_rock_voucher_9750_order_depth = state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_9750]
            volcanic_rock_mid_price = (
                min(volcanic_rock_order_depth.buy_orders.keys())
                + max(volcanic_rock_order_depth.sell_orders.keys())
            ) / 2
            volcanic_rock_voucher_9750_mid_price = self.get_volcanic_rock_voucher_9750_mid_price(
                volcanic_rock_voucher_9750_order_depth, traderObject[Product.VOLCANIC_ROCK_VOUCHER_9750]
            )
            tte = (
                self.params[Product.VOLCANIC_ROCK_VOUCHER_9750]["starting_time_to_expiry"]
                - (state.timestamp) / 1000000 / 250
            )
            volatility = BlackScholes.implied_volatility(
                volcanic_rock_voucher_9750_mid_price,
                volcanic_rock_mid_price,
                self.params[Product.VOLCANIC_ROCK_VOUCHER_9750]["strike"],
                tte,
            )
            delta = BlackScholes.delta(
                volcanic_rock_mid_price,
                self.params[Product.VOLCANIC_ROCK_VOUCHER_9750]["strike"],
                tte,
                volatility,
            )
    
            volcanic_rock_voucher_9750_take_orders, volcanic_rock_voucher_9750_make_orders = self.volcanic_rock_voucher_9750_orders(
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_9750],
                volcanic_rock_voucher_9750_position,
                traderObject[Product.VOLCANIC_ROCK_VOUCHER_9750],
                volatility,
            )

            volcanic_rock_orders = self.volcanic_rock_hedge_orders_9750(
                state.order_depths[Product.VOLCANIC_ROCK],
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_9750],
                volcanic_rock_voucher_9750_take_orders,
                volcanic_rock_position,
                volcanic_rock_voucher_9750_position,
                delta
            )

            if volcanic_rock_voucher_9750_take_orders != None or volcanic_rock_voucher_9750_make_orders != None:
                result[Product.VOLCANIC_ROCK_VOUCHER_9750] = volcanic_rock_voucher_9750_take_orders + volcanic_rock_voucher_9750_make_orders



        if Product.VOLCANIC_ROCK_VOUCHER_10000 not in traderObject:
            traderObject[Product.VOLCANIC_ROCK_VOUCHER_10000] = {
                "prev_coupon_price": 0,
                "past_coupon_vol": []
            }
        if (
            Product.VOLCANIC_ROCK_VOUCHER_10000 in self.params
            and Product.VOLCANIC_ROCK_VOUCHER_10000 in state.order_depths
        ):
            volcanic_rock_voucher_10000_position = (
                state.position[Product.VOLCANIC_ROCK_VOUCHER_10000]
                if Product.VOLCANIC_ROCK_VOUCHER_10000 in state.position
                else 0
            )

            volcanic_rock_position = (
                state.position[Product.VOLCANIC_ROCK]
                if Product.VOLCANIC_ROCK in state.position
                else 0
            )

            volcanic_rock_order_depth = state.order_depths[Product.VOLCANIC_ROCK]
            volcanic_rock_voucher_10000_order_depth = state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10000]
            volcanic_rock_mid_price = (
                min(volcanic_rock_order_depth.buy_orders.keys())
                + max(volcanic_rock_order_depth.sell_orders.keys())
            ) / 2
            volcanic_rock_voucher_10000_mid_price = self.get_volcanic_rock_voucher_10000_mid_price(
                volcanic_rock_voucher_10000_order_depth, traderObject[Product.VOLCANIC_ROCK_VOUCHER_10000]
            )
            tte = (
                self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]["starting_time_to_expiry"]
                - (state.timestamp) / 1000000 / 250
            )
            volatility = BlackScholes.implied_volatility(
                volcanic_rock_voucher_10000_mid_price,
                volcanic_rock_mid_price,
                self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]["strike"],
                tte,
            )
            delta = BlackScholes.delta(
                volcanic_rock_mid_price,
                self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]["strike"],
                tte,
                volatility,
            )
    
            volcanic_rock_voucher_10000_take_orders, volcanic_rock_voucher_10000_make_orders = self.volcanic_rock_voucher_10000_orders(
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10000],
                volcanic_rock_voucher_10000_position,
                traderObject[Product.VOLCANIC_ROCK_VOUCHER_10000],
                volatility,
            )

            volcanic_rock_orders = self.volcanic_rock_hedge_orders_10000(
                state.order_depths[Product.VOLCANIC_ROCK],
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10000],
                volcanic_rock_voucher_10000_take_orders,
                volcanic_rock_position,
                volcanic_rock_voucher_10000_position,
                delta
            )

            if volcanic_rock_voucher_10000_take_orders != None or volcanic_rock_voucher_10000_make_orders != None:
                result[Product.VOLCANIC_ROCK_VOUCHER_10000] = volcanic_rock_voucher_10000_take_orders + volcanic_rock_voucher_10000_make_orders

        if Product.VOLCANIC_ROCK_VOUCHER_10250 not in traderObject:
            traderObject[Product.VOLCANIC_ROCK_VOUCHER_10250] = {
                "prev_coupon_price": 0,
                "past_coupon_vol": []
            }
        if (
            Product.VOLCANIC_ROCK_VOUCHER_10250 in self.params
            and Product.VOLCANIC_ROCK_VOUCHER_10250 in state.order_depths
        ):
            volcanic_rock_voucher_10250_position = (
                state.position[Product.VOLCANIC_ROCK_VOUCHER_10250]
                if Product.VOLCANIC_ROCK_VOUCHER_10250 in state.position
                else 0
            )

            volcanic_rock_position = (
                state.position[Product.VOLCANIC_ROCK]
                if Product.VOLCANIC_ROCK in state.position
                else 0
            )

            volcanic_rock_order_depth = state.order_depths[Product.VOLCANIC_ROCK]
            volcanic_rock_voucher_10250_order_depth = state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10250]
            volcanic_rock_mid_price = (
                min(volcanic_rock_order_depth.buy_orders.keys())
                + max(volcanic_rock_order_depth.sell_orders.keys())
            ) / 2
            volcanic_rock_voucher_10250_mid_price = self.get_volcanic_rock_voucher_10250_mid_price(
                volcanic_rock_voucher_10250_order_depth, traderObject[Product.VOLCANIC_ROCK_VOUCHER_10250]
            )
            tte = (
                self.params[Product.VOLCANIC_ROCK_VOUCHER_10250]["starting_time_to_expiry"]
                - (state.timestamp) / 1025000 / 250
            )
            volatility = BlackScholes.implied_volatility(
                volcanic_rock_voucher_10250_mid_price,
                volcanic_rock_mid_price,
                self.params[Product.VOLCANIC_ROCK_VOUCHER_10250]["strike"],
                tte,
            )
            delta = BlackScholes.delta(
                volcanic_rock_mid_price,
                self.params[Product.VOLCANIC_ROCK_VOUCHER_10250]["strike"],
                tte,
                volatility,
            )
    
            volcanic_rock_voucher_10250_take_orders, volcanic_rock_voucher_10250_make_orders = self.volcanic_rock_voucher_10250_orders(
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10250],
                volcanic_rock_voucher_10250_position,
                traderObject[Product.VOLCANIC_ROCK_VOUCHER_10250],
                volatility,
            )

            volcanic_rock_orders = self.volcanic_rock_hedge_orders_10250(
                state.order_depths[Product.VOLCANIC_ROCK],
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10250],
                volcanic_rock_voucher_10250_take_orders,
                volcanic_rock_position,
                volcanic_rock_voucher_10250_position,
                delta
            )

            if volcanic_rock_voucher_10250_take_orders != None or volcanic_rock_voucher_10250_make_orders != None:
                result[Product.VOLCANIC_ROCK_VOUCHER_10250] = volcanic_rock_voucher_10250_take_orders + volcanic_rock_voucher_10250_make_orders
        
        if Product.VOLCANIC_ROCK_VOUCHER_10500 not in traderObject:
            traderObject[Product.VOLCANIC_ROCK_VOUCHER_10500] = {
                "prev_coupon_price": 0,
                "past_coupon_vol": []
            }
        if (
            Product.VOLCANIC_ROCK_VOUCHER_10500 in self.params
            and Product.VOLCANIC_ROCK_VOUCHER_10500 in state.order_depths
        ):
            volcanic_rock_voucher_10500_position = (
                state.position[Product.VOLCANIC_ROCK_VOUCHER_10500]
                if Product.VOLCANIC_ROCK_VOUCHER_10500 in state.position
                else 0
            )

            volcanic_rock_position = (
                state.position[Product.VOLCANIC_ROCK]
                if Product.VOLCANIC_ROCK in state.position
                else 0
            )
            volcanic_rock_order_depth = state.order_depths[Product.VOLCANIC_ROCK]
            volcanic_rock_voucher_10500_order_depth = state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10500]
            volcanic_rock_mid_price = (
                min(volcanic_rock_order_depth.buy_orders.keys())
                + max(volcanic_rock_order_depth.sell_orders.keys())
            ) / 2
            volcanic_rock_voucher_10500_mid_price = self.get_volcanic_rock_voucher_10500_mid_price(
                volcanic_rock_voucher_10500_order_depth, traderObject[Product.VOLCANIC_ROCK_VOUCHER_10500]
            )
            tte = (
                self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]["starting_time_to_expiry"]
                - (state.timestamp) / 1050000 / 250
            )
            volatility = BlackScholes.implied_volatility(
                volcanic_rock_voucher_10500_mid_price,
                volcanic_rock_mid_price,
                self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]["strike"],
                tte,
            )
            delta = BlackScholes.delta(
                volcanic_rock_mid_price,
                self.params[Product.VOLCANIC_ROCK_VOUCHER_10500]["strike"],
                tte,
                volatility,
            )
    
            volcanic_rock_voucher_10500_take_orders, volcanic_rock_voucher_10500_make_orders = self.volcanic_rock_voucher_10500_orders(
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10500],
                volcanic_rock_voucher_10500_position,
                traderObject[Product.VOLCANIC_ROCK_VOUCHER_10500],
                volatility,
            )

            volcanic_rock_orders = self.volcanic_rock_hedge_orders_10500(
                state.order_depths[Product.VOLCANIC_ROCK],
                state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10500],
                volcanic_rock_voucher_10500_take_orders,
                volcanic_rock_position,
                volcanic_rock_voucher_10500_position,
                delta
            )

            if volcanic_rock_voucher_10500_take_orders != None or volcanic_rock_voucher_10500_make_orders != None:
                result[Product.VOLCANIC_ROCK_VOUCHER_10500] = volcanic_rock_voucher_10500_take_orders + volcanic_rock_voucher_10500_make_orders

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
