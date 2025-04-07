from typing import Dict, List, Tuple
import numpy as np
from collections import deque
import math
import jsonpickle

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

class OrderDepth: #this class represents the market order book depth
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}  # Dictionary mapping price levels to volumes for buy orders
        self.sell_orders: Dict[int, int] = {}  # Dictionary mapping prices levels to volumes for sell orders, volumes being negative

class Order: #represents a single trading order 
    def __init__(self, symbol: str, price: int, quantity: int): 
        self.symbol = symbol #represents the trading instrument: kelp or Rainforest Resin
        self.price = price #order price level
        self.quantity = quantity  # Order size; positive for buy, negative for sell

class Trade: #represents a completed trade 
    def __init__(self, symbol: str, price: int, quantity: int, buyer: str = "", seller: str = ""):
        self.symbol = symbol #represents the trading instrument: kelp or rainforest resin
        self.price = price #execution, ie selling price
        self.quantity = quantity #trade size
        self.buyer = buyer #identifies the counterpart buyer
        self.seller = seller #identifies the counterpart seller 

#Market State Management

class Observation: #captures a snapshot of the market at a specifc timestamp 
    def __init__(self, 
                 order_depths: Dict[str, OrderDepth], #current order books for each instrument(kelp or Rainforest Resin)
                 own_trades: Dict[str, List[Trade]], #Trades executed by this trading bot 
                 market_trades: Dict[str, List[Trade]], #all the trades in the market 
                 position: Dict[str, int], #current positions for kelp and rainforest resin
                 timestamp: int): #market time 
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.timestamp = timestamp
        
        
        self.features = {} #empty dictionary to store extracted features 
        
    def extract_features(self):
        for symbol, depth in self.order_depths.items():
            if not depth.buy_orders or not depth.sell_orders:
                continue
                
            best_bid = max(depth.buy_orders.keys()) #highest available buy price- the bid
            best_ask = min(depth.sell_orders.keys()) #lowest available sell price - the ask
            spread = best_ask - best_bid #bid ask spread implementation
            mid_price = (best_ask + best_bid) / 2 #middle price between the bid and the ask taken as a mean of the bid and the ask 
            
#calculates the market pressure by specific metrics 

            # Calculate buy and sell volume imbalance
            buy_volume = sum(depth.buy_orders.values()) #total volume of all buy orders
            sell_volume = sum(-v for v in depth.sell_orders.values()) #total volume of sell orders (converts sell orders to positive as sell orders are given as negatives)
            volume_imbalance = buy_volume / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0.5 #ratio of buy volume to total volume indicating the buying/selling pressure - sentiment analysis. Defaults to 0.5 as a neutral when no volume
            
            # stores the extracted features for analysis later
            self.features[symbol] = { 
                'mid_price': mid_price,
                'spread': spread,
                'volume_imbalance': volume_imbalance,
                'position': self.position.get(symbol, 0)
            }
        
        return self.features

#represents the complete trading system rate
class TradingState:
    def __init__(self, 
                 timestamp: int, #current maket time 
                 listings: Dict[str, int], #listings of trading instruments(kelp or rainforest resin )
                 order_depths: Dict[str, OrderDepth], #current order books 
                 own_trades: Dict[str, List[Trade]],  #recent trading activity of this algo 
                 market_trades: Dict[str, List[Trade]], #recent trading activity on the market 
                 position: Dict[str, int], #current holdings
                 observations: List[Observation] = None, 
                 traderData: str = ""): #serialised trader state
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations or []
        self.traderData = traderData
    
#creates a new observation from current state and adds it to history

    def add_observation(self):
        """Add current state as an observation"""
        observation = Observation(
            self.order_depths,
            self.own_trades,
            self.market_trades,
            self.position,
            self.timestamp
        )
        observation.extract_features()
        self.observations.append(observation)
        return observation

class Trader:
    def __init__(self):
        # Product position limits
        self.position_limits = {
            "RAINFOREST_RESIN": 50, #maximum allowed position for Rainforest Resin
            "KELP": 50 #maximum allowed position for Kelp 
        }
        
        # Store historical price data
        self.price_history = { #fixed length queues (100 most recently calculated fair prices)
            "RAINFOREST_RESIN": deque(maxlen=100), 
            "KELP": deque(maxlen=100)
        }
        
        # Store VWAP (Volume-Weighted Average Price) history
        self.vwap_history = { #Volume Weighted Average Price history (20 most recenly calculated VWAP points)
            "RAINFOREST_RESIN": deque(maxlen=20),
            "KELP": deque(maxlen=20)
        }
        
        # Dynamic spread adjustment based on volatility
        self.volatility = { #initial volatility estimates
            "RAINFOREST_RESIN": 1,
            "KELP": 2  # Start with higher volatility for Kelp as it fluctuates more
        }
        
        # Time window for analysis
        self.time_window = 15 #rolling window for analysis 
        
        # Order size multiplier - aggressively increased for maximum profits 
        self.order_size_multiplier = 100.0  # Increase order sizes by 100x
        
        # Liquidity consumption percentage - aggressively takes all available liquidity for maximum profits 
        self.liquidity_consumption = 0.98  # Take up to 98% of available liquidity
    
    def calculate_fair_value(self, symbol: str, order_depth: OrderDepth, min_vol: int = 0) -> float: #Calculate fair value based on order book and historical data
        if not order_depth.buy_orders or not order_depth.sell_orders:
            # If we have historical data, use the last price
            if self.price_history[symbol]:
                return self.price_history[symbol][-1]
            return 0  #fair value returns to 0 if we have no data
        
       
        best_ask = min(order_depth.sell_orders.keys()) #find the ask price (lowest selling price)
        best_bid = max(order_depth.buy_orders.keys()) #find the bid price (highest buying price)
        mid_price = (best_ask + best_bid) / 2 #calculate the mean of the bid and the ask to find a middle price 
        
        # Method 2: Volume-weighted mid-price (similar to the tutorial)
        if min_vol > 0: 
            filtered_asks = [price for price in order_depth.sell_orders.keys() 
                             if abs(order_depth.sell_orders[price]) >= min_vol] #filters out sell orders with less than the minimum volume, hence ignoring the noise 
            filtered_bids = [price for price in order_depth.buy_orders.keys()  
                             if abs(order_depth.buy_orders[price]) >= min_vol] #filters out buy orders with less than the minimum volume, hence ignoring the noise 
            
            if filtered_asks and filtered_bids: 
                mm_ask = min(filtered_asks)
                mm_bid = max(filtered_bids)
                vw_mid_price = (mm_ask + mm_bid) / 2 #calculates the mid-price of the volume-filtered bid and ask to find a middle price without being affected by the noise
                
                # Calculate VWAP for recent trades
                if len(self.vwap_history[symbol]) > 0: 
                    vwap = sum([item["vwap"] * item["vol"] for item in self.vwap_history[symbol]]) / \
                           sum([item["vol"] for item in self.vwap_history[symbol]]) #calculates the Volume-Weighted Average Price (VWAP) from historical data: Σ(price * volume) / Σ(volume)
                    
                    # Blend mid-price and VWAP
                    fair_value = 0.7 * vw_mid_price + 0.3 * vwap #fair value is calculated by a 70% weighting of the current filtered mid price and 30% weight of the historical filtered average price 
                else:
                    fair_value = vw_mid_price 
            else:
                fair_value = mid_price #if the filtered data is insufficient, then it fallsback on the simple middle price 
        else:
            fair_value = mid_price #if the filtered data is insufficient, then it fallsback on the simple middle price
        
        
        self.price_history[symbol].append(fair_value) #adds calculated fair price to historical price data 
        
        # Calculate new volume-weighted info
        if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0: 
            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid] #calculates total volume at best bid and best ask 
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] +  #calculates the current Volume-Weighted Average price by weighting prices by their volumes 
                    best_ask * order_depth.buy_orders[best_bid]) / volume
            
            self.vwap_history[symbol].append({"vol": volume, "vwap": vwap}) #stores both volume and Volume-Weighted Average price in history
        
        return fair_value
    
    def calculate_optimal_spread(self, symbol: str, fair_value: float) -> float: #determines the optimal bid-ask spread based on volatility 
        # Higher volatility -> wider spread
        base_spread = self.volatility[symbol] #base spread is equal to the volatility estimate 
        
        # Adjust spread based on position - wider when positions are large
        position = 0  #the position will be filled during run()
        position_factor = abs(position) / self.position_limits[symbol] #position factor is a ratio of the current position to the limit 
        position_adjustment = position_factor * 2 #as the positions are larger, position adjustment increases the spread 
        
        # Final spread calculation
        optimal_spread = base_spread + position_adjustment 
        
        return max(1, optimal_spread)  # Minimum spread of 1 to avoid negative returns 
    
    def update_volatility(self, symbol: str): #Update volatility estimate based on recent price movements
        if len(self.price_history[symbol]) > 10: #requires at least 10 historical data points 
            prices = list(self.price_history[symbol]) 
            price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))] #calculates the change in price 
            std_dev = np.std(price_changes) #calculates the standard deviation of price changes 
            
            
            self.volatility[symbol] = 0.9 * self.volatility[symbol] + 0.1 * std_dev #moving average for volatility using 0.9 x old volatility + 0.1 x standard deviation to smooth out the volatility and prevent overreaction to temporary market changes 
    
    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, #inventory management 
                           position: int, position_limit: int, symbol: str, 
                           buy_order_volume: int, sell_order_volume: int, 
                           fair_value: float, width: int) -> Tuple[int, int]:
        position_after_take = position + buy_order_volume - sell_order_volume #calculates what the position will be after all currently planned orders execute 
        fair = round(fair_value) #rounds to nearest integer price 
        fair_for_bid = math.floor(fair_value) #rounds down to make sure buy orders are more likely to execute 
        fair_for_ask = math.ceil(fair_value) #rounds up to make sure sell orders are more likely to execute
 
        buy_quantity = position_limit - (position + buy_order_volume)  #check how much buy capacity remains 
        sell_quantity = position_limit + (position - sell_order_volume) #check how much selling capacity remains 

        #reduces long positions
        if position_after_take > 0: #checks if algorithm is currently long 
            if fair_for_ask in order_depth.buy_orders.keys():  #checks if there are any sellers that are willing to sell exactly at calculated fair bid price - only execute if we get a fair price
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], #gives the available volume being offered by sellers at our targeted buy price - take absolute value as short values are negative
                                    math.ceil(position_after_take * self.order_size_multiplier)) #extremely aggressive order scaling
                sent_quantity = min(sell_quantity, clear_quantity) #min property to ensure we don't sell more than availability
                if sent_quantity > 0: #ensure sell order 
                    orders.append(Order(symbol, fair_for_ask, -abs(sent_quantity))) 
                    sell_order_volume += abs(sent_quantity)

        #reduces short positons 
        if position_after_take < 0: #checks if algorithm is currently short to prevent squeeze 
            if fair_for_bid in order_depth.sell_orders.keys(): #checks if there are any buyers that are willing to buy exactly at calculated fair ask price - only execute if we get a fair price
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), #completely eliminate short positons and potentiall go long with aggressive scaling 
                                    math.ceil(abs(position_after_take) * self.order_size_multiplier)) 
                sent_quantity = min(buy_quantity, clear_quantity) #min property to ensure we don't buy more than availability 
                if sent_quantity > 0: #ensure buy order 
                    orders.append(Order(symbol, fair_for_bid, abs(sent_quantity)))
                    buy_order_volume += abs(sent_quantity)
    
        return buy_order_volume, sell_order_volume
    
    def generate_orders(self, symbol: str, order_depth: OrderDepth, #defines a method to create trading orders with 4 key inputs: symbol, order_depth, position and take_width 
                       position: int, take_width: float) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0 #running counter of the all the buy volume we're generating in this function call
        sell_order_volume = 0 #running counter of all the sell volume we're generating in this function call 
        
        if not order_depth.buy_orders or not order_depth.sell_orders: #market quality check - checks that there are both sell and buy orders, ensures liquidity 
            return orders
        
        # Calculate fair value
        fair_value = self.calculate_fair_value(symbol, order_depth, min_vol=15) #calculates fair value with a minimum volume of 15
        
        
        self.update_volatility(symbol) # Update volatility estimate
        
        
        spread = self.calculate_optimal_spread(symbol, fair_value)# Calculate optimal spread
        
        
        best_ask = min(order_depth.sell_orders.keys())# Get best ask price from order book
        best_bid = max(order_depth.buy_orders.keys()) # Get best bid price from order book
        
        # Statistical arbitrage on buy side - if market is underprice compared to our fair value, take liquidity aggressively 
        if best_ask <= fair_value - take_width:
            ask_amount = -1 * order_depth.sell_orders[best_ask] #calculates available volume at best ask 
            quantity = min(math.ceil(ask_amount * self.liquidity_consumption), #aggression in taking advantage of opportunities, takes 98% of available volume 
                          self.position_limits[symbol] - position) #calculates remaining capacity before hitting our maximum position limits 
            if quantity > 0: 
                # Apply maximum order size based on position limit
                max_possible_order = max(1, self.position_limits[symbol] - position)  
                scaled_quantity = min(quantity * 100, max_possible_order) #scales quantity by 100x for extreme aggression
                orders.append(Order(symbol, best_ask, scaled_quantity)) #creates order at best ask price to take liquidity
                buy_order_volume += scaled_quantity #tracks the accumalated buy volume
        
        if best_bid >= fair_value + take_width: #Statistical arbitrage on sell side - if market is overpriced compared to our fair value, take liqduity aggressively 
            
            bid_amount = order_depth.buy_orders[best_bid] #calculates available volume at best bid 
            
            quantity = min(math.ceil(bid_amount * self.liquidity_consumption), #aggression in taking advantage of opportunities, takes 98% of available volume 
                          self.position_limits[symbol] + position) #calculates remaining capacity before hitting our maximum position limits 
            if quantity > 0:
                # Apply maximum order size based on position limit
                max_possible_order = max(1, self.position_limits[symbol] + position)
                scaled_quantity = min(quantity * 100, max_possible_order) #scales quantity by 100x for extreme aggression 
                orders.append(Order(symbol, best_bid, -1 * scaled_quantity)) #creates order at best bid price to take liquidity 
                sell_order_volume += scaled_quantity #tracks the accumalated sell volume 
        
        # Position management - try to clear inventory at fair price
        buy_order_volume, sell_order_volume = self.clear_position_order( #occurs after new statistical arbitrage positions but before market making orders
            orders, order_depth, position, self.position_limits[symbol], #manages positons to reduce risk
            symbol, buy_order_volume, sell_order_volume, fair_value, 1 #tries to exit positions are the calculated fair value 
        )
        
        # Market making - place limit orders around fair value
        # Find prices just outside existing orders 
        ask_prices = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1] #market making price filter, retrieves all price levels where someone is willing to sell with the condition that they are at least 1 unit above calculated fair value 
        bid_prices = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1] #market making price filter, retrieves all price levels where someone is willing to buy with the condition that they are at least 1 unit below calculated fair value 
        
        best_ask_to_place = min(ask_prices) - 1 if ask_prices else math.ceil(fair_value + spread) #undercuts existing order by placing sell order 1 under lowest price for queue priority 
        best_bid_to_place = max(bid_prices) + 1 if bid_prices else math.floor(fair_value - spread) #undercuts existing order by placing buy order 1 over highest price for queue priority 
        
        # Maximize base size - use almost the entire position limit
        base_size = math.ceil(self.position_limits[symbol] * 0.95)  # Use 95% of position limit 
        
        # Adjust order size based on position
        position_ratio = abs(position) / self.position_limits[symbol] #indicate how close to position limits we are 
        
        # Maximum aggression on order sizing regardless of position
        # Always maintain large order sizes even at max position
        buy_size_adjustment = max(0.7, 1 - (position_ratio * 0.3 if position > 0 else 0)) 
        sell_size_adjustment = max(0.7, 1 - (position_ratio * 0.3 if position < 0 else 0))
        
        # Place buy order with increased size
        buy_quantity = min(
            math.floor(base_size * buy_size_adjustment * self.order_size_multiplier),
            self.position_limits[symbol] - (position + buy_order_volume) #order caps at remaing capacity
        )
        
        if buy_quantity > 0:
            orders.append(Order(symbol, best_bid_to_place, buy_quantity))
        
        # Place sell order with increased size
        sell_quantity = min(
            math.floor(base_size * sell_size_adjustment * self.order_size_multiplier),
            self.position_limits[symbol] + (position - sell_order_volume)
        )
        
        if sell_quantity > 0:
            orders.append(Order(symbol, best_ask_to_place, -sell_quantity))
        
        return orders #return complete list of orders for the symbol
    
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]: 
        """Main method to generate trading decisions"""
        result = {} #creating an empty dictionary that will eventually hold all the orders to be executed 
        
        # Load trader data if available
        if state.traderData:
            try:
                trader_data = jsonpickle.decode(state.traderData) #converts stored JSON string back into Python Objects 
                self.price_history = trader_data.get("price_history", self.price_history) 
                self.vwap_history = trader_data.get("vwap_history", self.vwap_history)
                self.volatility = trader_data.get("volatility", self.volatility)
            except Exception:
                # If there's an error decoding, just continue with current state
                pass
        
        # Ultra-aggressive strategy parameters, more aggressive arbitrage 
        kelp_take_width = 0.5  # Extremely aggressive for volatile product, 
        resin_take_width = 0.3  # Extremely aggressive for stable product 
        
        # Generate orders for each product
        if "KELP" in state.order_depths:
            kelp_position = state.position.get("KELP", 0) #gets current position 
            kelp_orders = self.generate_orders( 
                "KELP", 
                state.order_depths["KELP"], #call order generation with product-specific parameters 
                kelp_position, 
                kelp_take_width
            )
            result["KELP"] = kelp_orders #stores in dictionary 
        
        if "RAINFOREST_RESIN" in state.order_depths:
            resin_position = state.position.get("RAINFOREST_RESIN", 0)
            resin_orders = self.generate_orders(
                "RAINFOREST_RESIN", 
                state.order_depths["RAINFOREST_RESIN"], #call order generation with product-specific parameters. 
                resin_position, 
                resin_take_width
            )
            result["RAINFOREST_RESIN"] = resin_orders #stores in dictionary 
        
        # Save state for next iteration
        trader_data = {
            "price_history": self.price_history,
            "vwap_history": self.vwap_history,
            "volatility": self.volatility
        }
        
        trader_data_json = jsonpickle.encode(trader_data)
        
        # No conversions in this example
        conversions = 0
        logger.flush(state, result, conversions, trader_data_json)
        
        return result, conversions, trader_data_json
