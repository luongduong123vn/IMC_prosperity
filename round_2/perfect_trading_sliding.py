import pandas as pd
import numpy as np
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Any, Tuple
import json
from collections import defaultdict

class PerfectTraderSliding:
    def __init__(self):
        self.log_data = pd.read_csv('round_2/round_2_data/log_data_1.csv', sep=';')
        self.log_data['timestamp'] = pd.to_numeric(self.log_data['timestamp'], errors='coerce')
        self.log_data = self.log_data.dropna(subset=['timestamp'])
        self.log_data['timestamp'] = self.log_data['timestamp'].astype(int)
        
        # Position limits for each product
        self.position_limits = {
            "JAMS": 350,
            "CROISSANTS": 250,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "DJEMBES": 60
        }
        
        # Get unique timestamps and products
        self.timestamps = sorted(self.log_data['timestamp'].unique())
        self.products = self.log_data['product'].unique()
        print(f"Loaded {len(self.timestamps)} timestamps for {len(self.products)} products")

    def _get_current_order_depth(self, timestamp: int, product: str) -> OrderDepth:
        current_data = self.log_data[
            (self.log_data['timestamp'] == timestamp) & 
            (self.log_data['product'] == product)
        ]
        
        if current_data.empty:
            return None
            
        order_depth = OrderDepth()
        
        # Add buy orders
        for i in range(1, 4):
            price_col = f'bid_price_{i}'
            volume_col = f'bid_volume_{i}'
            if price_col in current_data.columns and volume_col in current_data.columns:
                price = current_data[price_col].iloc[0]
                volume = current_data[volume_col].iloc[0]
                if not pd.isna(price) and not pd.isna(volume) and volume > 0:
                    order_depth.buy_orders[price] = volume
        
        # Add sell orders
        for i in range(1, 4):
            price_col = f'ask_price_{i}'
            volume_col = f'ask_volume_{i}'
            if price_col in current_data.columns and volume_col in current_data.columns:
                price = current_data[price_col].iloc[0]
                volume = current_data[volume_col].iloc[0]
                if not pd.isna(price) and not pd.isna(volume) and volume > 0:
                    order_depth.sell_orders[price] = volume
            
        return order_depth
    
    def _get_next_price(self, timestamp: int, product: str) -> float:
        next_data = self.log_data[
            (self.log_data['timestamp'] > timestamp) & 
            (self.log_data['product'] == product)
        ].sort_values('timestamp').head(1)
        
        if next_data.empty or 'mid_price' not in next_data.columns:
            return None
            
        return next_data['mid_price'].iloc[0]

    def _get_all_possible_trades(self, timestamp: int, product: str, current_position: int) -> List[Tuple[float, int, float]]:
        """Get all possible trades at a timestamp considering position limits."""
        trades = []
        try:
            order_depth = self._get_current_order_depth(timestamp, product)
            if order_depth is None:
                return trades
                
            next_price = self._get_next_price(timestamp, product)
            if next_price is None:
                return trades
            
            position_limit = self.position_limits[product]
            
            # Find best bid and ask
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                
                # Calculate profits
                buy_profit = next_price - best_ask
                sell_profit = best_bid - next_price
                
                # Add possible buy trades
                if buy_profit > 0:
                    max_buy = min(
                        position_limit - current_position,
                        order_depth.sell_orders[best_ask]
                    )
                    if max_buy > 0:
                        trades.append((best_ask, max_buy, buy_profit * max_buy))
                
                # Add possible sell trades
                if sell_profit > 0:
                    max_sell = min(
                        position_limit + current_position,
                        order_depth.buy_orders[best_bid]
                    )
                    if max_sell > 0:
                        trades.append((best_bid, -max_sell, sell_profit * max_sell))
                        
        except Exception as e:
            print(f"Error getting trades for {product} at {timestamp}: {str(e)}")
            
        return trades

    def optimize_trades(self) -> Dict[int, tuple]:
        """Optimize trades using sliding window approach."""
        optimal_trades = {}
        positions = {product: 0 for product in self.products}
        window_size = 50  # Look ahead window size
        
        # Process timestamps in windows
        for start_idx in range(0, len(self.timestamps), window_size // 2):
            end_idx = min(start_idx + window_size, len(self.timestamps))
            window_timestamps = self.timestamps[start_idx:end_idx]
            
            print(f"Processing window {start_idx}-{end_idx} ({len(window_timestamps)} timestamps)")
            
            # Store best trades and their cumulative PnL for the window
            dp = defaultdict(lambda: defaultdict(lambda: float('-inf')))
            prev_trade = {}
            
            # Initialize with current positions
            dp[0][tuple(sorted(positions.items()))] = 0
            
            # For each timestamp in window
            for t_idx, timestamp in enumerate(window_timestamps):
                current_positions = list(dp[t_idx].keys())
                
                # For each current position state
                for pos_state in current_positions:
                    pos_dict = dict(pos_state)
                    current_pnl = dp[t_idx][pos_state]
                    
                    # Try trading each product
                    for product in self.products:
                        current_pos = pos_dict[product]
                        possible_trades = self._get_all_possible_trades(timestamp, product, current_pos)
                        
                        # For each possible trade
                        for price, quantity, profit in possible_trades:
                            # Create new position state
                            new_pos_dict = pos_dict.copy()
                            new_pos_dict[product] = current_pos + quantity
                            new_pos_state = tuple(sorted(new_pos_dict.items()))
                            
                            # Update if better PnL found
                            new_pnl = current_pnl + profit
                            if new_pnl > dp[t_idx + 1][new_pos_state]:
                                dp[t_idx + 1][new_pos_state] = new_pnl
                                prev_trade[(t_idx + 1, new_pos_state)] = (timestamp, product, price, quantity)
                    
                    # Also consider not trading
                    no_trade_state = pos_state
                    if current_pnl > dp[t_idx + 1][no_trade_state]:
                        dp[t_idx + 1][no_trade_state] = current_pnl
                        prev_trade[(t_idx + 1, no_trade_state)] = None
            
            # Find best final state in this window
            best_final_pnl = float('-inf')
            best_final_state = None
            
            for final_state in dp[len(window_timestamps)].keys():
                final_pnl = dp[len(window_timestamps)][final_state]
                if final_pnl > best_final_pnl:
                    best_final_pnl = final_pnl
                    best_final_state = final_state
            
            # Reconstruct optimal trades for first half of window
            current_state = best_final_state
            window_trades = {}
            
            for t in range(len(window_timestamps), 0, -1):
                trade_info = prev_trade.get((t, current_state))
                if trade_info is not None:
                    timestamp, product, price, quantity = trade_info
                    window_trades[timestamp] = (product, price, quantity)
                
                # Update current state based on previous trade
                if trade_info is not None:
                    pos_dict = dict(current_state)
                    pos_dict[product] -= quantity
                    current_state = tuple(sorted(pos_dict.items()))
            
            # Only keep trades from first half of window (except for last window)
            cutoff_idx = len(window_timestamps) if end_idx == len(self.timestamps) else window_size // 2
            window_trades = {t: trade for t, trade in window_trades.items() 
                           if self.timestamps.index(t) - start_idx < cutoff_idx}
            
            # Update optimal trades and positions
            optimal_trades.update(window_trades)
            
            # Update positions for next window
            if window_trades:
                last_timestamp = max(window_trades.keys())
                for timestamp in sorted(window_trades.keys()):
                    if timestamp <= last_timestamp:
                        product, _, quantity = window_trades[timestamp]
                        positions[product] += quantity
            
            print(f"Window PnL: {best_final_pnl}")
        
        return optimal_trades

    def save_trades(self, optimal_trades: Dict[int, tuple]):
        """Save trades to JSON file with string timestamps."""
        # Convert trades to JSON format with string timestamps
        trades_json = {
            str(timestamp): {  # Convert to string
                "product": trade[0],
                "price": int(trade[1]),  # Ensure Python int
                "quantity": int(trade[2])  # Ensure Python int
            }
            for timestamp, trade in optimal_trades.items()
        }
        
        # Save to JSON file
        with open('round_2/optimal_trades_sliding.json', 'w') as f:
            json.dump(trades_json, f, indent=2)
        
        print(f"\nGenerated {len(optimal_trades)} trades")
        print("Saved to optimal_trades_sliding.json")

if __name__ == "__main__":
    trader = PerfectTraderSliding()
    optimal_trades = trader.optimize_trades()
    trader.save_trades(optimal_trades) 