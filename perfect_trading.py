import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Order:
    type: OrderType
    price: int
    quantity: int

@dataclass
class OrderDepth:
    buy_orders: Dict[int, int]
    sell_orders: Dict[int, int]

class PerfectTrader:
    def __init__(self):
        self.data = None
        self.current_timestamp = 0
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess the log data."""
        try:
            self.data = pd.read_csv('round_2/round_2_data/log_data_1.csv', sep=';')
            logger.info(f"Loaded data with shape: {self.data.shape}")
            
            # Convert timestamp to integer
            self.data['timestamp'] = self.data['timestamp'].astype(int)
            
            # Sort by timestamp
            self.data = self.data.sort_values('timestamp')
            
            # Initialize current timestamp
            self.current_timestamp = self.data['timestamp'].min()
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def _get_next_price(self, product: str) -> Optional[float]:
        """Get the next price for a product."""
        future_data = self.data[
            (self.data['product'] == product) & 
            (self.data['timestamp'] > self.current_timestamp)
        ]
        if not future_data.empty:
            return future_data.iloc[0]['mid_price']
        return None
        
    def _get_current_order_depth(self, product: str) -> OrderDepth:
        """Get current order depth for a product."""
        current_data = self.data[
            (self.data['product'] == product) & 
            (self.data['timestamp'] == self.current_timestamp)
        ]
        
        if current_data.empty:
            return OrderDepth(buy_orders={}, sell_orders={})
            
        row = current_data.iloc[0]
        
        # Create buy orders dictionary
        buy_orders = {}
        for i in range(1, 4):
            price_col = f'bid_price_{i}'
            volume_col = f'bid_volume_{i}'
            if pd.notna(row[price_col]) and pd.notna(row[volume_col]):
                buy_orders[int(row[price_col])] = int(row[volume_col])
                
        # Create sell orders dictionary
        sell_orders = {}
        for i in range(1, 4):
            price_col = f'ask_price_{i}'
            volume_col = f'ask_volume_{i}'
            if pd.notna(row[price_col]) and pd.notna(row[volume_col]):
                sell_orders[int(row[price_col])] = int(row[volume_col])
                
        return OrderDepth(buy_orders=buy_orders, sell_orders=sell_orders)
        
    def generate_optimal_trades(self, product: str) -> List[Order]:
        """Generate optimal trades based on future price information."""
        try:
            next_price = self._get_next_price(product)
            if next_price is None:
                return []
                
            current_depth = self._get_current_order_depth(product)
            trades = []
            
            # Check if we should buy
            best_ask = min(current_depth.sell_orders.keys()) if current_depth.sell_orders else None
            if best_ask and best_ask < next_price:
                quantity = current_depth.sell_orders[best_ask]
                trades.append(Order(type=OrderType.BUY, price=best_ask, quantity=quantity))
                
            # Check if we should sell
            best_bid = max(current_depth.buy_orders.keys()) if current_depth.buy_orders else None
            if best_bid and best_bid > next_price:
                quantity = current_depth.buy_orders[best_bid]
                trades.append(Order(type=OrderType.SELL, price=best_bid, quantity=quantity))
                
            return trades
            
        except Exception as e:
            logger.error(f"Error generating trades: {str(e)}")
            return []
            
    def calculate_pnl(self, product: str, trades: List[Order]) -> float:
        """Calculate PnL for given trades."""
        try:
            next_price = self._get_next_price(product)
            if next_price is None:
                return 0.0
                
            pnl = 0.0
            for trade in trades:
                if trade.type == OrderType.BUY:
                    pnl += (next_price - trade.price) * trade.quantity
                else:
                    pnl += (trade.price - next_price) * trade.quantity
                    
            return pnl
            
        except Exception as e:
            logger.error(f"Error calculating PnL: {str(e)}")
            return 0.0
            
    def advance_timestamp(self):
        """Advance to the next timestamp."""
        self.current_timestamp += 100  # Assuming 100ms intervals
        
def main():
    try:
        trader = PerfectTrader()
        total_pnl = 0.0
        
        # Get unique products
        products = trader.data['product'].unique()
        
        # Process each timestamp
        while trader.current_timestamp <= trader.data['timestamp'].max():
            logger.info(f"Processing timestamp: {trader.current_timestamp}")
            
            for product in products:
                trades = trader.generate_optimal_trades(product)
                if trades:
                    pnl = trader.calculate_pnl(product, trades)
                    total_pnl += pnl
                    logger.info(f"Product: {product}, Trades: {len(trades)}, PnL: {pnl}")
                    
            trader.advance_timestamp()
            
        logger.info(f"Total PnL: {total_pnl}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 