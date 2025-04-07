import itertools

def find_best_trades_start_end_d(exchange_matrix, products, num_trades=4, max_sequences=5):
    n = len(products)
    product_index = {p: i for i, p in enumerate(products)}
    d_index = product_index['d']
    
    profitable_sequences = []
    
    # Generate all possible sequences of length num_trades + 1 that start and end with 'd'
    # The sequence length is num_trades + 1 because num_trades is the number of exchanges,
    # which connects num_trades + 1 products (including start and end)
    for intermediate in itertools.product(products, repeat=num_trades - 1):
        # The full sequence is 'd' followed by the intermediate products and ending with 'd'
        sequence = ('d',) + intermediate + ('d',)
        # Calculate the product of exchange rates along the sequence
        product = 1.0
        valid = True
        for i in range(len(sequence) - 1):
            from_p = sequence[i]
            to_p = sequence[i + 1]
            from_idx = product_index[from_p]
            to_idx = product_index[to_p]
            rate = exchange_matrix[from_idx][to_idx]
            product *= rate
            if product == 0:  # Avoid zero multiplication (though not present here)
                valid = False
                break
        if valid and product > 1.0:
            profitable_sequences.append((sequence, product))
    
    # Sort the profitable sequences by product in descending order
    profitable_sequences.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top max_sequences sequences
    top_sequences = profitable_sequences[:max_sequences]
    
    return top_sequences

# Given data
products = ['a', 'b', 'c', 'd']
exchange_matrix = [
    [1, 1.45, 0.52, 0.72],
    [0.7, 1, 0.31, 0.48],
    [1.95, 3.1, 1, 1.49],
    [1.34, 1.98, 0.64, 1]
]

# Find the best sequences starting and ending with 'd' with exactly 4 trades
best_trades = find_best_trades_start_end_d(exchange_matrix, products, num_trades=5, max_sequences=6)

# Print the results
for i, (sequence, profit) in enumerate(best_trades, 1):
    print(f"{i}. Sequence: {' → '.join(sequence)} | Profit multiplier: {profit:.4f}")