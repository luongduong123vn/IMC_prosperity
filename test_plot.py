import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np

# Read the data
market_data = pd.read_csv('./round_1/round_1_data/prices_round_1_day_0.csv')

# Filter for KELP data
kelp_data = market_data[market_data['product'] == 'KELP']

# Print data info for debugging
print("Data Info:")
print(f"Number of rows: {len(kelp_data)}")
print(f"Timestamp range: {kelp_data['timestamp'].min()} to {kelp_data['timestamp'].max()}")
print(f"Columns: {kelp_data.columns.tolist()}")

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)  # Make room for the sliders

# Convert timestamps to numpy array
timestamps = np.array(kelp_data['timestamp'])
print(f"\nTimestamps array shape: {timestamps.shape}")
print(f"Timestamps dtype: {timestamps.dtype}")

# Plot bid prices
bid1_line, = ax.plot(timestamps, kelp_data['bid_price_1'], label='Bid Price 1', color='blue')
bid2_line, = ax.plot(timestamps, kelp_data['bid_price_2'], label='Bid Price 2', color='lightblue')
bid3_line, = ax.plot(timestamps, kelp_data['bid_price_3'], label='Bid Price 3', color='cyan')

# Plot ask prices
ask1_line, = ax.plot(timestamps, kelp_data['ask_price_1'], label='Ask Price 1', color='red')
ask2_line, = ax.plot(timestamps, kelp_data['ask_price_2'], label='Ask Price 2', color='pink')
ask3_line, = ax.plot(timestamps, kelp_data['ask_price_3'], label='Ask Price 3', color='salmon')

# Add labels and title
plt.xlabel('Timestamp')
plt.ylabel('Price')
plt.title('KELP Bid and Ask Prices')
plt.legend()
plt.grid(True)

# Create start time slider
ax_start = plt.axes([0.25, 0.1, 0.65, 0.03])
start_slider = Slider(
    ax=ax_start,
    label='Start Time',
    valmin=float(timestamps.min()),
    valmax=float(timestamps.max()),
    valinit=float(timestamps.min()),
    valstep=100.0
)

# Create end time slider
ax_end = plt.axes([0.25, 0.05, 0.65, 0.03])
end_slider = Slider(
    ax=ax_end,
    label='End Time',
    valmin=float(timestamps.min()),
    valmax=float(timestamps.max()),
    valinit=float(timestamps.max()),
    valstep=100.0
)

# Create reset button
reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_ax, 'Reset', color='lightgray', hovercolor='0.975')

def update(val):
    # Get the current slider values
    start_time = start_slider.val
    end_time = end_slider.val
    
    # Ensure start time is not greater than end time
    if start_time > end_time:
        if val == start_slider.val:  # If start slider was moved
            end_slider.set_val(start_time)
        else:  # If end slider was moved
            start_slider.set_val(end_time)
    
    print(f"\nSlider values: start={start_time}, end={end_time}")
    
    # Update the x-axis limits
    ax.set_xlim(start_time, end_time)
    
    # Update the plot
    fig.canvas.draw_idle()

def reset(event):
    print("\nResetting view to full range")
    # Reset the sliders to initial values
    start_slider.set_val(float(timestamps.min()))
    end_slider.set_val(float(timestamps.max()))
    # Reset the view to show all data
    ax.set_xlim(timestamps.min(), timestamps.max())
    fig.canvas.draw_idle()

# Register the update function with both sliders
start_slider.on_changed(update)
end_slider.on_changed(update)

# Register the reset function with the button
reset_button.on_clicked(reset)

# Show the plot
plt.show() 