import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Define the container data
containers = [
    {'id': 0, 'multiplier': 10, 'inhabitants': 1},
    {'id': 1, 'multiplier': 80, 'inhabitants': 6},
    {'id': 2, 'multiplier': 37, 'inhabitants': 13},
    {'id': 3, 'multiplier': 17, 'inhabitants': 1},
    {'id': 4, 'multiplier': 90, 'inhabitants': 10},
    {'id': 5, 'multiplier': 31, 'inhabitants': 2},
    {'id': 6, 'multiplier': 50, 'inhabitants': 4},
    {'id': 7, 'multiplier': 20, 'inhabitants': 2},
    {'id': 8, 'multiplier': 73, 'inhabitants': 4},
    {'id': 9, 'multiplier': 89, 'inhabitants': 8}
]

BASE_TREASURE = 10000
TOTAL_PLAYERS = 2000
SECOND_CONTAINER_COST = 50000
NUM_SIMULATIONS = 200

def distribute_players_by_strategy():
    """Randomly distribute players among the three strategies"""
    strategy1_percent = random.random() * 0 + 0  # 20-80% of players
    strategy2_percent = random.random() * (1 - strategy1_percent) * 0  # 0-80% of remaining players
    strategy3_percent = 1 - strategy1_percent - strategy2_percent  # Rest are random players
    
    return {
        'strategy1': int(TOTAL_PLAYERS * strategy1_percent),
        'strategy2': int(TOTAL_PLAYERS * strategy2_percent),
        'strategy3': TOTAL_PLAYERS - int(TOTAL_PLAYERS * strategy1_percent) - int(TOTAL_PLAYERS * strategy2_percent)
    }

def strategy1(container_choices):
    """Maximize expected value considering both inhabitants and popularity"""
    expected_values = []
    total_choices = sum(container_choices)
    
    for container in containers:
        total_treasure = BASE_TREASURE * container['multiplier']
        # Calculate the percentage this container was chosen out of all choices
        popularity = container_choices[container['id']] / total_choices if total_choices > 0 else 0
        # Calculate expected value considering both inhabitants and popularity
        divisor = container['inhabitants'] + popularity * 100  # Convert popularity to percentage
        expected_values.append({
            'id': container['id'],
            'ev': total_treasure / divisor
        })
    
    # Sort containers by expected value
    expected_values_sorted = sorted(expected_values, key=lambda x: x['ev'], reverse=True)
    
    # Choose the container with highest expected value for free first choice
    first_choice = expected_values_sorted[0]['id']
    
    # For second choice, exclude the first choice and find best remaining option
    second_choice_evs = [{'id': container['id'], 'ev': container['ev'] - SECOND_CONTAINER_COST} 
                         for container in expected_values if container['id'] != first_choice]
    
    # Sort by net expected value
    second_choice_evs.sort(key=lambda x: x['ev'], reverse=True)
    
    # Always choose a second container, even if negative EV
    second_choice = second_choice_evs[0]['id']
    
    return [first_choice, second_choice]

def strategy2():
    """Naively divide by inhabitants only"""
    expected_values = []
    
    for container in containers:
        total_treasure = BASE_TREASURE * container['multiplier']
        expected_values.append({
            'id': container['id'],
            'ev': total_treasure / container['inhabitants']
        })
    
    # Sort containers by expected value
    expected_values_sorted = sorted(expected_values, key=lambda x: x['ev'], reverse=True)
    
    # Choose the container with highest expected value for free first choice
    first_choice = expected_values_sorted[0]['id']
    
    # For second choice, exclude the first choice
    second_choice_evs = [{'id': container['id'], 'ev': container['ev'] - SECOND_CONTAINER_COST} 
                         for container in expected_values if container['id'] != first_choice]
    
    # Sort by net expected value
    second_choice_evs.sort(key=lambda x: x['ev'], reverse=True)
    
    # Always choose a second container, even if negative EV
    second_choice = second_choice_evs[0]['id']
    
    return [first_choice, second_choice]

def strategy3():
    """Choose randomly"""
    first_choice = random.randint(0, len(containers) - 1)
    
    # Second choice must be different from first
    second_choice = random.randint(0, len(containers) - 1)
    while second_choice == first_choice:
        second_choice = random.randint(0, len(containers) - 1)
        
    return [first_choice, second_choice]

def run_simulation():
    """Run the simulation"""
    # Distribute players by strategy
    player_distribution = distribute_players_by_strategy()
    print("Player distribution:", player_distribution)
    
    # Initialize container choices counter
    container_choices = [0] * len(containers)
    
    # First pass: have all players make initial choices to establish baseline popularity
    # Strategy 1 players (initial guess)
    for i in range(player_distribution['strategy1']):
        initial_choices = strategy2()  # Start with strategy2 for initial estimates
        container_choices[initial_choices[0]] += 1
        container_choices[initial_choices[1]] += 1
    
    # Strategy 2 players
    for i in range(player_distribution['strategy2']):
        choices = strategy2()
        container_choices[choices[0]] += 1
        container_choices[choices[1]] += 1
    
    # Strategy 3 players
    for i in range(player_distribution['strategy3']):
        choices = strategy3()
        container_choices[choices[0]] += 1
        container_choices[choices[1]] += 1
    
    # Second pass: Strategy 1 players update based on current choices
    strategy1_choices = []
    for i in range(player_distribution['strategy1']):
        choices = strategy1(container_choices)
        strategy1_choices.append(choices)
    
    # Reset and recount with updated choices
    container_choices = [0] * len(containers)
    
    # Strategy 1 players (updated)
    for choices in strategy1_choices:
        container_choices[choices[0]] += 1
        container_choices[choices[1]] += 1
    
    # Strategy 2 players
    for i in range(player_distribution['strategy2']):
        choices = strategy2()
        container_choices[choices[0]] += 1
        container_choices[choices[1]] += 1
    
    # Strategy 3 players
    for i in range(player_distribution['strategy3']):
        choices = strategy3()
        container_choices[choices[0]] += 1
        container_choices[choices[1]] += 1
    
    print("Final container choices distribution:", container_choices)
    
    # Calculate actual expected values based on final distribution
    total_choices = sum(container_choices)
    expected_values = []
    
    for index, container in enumerate(containers):
        total_treasure = BASE_TREASURE * container['multiplier']
        popularity = container_choices[index] / total_choices if total_choices > 0 else 0
        divisor = container['inhabitants'] + popularity * 100  # Convert to percentage
        ev = total_treasure / divisor
        
        expected_values.append({
            'id': container['id'],
            'multiplier': container['multiplier'],
            'inhabitants': container['inhabitants'],
            'choices': container_choices[index],
            'popularity': popularity * 100,  # Display as percentage
            'divisor': divisor,
            'rawEV': ev,
            'firstChoiceEV': ev,
            'secondChoiceEV': ev - SECOND_CONTAINER_COST
        })
    
    # Find optimal pair of containers
    best_total_ev = float('-inf')
    best_combo = (0, 0)
    
    for i in range(len(containers)):
        for j in range(len(containers)):
            if i != j:  # Ensure different containers
                first_ev = expected_values[i]['firstChoiceEV']
                second_ev = expected_values[j]['secondChoiceEV']
                total_ev = first_ev + second_ev
                if total_ev > best_total_ev:
                    best_total_ev = total_ev
                    best_combo = (i, j)
    
    # Return the optimal choices
    return {
        'firstChoice': best_combo[0],
        'secondChoice': best_combo[1],
        'allContainerEVs': expected_values,
        'containerChoices': container_choices,
        'totalEV': best_total_ev
    }

def visualize_expected_values(ev_data, optimal_combo=None):
    """Create visualizations of the expected values for containers"""
    # Create a DataFrame for easier plotting
    df = pd.DataFrame(ev_data)
    
    # Set up the figure and axes
    fig, axs = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 1]})
    
    # 1. First Choice EV Visualization
    ax1 = axs[0]
    sns.barplot(x='id', y='firstChoiceEV', data=df, ax=ax1, palette='viridis')
    ax1.set_title('Expected Value for First Container Choice (Free)', fontsize=16)
    ax1.set_xlabel('Container ID', fontsize=14)
    ax1.set_ylabel('Expected Value', fontsize=14)
    
    if optimal_combo:
        # Highlight the optimal first choice
        bar_color = ax1.patches[optimal_combo[0]].get_facecolor()
        ax1.patches[optimal_combo[0]].set_facecolor('red')
        ax1.patches[optimal_combo[0]].set_edgecolor('black')
        ax1.patches[optimal_combo[0]].set_linewidth(2)
    
    # Add value labels on top of each bar
    for i, p in enumerate(ax1.patches):
        ax1.annotate(f'{p.get_height():.0f}', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='bottom', fontsize=10)
    
    # 2. Second Choice EV Visualization (after paying cost)
    ax2 = axs[1]
    sns.barplot(x='id', y='secondChoiceEV', data=df, ax=ax2, palette='viridis')
    ax2.set_title(f'Expected Value for Second Container Choice (After {SECOND_CONTAINER_COST} Cost)', fontsize=16)
    ax2.set_xlabel('Container ID', fontsize=14)
    ax2.set_ylabel('Expected Value (After Cost)', fontsize=14)
    
    if optimal_combo:
        # Highlight the optimal second choice
        bar_color = ax2.patches[optimal_combo[1]].get_facecolor()
        ax2.patches[optimal_combo[1]].set_facecolor('red')
        ax2.patches[optimal_combo[1]].set_edgecolor('black')
        ax2.patches[optimal_combo[1]].set_linewidth(2)
    
    # Add value labels on top of each bar
    for i, p in enumerate(ax2.patches):
        ax2.annotate(f'{p.get_height():.0f}', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='bottom', fontsize=10)
    
    # Add a horizontal line at y=0 for second choice to show break-even point
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax2.text(0, 0, 'Break-even point', fontsize=10, va='bottom', ha='left', alpha=0.7)
    
    plt.tight_layout()
    
    # 3. Create a third visualization showing container properties
    fig2, ax3 = plt.subplots(figsize=(14, 6))
    
    # Create a DataFrame with container properties
    prop_df = pd.DataFrame([
        {'id': i, 'multiplier': c['multiplier'], 'inhabitants': c['inhabitants'], 
         'popularity': next(item['popularity'] for item in ev_data if item['id'] == i)}
        for i, c in enumerate(containers)
    ])
    
    # Create a bubble chart where:
    # - x-axis is inhabitants
    # - y-axis is multiplier
    # - size is popularity
    # - color gradient is first choice EV
    scatter = ax3.scatter(prop_df['inhabitants'], prop_df['multiplier'], 
                         s=prop_df['popularity'] * 10, # Scale for better visibility
                         c=[item['firstChoiceEV'] for item in ev_data],
                         cmap='viridis', alpha=0.7)
    
    # Add container IDs as labels
    for i, row in prop_df.iterrows():
        ax3.annotate(f"{row['id']}", 
                    (row['inhabitants'], row['multiplier']),
                    xytext=(5, 5), textcoords='offset points')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('First Choice Expected Value')
    
    # Set labels and title
    ax3.set_xlabel('Number of Inhabitants', fontsize=14)
    ax3.set_ylabel('Treasure Multiplier', fontsize=14)
    ax3.set_title('Container Properties Overview', fontsize=16)
    ax3.grid(True, alpha=0.3)
    
    if optimal_combo:
        # Highlight optimal containers
        first_choice = prop_df[prop_df['id'] == optimal_combo[0]].iloc[0]
        second_choice = prop_df[prop_df['id'] == optimal_combo[1]].iloc[0]
        
        ax3.scatter(first_choice['inhabitants'], first_choice['multiplier'], 
                   s=first_choice['popularity'] * 10,
                   color='red', edgecolor='black', linewidth=2)
        
        ax3.scatter(second_choice['inhabitants'], second_choice['multiplier'], 
                   s=second_choice['popularity'] * 10,
                   color='orange', edgecolor='black', linewidth=2)
        
        # Add a legend for optimal choices
        ax3.scatter([], [], s=100, color='red', edgecolor='black', linewidth=2, label=f'First Choice (Container {optimal_combo[0]})')
        ax3.scatter([], [], s=100, color='orange', edgecolor='black', linewidth=2, label=f'Second Choice (Container {optimal_combo[1]})')
        ax3.legend()
    
    plt.tight_layout()
    
    return fig, fig2

def main():
    # Run multiple simulations to get a more robust result
    combo_counts = {}
    all_simulation_evs = []
    
    for i in range(NUM_SIMULATIONS):
        result = run_simulation()
        combo = (result['firstChoice'], result['secondChoice'])
        combo_counts[combo] = combo_counts.get(combo, 0) + 1
        all_simulation_evs.append(result['allContainerEVs'])
    
    print("\nContainer combination distribution across simulations:")
    for combo, count in sorted(combo_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"First choice: {combo[0]}, Second choice: {combo[1]} - Count: {count}")
    
    # Find the most frequently optimal combination
    optimal_combo = max(combo_counts.items(), key=lambda x: x[1])[0]
    
    print("\nMost optimal combination:")
    print(f"First choice: Container {optimal_combo[0]}")
    print(f"Second choice: Container {optimal_combo[1]}")
    
    # Run one more detailed simulation to show the full breakdown
    print("\nDetailed breakdown of final simulation:")
    final_result = run_simulation()
    print("Detailed EV calculations for all containers:")
    for container in final_result['allContainerEVs']:
        print(f"Container {container['id']}: Multiplier={container['multiplier']}, "
              f"Inhabitants={container['inhabitants']}, Choices={container['choices']}, "
              f"Popularity={container['popularity']:.2f}%, Divisor={container['divisor']:.2f}, "
              f"First Choice EV={container['firstChoiceEV']:.2f}, "
              f"Second Choice EV={container['secondChoiceEV']:.2f}")
    
    # Create visualization
    fig1, fig2 = visualize_expected_values(final_result['allContainerEVs'], optimal_combo)
    
    # Return the final recommendation
    print("\nFINAL RECOMMENDATION:")
    print(f"You should choose Container {optimal_combo[0]} as your first (free) choice.")
    print(f"You should choose Container {optimal_combo[1]} as your second choice, paying 50,000 SeaShells.")
    print(f"Expected total EV: {final_result['totalEV']:.2f}")
    
    # Save or show the figures
    # fig1.savefig('container_expected_values.png')
    # fig2.savefig('container_properties.png')
    plt.show()

if __name__ == "__main__":
    main()