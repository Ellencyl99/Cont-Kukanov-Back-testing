import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
import itertools
import time

class Venue:
    """Class representing a trading venue with L1 market data"""
    def __init__(self, publisher_id: str, ask: float, ask_size: int, fee: float = 0, rebate: float = 0):
        self.publisher_id = publisher_id
        self.ask = ask
        self.ask_size = ask_size
        self.fee = fee
        self.rebate = rebate

def allocate(order_size: int, venues: List[Venue], lambda_over: float, lambda_under: float, theta_queue: float) -> Tuple[List[int], float]:
    """
    Static Cont-Kukanov allocation algorithm as specified in the pseudocode
    Returns the best allocation split and its associated cost
    """
    step = 100
    splits = [[]]
    
    for v in range(len(venues)):
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size - used, venues[v].ask_size)
            for q in range(0, max_v + 1, step):
                new_splits.append(alloc + [q])
        splits = new_splits

    best_cost = float('inf')
    best_split = []
    
    for alloc in splits:
        if sum(alloc) != order_size:
            continue
        cost = compute_cost(alloc, venues, order_size, lambda_over, lambda_under, theta_queue)
        if cost < best_cost:
            best_cost = cost
            best_split = alloc
            
    return best_split, best_cost

def compute_cost(split: List[int], venues: List[Venue], order_size: int, lambda_o: float, lambda_u: float, theta: float) -> float:
    """
    Calculate the expected cost of a split as specified in the pseudocode
    """
    executed = 0
    cash_spent = 0
    
    for i in range(len(venues)):
        exe = min(split[i], venues[i].ask_size)
        executed += exe
        cash_spent += exe * (venues[i].ask + venues[i].fee)
        maker_rebate = max(split[i] - exe, 0) * venues[i].rebate
        cash_spent -= maker_rebate

    underfill = max(order_size - executed, 0)
    overfill = max(executed - order_size, 0)
    risk_pen = theta * (underfill + overfill)
    cost_pen = lambda_u * underfill + lambda_o * overfill
    
    return cash_spent + risk_pen + cost_pen

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the market data to get one Level-1 snapshot per venue per timestamp
    """
    # Keep only the first message per publisher_id for each unique ts_event
    df = df.sort_values(['ts_event', 'ts_recv'])
    df = df.drop_duplicates(subset=['ts_event', 'publisher_id'], keep='first')
    
    # Make sure we have data sorted by timestamp
    df = df.sort_values('ts_event')
    
    return df

def execute_best_ask_strategy(snapshots: pd.DataFrame, order_size: int) -> Tuple[float, float, List]:
    """
    Baseline strategy: Always take the best ask price
    """
    remaining = order_size
    cash_spent = 0
    execution_history = []
    
    # Group by timestamp to process one snapshot at a time
    for ts, group in snapshots.groupby('ts_event'):
        if remaining <= 0:
            break
            
        # Find the venue with the best ask price in this snapshot
        if not group.empty:
            best_venue = group.loc[group['ask_px_00'].idxmin()]
            
            executed = min(remaining, int(best_venue['ask_sz_00']))
            if executed > 0:
                cash_spent += executed * best_venue['ask_px_00']
                remaining -= executed
                
                execution_history.append({
                    'ts_event': ts,
                    'executed': executed,
                    'price': best_venue['ask_px_00'],
                    'cash_spent': executed * best_venue['ask_px_00'],
                    'remaining': remaining
                })
    
    # Calculate average fill price
    avg_price = cash_spent / (order_size - remaining) if order_size > remaining else 0
    
    return cash_spent, avg_price, execution_history

def execute_twap_strategy(snapshots: pd.DataFrame, order_size: int, bucket_seconds: int = 60) -> Tuple[float, float, List]:
    """
    Time-Weighted Average Price (TWAP) strategy with 60-second buckets
    """
    # Convert ts_event to numeric if it's not already
    snapshots = snapshots.copy()
    if not pd.api.types.is_numeric_dtype(snapshots['ts_event']):
        # Try to parse ISO format timestamps to datetime and then to timestamp
        try:
            snapshots['ts_event'] = pd.to_datetime(snapshots['ts_event']).astype(np.int64)
        except:
            # Fallback to direct numeric conversion if possible
            snapshots['ts_event'] = pd.to_numeric(snapshots['ts_event'], errors='coerce')
    
    # Group snapshots into 60-second buckets - nanoseconds to seconds conversion
    bucket_ns = bucket_seconds * 1_000_000_000
    snapshots['bucket'] = (snapshots['ts_event'] // bucket_ns).astype(int)
    buckets = snapshots.groupby('bucket')
    
    # Calculate number of shares to execute in each bucket
    buckets_count = len(buckets)
    shares_per_bucket = order_size // buckets_count
    remaining_shares = order_size % buckets_count
    
    # Adjust shares for the last bucket
    shares_allocation = [shares_per_bucket] * buckets_count
    if remaining_shares > 0:
        shares_allocation[-1] += remaining_shares
    
    remaining = order_size
    cash_spent = 0
    execution_history = []
    
    for (bucket_id, bucket_data), bucket_size in zip(buckets, shares_allocation):
        if remaining <= 0:
            break
        
        shares_this_bucket = min(bucket_size, remaining)
        
        # For each timestamp in the bucket, find the best ask
        for ts, group in bucket_data.groupby('ts_event'):
            if shares_this_bucket <= 0:
                break
                
            if not group.empty:
                best_venue = group.loc[group['ask_px_00'].idxmin()]
                
                executed = min(shares_this_bucket, int(best_venue['ask_sz_00']))
                if executed > 0:
                    cash_spent += executed * best_venue['ask_px_00']
                    shares_this_bucket -= executed
                    remaining -= executed
                    
                    execution_history.append({
                        'ts_event': ts,
                        'executed': executed,
                        'price': best_venue['ask_px_00'],
                        'cash_spent': executed * best_venue['ask_px_00'],
                        'remaining': remaining
                    })
    
    # Calculate average fill price
    avg_price = cash_spent / (order_size - remaining) if order_size > remaining else 0
    
    return cash_spent, avg_price, execution_history

def execute_vwap_strategy(snapshots: pd.DataFrame, order_size: int) -> Tuple[float, float, List]:
    """
    Volume-Weighted Average Price (VWAP) strategy that weights prices by displayed ask size
    """
    remaining = order_size
    cash_spent = 0
    execution_history = []
    
    for ts, group in snapshots.groupby('ts_event'):
        if remaining <= 0:
            break
        
        # Skip empty groups or groups with no valid ask prices
        if group.empty or group['ask_sz_00'].sum() == 0:
            continue
            
        # Calculate size weights for each venue
        group = group.copy()
        group['size_weight'] = group['ask_sz_00'] / group['ask_sz_00'].sum()
        
        # Calculate shares to execute per venue based on weights
        group['shares_to_execute'] = (order_size * group['size_weight']).astype(int)
        
        # Execute at each venue
        for _, venue in group.iterrows():
            shares_to_execute = min(int(venue['shares_to_execute']), remaining, int(venue['ask_sz_00']))
            
            if shares_to_execute > 0:
                cash_spent += shares_to_execute * venue['ask_px_00']
                remaining -= shares_to_execute
                
                execution_history.append({
                    'ts_event': ts,
                    'executed': shares_to_execute,
                    'price': venue['ask_px_00'],
                    'cash_spent': shares_to_execute * venue['ask_px_00'],
                    'remaining': remaining
                })
    
    # Calculate average fill price
    avg_price = cash_spent / (order_size - remaining) if order_size > remaining else 0
    
    return cash_spent, avg_price, execution_history

def execute_cont_kukanov_strategy(snapshots: pd.DataFrame, order_size: int, 
                                  lambda_over: float, lambda_under: float, 
                                  theta_queue: float) -> Tuple[float, float, List, int]:
    """
    Execute the Cont-Kukanov allocation strategy
    """
    remaining = order_size
    cash_spent = 0
    execution_history = []
    
    for ts, group in snapshots.groupby('ts_event'):
        if remaining <= 0:
            break
            
        # Convert snapshot to venues list
        venues = []
        for _, row in group.iterrows():
            if pd.notna(row['ask_px_00']) and pd.notna(row['ask_sz_00']) and row['ask_sz_00'] > 0:
                venues.append(Venue(
                    publisher_id=row['publisher_id'],
                    ask=row['ask_px_00'],
                    ask_size=int(row['ask_sz_00']),
                    fee=0,  # Assuming no fees for this test
                    rebate=0  # Assuming no rebates for this test
                ))
        
        if not venues:
            continue
            
        # Allocate the remaining shares using the Cont-Kukanov algorithm
        try:
            alloc, _ = allocate(remaining, venues, lambda_over, lambda_under, theta_queue)
            
            # Execute the allocation
            for venue_idx, allocated_shares in enumerate(alloc):
                venue = venues[venue_idx]
                executed = min(allocated_shares, venue.ask_size)
                
                if executed > 0:
                    cash_spent += executed * venue.ask
                    remaining -= executed
                    
                    execution_history.append({
                        'ts_event': ts,
                        'venue': venue.publisher_id,
                        'allocated': allocated_shares,
                        'executed': executed,
                        'price': venue.ask,
                        'cash_spent': executed * venue.ask,
                        'remaining': remaining
                    })
        except Exception as e:
            print(f"Error allocating at timestamp {ts}: {e}")
            continue
    
    # Calculate average fill price
    avg_price = cash_spent / (order_size - remaining) if order_size > remaining else 0
    
    return cash_spent, avg_price, execution_history, remaining

def grid_search(snapshots: pd.DataFrame, order_size: int) -> Dict:
    """
    Perform a grid search for the optimal parameters
    """
    # Define parameter ranges - reduced for faster execution
    lambda_over_range = [0.01, 0.05, 0.1, 0.5, 1.0]
    lambda_under_range = [0.01, 0.05, 0.1, 0.5, 1.0]
    theta_queue_range = [0.001, 0.01, 0.05, 0.1, 0.5]
    
    best_cost = float('inf')
    best_params = None
    best_avg_price = None
    best_remaining = None
    best_history = None
    
    total_combinations = len(lambda_over_range) * len(lambda_under_range) * len(theta_queue_range)
    # print(f"Starting grid search with {total_combinations} parameter combinations...")
    
    # Run grid search
    combo_count = 0
    for lambda_over, lambda_under, theta_queue in itertools.product(
            lambda_over_range, lambda_under_range, theta_queue_range):
        
        # combo_count += 1
        # if combo_count % 5 == 0:
        #     print(f"Testing combination {combo_count}/{total_combinations}")
        
        cash_spent, avg_price, history, remaining = execute_cont_kukanov_strategy(
            snapshots, order_size, lambda_over, lambda_under, theta_queue)
        
        # Prioritize combinations that fill the entire order
        if remaining == 0 and cash_spent < best_cost:
            best_cost = cash_spent
            best_params = {
                'lambda_over': lambda_over,
                'lambda_under': lambda_under,
                'theta_queue': theta_queue
            }
            best_avg_price = avg_price
            best_remaining = remaining
            best_history = history
            
    # If no parameter set fills the entire order, select the one with minimum cost
    if best_params is None:
        best_fill_amount = 0
        
        for lambda_over, lambda_under, theta_queue in itertools.product(
                lambda_over_range, lambda_under_range, theta_queue_range):
            
            cash_spent, avg_price, history, remaining = execute_cont_kukanov_strategy(
                snapshots, order_size, lambda_over, lambda_under, theta_queue)
            
            fill_amount = order_size - remaining
            
            # Prioritize larger fills, then lower cost
            if fill_amount > best_fill_amount or (fill_amount == best_fill_amount and cash_spent < best_cost):
                best_cost = cash_spent
                best_params = {
                    'lambda_over': lambda_over,
                    'lambda_under': lambda_under,
                    'theta_queue': theta_queue
                }
                best_avg_price = avg_price
                best_remaining = remaining
                best_history = history
                best_fill_amount = fill_amount
    
    # print(f"Best parameters found: {best_params}")
    # print(f"Remaining shares: {best_remaining}")
    
    return {
        'params': best_params,
        'cash_spent': best_cost,
        'avg_price': best_avg_price,
        'remaining': best_remaining,
        'history': best_history
    }


def create_cumulative_cost_plot(cont_kukanov_history, best_ask_history, twap_history, vwap_history, order_size: int) -> None:
    """
    Create a cumulative cost plot comparing all strategies, each in a separate subplot.
    """
    # Create a figure with 4 subplots (2 rows, 2 columns)
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Cumulative Execution Cost Comparison', fontsize=16)

    # Plot Cont-Kukanov
    if cont_kukanov_history:
        df_ck = pd.DataFrame(cont_kukanov_history)
        if not df_ck.empty:
            df_ck = df_ck.sort_values('ts_event')
            if not pd.api.types.is_numeric_dtype(df_ck['ts_event']):
                try:
                    df_ck['ts_event'] = pd.to_datetime(df_ck['ts_event']).astype(np.int64)
                except:
                    df_ck['ts_event'] = pd.to_numeric(df_ck['ts_event'], errors='coerce')

            df_ck['timestamp_norm'] = (df_ck['ts_event'] - df_ck['ts_event'].min()) / 1_000_000_000
            df_ck['cumulative_spent'] = df_ck['cash_spent'].cumsum()
            
            filled_shares_ck = order_size - df_ck['remaining'].iloc[-1] if not df_ck.empty else 0
            avg_price_ck = df_ck['cumulative_spent'].iloc[-1] / filled_shares_ck if filled_shares_ck > 0 else 0
            if filled_shares_ck > 0:
                df_ck_filtered = df_ck[df_ck['remaining'] >= 0]
                axs[0, 0].plot(df_ck_filtered['timestamp_norm'], df_ck_filtered['cumulative_spent'], 
                               label=f"Cont-Kukanov (Avg: ${avg_price_ck:.4f})")
                axs[0, 0].set_title('Cont-Kukanov')
                axs[0, 0].set_xlabel('Time (seconds from start)')
                axs[0, 0].set_ylabel('Cumulative Cost ($)')
                axs[0, 0].grid(True, alpha=0.3)

    # Plot Best Ask
    if best_ask_history:
        df_ba = pd.DataFrame(best_ask_history)
        if not df_ba.empty:
            df_ba = df_ba.sort_values('ts_event')
            if not pd.api.types.is_numeric_dtype(df_ba['ts_event']):
                try:
                    df_ba['ts_event'] = pd.to_datetime(df_ba['ts_event']).astype(np.int64)
                except:
                    df_ba['ts_event'] = pd.to_numeric(df_ba['ts_event'], errors='coerce')

            df_ba['timestamp_norm'] = (df_ba['ts_event'] - df_ba['ts_event'].min()) / 1_000_000_000
            df_ba['cumulative_spent'] = df_ba['cash_spent'].cumsum()
            filled_shares_ba = order_size - df_ba['remaining'].iloc[-1] if not df_ba.empty else 0
            avg_price_ba = df_ba['cumulative_spent'].iloc[-1] / filled_shares_ba if filled_shares_ba > 0 else 0
            if filled_shares_ba > 0:
                df_ba_filtered = df_ba[df_ba['remaining'] > 0]
                axs[0, 1].plot(df_ba_filtered['timestamp_norm'], df_ba_filtered['cumulative_spent'], 
                               label=f"Best Ask (Avg: ${avg_price_ba:.4f})")
                axs[0, 1].set_title('Best Ask')
                axs[0, 1].set_xlabel('Time (seconds from start)')
                axs[0, 1].set_ylabel('Cumulative Cost ($)')
                axs[0, 1].grid(True, alpha=0.3)

    # Plot TWAP
    if twap_history:
        df_twap = pd.DataFrame(twap_history)
        if not df_twap.empty:
            df_twap = df_twap.sort_values('ts_event')
            if not pd.api.types.is_numeric_dtype(df_twap['ts_event']):
                try:
                    df_twap['ts_event'] = pd.to_datetime(df_twap['ts_event']).astype(np.int64)
                except:
                    df_twap['ts_event'] = pd.to_numeric(df_twap['ts_event'], errors='coerce')

            df_twap['timestamp_norm'] = (df_twap['ts_event'] - df_twap['ts_event'].min()) / 1_000_000_000
            df_twap['cumulative_spent'] = df_twap['cash_spent'].cumsum()
            filled_shares_twap = order_size - df_twap['remaining'].iloc[-1] if not df_twap.empty else 0
            avg_price_twap = df_twap['cumulative_spent'].iloc[-1] / filled_shares_twap if filled_shares_twap > 0 else 0
            if filled_shares_twap > 0:
                df_twap_filtered = df_twap[df_twap['remaining'] > 0]
                axs[1, 0].plot(df_twap_filtered['timestamp_norm'], df_twap_filtered['cumulative_spent'], 
                               label=f"TWAP (Avg: ${avg_price_twap:.4f})")
                axs[1, 0].set_title('TWAP')
                axs[1, 0].set_xlabel('Time (seconds from start)')
                axs[1, 0].set_ylabel('Cumulative Cost ($)')
                axs[1, 0].grid(True, alpha=0.3)

    # Plot VWAP
    if vwap_history:
        df_vwap = pd.DataFrame(vwap_history)
        if not df_vwap.empty:
            df_vwap = df_vwap.sort_values('ts_event')
            if not pd.api.types.is_numeric_dtype(df_vwap['ts_event']):
                try:
                    df_vwap['ts_event'] = pd.to_datetime(df_vwap['ts_event']).astype(np.int64)
                except:
                    df_vwap['ts_event'] = pd.to_numeric(df_vwap['ts_event'], errors='coerce')

            df_vwap['timestamp_norm'] = (df_vwap['ts_event'] - df_vwap['ts_event'].min()) / 1_000_000_000
            df_vwap['cumulative_spent'] = df_vwap['cash_spent'].cumsum()
            filled_shares_vwap = order_size - df_vwap['remaining'].iloc[-1] if not df_vwap.empty else 0
            avg_price_vwap = df_vwap['cumulative_spent'].iloc[-1] / filled_shares_vwap if filled_shares_vwap > 0 else 0
            if filled_shares_vwap > 0:
                df_vwap_filtered = df_vwap[df_vwap['remaining'] > 0]
                axs[1, 1].plot(df_vwap_filtered['timestamp_norm'], df_vwap_filtered['cumulative_spent'], 
                               label=f"VWAP (Avg: ${avg_price_vwap:.4f})")
                axs[1, 1].set_title('VWAP')
                axs[1, 1].set_xlabel('Time (seconds from start)')
                axs[1, 1].set_ylabel('Cumulative Cost ($)')
                axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('results.png', dpi=300, bbox_inches='tight')
    print("Cumulative cost plot saved as results.png")

def calculate_basis_points_savings(optimal_avg_price: float, baseline_avg_price: float) -> float:
    """
    Calculate savings in basis points
    """
    if baseline_avg_price <= 0 or optimal_avg_price <= 0:
        return 0.0
    return (baseline_avg_price - optimal_avg_price) / baseline_avg_price * 10000

def main():
    start_time = time.time()
    
    # Load and preprocess data
    df = pd.read_csv('l1_day.csv')
    
    # Handle timestamp conversion before preprocessing
    if 'ts_event' in df.columns and not pd.api.types.is_numeric_dtype(df['ts_event']):
        try:
            df['ts_event'] = pd.to_datetime(df['ts_event']).astype(np.int64)
        except:
            df['ts_event'] = pd.to_numeric(df['ts_event'], errors='coerce')
    
    processed_df = preprocess_data(df)
    
    # Double-check that ts_event is numeric after preprocessing
    if not pd.api.types.is_numeric_dtype(processed_df['ts_event']):
        try:
            processed_df['ts_event'] = pd.to_datetime(processed_df['ts_event']).astype(np.int64)
        except:
            processed_df['ts_event'] = pd.to_numeric(processed_df['ts_event'], errors='coerce')
    
    order_size = 5000
    
    # Run grid search for optimal parameters
    optimal_result = grid_search(processed_df, order_size)
    
    # Execute baseline strategies
    best_ask_cost, best_ask_avg_price, best_ask_history = execute_best_ask_strategy(processed_df, order_size)
    twap_cost, twap_avg_price, twap_history = execute_twap_strategy(processed_df, order_size)
    vwap_cost, vwap_avg_price, vwap_history = execute_vwap_strategy(processed_df, order_size)
    
    # Calculate savings in basis points
    best_ask_savings = calculate_basis_points_savings(optimal_result['avg_price'], best_ask_avg_price)
    twap_savings = calculate_basis_points_savings(optimal_result['avg_price'], twap_avg_price)
    vwap_savings = calculate_basis_points_savings(optimal_result['avg_price'], vwap_avg_price)
    
    # Create cumulative cost plot
    create_cumulative_cost_plot(
        optimal_result['history'], 
        best_ask_history, 
        twap_history, 
        vwap_history, 
        order_size
    )
    
    # Format JSON output
    output = {
        'best_parameters': optimal_result['params'],
        'cont_kukanov': {
            'cash_spent': optimal_result['cash_spent'],
            'avg_price': optimal_result['avg_price'],
            'shares_filled': order_size - optimal_result['remaining']
        },
        'best_ask': {
            'cash_spent': best_ask_cost,
            'avg_price': best_ask_avg_price
        },
        'twap': {
            'cash_spent': twap_cost,
            'avg_price': twap_avg_price
        },
        'vwap': {
            'cash_spent': vwap_cost,
            'avg_price': vwap_avg_price
        },
        'savings_bps': {
            'vs_best_ask': best_ask_savings,
            'vs_twap': twap_savings,
            'vs_vwap': vwap_savings
        }
    }
    
    # Print JSON result
    print(json.dumps(output, indent=2))
    
    print(f"Execution completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()