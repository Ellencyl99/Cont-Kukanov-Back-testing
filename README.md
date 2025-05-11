# Cont & Kukanov Back-testing Implementation

This repository contains a Smart Order Router back-testing implementation that follows the static cost model introduced by Cont & Kukanov. The implementation splits a 5,000-share buy order across multiple venues using three risk parameters and aims to minimize the total execution cost.

## Code Structure

The back-testing framework consists of the following components:

1. **Data Preprocessing**
   - Filters and organizes the market data to maintain one snapshot per venue per timestamp
   - Sorts data chronologically to facilitate the back-testing process

2. **Allocation Algorithm**
   - Implements the Cont-Kukanov allocator exactly as specified in the pseudocode
   - Dynamically calculates cost-optimal allocations across venues based on current market conditions and risk parameters

3. **Strategy Implementations**
   - **Cont-Kukanov:** The main strategy that dynamically allocates orders across venues
   - **Best Ask:** Always executes at the venue with the lowest ask price
   - **TWAP:** Time-Weighted Average Price strategy with 60-second buckets
   - **VWAP:** Volume-Weighted Average Price strategy with weights based on displayed ask sizes

4. **Parameter Search**
   - Implements grid search across defined parameter ranges
   - Evaluates each parameter combination and selects the one with the lowest total cost

5. **Performance Evaluation**
   - Calculates execution costs and average prices for all strategies
   - Computes basis point savings versus each baseline
   - Generates a cumulative cost plot showing execution costs over time

## Parameter Range Selection

I selected the following parameter ranges based on the scale of prices in the dataset and the relative importance of each parameter:

- **lambda_over:** [0.01, 0.05, 0.1, 0.5, 1.0]
  - Represents the cost penalty for over-execution
  - Range covers both minimal and significant penalties

- **lambda_under:** [0.01, 0.05, 0.1, 0.5, 1.0]
  - Represents the cost penalty for under-execution
  - Matched with lambda_over to explore symmetric and asymmetric cost models

- **theta_queue:** [0.001, 0.01, 0.05, 0.1, 0.5]
  - Represents the queue-risk penalty
  - Lower range than the other parameters to account for its multiplicative effect on total mis-execution

These ranges allow the model to explore different risk preferences:
- Conservative execution (higher lambda values)
- Aggressive execution (lower lambda values)
- Different weightings between over-execution and under-execution risks

## Suggested Improvement: Adaptive Queue Position Modeling

One significant way to improve fill realism would be to implement an adaptive queue position model. The current implementation assumes that any order sent to a venue will execute immediately up to the displayed size, but in reality, our order would enter a queue behind other orders.

**Proposed Enhancement:**
   - Track queue changes between snapshots to estimate queue consumption rates
   - Adjust the execution probability based on observed market dynamics
   - Update the `compute_cost` function to include time-value penalties for delayed executions
