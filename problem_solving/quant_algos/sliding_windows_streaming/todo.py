"""
1. Sliding Window & Streaming / Online Algorithms
Financial time-series data arrives as a continuous stream of tick or bar data, making real-time calculation essential.

Key Topics:
Monotonic Deques / Stacks: Sliding window minimum/maximum (e.g., rolling high/low indicators in O(N) time).
Two-Heap Patterns: Finding running/streaming median and rolling percentiles.
Online Variance / Moments: Algorithms like Welford’s Algorithm for single-pass online mean, variance, and covariance calculation without numerical instability.
Streaming Sketches: Count-Min Sketch, HyperLogLog, and Reservoir Sampling for handling high-throughput market feeds.
Financial Context: Calculating rolling volatility, dynamic slippage estimates, and online drawdown metrics without recalculating over the full history at each tick.


"""