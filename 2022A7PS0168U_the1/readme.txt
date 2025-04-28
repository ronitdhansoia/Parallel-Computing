Parallel Matrix-Vector Multiplication Performance Study

Author
Name: Ronit Dhansoia
Course: CS F422 - Parallel Computing
Semester: II Semester 2024-2025

Project Overview
This project is about performance analysis, false sharing prevention, and synchronization of parallel matrix-vector multiplication using pthreads.

Objectives and Implementations

(a) Performance Measurement
- Goal: Measure GFlops/sec across different thread counts and matrix dimensions
- Test Configurations**:
  1. Thread Counts: 1, 2, 4, 8 threads
  2. Matrix Dimensions:
     - 8000000 × 8
     - 8000 × 8000
     - 8 × 8000000
Performance metric is calculated by using below formula 
  GFlops/sec = (2 * m * n * operations) / (time * 10^9)

(b) False Sharing Prevention
The problem is cache coherence issues managing performance deterioration.
- Solution: 
  Aligned double was implemented `struct aligned_double` padded with cache-line size.
  - Guarantees that result of each thread resides on different cache line
  Provides no unnecessary cache invalidations and coherence traffic.
- Technique: 
  #define CACHE_LINE 64
   struct aligned_double {
       double value;
       char padding[CACHE_LINE - sizeof(double)];
   };


(c) Barrier Synchronization
The goal is to support row-level thread synchronization.
- Implementation: 
  Using pthread mutex and condition variable to realize custom barrier.
  It waits as each matrix row is processed with each thread.
  Provides synchronization overhead and coordination.

Performance Analysis Methodology
- Random matrix and vector generation
It can then be computed in parallel using block distribution.
- Time measurement with microsecond accuracy
- GFlops/sec calculation for each configuration

Compilation Instructions

make modified_pth_mat_vect_rand_split

Execution

./modified_pth_mat_vect_rand_split


Expected Outputs
Execution times for various numerical matrix dimensions
- GFlops/sec for different thread counts
- Performance variations and scalability insights

Key Performance Considerations
- Thread count vs. computational efficiency
- Impact of matrix dimensions on parallelism
- False sharing mitigation effects
- Synchronization overhead

Limitations and Future Work
- Fixed thread-to-matrix row mapping
Load balancer can enable more advanced load balancing.
- Exploration of other synchronization techniques

References
CS F422 Parallel Computing – Course Material
- Textbook Concepts on Parallel Matrix Multiplication