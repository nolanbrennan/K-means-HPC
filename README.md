**MPI Parallel K-Means Clustering (HPC / Distributed Systems)**
Developed a distributed-memory implementation of the K-Means clustering algorithm in C with MPI and evaluated performance on the Zaratan high-performance computing cluster.

Key features:

- Parallelized a serial K-Means implementation using input data partitioning across processors.
- Implemented MPI communication patterns including MPI_Bcast, MPI_Scatterv, MPI_Reduce, and MPI_Gather to synchronize centroid updates and cluster assignments.
- Designed root-worker architecture where Processor 0 handles I/O while worker processes perform distributed clustering computations.
- Executed large-scale performance experiments (1–128 processes) on MNIST datasets to evaluate scalability and communication overhead.
- Ensured correctness through automated testing frameworks and memory safety verification using Valgrind.
