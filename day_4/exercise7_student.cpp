/**
 * @file main.cpp
 * @brief Demonstrates collective communication using MPI in a multi-process application.
 *
 * This program initializes multiple workers, each potentially on a different process,
 * and performs collective operations like broadcast, all-reduce, and all-gather using MPI.
 * The code includes extensive documentation for educational purposes.
 *
 * Compile with:
 * ```sh
 * $ mpicxx -o exercise exercise7.cpp  -lm
 * ```
 */

#include <cassert>
#include <functional>
#include <iostream>
#include <numeric> // For std::iota
#include <random>
#include <stdexcept>
#include <vector>
#include <mpi.h>

// Macros for error checking
/**
 * @brief Checks the result of an MPI API call and throws an exception on error.
 *
 * @param cmd The MPI API function call.
 */
#define MPI_CHECK(cmd)                                                                                                   \
    {                                                                                                                    \
        int error = cmd;                                                                                                 \
        if (error != MPI_SUCCESS)                                                                                        \
        {                                                                                                                \
            char error_string[MPI_MAX_ERROR_STRING];                                                                     \
            int length_of_error_string;                                                                                  \
            MPI_Error_string(error, error_string, &length_of_error_string);                                              \
            throw std::runtime_error(std::string("MPI Error: ") + error_string + " at " + __FILE__ + ":" +               \
                                     std::to_string(__LINE__));                                                          \
        }                                                                                                                \
    }

/**
 * @brief Function to add two arrays element-wise.
 *
 * @param A Reference to the first input array and output array.
 * @param B Reference to the second input array.
 */
void add_arrays(std::vector<float>& A, const std::vector<float>& B)
{
    for (size_t i = 0; i < A.size(); ++i)
    {
        A[i] += B[i];
    }
}

/**
 * @brief Function to compute the mean squared error (MSE) of an array.
 *
 * @param A Reference to the input array.
 * @return float The computed MSE value.
 */
float compute_mse(const std::vector<float>& A)
{
    float mse = 0.0f;
    for (size_t i = 0; i < A.size(); ++i)
    {
        mse += A[i] * A[i];
    }
    mse /= A.size();
    return mse;
}

/**
 * @class Worker
 * @brief Represents a worker that performs computations.
 *
 * This class encapsulates the initialization of resources,
 * the execution of computational functions, and the use of MPI for collective communication.
 */
class Worker
{
public:
    /**
     * @brief Constructs a Worker object.
     *
     * @param rank The rank of the worker.
     * @param world_size The total number of workers.
     */
    Worker(int rank, int world_size)
        : rank_(rank), world_size_(world_size)
    {
        // Initialize host data
        h_A_.resize(ARRAY_SIZE_);
        h_B_.resize(ARRAY_SIZE_);
        h_C_.resize(256);

        // Initialize random number generator with a different seed per rank
        std::random_device rd;
        std::mt19937 gen(rd() + rank_);
        std::uniform_real_distribution<> dis(0.0, 1.0);

        // Initialize arrays A and B
        for (auto& val : h_A_)
            val = dis(gen);
        std::fill(h_B_.begin(), h_B_.end(), 1.0f);

        // Initialize array C
        if (rank_ == 0)
        {
            // Rank 0 initializes array C with values 1 to 256
            std::iota(h_C_.begin(), h_C_.end(), 1.0f);
        }
        else
        {
            // Other ranks initialize array C with their rank
            std::fill(h_C_.begin(), h_C_.end(), static_cast<float>(rank_));
        }

        // Print initialization message
        std::cout << "Rank " << rank_ << " initialized data\n";
    }

    /**
     * @brief Executes the main computation and communication tasks.
     *
     * This method performs the following steps:
     * - Performs collective communications (broadcast, all-reduce, all-gather) using MPI.
     * - Executes computational functions (addition, MSE computation).
     */
    void run()
    {
        // Perform a broadcast operation to distribute B from rank 0 to all other ranks
        std::cout << "Rank " << rank_ << ": Starting MPI_Bcast\n";
        // TODO 2: Implement the broadcast of h_b operation

        // Perform element-wise addition of A and B
        add_arrays(h_A_, h_B_);

        std::cout << "Rank " << rank_ << ": Completed add_arrays function\n";

        // Compute the mean squared error (MSE)
        float local_mse = compute_mse(h_A_);

        std::cout << "Rank " << rank_ << ": Local MSE value: " << local_mse << "\n";

        // Perform an all-reduce operation to sum the MSE values across all ranks
        std::cout << "Rank " << rank_ << ": Starting MPI_Allreduce\n";
        // TODO 3: Implement the all-reduce operation from local_mse to global_mse_

        std::cout << "Rank " << rank_ << ": Global MSE value after AllReduce: " << global_mse_ << "\n";

        // Prepare receive buffer for all-gather
        std::vector<float> h_C_recv(256 * world_size_);

        // Perform an all-gather operation on array C
        std::cout << "Rank " << rank_ << ": Starting MPI_Allgather\n";
        // TODO 4: Implement the all-gather operation from h_C_ to h_C_recv

        // Optionally, only rank 0 prints the final results
        if (rank_ == 0)
        {
            std::cout << "Rank 0: First few elements of h_C_recv after AllGather:\n";
            for (int i = 0; i < 10; ++i)
            {
                std::cout << h_C_recv[i] << " ";
            }
            std::cout << "\n";
            std::cout << "Rank 0: Final Global MSE value: " << global_mse_ << "\n";
        }
    }

    // Disable copy semantics
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

private:
    int rank_;                   /**< The rank of the worker */
    int world_size_;             /**< The total number of workers */
    std::vector<float> h_A_;     /**< Host memory for array A */
    std::vector<float> h_B_;     /**< Host memory for array B */
    std::vector<float> h_C_;     /**< Host memory for array C */
    float global_mse_ = 0.0f;    /**< The global MSE value after all-reduce */

    // Constants
    const int ARRAY_SIZE_ = 1 << 10; /**< The number of elements in the arrays (2^10) */
};

/**
 * @brief The main function initializes the application and launches worker processes.
 *
 * @param argc The argument count.
 * @param argv The argument vector.
 * @return int Exit code indicating success or failure.
 */
int main(int argc, char* argv[])
{
    // Initialize MPI
    int provided;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));
    if (provided < MPI_THREAD_MULTIPLE)
    {
        std::cerr << "Error: The MPI library does not provide required threading level\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // TODO 1: Implement the MPI_Comm_size call and rank

    // Each process creates a Worker and runs it
    {
        Worker worker(rank, world_size);
        worker.run();
    }

    MPI_CHECK(MPI_Finalize());
    return 0;
}
