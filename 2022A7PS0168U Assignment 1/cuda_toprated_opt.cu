
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <utility>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <chrono>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(err)                                            \
    if (err != cudaSuccess)                                              \
    {                                                                    \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));                                \
        exit(EXIT_FAILURE);                                              \
    }
__global__ void
bitonicSortStep(float *keys, int *vals, int n, int stage,
                int sub_stage_pass)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int sort_increasing = 1;
    int distance = 1 << (stage - sub_stage_pass);
    int left_id = (tid % distance) + (tid / distance) * 2 * distance;
    int right_id = left_id + distance;
    int direction_stage = 1 << stage;
    int direction_mask = (tid / direction_stage);
    int sort_direction = (direction_mask % 2 == 0) ? sort_increasing : (1 - sort_increasing);
    sort_direction = 1 - sort_direction;
    if (right_id < n)
    {
        float left_key = keys[left_id];
        float right_key = keys[right_id];
        int left_val = vals[left_id];
        int right_val = vals[right_id];
        bool swap = (left_key < right_key && sort_direction) ||
                    (left_key > right_key && !sort_direction);
        if (swap)
        {
            keys[left_id] = right_key;
            keys[right_id] = left_key;
            vals[left_id] = right_val;
            vals[right_id] = left_val;
        }
    }
}
int nextPowerOf2(int n)
{
    if (n <= 0)
        return 1;
    int power = 1;
    while (power < n)
    {
        power *= 2;
    }
    return power;
}
int main(int argc, char **argv)
{
    auto total_start_time = std::chrono::high_resolution_clock::now();
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <path_to_reviews.json>"
                  << std::endl;
        return 1;
    }
    std::string filename = argv[1];
    std::ifstream infile(filename);
    if (!infile.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return 1;
    }
    auto cpu_read_start = std::chrono::high_resolution_clock::now();
    std::cout << "Reading reviews and calculating sums/counts (CPU using "
                 "RapidJSON)..."
              << std::endl;
    std::map<std::string, std::pair<double, int>> product_ratings;
    std::string line;
    long long line_count = 0;
    long long processed_count = 0;
    long long error_count = 0;
    rapidjson::Document document;
    while (std::getline(infile, line))
    {
        line_count++;
        document.Parse(line.c_str());
        if (document.HasParseError())
        {
            error_count++;
            continue;
        }
        if (document.IsObject() &&
            document.HasMember("asin") && document["asin"].IsString() &&
            document.HasMember("overall") && document["overall"].IsNumber())
        {
            try
            {
                std::string asin = document["asin"].GetString();
                float rating = document["overall"].GetFloat();
                product_ratings[asin].first += rating;
                product_ratings[asin].second++;
                processed_count++;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Warning: Exception accessing data on line " << line_count
                          << ": " << e.what() << std::endl;
                error_count++;
            }
        }
        else
        {
            error_count++;
        }
    }
    infile.close();
    auto cpu_read_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_read_duration = cpu_read_end - cpu_read_start;
    std::cout << "Finished reading. Processed " << processed_count
              << " valid reviews for " << product_ratings.size()
              << " unique products." << std::endl;
    if (error_count > 0)
    {
        std::cout << "  Skipped " << error_count
                  << " lines due to parsing errors or missing/invalid fields."
                  << std::endl;
    }
    if (product_ratings.empty())
    {
        std::cerr << "No valid product data found." << std::endl;
        return 1;
    }
    auto cpu_avg_start = std::chrono::high_resolution_clock::now();
    std::cout << "Calculating average ratings (CPU)..." << std::endl;
    std::vector<float> h_avg_ratings;
    std::vector<std::string> h_asins;
    std::vector<int> h_original_indices;
    h_avg_ratings.reserve(product_ratings.size());
    h_asins.reserve(product_ratings.size());
    h_original_indices.reserve(product_ratings.size());
    int current_index = 0;
    for (const auto &pair : product_ratings)
    {
        const std::string &asin = pair.first;
        double sum_ratings = pair.second.first;
        int count = pair.second.second;
        if (count > 0)
        {
            h_avg_ratings.push_back(static_cast<float>(sum_ratings / count));
            h_asins.push_back(asin);
            h_original_indices.push_back(current_index++);
        }
    }
    int num_elements = h_avg_ratings.size();
    auto cpu_avg_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_avg_duration = cpu_avg_end - cpu_avg_start;
    std::cout << "Calculated " << num_elements << " average ratings." << std::endl;
    if (num_elements == 0)
    {
        std::cerr << "No products with ratings found after averaging." << std::endl;
        return 1;
    }
    std::cout << "Sorting average ratings (GPU using Bitonic Sort)..." << std::endl;
    cudaEvent_t start_event, stop_event;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_event));
    float gpu_alloc_memcpy_h2d_ms = 0;
    float gpu_kernel_ms = 0;
    float gpu_memcpy_d2h_free_ms = 0;
    int padded_size = nextPowerOf2(num_elements);
    std::cout << "  Padding data from " << num_elements << " to " << padded_size << " elements." << std::endl;
    std::vector<float> h_padded_ratings(padded_size, -1.0f);
    std::vector<int> h_padded_indices(padded_size, -1);
    std::copy(h_avg_ratings.begin(), h_avg_ratings.end(), h_padded_ratings.begin());
    std::copy(h_original_indices.begin(), h_original_indices.end(), h_padded_indices.begin());
    float *d_keys = nullptr;
    int *d_vals = nullptr;
    CHECK_CUDA_ERROR(cudaEventRecord(start_event));
    CHECK_CUDA_ERROR(cudaMalloc(&d_keys, padded_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_vals, padded_size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_keys, h_padded_ratings.data(), padded_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_vals, h_padded_indices.data(), padded_size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(stop_event));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_alloc_memcpy_h2d_ms, start_event, stop_event));
    int num_stages = static_cast<int>(std::log2(padded_size));
    int threads_per_block = 256;
    int blocks_per_grid = (padded_size + threads_per_block - 1) / threads_per_block;
    CHECK_CUDA_ERROR(cudaEventRecord(start_event));
    for (int stage = 0; stage < num_stages; ++stage)
    {
        for (int sub_stage_pass = 0; sub_stage_pass <= stage; ++sub_stage_pass)
        {
            bitonicSortStep<<<blocks_per_grid, threads_per_block>>>(
                d_keys, d_vals, padded_size, stage + 1, sub_stage_pass + 1);
            CHECK_CUDA_ERROR(cudaGetLastError());
        }
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop_event));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_kernel_ms, start_event, stop_event));
    std::vector<int> h_sorted_indices(padded_size);
    CHECK_CUDA_ERROR(cudaEventRecord(start_event));
    CHECK_CUDA_ERROR(cudaMemcpy(h_sorted_indices.data(), d_vals, padded_size * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaFree(d_keys));
    CHECK_CUDA_ERROR(cudaFree(d_vals));
    CHECK_CUDA_ERROR(cudaEventRecord(stop_event));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_memcpy_d2h_free_ms, start_event, stop_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_event));
    std::cout << "Sorting complete." << std::endl;
    std::cout << "\n--- Top 10 Rated Products (ASIN) ---" << std::endl;
    int num_to_show = std::min(num_elements, 10);
    std::cout << std::fixed << std::setprecision(4);
    int displayed_count = 0;
    for (int i = 0; i < padded_size && displayed_count < num_to_show; ++i)
    {
        int original_index = h_sorted_indices[i];
        if (original_index >= 0 && original_index < num_elements)
        {
            float rating = h_avg_ratings[original_index];
            std::string asin = h_asins[original_index];
            std::cout << displayed_count + 1 << ". ASIN: " << asin << " (Avg Rating: " << rating
                      << ")" << std::endl;
            displayed_count++;
        }
    }
    if (displayed_count == 0 && num_elements > 0)
    {
        std::cerr << "Warning: Could not display any top products." << std::endl;
    }
    auto total_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_duration = total_end_time - total_start_time;
    std::cout << "\n--- Timing Information ---" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CPU Reading/Parsing/Aggregation: " << cpu_read_duration.count() << " ms" << std::endl;
    std::cout << "CPU Average Calculation:         " << cpu_avg_duration.count() << " ms" << std::endl;
    std::cout << "GPU Allocation & H2D Memcpy:     " << gpu_alloc_memcpy_h2d_ms << " ms" << std::endl;
    std::cout << "GPU Kernel Execution (Sort):     " << gpu_kernel_ms << " ms" << std::endl;
    std::cout << "GPU D2H Memcpy & Free:           " << gpu_memcpy_d2h_free_ms << " ms" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    std::cout << "Total Execution Time:            " << total_duration.count() << " ms" << std::endl;
    return 0;
}