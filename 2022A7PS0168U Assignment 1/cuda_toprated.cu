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
        fprintf(stderr, "cuda error in %s:%d: %s\n", __file__, __line__, \
                cudaGetErrorString(err));                                \
        exit(exit_failure);                                              \
    }

// gpu kernel for a single step of bitonic sort
__global__ void bitonicSortStep(float *keys, int *vals, int n, int stage,
                                int sub_stage_pass)
{
    int tid = blockidx.x * blockdim.x + threadidx.x;
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

// finds the next power of 2 greater than or equal to n
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
        std::cerr << "usage: " << argv[0] << " <path_to_reviews.json>"
                  << std::endl;
        return 1;
    }
    std::string filename = argv[1];
    std::ifstream infile(filename);

    if (!infile.is_open())
    {
        std::cerr << "error opening file: " << filename << std::endl;
        return 1;
    }

    auto cpu_read_start = std::chrono::high_resolution_clock::now();
    std::cout << "reading reviews and calculating sums/counts (cpu using "
                 "rapidjson)..."
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
        document.parse(line.c_str());
        if (document.hasparseerror())
        {
            error_count++;
            continue;
        }
        if (document.isobject() &&
            document.hasmember("asin") && document["asin"].isstring() &&
            document.hasmember("overall") && document["overall"].isnumber())
        {
            try
            {
                std::string asin = document["asin"].getstring();
                float rating = document["overall"].getfloat();
                product_ratings[asin].first += rating;
                product_ratings[asin].second++;
                processed_count++;
            }
            catch (const std::exception &e)
            {
                std::cerr << "warning: exception accessing data on line " << line_count
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

    std::cout << "finished reading. processed " << processed_count
              << " valid reviews for " << product_ratings.size()
              << " unique products." << std::endl;
    if (error_count > 0)
    {
        std::cout << "  skipped " << error_count
                  << " lines due to parsing errors or missing/invalid fields."
                  << std::endl;
    }
    if (product_ratings.empty())
    {
        std::cerr << "no valid product data found." << std::endl;
        return 1;
    }

    auto cpu_avg_start = std::chrono::high_resolution_clock::now();
    std::cout << "calculating average ratings (cpu)..." << std::endl;
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

    std::cout << "calculated " << num_elements << " average ratings." << std::endl;
    if (num_elements == 0)
    {
        std::cerr << "no products with ratings found after averaging." << std::endl;
        return 1;
    }

    std::cout << "sorting average ratings (gpu using bitonic sort)..." << std::endl;

    cudaEvent_t start_event, stop_event;
    check_cuda_error(cudaEventCreate(&start_event));
    check_cuda_error(cudaEventCreate(&stop_event));
    float gpu_alloc_memcpy_h2d_ms = 0;
    float gpu_kernel_ms = 0;
    float gpu_memcpy_d2h_free_ms = 0;

    int padded_size = nextPowerOf2(num_elements);
    std::cout << "  padding data from " << num_elements << " to " << padded_size << " elements." << std::endl;
    std::vector<float> h_padded_ratings(padded_size, -1.0f);
    std::vector<int> h_padded_indices(padded_size, -1);
    std::copy(h_avg_ratings.begin(), h_avg_ratings.end(), h_padded_ratings.begin());
    std::copy(h_original_indices.begin(), h_original_indices.end(), h_padded_indices.begin());

    float *d_keys = nullptr;
    int *d_vals = nullptr;

    check_cuda_error(cudaEventRecord(start_event));
    check_cuda_error(cudaMalloc(&d_keys, padded_size * sizeof(float)));
    check_cuda_error(cudaMalloc(&d_vals, padded_size * sizeof(int)));
    check_cuda_error(cudaMemcpy(d_keys, h_padded_ratings.data(), padded_size * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(d_vals, h_padded_indices.data(), padded_size * sizeof(int), cudaMemcpyHostToDevice));
    check_cuda_error(cudaEventRecord(stop_event));
    check_cuda_error(cudaEventSynchronize(stop_event));
    check_cuda_error(cudaEventElapsedTime(&gpu_alloc_memcpy_h2d_ms, start_event, stop_event));

    int num_stages = static_cast<int>(std::log2(padded_size));
    int threads_per_block = 256;
    int blocks_per_grid = (padded_size + threads_per_block - 1) / threads_per_block;

    check_cuda_error(cudaEventRecord(start_event));
    for (int stage = 0; stage < num_stages; ++stage)
    {
        for (int sub_stage_pass = 0; sub_stage_pass <= stage; ++sub_stage_pass)
        {
            bitonicSortStep<<<blocks_per_grid, threads_per_block>>>(
                d_keys, d_vals, padded_size, stage + 1, sub_stage_pass + 1);
            check_cuda_error(cudaGetLastError());
        }
    }
    check_cuda_error(cudaEventRecord(stop_event));
    check_cuda_error(cudaEventSynchronize(stop_event));
    check_cuda_error(cudaEventElapsedTime(&gpu_kernel_ms, start_event, stop_event));

    std::vector<int> h_sorted_indices(padded_size);
    check_cuda_error(cudaEventRecord(start_event));
    check_cuda_error(cudaMemcpy(h_sorted_indices.data(), d_vals, padded_size * sizeof(int), cudaMemcpyDeviceToHost));
    check_cuda_error(cudaFree(d_keys));
    check_cuda_error(cudaFree(d_vals));
    check_cuda_error(cudaEventRecord(stop_event));
    check_cuda_error(cudaEventSynchronize(stop_event));
    check_cuda_error(cudaEventElapsedTime(&gpu_memcpy_d2h_free_ms, start_event, stop_event));

    check_cuda_error(cudaEventDestroy(start_event));
    check_cuda_error(cudaEventDestroy(stop_event));

    std::cout << "sorting complete." << std::endl;

    std::cout << "\n--- top 10 rated products (asin) ---" << std::endl;
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
            std::cout << displayed_count + 1 << ". asin: " << asin << " (avg rating: " << rating
                      << ")" << std::endl;
            displayed_count++;
        }
    }
    if (displayed_count == 0 && num_elements > 0)
    {
        std::cerr << "warning: could not display any top products." << std::endl;
    }

    auto total_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_duration = total_end_time - total_start_time;

    std::cout << "\n--- timing information ---" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "cpu reading/parsing/aggregation: " << cpu_read_duration.count() << " ms" << std::endl;
    std::cout << "cpu average calculation:         " << cpu_avg_duration.count() << " ms" << std::endl;
    std::cout << "gpu allocation & h2d memcpy:     " << gpu_alloc_memcpy_h2d_ms << " ms" << std::endl;
    std::cout << "gpu kernel execution (sort):     " << gpu_kernel_ms << " ms" << std::endl;
    std::cout << "gpu d2h memcpy & free:           " << gpu_memcpy_d2h_free_ms << " ms" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    std::cout << "total execution time:            " << total_duration.count() << " ms" << std::endl;

    return 0;
}