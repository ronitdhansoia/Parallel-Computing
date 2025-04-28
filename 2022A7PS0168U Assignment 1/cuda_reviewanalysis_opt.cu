#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <stdexcept>
#include <iomanip>
#include <cctype>
#include <algorithm>
#include <chrono>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

#include <cuda_runtime.h>

// Simple macro to check CUDA errors
#define CHECK_CUDA_ERROR(err)                                       \
    if (err != cudaSuccess)                                         \
    {                                                               \
        fprintf(stderr, "CUDA error in %s:%d: %s (%d)\n", __FILE__, \
                __LINE__, cudaGetErrorString(err), err);            \
        exit(EXIT_FAILURE);                                         \
    }

// --- Helper Functions (Trim, Lowercase - same as before) ---
std::string trim(const std::string &str)
{
    size_t first = str.find_first_not_of(" \t\n\r\f\v.,!?;:\"'()[]{}");
    if (std::string::npos == first)
    {
        return str; // Return empty string if all whitespace/punctuation
    }
    size_t last = str.find_last_not_of(" \t\n\r\f\v.,!?;:\"'()[]{}");
    return str.substr(first, (last - first + 1));
}

std::string toLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c)
                   { return std::tolower(c); });
    return s;
}

// --- CUDA Kernel ---
__global__ void sentimentKernel(const int *packed_review_word_ids,
                                const int *review_offsets,
                                const int *review_lengths,
                                const float *lexicon_scores_gpu,
                                int num_reviews, float *output_scores,
                                int lexicon_size) // Pass lexicon size for bounds check
{
    int review_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (review_idx >= num_reviews)
    {
        return; // Thread is outside the bounds of reviews for this batch
    }

    int offset = review_offsets[review_idx];
    int length = review_lengths[review_idx];
    float local_score = 0.0f;

    for (int i = 0; i < length; ++i)
    {
        int word_id = packed_review_word_ids[offset + i];

        // Check if word_id is valid (e.g., >= 0 and within lexicon bounds)
        // Assumes word_id == -1 for unknown words during preprocessing
        if (word_id >= 0 && word_id < lexicon_size)
        {
            // Direct access using word_id as index
            local_score += lexicon_scores_gpu[word_id];
            // Note: Access to lexicon_scores_gpu is likely uncoalesced
        }
    }

    output_scores[review_idx] = local_score;
    // Note: Write to output_scores is likely uncoalesced
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <path_to_reviews.json> <path_to_lexicon.txt>"
                  << std::endl;
        return 1;
    }
    std::string reviews_filename = argv[1];
    std::string lexicon_filename = argv[2];

    auto start_time = std::chrono::high_resolution_clock::now();

    // --- 1. Load Lexicon and Build Word-to-ID Map (CPU) ---
    std::cout << "Loading lexicon and building ID map from: "
              << lexicon_filename << "..." << std::endl;
    std::unordered_map<std::string, int> word_to_id;
    std::vector<float> lexicon_scores_cpu;
    std::ifstream lexicon_file(lexicon_filename);
    if (!lexicon_file.is_open())
    {
        std::cerr << "Error opening lexicon file: " << lexicon_filename
                  << std::endl;
        return 1;
    }

    std::string line;
    int current_id = 0;
    while (std::getline(lexicon_file, line))
    {
        std::stringstream ss(line);
        std::string lexicon_term;
        float score;
        if (std::getline(ss, lexicon_term, '\t') && (ss >> score))
        {
            std::string cleaned_term = toLower(trim(lexicon_term)); // Use lowercase for map
            if (!cleaned_term.empty() && word_to_id.find(cleaned_term) == word_to_id.end())
            {
                word_to_id[cleaned_term] = current_id;
                lexicon_scores_cpu.push_back(score);
                current_id++;
            }
        }
    }
    lexicon_file.close();
    int lexicon_size = lexicon_scores_cpu.size();
    std::cout << "Loaded " << lexicon_size << " unique lexicon terms."
              << std::endl;

    if (lexicon_size == 0)
    {
        std::cerr << "Error: Lexicon is empty or failed to load." << std::endl;
        return 1;
    }

    // --- 2. Allocate and Transfer Lexicon to GPU ---
    float *lexicon_scores_gpu = nullptr;
    size_t lexicon_bytes = lexicon_size * sizeof(float);
    std::cout << "Allocating " << lexicon_bytes / (1024.0 * 1024.0)
              << " MB on GPU for lexicon..." << std::endl;
    CHECK_CUDA_ERROR(cudaMalloc(&lexicon_scores_gpu, lexicon_bytes));
    std::cout << "Copying lexicon to GPU..." << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpy(lexicon_scores_gpu, lexicon_scores_cpu.data(),
                                lexicon_bytes, cudaMemcpyHostToDevice));

    // --- 3. Process Reviews in Batches ---
    std::cout << "Processing reviews from: " << reviews_filename << "..." << std::endl;
    std::ifstream reviews_file(reviews_filename);
    if (!reviews_file.is_open())
    {
        std::cerr << "Error opening reviews file: " << reviews_filename
                  << std::endl;
        CHECK_CUDA_ERROR(cudaFree(lexicon_scores_gpu)); // Clean up GPU memory
        return 1;
    }

    long long line_count = 0;
    long long processed_count = 0;
    long long error_count = 0;
    long long positive_count = 0;
    long long negative_count = 0;
    long long neutral_count = 0;

    const size_t BATCH_SIZE = 262144; // Reviews per batch (tune this)
    std::vector<std::string> review_batch_text;
    review_batch_text.reserve(BATCH_SIZE);

    rapidjson::Document document;

    // --- Batch Processing Loop ---
    while (true)
    {
        review_batch_text.clear();
        // Read a batch of reviews
        while (review_batch_text.size() < BATCH_SIZE && std::getline(reviews_file, line))
        {
            line_count++;
            document.Parse(line.c_str());

            if (document.HasParseError())
            {
                error_count++;
                continue;
            }

            if (document.IsObject() && document.HasMember("reviewText") &&
                document["reviewText"].IsString())
            {
                review_batch_text.push_back(document["reviewText"].GetString());
            }
            else
            {
                error_count++;
            }
        }

        if (review_batch_text.empty())
        {
            break; // No more reviews or file ended
        }

        size_t current_batch_size = review_batch_text.size();

        // --- CPU Preprocessing for the Batch ---
        std::vector<int> batch_packed_ids;
        std::vector<int> batch_offsets;
        std::vector<int> batch_lengths;
        batch_offsets.reserve(current_batch_size);
        batch_lengths.reserve(current_batch_size);
        int current_offset = 0;

        for (const std::string &review_text : review_batch_text)
        {
            batch_offsets.push_back(current_offset);
            int word_count = 0;
            std::stringstream text_stream(review_text);
            std::string word;
            while (text_stream >> word)
            {
                std::string cleaned_word = toLower(trim(word));
                if (!cleaned_word.empty())
                {
                    auto it = word_to_id.find(cleaned_word);
                    if (it != word_to_id.end())
                    {
                        batch_packed_ids.push_back(it->second); // Store ID
                    }
                    else
                    {
                        batch_packed_ids.push_back(-1); // Mark unknown words
                    }
                    word_count++;
                }
            }
            batch_lengths.push_back(word_count);
            current_offset += word_count;
        }

        // --- Allocate and Transfer Batch Data to GPU ---
        int *packed_ids_gpu = nullptr;
        int *offsets_gpu = nullptr;
        int *lengths_gpu = nullptr;
        float *output_scores_gpu = nullptr;

        size_t packed_bytes = batch_packed_ids.size() * sizeof(int);
        size_t offsets_bytes = batch_offsets.size() * sizeof(int);
        size_t lengths_bytes = batch_lengths.size() * sizeof(int);
        size_t output_bytes = current_batch_size * sizeof(float);

        CHECK_CUDA_ERROR(cudaMalloc(&packed_ids_gpu, packed_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&offsets_gpu, offsets_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&lengths_gpu, lengths_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&output_scores_gpu, output_bytes));

        CHECK_CUDA_ERROR(cudaMemcpy(packed_ids_gpu, batch_packed_ids.data(),
                                    packed_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(offsets_gpu, batch_offsets.data(),
                                    offsets_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(lengths_gpu, batch_lengths.data(),
                                    lengths_bytes, cudaMemcpyHostToDevice));

        // --- Launch Kernel ---
        int threads_per_block = 256; // Common block size
        int blocks_per_grid =
            (current_batch_size + threads_per_block - 1) / threads_per_block;

        sentimentKernel<<<blocks_per_grid, threads_per_block>>>(
            packed_ids_gpu, offsets_gpu, lengths_gpu, lexicon_scores_gpu,
            current_batch_size, output_scores_gpu, lexicon_size);
        CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors

        // --- Copy Results Back to CPU ---
        std::vector<float> batch_scores_cpu(current_batch_size);
        CHECK_CUDA_ERROR(cudaMemcpy(batch_scores_cpu.data(), output_scores_gpu,
                                    output_bytes, cudaMemcpyDeviceToHost));

        // --- Free GPU Batch Memory ---
        CHECK_CUDA_ERROR(cudaFree(packed_ids_gpu));
        CHECK_CUDA_ERROR(cudaFree(offsets_gpu));
        CHECK_CUDA_ERROR(cudaFree(lengths_gpu));
        CHECK_CUDA_ERROR(cudaFree(output_scores_gpu));

        // --- CPU Postprocessing (Labeling and Counting) ---
        for (float score : batch_scores_cpu)
        {
            processed_count++;
            if (score > 0.0f)
            {
                positive_count++;
            }
            else if (score < 0.0f)
            {
                negative_count++;
            }
            else
            {
                neutral_count++;
            }
        }

        std::cout << "  Processed batch ending at line " << line_count
                  << " (Batch Size: " << current_batch_size << ")" << std::endl;

        // Check if we reached the end of the file in the last read attempt
        if (reviews_file.eof() && review_batch_text.size() < BATCH_SIZE)
        {
            break;
        }
    } // End batch processing loop

    reviews_file.close();

    // --- Clean up Lexicon GPU Memory ---
    CHECK_CUDA_ERROR(cudaFree(lexicon_scores_gpu));

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    // --- 4. Print Summary ---
    std::cout << "\n--- Sentiment Analysis Summary (CUDA Accelerated) ---"
              << std::endl;
    std::cout << "Total lines read:        " << line_count << std::endl;
    std::cout << "Successfully processed:  " << processed_count << std::endl;
    std::cout << "Skipped/Errors:        " << error_count << std::endl;
    std::cout << "----------------------------------" << std::endl;
    std::cout << "Positive reviews (score > 0): " << positive_count << std::endl;
    std::cout << "Negative reviews (score < 0): " << negative_count << std::endl;
    std::cout << "Neutral reviews (score = 0):  " << neutral_count << std::endl;
    std::cout << "----------------------------------" << std::endl;
    std::cout << "Total execution time:    " << std::fixed << std::setprecision(3)
              << elapsed_seconds.count() << " seconds" << std::endl;
    std::cout << "----------------------------------" << std::endl;

    return 0;
}
