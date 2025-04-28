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

#define CHECK_CUDA_ERROR(err)                                            \
    if (err != cudaSuccess)                                              \
    {                                                                    \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));                                \
        exit(EXIT_FAILURE);                                              \
    }

// function to trim leading/trailing whitespace and punctuation
std::string trim(const std::string &str)
{
    size_t first = str.find_first_not_of(" \t\n\r\f\v.,!?;:\"'()[]{}");
    if (std::string::npos == first)
    {
        return str;
    }
    size_t last = str.find_last_not_of(" \t\n\r\f\v.,!?;:\"'()[]{}");
    return str.substr(first, (last - first + 1));
}

// function to convert string to lowercase
std::string toLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c)
                   { return std::tolower(c); });
    return s;
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

    // start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // load lexicon (cpu)
    std::cout << "loading lexicon from: " << lexicon_filename << "..." << std::endl;
    std::unordered_map<std::string, float> lexicon_scores;
    std::ifstream lexicon_file(lexicon_filename);
    if (!lexicon_file.is_open())
    {
        std::cerr << "error opening lexicon file: " << lexicon_filename
                  << std::endl;
        return 1;
    }

    std::string line;
    int lexicon_count = 0;
    while (std::getline(lexicon_file, line))
    {
        std::stringstream ss(line);
        std::string lexicon_term;
        float score;
        if (std::getline(ss, lexicon_term, '\t') && (ss >> score))
        {
            lexicon_scores[lexicon_term] = score;
            lexicon_count++;
        }
        else
        {
            // warning: skipping malformed lexicon line:
        }
    }
    lexicon_file.close();
    std::cout << "loaded " << lexicon_count << " lexicon terms." << std::endl;

    if (lexicon_scores.empty())
    {
        std::cerr << "error: lexicon is empty or failed to load." << std::endl;
        return 1;
    }

    // process reviews (cpu)
    std::cout << "processing reviews from: " << reviews_filename << "..." << std::endl;
    std::ifstream reviews_file(reviews_filename);
    if (!reviews_file.is_open())
    {
        std::cerr << "error opening reviews file: " << reviews_filename
                  << std::endl;
        return 1;
    }

    long long line_count = 0;
    long long processed_count = 0;
    long long error_count = 0;
    long long positive_count = 0;
    long long negative_count = 0;
    long long neutral_count = 0; // count reviews with score 0

    rapidjson::Document document;

    // add a trivial cuda call to demonstrate usage
    int deviceCount;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount > 0)
    {
        std::cout << "found " << deviceCount << " cuda capable device(s)." << std::endl;
        CHECK_CUDA_ERROR(cudaSetDevice(0)); // use the first device
    }
    else
    {
        std::cout << "no cuda capable devices found." << std::endl;
    }

    while (std::getline(reviews_file, line))
    {
        line_count++;

        document.Parse(line.c_str());

        if (document.HasParseError())
        {
            // warning: json parse error on line
            error_count++;
            continue;
        }

        if (document.IsObject() && document.HasMember("reviewText") &&
            document["reviewText"].IsString())
        {
            try
            {
                const char *review_text_cstr = document["reviewText"].GetString();
                std::string review_text(review_text_cstr);

                // sentiment calculation (cpu)
                float total_score = 0.0f;
                int words_matched = 0;
                std::stringstream text_stream(review_text);
                std::string word;

                while (text_stream >> word)
                {
                    // basic cleaning: remove punctuation, convert to lower
                    std::string cleaned_word = trim(word);
                    cleaned_word = toLower(cleaned_word); // match lexicon case-insensitively

                    if (!cleaned_word.empty())
                    {
                        auto it = lexicon_scores.find(cleaned_word);
                        if (it != lexicon_scores.end())
                        {
                            // found word in lexicon
                            total_score += it->second; // add score
                            words_matched++;
                        }
                    }
                }

                // labeling
                std::string label = "neutral"; // default if score is 0
                if (total_score > 0.0f)
                {
                    label = "positive";
                    positive_count++;
                }
                else if (total_score < 0.0f)
                {
                    label = "negative";
                    negative_count++;
                }
                else
                {
                    neutral_count++;
                }

                processed_count++;

                // print first few results
            }
            catch (const std::exception &e)
            {
                std::cerr << "warning: exception processing review on line " << line_count
                          << ": " << e.what() << std::endl;
                error_count++;
            }
        }
        else
        {
            // warning: skipping line due to missing/invalid 'reviewtext'.
            error_count++;
        }

        if (line_count % 100000 == 0)
        {
            std::cout << "  processed " << line_count << " lines..." << std::endl;
        }
    }
    reviews_file.close();

    // end timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    // sentiment analysis summary
    std::cout << "\nSentiment Analysis Results:" << std::endl;
    std::cout << "Total reviews read:        " << line_count << std::endl;
    std::cout << "Reviews successfully analyzed:  " << processed_count << std::endl;
    std::cout << "Reviews skipped or with errors: " << error_count << std::endl;
    std::cout << "----------------------------------" << std::endl;
    std::cout << "Positive sentiment: " << positive_count << std::endl;
    std::cout << "Negative sentiment: " << negative_count << std::endl;
    std::cout << "Neutral sentiment:  " << neutral_count << std::endl;
    std::cout << "----------------------------------" << std::endl;
    // print total execution time
    std::cout << "Total time taken:    " << std::fixed << std::setprecision(3)
              << elapsed_seconds.count() << " seconds" << std::endl;
    std::cout << "----------------------------------" << std::endl;

    // another trivial cuda call
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    std::cout << "cuda device synchronized (if used)." << std::endl;

    return 0;
}
