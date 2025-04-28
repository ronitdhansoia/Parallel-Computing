#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <utility>
#include <stdexcept>
#include <chrono>
#include <iomanip>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

int countWords(const std::string &text)
{
    std::stringstream ss(text);
    std::string word;
    int count = 0;
    while (ss >> word)
    {
        count++;
    }
    return count;
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

    auto processing_start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Processing reviews sequentially (CPU)..." << std::endl;

    std::map<std::string, int> elaborate_review_counts;
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

        if (document.IsObject() && document.HasMember("reviewerID") &&
            document["reviewerID"].IsString() &&
            document.HasMember("reviewText") &&
            document["reviewText"].IsString())
        {
            try
            {
                std::string reviewerID = document["reviewerID"].GetString();
                std::string reviewText = document["reviewText"].GetString();
                int word_count = countWords(reviewText);
                if (word_count >= 50)
                {
                    elaborate_review_counts[reviewerID]++;
                }
                processed_count++;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Warning: Exception processing data on line "
                          << line_count << ": " << e.what() << std::endl;
                error_count++;
            }
        }
        else
        {
            error_count++;
        }

    }
    infile.close();
    auto processing_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> processing_duration = processing_end_time - processing_start_time;

    std::cout << "Finished processing " << line_count << " lines." << std::endl;
    std::cout << "  Found " << processed_count << " valid reviews." << std::endl;
    std::cout << "  Encountered " << error_count << " errors/skipped lines."
              << std::endl;
    std::cout << "  Tracking elaborate reviews for "
              << elaborate_review_counts.size() << " unique reviewers."
              << std::endl;

    auto filtering_start_time = std::chrono::high_resolution_clock::now();
    std::cout << "\n--- Elaborate Reviewers (>= 5 reviews with >= 50 words) ---"
              << std::endl;
    std::vector<std::string> elaborate_reviewers;
    elaborate_reviewers.reserve(elaborate_review_counts.size() / 10);

    for (const auto &pair : elaborate_review_counts)
    {
        if (pair.second >= 5)
        {
            elaborate_reviewers.push_back(pair.first);
        }
    }
    auto filtering_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> filtering_duration = filtering_end_time - filtering_start_time;

    std::cout << "Found " << elaborate_reviewers.size()
              << " elaborate reviewers." << std::endl;

    auto total_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_duration = total_end_time - total_start_time;

    std::cout << "\n--- Timing Information ---" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CPU Reading/Parsing/Aggregation: " << processing_duration.count() << " ms" << std::endl;
    std::cout << "CPU Filtering:                   " << filtering_duration.count() << " ms" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    std::cout << "Total Execution Time:            " << total_duration.count() << " ms" << std::endl;

    return 0;
}
