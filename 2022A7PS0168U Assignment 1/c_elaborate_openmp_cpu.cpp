#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <utility>
#include <stdexcept>
#include <omp.h> // OpenMP Header

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

    std::cout << "Processing reviews (CPU part - sequential)..." << std::endl;
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
            catch (...)
            {
                error_count++;
            }
        }
        else
        {
            error_count++;
        }
    }
    infile.close();
    std::cout << "Finished sequential processing. Found "
              << elaborate_review_counts.size() << " unique reviewers."
              << std::endl;

    if (elaborate_review_counts.empty())
    {
        std::cout << "No reviewers found to process." << std::endl;
        return 0;
    }

    int num_reviewers = elaborate_review_counts.size();
    std::vector<int> h_counts(num_reviewers);
    std::vector<std::string> h_ids(num_reviewers);

    int idx = 0;
    for (const auto &pair : elaborate_review_counts)
    {
        h_ids[idx] = pair.first;
        h_counts[idx] = pair.second;
        idx++;
    }

    std::cout << "Filtering reviewers using OpenMP CPU parallelism..." << std::endl;
    std::vector<std::string> elaborate_reviewers_omp;
    const int threshold = 5;

    double start_time = omp_get_wtime();

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_reviewers; ++i)
    {
        if (h_counts[i] >= threshold)
        {
#pragma omp critical
            {
                elaborate_reviewers_omp.push_back(h_ids[i]);
            }
        }
    }

    double end_time = omp_get_wtime();
    std::cout << "CPU parallel filtering completed in " << (end_time - start_time)
              << " seconds." << std::endl;

    std::cout << "\n--- Elaborate Reviewers (Identified by OpenMP CPU) ---"
              << std::endl;
    std::cout << "Found " << elaborate_reviewers_omp.size()
              << " elaborate reviewers." << std::endl;

    return 0;
}