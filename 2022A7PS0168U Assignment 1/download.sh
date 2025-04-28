#!/bin/bash

electronics_url="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Electronics_5.json.gz"
electronics_download_name="Electronics_5.json.gz"
electronics_intermediate_name="Electronics_5.json"
electronics_final_name="electronics.json"


vader_url="https://raw.githubusercontent.com/cjhutto/vaderSentiment/refs/heads/master/vaderSentiment/vader_lexicon.txt"

vader_final_name="lexicon.txt"

download_file() {
    local url="$1"
    local filename="$2"

    echo "Attempting to download: $filename from $url"

    if command -v curl >/dev/null 2>&1; then
        echo "Using curl..."
        curl -L -o "$filename" "$url"
        local download_status=$?
    elif command -v wget >/dev/null 2>&1; then
        echo "Using wget..."
        wget -O "$filename" "$url"
        local download_status=$?
    else
        echo "Error: Neither curl nor wget is installed. Please install one of them."
        return 1
    fi
    if [ $download_status -eq 0 ] && [ -f "$filename" ]; then
        echo "Successfully downloaded: $filename"
        return 0 
    else
        echo "Error: Failed to download $filename."
        rm -f "$filename"
        return 1 
    fi
}

download_file "$electronics_url" "$electronics_download_name"
if [ $? -ne 0 ]; then
    exit 1
fi

download_file "$vader_url" "$vader_final_name"
if [ $? -ne 0 ]; then
    rm -f "$electronics_download_name"
    exit 1
fi

echo "Extracting $electronics_download_name using gunzip..."
gunzip "$electronics_download_name"
gunzip_status=$?

if [ $gunzip_status -eq 0 ] && [ -f "$electronics_intermediate_name" ]; then
    echo "File extracted successfully to intermediate name: $electronics_intermediate_name"
    echo "Renaming $electronics_intermediate_name to $electronics_final_name..."
    mv "$electronics_intermediate_name" "$electronics_final_name"
    rename_status=$?

    if [ $rename_status -eq 0 ] && [ -f "$electronics_final_name" ]; then
        echo "Successfully renamed to: $electronics_final_name"
    else
        echo "Error: Failed to rename $electronics_intermediate_name to $electronics_final_name."
        rm -f "$electronics_intermediate_name"
        rm -f "$vader_final_name"
        exit 1
    fi
else
    echo "Error: File extraction failed for $electronics_download_name."
    if [ -f "$electronics_download_name" ]; then
         echo "The .gz file might be corrupted or gunzip encountered an error."
    else
         echo "Intermediate extracted file $electronics_intermediate_name not found after gunzip operation."
    fi
    rm -f "$vader_final_name"
    exit 1
fi

echo "Done! Final files are $electronics_final_name and $vader_final_name."
exit 0
