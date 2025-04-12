#!/bin/bash

# Set paths for the original and new directory
original_dir="/work/datasets/tdt4265/ad/open/Poles/lidar/"
new_dir="/work/tobiart/TDT4265-Mini-Project/data/lidar/"

# Create the new directory structure if it doesn't already exist
mkdir -p "$new_dir/train/images" "$new_dir/train/labels"
mkdir -p "$new_dir/valid/images" "$new_dir/valid/labels"
mkdir -p "$new_dir/test/images"

# Copy images and labels for the train set
cp -r "$original_dir/combined_color/train/"* "$new_dir/train/images/"
cp -r "$original_dir/labels/train/"* "$new_dir/train/labels/"

# Copy images and labels for the valid set
cp -r "$original_dir/combined_color/valid/"* "$new_dir/valid/images/"
cp -r "$original_dir/labels/valid/"* "$new_dir/valid/labels/"

# Copy images for the test set (no labels in the original test set)
cp -r "$original_dir/combined_color/test/"* "$new_dir/test/images/"

echo "Data has been successfully copied and restructured!"




# To run, execute the following commands:
# chmod +x load_dataset.sh
# ./load_dataset.sh