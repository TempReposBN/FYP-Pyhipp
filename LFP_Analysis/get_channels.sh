#!/bin/bash
day=$1
curr=$PWD
dest="/Volumes/Hippocampus/Data/picasso-misc/$day/session01"

# Navigate to the destination directory
cd "$dest" || { echo "Destination directory not found!"; exit 1; }

# Create channel_list.txt in the current working directory
find . -type d \( -name "channel0*" -o -name "channel1*" \) | cut -d "/" -f 2-4 > "$curr/channel_list.txt"

# Return to the original directory
cd "$curr"
