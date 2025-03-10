#!/bin/bash
day=$1
curr=$PWD
dest="/Volumes/Hippocampus/Data/picasso-misc/$day/session01"
cd $dest
find . -type d \( -name "channel0*" -o -name "channel1*" \) | cut -d "/" -f 2-4 > ~/Documents/GitHub/Pyhipp_T/channel_list.txt
cd $curr
