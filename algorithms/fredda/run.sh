#!/bin/bash

BASENAME=/tank/users/connor/eyra/data # this is /data in the master branch    
#BASENAME=/data
TMP_PATH="/opt/frbbench/fredda_tmp"
INPUT_PATH="$BASENAME/input/test_data"
OUTPUT_PATH="$BASENAME/fredda"

FREQ_REF_NAME='low' 
FREQ_REF=$(python3 ./get_fil_header.py $INPUT_PATH $FREQ_REF_NAME)

#g: cuda device
#t: samples per block
#x: S/N threshold
#d: number of dispersion trials in units of sampling time; ASKAP data has 1.2656 ms sampling, freq range 1165-1500
# DM 10000 results in delay of 12.13 s = 9584 trials 
# then round up to nearest multiple of samples per block: 9728
cudafdmt -g 0 -t 512 -d 9728 -x 6 -o $TMP_PATH $BASENAME/input/test_data

tail -n +2 $TMP_PATH | awk -F" " -v FREQ_REF="$FREQ_REF" '{ print $6 " " $1 " " $3 " " $4 " " FREQ_REF }' > $OUTPUT_PATH

