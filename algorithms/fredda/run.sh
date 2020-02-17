#!/bin/bash
TMP_PATH="/tmp/fredda"
INPUT_PATH="/data/input/test_data"
OUTPUT_PATH="/data/output"

FREQ_REF_NAME='low' 
FREQ_REF=$(python3 get_fil_header.py $INPUT_PATH $FREQ_REF_NAME)

cudafdmt -t 512 -d 4096 -x 8 -o $TMP_PATH /data/input/test_data

tail -n +2 $TMP_PATH | awk -F" " -v FREQ_REF="$FREQ_REF" '{ print $6 " " $1 " " $3 " " $4 " " FREQ_REF }' > /data/output

