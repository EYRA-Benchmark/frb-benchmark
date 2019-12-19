#!/bin/bash
TMP_PATH="/tmp/heimdall"
INPUT_PATH="/data/input/test_data"
OUTPUT_PATH="/data/output"
# Heimdall uses the highest frequency as its arrival time reference freq
FREQ_REF_NAME='high' 
FREQ_REF=$(python get_fil_header.py $INPUT_PATH $FREQ_REF_NAME)
mkdir -p $TMP_PATH
heimdall -v -f $INPUT_PATH -dm 10. 3000. -rfi_no_narrow -rfi_no_broad -output_dir $TMP_PATH
# cols: DM, S/N, TIME, LOG_2_DOWNSAMPLE, FREQ_REF
cat $TMP_PATH/*.cand | awk -F" " -v FREQ_REF="$FREQ_REF" '{ print $6 " " $1 " " $3 " " 2^$4 " " FREQ_REF }' > $OUTPUT_PATH
