#!/bin/bash
#BASENAME=/tank/users/connor/eyra/data/ # this is /data in the master branch                                 
BASENAME=/data

TMP_PATH="/opt/frbbench/heimdall_tmp"
INPUT_PATH="$BASENAME/input/test_data"
OUTPUT_PATH="$BASENAME/heimdall"

# Heimdall uses the highest frequency as its arrival time reference freq
FREQ_REF_NAME='high' 
FREQ_REF=$(python get_fil_header.py $INPUT_PATH $FREQ_REF_NAME)

mkdir -p $TMP_PATH
heimdall -v -detect_thresh 6.0 -dm_to 1.01 -nsamps_gulp 1048576 -f $INPUT_PATH -dm 0. 10000. -rfi_no_narrow -rfi_no_broad -output_dir $TMP_PATH
# cols: DM, S/N, TIME, LOG_2_DOWNSAMPLE, FREQ_REF
cat $TMP_PATH/*.cand | awk -F" " -v FREQ_REF="$FREQ_REF" '{ print $6 " " $1 " " $3 " " 2^$4 " " FREQ_REF }' > $OUTPUT_PATH
