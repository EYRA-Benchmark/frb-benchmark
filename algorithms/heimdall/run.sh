#!/bin/bash
BASENAME=/tank/users/connor/eyra/data/ # this is /data in the master branch                                 
TMP_PATH="/$BASENAME/heimdall"
INPUT_PATH="$BASENAME/input/test_data"
OUTPUT_PATH="$BASENAME/output"
# Heimdall uses the highest frequency as its arrival time reference freq
FREQ_REF_NAME='high' 
FREQ_REF=$(python get_fil_header.py $INPUT_PATH $FREQ_REF_NAME)
mkdir -p $TMP_PATH
heimdall -v -f $INPUT_PATH -dm 10. 3000. -rfi_no_narrow -rfi_no_broad -output_dir $TMP_PATH
# cols: DM, S/N, TIME, LOG_2_DOWNSAMPLE, FREQ_REF
cat $TMP_PATH/*.cand | awk -F" " -v FREQ_REF="$FREQ_REF" '{ print $6 " " $1 " " $3 " " 2^$4 " " FREQ_REF }' > $OUTPUT_PATH
