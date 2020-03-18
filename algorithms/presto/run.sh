#!/bin/bash
TMP_PATH="/tank/users/connor/eyra/presto"
BASE_PATH="/tank/users/connor/eyra/intermediate"
INPUT_PATH="/tank/users/connor/eyra/input/test_data"
OUTPUT_PATH="/tank/users/connor/eyra/output_presto"
# Heimdall uses the highest frequency as its arrival time reference freq
FREQ_REF_NAME='high' 
FREQ_REF=$(python get_fil_header.py $INPUT_PATH $FREQ_REF_NAME)
mkdir -p $TMP_PATH

python dedisp_FRB_challenge.py $BASE_PATH $INPUT_PATH

# cols: DM, S/N, TIME, LOG_2_DOWNSAMPLE, FREQ_REF
cat $BASE_PATH*.singlepulse | awk -F" " -v FREQ_REF="$FREQ_REF" '{ print $1 " " $2 " " $3 " " $4 " " FREQ_REF }' > $OUTPUT_PATH
#cat $BASE_PATH*.singlepulse | awk -F" " -v FREQ_REF="$FREQ_REF" '{ print $1 " " $2 " " $3 " " $4 " " FREQ_REF }' > $OUTPUT_PATH
