#!/bin/bash
#BASENAME=/tank/users/connor/eyra/data/ # this is /data in the master branch
BASENAME=/data
INPUT_FILE=$BASENAME/input/test_data
OUTPUT_PATH=$BASENAME/astro-accelerate

FREQ_REF_NAME='high' 
FREQ_REF=$(python3 get_fil_header.py $INPUT_FILE $FREQ_REF_NAME)

python3 ./find_candidates.py $INPUT_FILE


cat /tmp/output | awk -F" " -v FREQ_REF="$FREQ_REF" '{ print $0 " " FREQ_REF }' > $OUTPUT_PATH

