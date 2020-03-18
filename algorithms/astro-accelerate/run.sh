#!/bin/bash
FREQ_REF_NAME='high' 
FREQ_REF=$(python3 get_fil_header.py $file $FREQ_REF_NAME)
BASENAME=/tank/users/connor/eyra/data/ # this is /data in the master branch
OUTPUT_PATH=$BASENAME/output

python3 /app/find_candidates.py $BASENAME/input/test_data

cat /tmp/output | awk -F" " -v FREQ_REF="$FREQ_REF" '{ print $0 " " FREQ_REF }' > $OUTPUT_PATH

