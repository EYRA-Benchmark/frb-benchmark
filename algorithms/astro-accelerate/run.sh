#!/bin/bash
FREQ_REF_NAME='high' 
FREQ_REF=$(python3 get_fil_header.py $file $FREQ_REF_NAME)

python3 /app/find_candidates.py /data/input/test_data

cat /tmp/output | awk -F" " -v FREQ_REF="$FREQ_REF" '{ print $0 " " FREQ_REF }' > /data/output

