#!/bin/bash
ROOT="/data/"
BASE_NAME="/opt/frbbench/presto_run/output"  # has to match Dockerfile
INPUT_PATH="$ROOT/input/test_data"
OUTPUT_FILE="$ROOT/presto"

# Heimdall uses the highest frequency as its arrival time reference freq
FREQ_REF_NAME='high' 
FREQ_REF=$(python get_fil_header.py $INPUT_PATH $FREQ_REF_NAME)

# dedispersion
python gen_dedisp_commands.py -s $SYSTEM --filterbank $INPUT_PATH -o $BASE_NAME > dedisp.sh
./parallel.sh dedisp.sh

# single pulse search
python gen_sps_commands.py $BASE_NAME*dat > sps.sh
./parallel.sh sps.sh

# cols: DM, S/N, TIME, DOWNSAMPLE, FREQ_REF
cat $BASE_PATH*.singlepulse | awk -F" " -v FREQ_REF="$FREQ_REF" '{ print $1 " " $2 " " $3 " " $4 " " FREQ_REF }' > $OUTPUT_FILE
