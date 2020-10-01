#!/bin/bash

#BASENAME=/tank/users/connor/eyra/data # this is /data in the master branch                           
BASENAME=/data
file="$BASENAME/input/test_data"
nbatch=$NBATCH

conf_dir="/opt/frbbench/conf"
snrmin=6

# Amber uses the highest frequency as its arrival time reference freq
FREQ_REF_NAME='high' 
FREQ_REF=$(python get_fil_header.py $file $FREQ_REF_NAME)

# run AMBER
for step in {1..4}; do
    source scenarios/askap_step${step}.sh
    output="amber_step${step}"
    int_steps=integration_steps_askap_step${step}.conf
    if [ $DOWNSAMPLING -eq 1 ]; then
        ds_conf=""
    else
        ds_conf="-downsampling -downsampling_factor $DOWNSAMPLING"
    fi
    amber -debug -opencl_platform ${OPENCL_PLATFORM} -opencl_device ${OPENCL_DEVICE} -device_name ${DEVICE_NAME} -sync -print -snr_sc -padding_file $conf_dir/padding.conf -zapped_channels $conf_dir/zapped_channels.conf -subband_dedispersion -dedispersion_stepone_file $conf_dir/dedispersion_stepone.conf -dedispersion_steptwo_file $conf_dir/dedispersion_steptwo.conf -integration_file $conf_dir/integration.conf -integration_steps $conf_dir/$int_steps -snr_file $conf_dir/snr.conf -downsampling_configuration $conf_dir/downsampling.conf -output $output $ds_conf -subbands ${SUBBANDS} -dms ${DMS} -dm_first ${DM_FIRST} -dm_step ${DM_STEP} -subbanding_dms ${SUBBANDING_DMS} -subbanding_dm_first ${SUBBANDING_DM_FIRST} -subbanding_dm_step ${SUBBANDING_DM_STEP} -threshold $snrmin -sigproc -stream -data $file -batches $nbatch -channel_bandwidth ${CHANNEL_BANDWIDTH} -min_freq ${MIN_FREQ} -channels ${CHANNELS} -samples ${SAMPLES} -sampling_time ${SAMPLING_TIME} -nsigma ${SNR_SIGMA} -correction_factor ${CORRECTION_FACTOR} -compact_results &
done
wait


# cut headers
# select columns: DM, S/N, TIME, WIDTH, FREQ_REF
cat amber_step*.trigger | grep -v \# | awk -F" " -v FREQ_REF="$FREQ_REF" '{ print $8 " " $10 " " $6 " " $4 " " FREQ_REF }' > $BASENAME/amber
