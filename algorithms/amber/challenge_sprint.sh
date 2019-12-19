#!/bin/bash

# Example tuning scenario

# System
## CPU core to pin the processes to
CPU_CORE="1"
## OpenCL platform ID
OPENCL_PLATFORM="0"
## OpenCL device ID
OPENCL_DEVICE="0"
## Name of OpenCL device, used for configuration files
DEVICE_NAME="ChallengeDevice"
## Size, in bytes, of the OpenCL device's cache line
DEVICE_PADDING="128"
## Number of OpenCL work-items running simultaneously
DEVICE_THREADS="32"

# Tuning
## Number of iterations for each kernel run
ITERATIONS="5"
## Minimum number of work-items
MIN_THREADS="8"
## Maximum number of work-items
MAX_THREADS="1024"
## Maximum number of variables
MAX_ITEMS="63"
## Maximum unrolling
MAX_UNROLL="1"
## Maximum number of work-items in OpenCL dimension 0; dedispersion specific
MAX_DIM0="1024"
## Maximum number of work-items in OpenCL dimension 1; dedispersion specific
MAX_DIM1="128"
## Maximum number of variables in OpenCL dimension 0; dedispersion specific
MAX_ITEMS_DIM0="64"
## Maximum number of variables in OpenCL dimension 1; dedispersion specific
MAX_ITEMS_DIM1="32"

# Scenario
## Switch to select the subbanding mode; dedispersion specific
SUBBANDING=true
## Switch to select the SNR Mode ["SNR", "SNR_SC", "MOMAD", "MOMSIGMACUT"]
SNR="SNR_SC"
## Number of channels
CHANNELS="336"
## Frequency of the lowest channel, in MHz
MIN_FREQ="1214.700928"
## Bandwidth of a channel, in MHz
CHANNEL_BANDWIDTH="1.0"
## Number of samples per batch
SAMPLES="771"
## Sampling time
SAMPLING_TIME="0.001328"
## Downsampling factor
DOWNSAMPLING=1
## Number of subbands
SUBBANDS="32"
## Number of DMs to dedisperse in step one; subbanding mode only
SUBBANDING_DMS="128"
## First DM in step one; subbanding mode only
SUBBANDING_DM_FIRST="5.0"
## DM step in step one; subbanding mode only
SUBBANDING_DM_STEP="16.0"
## Number of DMs to dedisperse in either the single step or subbanding step two
DMS="32"
## First DM in either the single step or subbanding step two
DM_FIRST="0.0"
## DM step in either the single step or subbanding step two
DM_STEP="0.5"
## Number of input beams
BEAMS="1"
## Number of synthesized output beams
SYNTHESIZED_BEAMS="1"
## Sigma cut steps for the time domain sigma cut RFI mitigation
RFIM_TDSC_STEPS=""
## Sigma cut steps for the frequency domain sigma cut RFI mitigation
RFIM_FDSC_STEPS=""
## Number of bins for the bandpass correction in the frequency domain sigma cut
RFIM_FDSC_BINS=""
## Downsampling factors for pulse width test
INTEGRATION_STEPS="5 10 50 100"
## Zapped channels
ZAPPED_CHANNELS=""
## Median of medians step size; only for MOMAD and MOMSIGMACUT mode
MEDIAN_STEP=5
## Sigma cut value for MOMSIGMACUT mode; this value is currently harcoded in AMBER
NSIGMA=3
## Sigma cut for the SNR computation in SNR_SC mode
SNR_SIGMA=3.00
## Correction factor for the SNR computation in SNR_SC mode
CORRECTION_FACTOR=1.014
