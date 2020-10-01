
CPU_CORE=1
OPENCL_PLATFORM=0
OPENCL_DEVICE=3
DEVICE_PADDING=128
DEVICE_THREADS=32
ITERATIONS=1
MIN_THREADS=8
MAX_THREADS=1024
MAX_ITEMS=63
MAX_UNROLL=1
MAX_DIM0=1024
MAX_DIM1=128
MAX_ITEMS_DIM0=64
MAX_ITEMS_DIM1=32
# the following two options are not used, but avoid
# warnings during tuning
MEDIAN_STEP=5
NSIGMA=3

SUBBANDING=true
BEAMS=1
SYNTHESIZED_BEAMS=1
SNR=SNR_SC
SNR_SIGMA=3
CORRECTION_FACTOR=1.014
DM_FIRST=0
RFIM_TDSC_STEPS=""
RFIM_FDSC_STEPS=""
RFIM_FDSC_BINS=""
ZAPPED_CHANNELS=""

DEVICE_NAME=askap_step4
CHANNELS=336
MIN_FREQ=1165.0
CHANNEL_BANDWIDTH=1.0
SAMPLES=10240
SAMPLING_TIME=0.0012656
DOWNSAMPLING=8
SUBBANDS=112
SUBBANDING_DMS=4
SUBBANDING_DM_STEP=1520.0
SUBBANDING_DM_FIRST=5208.0
DMS=76
DM_STEP=20.0
INTEGRATION_STEPS="2"
