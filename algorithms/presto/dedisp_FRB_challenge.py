from __future__ import print_function
from builtins import zip
from builtins import range
import os
import sys

import glob

try:
    basename = sys.argv[1]
except:
    basename = "/tank/users/connor/eyra/intermediate"

try:
    rawfiles = sys.argv[2]
except:
    rawfiles = "/tank/users/connor/eyra/input/test_data"
    
system = "ASKAP" # Needs to be one of the following

systems = ["ASKAP", "APERTIF", "CHIME"]
index = systems.index(system)

#
# Made using input from:
#
# For ASKAP
# DDplan.py -d 4000 -t 0.0012656 -s 112 -n 336 -b 336.0 -f 1332.0
#
# For APERTIF
# DDplan.py -d 4000 -t 8.192e-5 -s 128 -n 1536 -b 300.0 -f 1369.5 
#
# For CHIME
# DDplan.py -d 4000 -t 0.000983 -s 128 -n 16384 -b 400.0 -f 600.0 
#

def myexecute(cmd):
    print("'%s'"%cmd)
    os.system(cmd)

# dDM steps from DDplan.py
dDMs      = [[2.0, 5.0, 10.0],
             [0.2, 0.3, 0.5, 1.0, 3.0],
             [0.1, 0.3]]
# dsubDM steps
dsubDMs   = [[176.0, 380.0, 760.0],
             [17.2, 30.6, 51.0, 102.0, 276.0],
             [10.2, 25.8]]
# downsample factors
downsamps = [[1, 2, 4],
             [1, 2, 4, 8, 16],
             [1, 2]]
# number of calls per set of subbands
subcalls  = [[8, 4, 2],
             [31, 10, 13, 14, 4],
             [356, 15]]
# The low DM for each set of DMs 
startDMs  = [[0.0, 1408.0, 2928.0],
             [0.0, 533.2, 839.2, 1502.2, 2930.2],
             [0.0, 3631.2]]
# DMs/call
dmspercalls = [[88, 76, 76],
               [86, 102, 102, 102, 92],
               [102, 86]]

# Number of subbands
nsub = 112 if system=="ASKAP" else 128 
max_width_sec = 0.1
snr_thresh = 6.0
# The basename of the output files you want to use
#os.system("ln -s %s %s.fil" % (rawfiles, rawfiles))
#rawfiles = rawfiles+'.fil'

# Loop over the DDplan plans
for dDM, dsubDM, dmspercall, downsamp, subcall, startDM in \
        zip(dDMs[index], dsubDMs[index], dmspercalls[index], downsamps[index], subcalls[index], startDMs[index]):
    # Get our downsampling right
    print(downsamp, downsamps)
    subdownsamp = downsamp/2
    datdownsamp = 2
    if downsamp < 2: subdownsamp = datdownsamp = 1
    # Loop over the number of calls
    for ii in range(subcall):
        subDM = startDM + (ii+0.5)*dsubDM
        # First create the subbands
        myexecute("prepsubband -nobary -filterbank -sub -subdm %.2f -nsub %d -downsamp %d -o %s %s" %
                  (subDM, nsub, subdownsamp, basename, rawfiles))
        # And now create the time series
        loDM = startDM + ii*dsubDM
        subnames = basename+"_DM%.2f.sub[0-9]*"%subDM
        myexecute("prepsubband -nobary -lodm %.2f -dmstep %.2f -numdms %d -downsamp %d -o %s %s" %
                  (loDM, dDM, dmspercall, datdownsamp, basename, subnames))
myexecute("single_pulse_search.py -m %f -t %f -p -g %s*.dat" % (max_width_sec, snr_thresh, basename))
