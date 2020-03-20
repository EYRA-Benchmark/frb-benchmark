from __future__ import print_function
from builtins import zip
from builtins import range
import os
import sys
import argparse
import glob


if __name__ == '__main__':
    systems = ["ASKAP", "APERTIF", "CHIME"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', required=True, help='Output prefix (incl. filename prefix, e.g. /data/output_apertif)')
    parser.add_argument('--system', '-s', choices=systems, default='ASKAP', help='System (default: %(default)s)')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--filterbank', '-f', help='Input filterbank file')
    group.add_argument('--psrfits', '-p', help='Input psrfits file')

    args = parser.parse_args()

    basename = args.output
    system = args.system
    if args.filterbank is not None:
        rawfiles = args.filterbank
        dtype = '-filterbank'
    else:
        rawfiles = args.psrfits
        dtype = '-psrfits'
        
    index = systems.index(system)

    #
    # Made using input from:
    #
    # For ASKAP
    # DDplan.py -d 10000 -t 0.0012656 -s 112 -n 336 -b 336.0 -f 1332.0
    #
    # For APERTIF
    # DDplan.py -d 10000 -t 8.192e-5 -s 128 -n 1536 -b 300.0 -f 1369.5 
    #
    # For CHIME
    # DDplan.py -d 10000 -t 0.000983 -s 128 -n 16384 -b 400.0 -f 600.0 
    #


    # dDM steps from DDplan.py
    dDMs      = [[2.0, 5.0, 10.0, 20.0],
                 [0.2, 0.3, 0.5, 1.0, 3.0, 5.0],
                 [0.1, 0.3, 0.5]]
    # dsubDM steps
    dsubDMs   = [[176.0, 380.0, 760.0, 1520.0],
                 [17.2, 30.6, 51.0, 102.0, 276.0, 510.0],
                 [10.2, 25.8, 51.0]]
    # downsample factors
    downsamps = [[1, 2, 4, 8],
                 [1, 2, 4, 8, 16, 32],
                 [1, 2, 4]]
    # number of calls per set of subbands
    subcalls  = [[8, 4, 3, 4],
                 [31, 10, 13, 14, 15, 6],
                 [356, 152, 48]]
    # The low DM for each set of DMs 
    startDMs  = [[0.0, 1408.0, 2928.0, 5208.0],
                 [0.0, 533.2, 839.2, 1502.2, 2930.2, 7070.2],
                 [0.0, 3631.2, 7552.8]]
    # DMs/call
    dmspercalls = [[88, 76, 76, 76],
                   [86, 102, 102, 102, 92, 102],
                   [102, 86, 102]]

    # Number of subbands
    if system == 'ASKAP':
        nsub = 112
    else:
        nsub = 128

    # The basename of the output files you want to use
    #os.system("ln -s %s %s.fil" % (rawfiles, rawfiles))
    #rawfiles = rawfiles+'.fil'

    # Loop over the DDplan plans
    for dDM, dsubDM, dmspercall, downsamp, subcall, startDM in \
            zip(dDMs[index], dsubDMs[index], dmspercalls[index], downsamps[index], subcalls[index], startDMs[index]):
        # Loop over the number of calls
        for ii in range(subcall):
            subDM = startDM + (ii+0.5)*dsubDM
            loDM = startDM + ii*dsubDM
            # create the subbands and time series
            vals = {'nsub': nsub, 'lodm': loDM, 'subdm': subDM, 'downsamp': downsamp, 'numdm': dmspercall, 
                    'dmstep': dDM, 'out': basename, 'in': rawfiles, 'dtype': dtype}
            print("prepsubband -nobary {dtype} -nsub {nsub} -lodm {lodm} -dmstep {dmstep} " \
                  "-numdms {numdm} -downsamp {downsamp} -o {out} {in}".format(**vals))
