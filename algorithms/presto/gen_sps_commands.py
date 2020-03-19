from __future__ import print_function
from builtins import zip
from builtins import range
import os
import sys
import argparse
import glob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snrmin', type=float, default=6.0, help='S/N threshold (default: %(default)s)')
    parser.add_argument('--maxwidth', type=float, default=0.1, help='Max width in seconds (default: %(default)s)')
    parser.add_argument('files', nargs='+', help='.dat files')

    args = parser.parse_args()

    # parse input files, might be mix of files and glob expressions
    files = []
    for f in args.files:
        if not os.path.isfile(f):
            # assume it is a glob expression
            expanded = glob.glob(f)
            # add any files found to the file list
            if expanded:
                files.extent(expanded)
        else:
            # check if valid file
            assert f.endswith('.dat')
            # add to file list
            files.append(f)
    # remove any duplicates
    files = list(set(files))
    # sort (not necessary, but can be useful)
    files.sort()

    if not files:
        print("No .dat files found, verify input")
        sys.exit(1)

    # generate the single pulse search commands
    vals = {'snrmin': args.snrmin, 'maxwidth': args.maxwidth}
    for f in files:
        print("single_pulse_search.py -b -p -m {maxwidth} -t {snrmin} {fname}".format(fname=f, **vals))
