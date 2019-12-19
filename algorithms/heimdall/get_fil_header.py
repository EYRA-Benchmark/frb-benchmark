#!/usr/bin/env python
from __future__ import unicode_literals
from __future__ import print_function
import os
import struct
import sys
import math
import warnings

telescope_ids = {"Fake": 0, "Arecibo": 1, "ARECIBO 305m": 1, 
                 "Ooty": 2, "Nancay": 3, "Parkes": 4, "Jodrell": 5,
                 "GBT": 6, "GMRT": 7, "Effelsberg": 8, "ATA": 9,
                 "SRT": 10, "LOFAR": 11, "VLA": 12, "CHIME": 20,
                 "FAST": 21, "MeerKAT": 64, "KAT-7": 65}
ids_to_telescope = dict(zip(telescope_ids.values(), telescope_ids.keys()))

machine_ids = {"FAKE": 0, "PSPM": 1, "Wapp": 2, "WAPP": 2, "AOFTM": 3,
               "BCPM1": 4, "BPP": 4, "OOTY": 5, "SCAMP": 6,
               "GBT Pulsar Spigot": 7, "SPIGOT": 7, "BG/P": 11,
               "PDEV": 12, "CHIME+PSR": 20, "KAT": 64, "KAT-DC2": 65}
ids_to_machine = dict(zip(machine_ids.values(), machine_ids.keys()))

header_params = {
    "HEADER_START": 'flag',
    "telescope_id": 'i',
    "machine_id": 'i',
    "data_type": 'i', 
    "rawdatafile": 'str',
    "source_name": 'str', 
    "barycentric": 'i', 
    "pulsarcentric": 'i', 
    "az_start": 'd',  
    "za_start": 'd',  
    "src_raj": 'd',  
    "src_dej": 'd',  
    "tstart": 'd',  
    "tsamp": 'd',  
    "nbits": 'i', 
    "nsamples": 'i', 
    "nbeams": "i",
    "ibeam": "i",
    "fch1": 'd',  
    "foff": 'd',
    "FREQUENCY_START": 'flag',
    "fchannel": 'd',  
    "FREQUENCY_END": 'flag',
    "nchans": 'i', 
    "nifs": 'i', 
    "refdm": 'd',  
    "period": 'd',  
    "npuls": 'q',
    "nbins": 'i', 
    "HEADER_END": 'flag'}

def dec2radians(src_dej):
    """
    dec2radians(src_dej):
       Convert the SIGPROC-style DDMMSS.SSSS declination to radians
    """
    sign = 1.0
    if (src_dej < 0): sign = -1.0;
    xx = math.fabs(src_dej)
    dd = int(math.floor(xx / 10000.0))
    mm = int(math.floor((xx - dd * 10000.0) / 100.0))
    ss = xx - dd * 10000.0 - mm * 100.0
    return sign * ARCSECTORAD * (60.0 * (60.0 * dd + mm) + ss)

def ra2radians(src_raj):
    """
    ra2radians(src_raj):
       Convert the SIGPROC-style HHMMSS.SSSS right ascension to radians
    """
    return 15.0 * dec2radians(src_raj)

def read_doubleval(filfile, stdout=False):
    dblval = struct.unpack('d', filfile.read(8))[0]
    if stdout:
        print("  double value = '%20.15f'"%dblval)
    return dblval

def read_intval(filfile, stdout=False):
    intval = struct.unpack('i', filfile.read(4))[0]
    if stdout:
        print("  int value = '%d'"%intval)
    return intval

def read_longintval(filfile, stdout=False):
    longintval = struct.unpack('q', filfile.read(8))[0]
    if stdout:
        print("  long int value = '%d'"%longintval)
    return longintval

def read_string(filfile, stdout=False):
    strlen = struct.unpack('i', filfile.read(4))[0]
    strval = filfile.read(strlen)
    if stdout:
        print("  string = '%s'"%strval)
    return strval.decode('ascii')

def read_paramname(filfile, stdout=False):
    paramname = read_string(filfile, stdout=False)
    if stdout:
        print("Read '%s'"%paramname)
    return paramname

def read_hdr_val(filfile, stdout=False):
    paramname = read_paramname(filfile, stdout)
    if header_params[paramname] == 'd':
        return paramname, read_doubleval(filfile, stdout)
    elif header_params[paramname] == 'i':
        return paramname, read_intval(filfile, stdout)
    elif header_params[paramname] == 'q':
        return paramname, read_longintval(filfile, stdout)
    elif header_params[paramname] == 'str':
        return paramname, read_string(filfile, stdout)
    elif header_params[paramname] == 'flag':
        return paramname, None
    else:
        print("Warning:  key '%s' is unknown!" % paramname)
        return None, None

def samples_per_file(infile, hdrdict, hdrlen):
    """
    samples_per_file(infile, hdrdict, hdrlen):
       Given an input SIGPROC-style filterbank file and a header
           dictionary and length (as returned by read_header()),
           return the number of (time-domain) samples in the file.
    """
    numbytes = os.stat(infile)[6] - hdrlen
    bytes_per_sample = hdrdict['nchans'] * (hdrdict['nbits']/8)
    if numbytes % bytes_per_sample:
        print("Warning!:  File does not appear to be of the correct length!")
    numsamples = numbytes / bytes_per_sample
    return numsamples

def read_header(filename, verbose=False):
    """Read the header of a filterbank file, and return
        a dictionary of header paramters and the header's
        size in bytes.
        Inputs:
            filename: Name of the filterbank file.
            verbose: If True, be verbose. (Default: be quiet)
        Outputs:
            header: A dictionary of header paramters.
            header_size: The size of the header in bytes.
    """
    header = {}
    filfile = open(filename, 'rb')
    filfile.seek(0)
    paramname = ""
    while (paramname != 'HEADER_END'):
        if verbose:
            print("File location: %d" % filfile.tell())
        paramname, val = read_hdr_val(filfile, stdout=verbose)
        if verbose:
            print("Read param %s (value: %s)" % (paramname, val))
        if paramname not in ["HEADER_START", "HEADER_END"]:
            header[paramname] = val
    header_size = filfile.tell()
    filfile.close()
    return header, header_size

if __name__=='__main__':
    try:
        fn_fil = sys.argv[1]
        freq_ref = sys.argv[2]
    except:
        print("Expected fnfil freq_ref as arguments")
        exit()

    header, header_nbyte = read_header(fn_fil, verbose=False)
    freq_low = min(header['fch1'], header['fch1'] + header['foff']*header['nchans'])
    freq_high = max(header['fch1'], header['fch1'] + header['foff']*header['nchans'])
    freq_mid = 0.5*(freq_low+freq_high)

    if freq_ref in ['hi', 'HIGH', 'high', 'top']:
        print(freq_high)
    elif freq_ref in ['bot', 'Low', 'low', 'bottom']:
        print(freq_low)
    elif freq_ref in ['mid', 'MID', 'Mid', 'middle']:
        print(freq_mid)
    else:
        print("freq_ref should be high, low, or mid")
