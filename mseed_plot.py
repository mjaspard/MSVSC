#!/usr/bin/env python

# -*-coding:UTF-8 -*
#
# Script to plot a miniseed file:
# Mandatory arguments: mseed files (announce withst.plot -f or --miniseed)




from obspy import read, Stream
import argparse
import sys


# construction des arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--miniseed", required=True, help="Miniseed file")
# args = vars(ap.parse_args())	

#mseedFile = sys.argv[1]
	

#st = read(args["miniseed"])
st = read(sys.argv[1])
st.plot()




