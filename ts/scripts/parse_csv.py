import argparse

import numpy as np
from numpy import genfromtxt


def parse_and_save(inpath, outpath):
    data = genfromtxt(inpath, delimiter=',')
    with open(outpath, 'wb') as outfile:
        np.save(outfile, data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath')
    parser.add_argument('outpath')
    return parser.parse_args()

def main():
    args = parse_args()
    parse_and_save(inpath=args.inpath, outpath=args.outpath)

if __name__ == '__main__':
    main()
