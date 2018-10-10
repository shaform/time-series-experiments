import argparse

import numpy as np
import scipy.io as sio


def parse_and_save(inpaths, outpath):
    Y = []
    L = []

    for inpath in inpaths:
        mat = sio.loadmat(inpath)
        Y.append(mat['Y'])
        L.append(mat['L'])
    Y = np.concatenate(Y)
    L = np.concatenate(L)
    data = {'Y': Y, 'L': L}

    with open(outpath, 'wb') as outfile:
        np.save(outfile, data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpaths', nargs='+')
    parser.add_argument('outpath')
    return parser.parse_args()


def main():
    args = parse_args()
    parse_and_save(inpaths=args.inpaths, outpath=args.outpath)


if __name__ == '__main__':
    main()
