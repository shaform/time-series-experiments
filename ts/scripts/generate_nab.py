"""
This scripts reads in a result file and generates NAB results according to
different algorithms to detect the errors
"""
import argparse
import os
import csv


def main():
    args = parse_args()

    # load results
    results = {}
    with open(args.result_path) as infile:
        for line in infile:
            name, values = line.strip().split('\t')
            values = values.split()
            name = name[len('nab_data/'):-len('.npy')]
            results[name] = values

    data = {}
    # load data
    for root, _, names in os.walk(args.data_dir):
        for name in names:
            if not name.endswith('.csv'):
                continue
            base = root.split(args.data_dir)[-1].strip('/')
            basename = os.path.join(base, name)

            path = os.path.join(root, name)
            with open(path) as infile:
                data[basename] = [line + ['0'] for line in csv.reader(infile)]
                data[basename][0][-1] = 'anomaly_score'

    # fill results
    for name, fields in data.items():
        values = results[name]
        start = len(fields) - len(values)
        for i, v in enumerate(values, start=start):
            fields[i][-1] = v

    # write csv
    for name, fields in data.items():
        path = os.path.join(args.out_dir, name)
        dirname, fname = os.path.split(path)
        path = os.path.join(dirname, args.name + '_' + fname)
        with open(path, 'w') as outfile:
            for line in fields:
                outfile.write(','.join(line) + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-path', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--name', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
