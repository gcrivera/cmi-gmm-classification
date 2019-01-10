# addpath(genpath('matlab-reqs')) to load toolkit
import argparse
from scripts import data

parser = argparse.ArgumentParser(description='Baseline Code-switching Classifier')

parser.add_argument('--num_features', type=int, default=13, help='Number of MFCC features')
parser.add_argument('--phoneme', action='store_true', help='Features will include one hot phoneme features')

args = parser.parse_args()

if __name__ == '__main__':
    # data.extract_phoneme_alone()
    data.extract(args.num_features, args.phoneme)