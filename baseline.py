import argparse

from baselines.gla.gla_train import gla_main
from baselines.gla.gla_test import meta_test

def main():

    parser = argparse.ArgumentParser(description='Run GLA')
    parser.add_argument('-m', '--mode', type=str, default='train', help='train or test')

    args = parser.parse_args()

    if args.mode == 'train':
        gla_main()
    elif args.mode == 'test':
        meta_test("HalfCheetah-v5", "HalfCheetah-v5")

if __name__ == '__main__':
    main()