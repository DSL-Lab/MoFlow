#### This script is used to store the pickle files for the ETH dataset from the txt files.
from data.dataloader_eth_ucy import ETHDatasetSocialGAN
import argparse


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_dir', type=str, required=True, help='Directory where the data is stored.')
    return argparser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    for subset in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
        dataset = ETHDatasetSocialGAN(args.data_dir + f'/{subset}', subset=subset)
    print("Done storing pickle files.")