import os
import torch
import argparse
import copy
from glob import glob

from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter

from data.dataloader_nba import NBADatasetMinMax as NBADatasetMinMax
from data.dataloader_nba import seq_collate_nba, seq_collate_imle_train

from utils.config import Config
from utils.utils import back_up_code_git, set_random_seed, log_config_to_file

from models.flow_matching import FlowMatcher
from models.imle import IMLE
from models.backbone import IMLETransformer
from trainer.imle_trainers import IMLETrainer


def parse_config():
    """
    Parse the command line arguments and return the configuration options.
    """

    parser = argparse.ArgumentParser()

    # Basic configuration
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the checkpoint to load the model from.')
    parser.add_argument('--cfg', default='auto', type=str, help="Config file path")
    parser.add_argument('--exp', default='', type=str, help='Experiment description for each run, name of the saving folder.')
    parser.add_argument('--save_samples', default=False, action='store_true', help='Save the samples during evaluation.')
    parser.add_argument('--eval_on_train', default=False, action='store_true', help='Evaluate the model on the training set.')

    # Data configuration
    parser.add_argument('--batch_size', default=None, type=int, help='Override the batch size in the config file.')
    parser.add_argument('--data_dir', type=str, default='./data/nba', help='Directory where the data is stored.')
    parser.add_argument('--n_train', type=int, default=32500, help='Number training scenes used.')
    parser.add_argument('--n_test', type=int, default=12500, help='Number testing scenes used.')
    parser.add_argument('--rotate', default=False, action='store_true', help='Whether to rotate the data to canonical x-axis or not.')
    parser.add_argument('--data_norm', default='min_max', choices=['min_max', 'sqrt'], help='Normalization method for the data.')

    # Reproducibility configuration
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='fix random seed for reproducibility')
    parser.add_argument('--seed', type=int, default=42, help='Set the random seed to split the testing set for training evaluation.')


    return parser.parse_args()


def init_basics(args):
    """
    Init the basic configurations for the experiment.
    """

    """Load the config file"""
    result_dir = os.path.abspath(os.path.join(args.ckpt_path, '../../'))
    if args.cfg == 'auto':
        yml_ls = glob(result_dir+'/*.yml')
        assert len(yml_ls) >= 1, 'At least one config file should be found in the directory.'
        yml_path = [f for f in yml_ls if '_updated.yml' in os.path.basename(f)][0]
        args.cfg = yml_path
    cfg = Config(args.cfg, f'{args.exp}', train_mode=False)

    tag = '_'


    ### Update data configuration ###
    def _update_data_params(args, cfg, tag):	

        if args.n_train != 32500:
            tag += f'_subset{args.n_train}'

        return cfg, tag

    cfg, tag = _update_data_params(args, cfg, tag)


    def _update_optimization_params(args, cfg, tag):
        if args.batch_size is not None:
            # override the batch size
            cfg.train_batch_size = args.batch_size
            cfg.test_batch_size = args.batch_size
        return cfg, tag

    cfg, tag = _update_optimization_params(args, cfg, tag)
    
    ### voila, create the saving directory ###
    tag += '_train_set' if args.eval_on_train else '_test_set'
    tag = tag.replace('__', '_')
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = cfg.create_dirs(tag_suffix=tag)


    """fix random seed"""
    if args.fix_random_seed:
        set_random_seed(args.seed)


    """set up tensorboard and text log"""
    tb_dir = os.path.abspath(os.path.join(cfg.log_dir, '../tb_eval'))
    os.makedirs(tb_dir, exist_ok=True)
    tb_log = SummaryWriter(log_dir=tb_dir)

    
    """print the config file"""
    log_config_to_file(cfg.yml_dict, logger=logger)
    return cfg, logger, tb_log


def build_data_loader(cfg, args):
    """
    Build the data loader for the NBA dataset.
    """
    train_dset = NBADatasetMinMax(
        data_dir=args.data_dir,
        obs_len=cfg.past_frames,
        pred_len=cfg.future_frames,
        training=True,
        num_scenes=args.n_train,
        overfit=False,
        cfg=cfg,
        rotate=args.rotate,
        data_norm=args.data_norm,
        imle=True)

    train_loader = DataLoader(
        train_dset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_imle_train,
        pin_memory=True)
    

    test_dset = NBADatasetMinMax(
    data_dir=args.data_dir,
    obs_len=cfg.past_frames,
    pred_len=cfg.future_frames,
    training=False,
    overfit=False,
    test_scenes=args.n_test,
    cfg=cfg,
    rotate=args.rotate,
    data_norm=args.data_norm,
    imle=False)
        
    test_loader = DataLoader(
        test_dset,
        batch_size=cfg.test_batch_size, ### change it from 500 
        shuffle=False,
        num_workers=4,
        collate_fn=seq_collate_nba,
        pin_memory=True)
    
    return train_loader, test_loader


def build_network(cfg, args, logger):
    """
    Build the network for the denoising model.
    """
    model = IMLETransformer(
        model_config=cfg.MODEL,
        logger=logger,
        config=cfg,
    )

    imle_model = IMLE(
        cfg=cfg,
        model=model,
        logger=logger,
    )

    return imle_model


def main():
    """
    Main function to train the model.
    """

    """Init everything"""
    args = parse_config()

    cfg, logger, tb_log = init_basics(args)

    train_loader, test_loader = build_data_loader(cfg, args)

    imle_model = build_network(cfg, args, logger)

    """Train or evaluate the model"""
    trainer = IMLETrainer(
        cfg,
        imle_model, 
        train_loader, 
        test_loader, 
        tb_log=tb_log,
        logger=logger,
        gradient_accumulate_every=1,
        ema_decay = 0.995,
        ema_update_every = 1,
        save_samples=args.save_samples
        ) ### grid search

    trainer.test(mode='best', eval_on_train=args.eval_on_train) 


if __name__ == "__main__":
    main()
