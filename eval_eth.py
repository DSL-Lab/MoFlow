import os
import torch
import argparse
import copy
from glob import glob

from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter

from data.dataloader_eth_ucy import ETHDataset, seq_collate_eth

from utils.config import Config
from utils.utils import back_up_code_git, set_random_seed, log_config_to_file

from models.flow_matching import FlowMatcher
from models.backbone_eth_ucy import ETHMotionTransformer
from trainer.denoising_model_trainers import Trainer


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
	parser.add_argument('--data_dir', type=str, default='./data/eth_ucy', help='Directory where the data is stored.')
	parser.add_argument('--n_train', type=int, default=32500, help='Number training scenes used.')
	parser.add_argument('--n_test', type=int, default=12500, help='Number testing scenes used.')
	parser.add_argument('--data_norm', default='min_max', choices=['min_max', 'sqrt'], help='Normalization method for the data.')
	parser.add_argument('--subset', type=str, required=True, choices=['eth', 'hotel', 'univ', 'zara1', 'zara2'], help='Trajectory subset to run experiment')
	parser.add_argument('--rotate', default=False, action='store_true', help="Whether to rotate the trajectories in the dataset")
	parser.add_argument('--rotate_time_frame', type=int, default=0, help='Index of time frames to rotate the trajectories.')

	# Reproducibility configuration
	parser.add_argument('--fix_random_seed', action='store_true', default=False, help='fix random seed for reproducibility')
	parser.add_argument('--seed', type=int, default=42, help='Set the random seed to split the testing set for training evaluation.')

	### FM parameters ###
	parser.add_argument('--sampling_steps', type=int, default=10, help='Number of sampling timesteps for the FlowMatcher.')
	parser.add_argument('--solver', type=str, default='euler', choices=['euler', 'lin_poly'], help='Solver for the FlowMatcher.')
	parser.add_argument('--lin_poly_p', type=int, default=2, help='Degree of the polynomial in the linear stage.')
	parser.add_argument('--lin_poly_long_step', type=int, default=1000, help='Number of steps to mimic slope in the linear stage.')
	### FM parameters ###


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


	### Update FM parameters ###
	def _update_fm_params(args, cfg, tag):
		if cfg.denoising_method == 'fm':
			cfg.sampling_steps = args.sampling_steps
			cfg.solver = args.solver

			if args.solver == 'euler':
				solver_tag_ = args.solver
			elif args.solver == 'lin_poly':
				cfg.lin_poly_p = args.lin_poly_p
				cfg.lin_poly_long_step = args.lin_poly_long_step
				solver_tag_ = f'lin_poly_p{args.lin_poly_p}_long{args.lin_poly_long_step}'
			
			fm_tag_ = f'FM_S{cfg.sampling_steps}_{solver_tag_}'
			tag += fm_tag_
			cfg.solver_tag = fm_tag_

		return cfg, tag

	cfg, tag = _update_fm_params(args, cfg, tag)


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
	Build the data loader for the ETH-UCY dataset. [including 5 subsets: ETH, HOTEL, UNIV, ZARA1, ZARA2]
	"""
	train_dset = ETHDataset(
		cfg=cfg,
		training=True,
		data_dir=args.data_dir,
		subset=cfg.subset,
		rotate_time_frame=args.rotate_time_frame)

	train_loader = DataLoader(
		train_dset,
		batch_size=cfg.train_batch_size,
		shuffle=False,
		num_workers=4,
		collate_fn=seq_collate_eth,
		pin_memory=True)
	

	test_dset = ETHDataset(
		cfg=cfg,
		training=False,
		data_dir=args.data_dir,
		subset=cfg.subset,
		rotate_time_frame=args.rotate_time_frame)
		
	test_loader = DataLoader(
		test_dset,
		batch_size=cfg.test_batch_size, ### change it from 500 
		shuffle=False,
		num_workers=4,
		collate_fn=seq_collate_eth,
		pin_memory=True)
	
	return train_loader, test_loader


def build_network(cfg, args, logger):
	"""
	Build the network for the denoising model.
	"""
	model = ETHMotionTransformer(
		model_config=cfg.MODEL,
		logger=logger,
		config=cfg,
	)

	if cfg.denoising_method == 'fm':
		denoiser = FlowMatcher(
			cfg,
			model,
			logger=logger,
		)
	else:
		raise NotImplementedError(f'Denoising method [{cfg.denoising_method}] is not implemented yet.')

	return denoiser


def main():
	"""
	Main function to train the model.
	"""

	"""Init everything"""
	args = parse_config()

	cfg, logger, tb_log = init_basics(args)

	train_loader, test_loader = build_data_loader(cfg, args)

	denoiser = build_network(cfg, args, logger)

	"""Train or evaluate the model"""
	trainer = Trainer(
		cfg,
		denoiser, 
		train_loader, 
		test_loader, 
		tb_log=tb_log,
		logger=logger,
		gradient_accumulate_every=1,
		ema_decay = 0.995,
		ema_update_every = 1,
		save_samples=args.save_samples,
		) ### grid search

	trainer.test(mode='best', eval_on_train=args.eval_on_train) 


if __name__ == "__main__":
	main()
