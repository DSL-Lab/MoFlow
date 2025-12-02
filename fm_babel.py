import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import argparse
import copy
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter

# from data.dataloader_nba import NBADatasetMinMax as NBADatasetMinMax
# from data.dataloader_nba import seq_collate_nba

from data.dataloader_babel import BabelDatasetMinMax as BabelDatasetMinMax
from data.dataloader_babel import seq_collate_rat

from utils.config import Config
from utils.utils import back_up_code_git, set_random_seed, log_config_to_file

from models.flow_matching import FlowMatcher
from models.backbone import MotionTransformer
from trainer.denoising_model_trainers import Trainer


def parse_config():
	"""
	Parse the command line arguments and return the configuration options.
	"""

	parser = argparse.ArgumentParser()

	# Basic configuration
	parser.add_argument('--cfg', default='cfg/babel/cor_fm.yml', type=str, help="Config file path")
	parser.add_argument('--exp', default='', type=str, help='Experiment description for each run, name of the saving folder.')

	# Data configuration
	parser.add_argument('--epochs', default=None, type=int, help='Override the number of epochs in the config file.')
	parser.add_argument('--batch_size', default=128, type=int, help='Override the batch size in the config file.')
	parser.add_argument('--data_dir', type=str, default='./data/babel', help='Directory where the data is stored.')
	parser.add_argument('--overfit', default=False, action='store_true', help='Overfit the testing set by setting it to the same entries as the training set.')
	parser.add_argument('--n_train', type=int, default=9536, help='Number training scenes used.')
	parser.add_argument('--n_test', type=int, default=2724, help='Number testing scenes used.')
	parser.add_argument('--rotate', default=False, action='store_true', help='Whether to rotate the data to canonical x-axis or not.')
	parser.add_argument('--checkpt_freq', default=5, type=int, help='Override the checkpt_freq in the config file.')
	parser.add_argument('--max_num_ckpts', default=5, type=int, help='Override the max_num_ckpts in the config file.')
	parser.add_argument('--data_norm', default='min_max', choices=['min_max', 'sqrt'], help='Normalization method for the data.')

	# Reproducibility configuration
	parser.add_argument('--fix_random_seed', action='store_true', default=False, help='fix random seed for reproducibility')
	parser.add_argument('--seed', type=int, default=42, help='Set the random seed.')

	### FM parameters ###
	parser.add_argument('--sampling_steps', type=int, default=10, help='Number of sampling timesteps for the FlowMatcher.')

	# time scheduler during training
	parser.add_argument('--t_schedule', type=str, choices=['uniform', 'logit_normal'], default='logit_normal', help='Time schedule for the FlowMatcher.')
	parser.add_argument('--fm_skewed_t', default=None, type=str, help='Skewed time schedule for the FlowMatcher.')
	parser.add_argument('--logit_norm_mean', default=-0.5, type=float, help='Mean for the logit normal distribution.')
	parser.add_argument('--logit_norm_std', default=1.5, type=float, help='Standard deviation for the logit normal distribution.')

	parser.add_argument('--fm_wrapper', type=str, default='direct', choices=['direct', 'velocity', 'precond'], help='Wrapper for the FlowMatcher.')
	parser.add_argument('--fm_rew_sqrt', default=False, action='store_true', help='Whether to apply square root to the reweighting factor.')
	parser.add_argument('--fm_in_scaling', default=False, action='store_true', help='Whether to scale the input to the FlowMatcher.')

	# input dropout / masking rate
	parser.add_argument('--drop_method', default='emb', type=str, choices=['None', 'input', 'emb'], help='Dropout method for the FlowMatcher.')
	parser.add_argument('--drop_logi_k', default=20.0, type=float, help='Logistic growth rate for masking rate at different timesteps.')
	parser.add_argument('--drop_logi_m', default=0.5, type=float, help='Logistic midpoint for masking rate at different timesteps.')
	### FM parameters ###

	### Architecture configuration ###
	parser.add_argument('--use_pre_norm', default=False, action='store_true', help='Where to normalize the input trajectories in the Transformer Encoders.')
	### Architecture configuration ###

	### General denoising objective configuration ###
	parser.add_argument('--tied_noise', default=False, action='store_true', help='Whether to use tied noise for the denoiser.')
	### General denoising objective configuration ###

	### Regression loss configuration ###
	parser.add_argument('--loss_nn_mode', type=str, default='agent', choices=['agent', 'scene', 'both'], help='Whether to use the agent-wise or scene-wise NN loss.')
	parser.add_argument('--loss_reg_reduction', type=str, default='sum', choices=['mean', 'sum'], help='Reduction method for the regression loss.')
	parser.add_argument('--loss_reg_squared', default=False, action='store_true', help='Whether to use the squared regression loss.')
	parser.add_argument('--loss_velocity', default=False, action='store_true', help='Whether to use the regression loss for velocity.')
	### Regression loss configuration ###

	### Optimization configuration ###
	parser.add_argument('--init_lr', type=float, default=None, help='Override the peak learning rate in the config file.')
	parser.add_argument('--weight_decay', type=float, default=None, help='Override the weight decay in the config file.')
	### Optimization configuration ###

	parser.add_argument('--load_pretrained', default=False, action='store_true', help='Whether to load a pretrained encoder.')
	parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the pretrained encoder checkpoint.')

	return parser.parse_args()


def init_basics(args):
	"""
	Init the basic configurations for the experiment.
	"""

	"""Load the config file"""
	cfg = Config(args.cfg, f'{args.exp}')

	tag = '_'

	### Update FM parameters ###
	def _update_fm_params(args, cfg, tag):
		if cfg.denoising_method == 'fm':
			cfg.sampling_steps = args.sampling_steps
			cfg.load_pretrained = args.load_pretrained
			if cfg.load_pretrained:
				cfg.ckpt_path = args.ckpt_path
				# tag += '_load_enc'

			if args.fm_skewed_t is not None:
				cfg.t_schedule = args.fm_skewed_t
			else:
				cfg.t_schedule = args.t_schedule

			if args.t_schedule == 'logit_normal':
				cfg.logit_norm_mean = args.logit_norm_mean
				cfg.logit_norm_std = args.logit_norm_std

			cfg.fm_wrapper = args.fm_wrapper
			cfg.fm_rew_sqrt = args.fm_rew_sqrt
			cfg.fm_in_scaling = args.fm_in_scaling

			# if args.fm_skewed_t is not None:
			# 	tag += f'FM_S{cfg.sampling_steps}_{cfg.t_schedule}_{cfg.fm_wrapper[:4]}'
			# elif args.t_schedule == 'logit_normal':
			# 	tag += f'FM_S{cfg.sampling_steps}_{cfg.t_schedule[:3]}_m{cfg.logit_norm_mean}_s{cfg.logit_norm_std}_{cfg.fm_wrapper[:4]}'
			# elif args.t_schedule == 'uniform':
			# 	tag += f'FM_S{cfg.sampling_steps}_{cfg.t_schedule[:3]}_{cfg.fm_wrapper[:4]}'

			if args.drop_method is not None and args.drop_logi_k is not None and args.drop_logi_m is not None:
				cfg.drop_method = args.drop_method
				cfg.drop_logi_k = args.drop_logi_k
				cfg.drop_logi_m = args.drop_logi_m
				# tag += f'_drop_{cfg.drop_method}_m{cfg.drop_logi_m}_k{cfg.drop_logi_k}'

			# if cfg.fm_rew_sqrt:
			# 	tag += '_RESQ'
			# if cfg.fm_in_scaling:
			# 	tag += '_IS'
		return cfg, tag

	cfg, tag = _update_fm_params(args, cfg, tag)


	### Architecture configuration ###
	def _update_architecture_params(args, cfg, tag):
		cfg.MODEL.USE_PRE_NORM = args.use_pre_norm
		
		return cfg, tag

	cfg, tag = _update_architecture_params(args, cfg, tag)

	### General denoising objective configuration ###
	def _update_general_denoising_params(args, cfg, tag):
		cfg.tied_noise = args.tied_noise
		if args.tied_noise:
			tag += '_TN'
		return cfg, tag

	cfg, tag = _update_general_denoising_params(args, cfg, tag)


	### Regression loss configuration ###
	def _update_regression_loss_params(args, cfg, tag):
		cfg.LOSS_NN_MODE = args.loss_nn_mode
		cfg.LOSS_REG_REDUCTION = args.loss_reg_reduction
		cfg.LOSS_REG_SQUARED = args.loss_reg_squared
		cfg.LOSS_VELOCITY = args.loss_velocity

		tag += f'_NN_{cfg.LOSS_NN_MODE[:1].upper()}'
		tag += f'_REG_{cfg.LOSS_REG_REDUCTION[:1].upper()}'

		if args.loss_reg_squared:
			tag += '_SQ'
		if args.loss_velocity:
			tag += '_VEL'
			cfg.MODEL.REGRESSION_MLPS[-1] += cfg.MODEL.MODEL_OUT_DIM

		return cfg, tag

	cfg, tag = _update_regression_loss_params(args, cfg, tag)


	### Update data configuration ###
	def _update_data_params(args, cfg, tag):	
		if args.overfit:
			tag += '_overfit'

		if args.n_train != 32500:
			tag += f'_subset{args.n_train}'

		cfg.data_norm = args.data_norm
		tag += f'_{args.data_norm}'

		return cfg, tag

	cfg, tag = _update_data_params(args, cfg, tag)


	### Update optimization configs ###
	def _update_optimization_params(args, cfg, tag):
		if args.init_lr is not None:
			cfg.OPTIMIZATION.LR = args.init_lr

		if args.weight_decay is not None:
			cfg.OPTIMIZATION.WEIGHT_DECAY = args.weight_decay

		tag += f'_LR{cfg.OPTIMIZATION.LR}_WD{cfg.OPTIMIZATION.WEIGHT_DECAY}'

		if args.epochs is not None:
			# override the number of epochs
			cfg.OPTIMIZATION.NUM_EPOCHS = args.epochs

		if args.batch_size is not None:
			# override the batch size
			cfg.train_batch_size = args.batch_size
			cfg.test_batch_size = args.batch_size  # larger BS for during-training evaluation

		if args.checkpt_freq is not None:
			# override the checkpt_freq
			cfg.checkpt_freq = args.checkpt_freq
		
		cfg.max_num_ckpts = args.max_num_ckpts

		tag += f'_BS{cfg.train_batch_size}_EP{cfg.OPTIMIZATION.NUM_EPOCHS}'

		return cfg, tag

	cfg, tag = _update_optimization_params(args, cfg, tag)
		

	### voila, create the saving directory ###
	tag = tag.replace('__', '_')
	cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
	# cfg.device = torch.device("cpu")

	logger = cfg.create_dirs(tag_suffix=tag)


	"""fix random seed"""
	if args.fix_random_seed:
		set_random_seed(args.seed)


	"""set up tensorboard and text log"""
	tb_dir = os.path.abspath(os.path.join(cfg.log_dir, '../tb'))
	os.makedirs(tb_dir, exist_ok=True)
	tb_log = SummaryWriter(log_dir=tb_dir)

		
	"""back up the code"""
	back_up_code_git(cfg, logger=logger)
	
	"""print the config file"""
	log_config_to_file(cfg.yml_dict, logger=logger)
	return cfg, logger, tb_log


def build_data_loader(cfg, args):
	"""
	Build the data loader for the Rat dataset.
	"""
	train_dset = BabelDatasetMinMax(
		data_dir=args.data_dir,
		obs_len=cfg.past_frames,
		pred_len=cfg.future_frames,
		training=True,
		num_scenes=args.n_train,
		overfit=args.overfit,
		cfg=cfg,
		# rotate=args.rotate,
		data_norm=args.data_norm)

	train_loader = DataLoader(
		train_dset,
		batch_size=cfg.train_batch_size,
		shuffle=True,
		num_workers=4,
		collate_fn=seq_collate_rat,
		pin_memory=True)

	if args.overfit:
		test_dset = copy.deepcopy(train_dset)
	else:
		test_dset = BabelDatasetMinMax(
		data_dir=args.data_dir,
		obs_len=cfg.past_frames,
		pred_len=cfg.future_frames,
		training=False,
		overfit=args.overfit,
		test_scenes=args.n_test,
		cfg=cfg,
		# rotate=args.rotate,
		data_norm=args.data_norm)
		
	test_loader = DataLoader(
		test_dset,
		batch_size=cfg.test_batch_size, ### change it from 500 
		shuffle=False,
		num_workers=4,
		collate_fn=seq_collate_rat,
		pin_memory=True)
	
	return train_loader, test_loader


def build_network(cfg, args, logger):
	"""
	Build the network for the denoising model.
	"""
	model = MotionTransformer(
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
	def set_requires_grad(module, flag: bool):
		for p in module.parameters():
			p.requires_grad = flag

	"""Init everything"""
	args = parse_config()

	cfg, logger, tb_log = init_basics(args)
	# 构建数据加载器
	train_loader, test_loader = build_data_loader(cfg, args)
	# 构建模型网络
	denoiser = build_network(cfg, args, logger).to(device=cfg.device)

	# # 1) 先把全模型都冻住
	# set_requires_grad(denoiser, False)
	#
	# # 2) 只打开以下几块（Stage A）
	# set_requires_grad(denoiser.model.z_encoder.head_z, True)  # 只训练 z 头
	# set_requires_grad(denoiser.model.z_proj, True)
	# set_requires_grad(denoiser.model.z_gamma, True)
	# set_requires_grad(denoiser.model.z_beta, True)

	"""Train the model"""
	trainer = Trainer(
		cfg,
		denoiser, 
		train_loader, 
		test_loader,  # 其实是val loader
		tb_log=tb_log,
		logger=logger,
		gradient_accumulate_every=1,
		ema_decay = 0.995,
		ema_update_every = 1,
		) ### grid search

	if args.load_pretrained:
		print(cfg.model_dir)
		trainer.load(cfg.ckpt_path)

	trainer.train()


if __name__ == "__main__":
	main()

# python fm_babel.py --exp rat_1202_CNDS_babel --tied_noise --fm_in_scaling --checkpt_freq 5 --batch_size 128 --init_lr 1e-3

