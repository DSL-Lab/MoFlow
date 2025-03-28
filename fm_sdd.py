import os
import torch
import argparse
import copy
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter

from data.dataloader_sdd import SDDDataset, seq_collate_sdd

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
    parser.add_argument('--cfg', default='cfg/sdd/cor_fm.yml', type=str, help="Config file path")
    parser.add_argument('--exp', default='', type=str, help='Experiment description for each run, name of the saving folder.')

    # Data configuration
    parser.add_argument('--epochs', default=None, type=int, help='Override the number of epochs in the config file.')
    parser.add_argument('--batch_size', default=None, type=int, help='Override the batch size in the config file.')
    parser.add_argument('--data_dir', type=str, default='./data/sdd', help='Directory where the data is stored.')
    parser.add_argument('--overfit', default=False, action='store_true', help='Overfit the testing set by setting it to the same entries as the training set.')
    parser.add_argument('--n_train', type=int, default=None, help='Override the number training scenes used.')
    parser.add_argument('--n_test', type=int, default=None, help='Override the number testing scenes used.')
    parser.add_argument('--checkpt_freq', default=5, type=int, help='Override the checkpt_freq in the config file.')
    parser.add_argument('--max_num_ckpts', default=5, type=int, help='Override the max_num_ckpts in the config file.')
    parser.add_argument('--data_norm', default='min_max', choices=['min_max', 'original'], help='Normalization method for the data.')
    parser.add_argument('--rotate', default=False, action='store_true', help="Whether to rotate the trajectories in the dataset")
    parser.add_argument('--rotate_time_frame', type=int, default=0, help='Index of time frames to rotate the trajectories.')
    parser.add_argument('--rotate_aug', default=False, action='store_true', help='Whether to use rotation as data augmentation.')

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

    parser.add_argument('--perturb_ctx', default=0.0, type=float, help='The scale of perturbation applied to the contextual input to the network.')

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
    parser.add_argument('--num_layers', type=int, default=None, help='Overwrite the number of layers in the config file.')
    parser.add_argument('--dropout', default=None, type=float, help='Overwrite the dropout rate in the config file.')
    ### Architecture configuration ###

    ### General denoising objective configuration ###
    parser.add_argument('--tied_noise', default=False, action='store_true', help='Whether to use tied noise for the denoiser.')
    ### General denoising objective configuration ###

    ### Regression loss configuration ###
    parser.add_argument('--loss_nn_mode', type=str, default='agent', choices=['agent', 'scene'], help='Whether to use the agent-wise or scene-wise NN loss.')
    parser.add_argument('--loss_reg_reduction', type=str, default='sum', choices=['mean', 'sum'], help='Reduction method for the regression loss.')
    parser.add_argument('--loss_reg_squared', default=False, action='store_true', help='Whether to use the squared regression loss.')
    parser.add_argument('--loss_velocity', default=False, action='store_true', help='Whether to use the regression loss for velocity.')
    ### Regression loss configuration ###

    ### Classification loss configuration ###
    parser.add_argument('--loss_cls_weight', type=float, default=1.0, help='Weight for the classification loss.')
    ### Classification loss configuration ###


    ### Optimization configuration ###
    parser.add_argument('--init_lr', type=float, default=None, help='Override the peak learning rate in the config file.')
    parser.add_argument('--weight_decay', type=float, default=None, help='Override the weight decay in the config file.')
    ### Optimization configuration ###

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

            if args.fm_skewed_t is not None:
                tag += f'FM_S{cfg.sampling_steps}_{cfg.t_schedule}_{cfg.fm_wrapper[:4]}'
            elif args.t_schedule == 'logit_normal':
                tag += f'FM_S{cfg.sampling_steps}_{cfg.t_schedule[:3]}_m{cfg.logit_norm_mean}_s{cfg.logit_norm_std}_{cfg.fm_wrapper[:4]}'
            elif args.t_schedule == 'uniform':
                tag += f'FM_S{cfg.sampling_steps}_{cfg.t_schedule[:3]}_{cfg.fm_wrapper[:4]}'


            tag += '_PC_{:.2f}'.format(args.perturb_ctx)

            if args.drop_method is not None and args.drop_logi_k is not None and args.drop_logi_m is not None:
                cfg.drop_method = args.drop_method
                cfg.drop_logi_k = args.drop_logi_k
                cfg.drop_logi_m = args.drop_logi_m
                tag += f'_drop_{cfg.drop_method}_m{cfg.drop_logi_m}_k{cfg.drop_logi_k}'

            if cfg.fm_rew_sqrt:
                tag += '_RESQ'
            if cfg.fm_in_scaling:
                tag += '_IS'
        return cfg, tag

    cfg, tag = _update_fm_params(args, cfg, tag)


    ### Architecture configuration ###
    def _update_architecture_params(args, cfg, tag):
        cfg.MODEL.USE_PRE_NORM = args.use_pre_norm
        cfg.MODEL.NUM_LAYERS = args.num_layers
        cfg.MODEL.DROPOUT = args.dropout
        
        if args.num_layers is not None:
            tag += f'_L{args.num_layers}'
            cfg.MODEL.CONTEXT_ENCODER.NUM_ATTN_LAYERS = args.num_layers
            cfg.MODEL.MOTION_DECODER.NUM_DECODER_BLOCKS = args.num_layers

        if args.dropout is not None:
            tag += f'_DO{args.dropout}'
            cfg.MODEL.CONTEXT_ENCODER.DROPOUT_OF_ATTN = args.dropout
            cfg.MODEL.MOTION_DECODER.DROPOUT_OF_ATTN = args.dropout
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

        cfg.rotate = args.rotate
        if args.rotate:
            cfg.rotate_time_frame = args.rotate_time_frame
            cfg.rotate_aug = args.rotate_aug
            tag += f'_rot_{cfg.rotate_time_frame}'
            if cfg.rotate_aug:
                tag += '_aug'

        if args.overfit:
            tag += '_overfit'

        if args.n_train is not None:
            tag += f'_subset_train_{args.n_train}'

        if args.n_test is not None:
            tag += f'_test{args.n_test}'

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

        cfg.OPTIMIZATION.LOSS_WEIGHTS['cls'] = args.loss_cls_weight

        tag += f'_LR{cfg.OPTIMIZATION.LR}_WD{cfg.OPTIMIZATION.WEIGHT_DECAY}_CLS_{args.loss_cls_weight}'

        if args.epochs is not None:
            # override the number of epochs
            cfg.OPTIMIZATION.NUM_EPOCHS = args.epochs

        if args.batch_size is not None:
            # override the batch size
            cfg.train_batch_size = args.batch_size
            cfg.test_batch_size = args.batch_size * 2  # larger BS for during-training evaluation

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
    Build the data loader for the NBA dataset.
    """
    train_dset = SDDDataset(
        cfg=cfg,
        training=True,
        data_dir=args.data_dir,
        rotate_time_frame=args.rotate_time_frame)

    train_loader = DataLoader(
        train_dset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=seq_collate_sdd,
        pin_memory=True)
    
    if args.overfit:
        test_dset = copy.deepcopy(train_dset)
    else:
        test_dset = SDDDataset(
        cfg=cfg,
        training=False,
        data_dir=args.data_dir,
        rotate_time_frame=args.rotate_time_frame)
        
    test_loader = DataLoader(
        test_dset,
        batch_size=cfg.test_batch_size, ### change it from 500 
        shuffle=False,
        num_workers=4,
        collate_fn=seq_collate_sdd,
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

    """Train the model"""
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
        ) 

    trainer.train()


if __name__ == "__main__":
    main()
