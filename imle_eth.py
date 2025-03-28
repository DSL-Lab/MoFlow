import os
import torch
import argparse
import copy
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter

from data.dataloader_eth_ucy import ETHDataset, seq_collate_eth, seq_collate_imle_train


from utils.config import Config
from utils.utils import back_up_code_git, set_random_seed, log_config_to_file

from models.flow_matching import FlowMatcher
from models.imle import IMLE
from models.backbone_eth_ucy import ETHIMLETransformer
from trainer.imle_trainers import IMLETrainer



def parse_config():
    """
    Parse the command line arguments and return the configuration options.
    """

    parser = argparse.ArgumentParser()

    # Basic configuration
    parser.add_argument('--cfg', default='cfg/eth_ucy/imle.yml', type=str, help="Config file path")
    parser.add_argument('--exp', default='', type=str, help='Experiment description for each run, name of the saving folder.')

    # Data configuration
    parser.add_argument('--data_source', type=str, default='original', choices=['original', 'LED'], help='Data source for the training data.')
    parser.add_argument('--epochs', default=None, type=int, help='Override the number of epochs in the config file.')
    parser.add_argument('--batch_size', default=None, type=int, help='Override the batch size in the config file.')
    parser.add_argument('--data_dir', type=str, default='./data/eth_ucy', help='Directory where the data is stored.')
    parser.add_argument('--overfit', default=False, action='store_true', help='Overfit the testing set by setting it to the same entries as the training set.')
    parser.add_argument('--n_train', type=int, default=32500, help='Number training scenes used.')
    parser.add_argument('--n_test', type=int, default=12500, help='Number testing scenes used.')
    parser.add_argument('--checkpt_freq', default=5, type=int, help='Override the checkpt_freq in the config file.')
    parser.add_argument('--max_num_ckpts', default=5, type=int, help='Override the max_num_ckpts in the config file.')
    parser.add_argument('--data_norm', default='min_max', choices=['min_max', 'sqrt'], help='Normalization method for the data.')
    parser.add_argument('--subset', type=str, required=True, choices=['eth', 'hotel', 'univ', 'zara1', 'zara2'], help='Trajectory subset to run experiment')
    parser.add_argument('--rotate', default=False, action='store_true', help="Whether to rotate the trajectories in the dataset")
    parser.add_argument('--rotate_time_frame', type=int, default=0, help='Index of time frames to rotate the trajectories.')
    parser.add_argument('--rotate_aug', default=False, action='store_true', help='Whether to use rotation as data augmentation.')

    # Reproducibility configuration
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='fix random seed for reproducibility')
    parser.add_argument('--seed', type=int, default=42, help='Set the random seed.')


    ### IMLE parameters ###
    parser.add_argument('--load_pretrained', default=False, action='store_true', help='Whether to load a pretrained encoder.')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the pretrained encoder checkpoint.')
    parser.add_argument('--latent_tau', type=int, default=0, help='The remaining denoising steps needed for the distillation latents')
    parser.add_argument('--objective', type=str, default='set', choices=['set', 'single'], help='Whether to generate a set of latents or a single latent for one noise vector.')
    parser.add_argument('--num_to_gen', type=int, default=30, help='Number of samples to generate to compute IMLE loss.')
    ### IMLE parameters ###


    ### Architecture configuration ###
    parser.add_argument("--use_pre_norm", default=False, action='store_true', help='Where to normalize the input trajectories in the Transformer Encoders.')
    ### Architecture configuration ###


    ### Regression loss configuration ###
    parser.add_argument('--loss_reg_gt_weight', default=0.0, type=float, help='Weight for the ground truth supervision loss.')
    parser.add_argument('--loss_reg_chamfer_weight', default=1.0, type=float, help='Weight for the chamfer distance loss.')
    parser.add_argument('--loss_reg_reduction', type=str, default='sum', choices=['mean', 'sum'], help='Reduction method for the regression loss.')
    parser.add_argument('--loss_reg_squared', default=False, action='store_true', help='Whether to use the squared regression loss.')
    ### Regression loss configuration ###

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

    ### Architecture configuration ###
    def _update_architecture_params(args, cfg, tag):
        cfg.MODEL.USE_PRE_NORM = args.use_pre_norm
        return cfg, tag

    cfg, tag = _update_architecture_params(args, cfg, tag)

    ### General denoising objective configuration ###
    def _update_general_imle_params(args, cfg, tag):
        cfg.latent_tau = args.latent_tau
        cfg.objective = args.objective
        cfg.num_to_gen = args.num_to_gen
        cfg.load_pretrained = args.load_pretrained

        tag += f'_IMLE_gen_{cfg.objective}_M_{cfg.num_to_gen}'

        if cfg.load_pretrained:
            cfg.ckpt_path = args.ckpt_path
            tag += '_load_enc'

        return cfg, tag

    cfg, tag = _update_general_imle_params(args, cfg, tag)


    ### Regression loss configuration ###
    def _update_regression_loss_params(args, cfg, tag):
        cfg.loss_reg_gt_weight = args.loss_reg_gt_weight
        cfg.loss_reg_chamfer_weight = args.loss_reg_chamfer_weight
        tag += '_GT_{:.2f}_Chamfer_{:.2f}'.format(cfg.loss_reg_gt_weight, cfg.loss_reg_chamfer_weight)

        cfg.loss_reg_reduction = args.loss_reg_reduction
        cfg.loss_reg_squared = args.loss_reg_squared

        tag += f'_REG_{cfg.loss_reg_reduction[:1].upper()}'

        if args.loss_reg_squared:
            tag += '_SQ'

        return cfg, tag

    cfg, tag = _update_regression_loss_params(args, cfg, tag)


    ### Update data configuration ###
    def _update_data_params(args, cfg, tag):	
        cfg.subset = args.subset
        tag += f'_{cfg.subset}'
    

        if args.overfit:
            tag += '_overfit'

        cfg.rotate = args.rotate
        if args.rotate:
            cfg.rotate_time_frame = args.rotate_time_frame
            cfg.rotate_aug = args.rotate_aug
            tag += f'_rot_{cfg.rotate_time_frame}'
            if cfg.rotate_aug:
                tag += '_aug'

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
            cfg.test_batch_size = args.batch_size

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
    train_dset = ETHDataset(
        cfg=cfg,
        training=True,
        data_dir=args.data_dir,
        subset=args.subset,
        rotate_time_frame=args.rotate_time_frame,
        imle=True,
        type=args.data_source,)

    train_loader = DataLoader(
        train_dset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=seq_collate_imle_train,
        pin_memory=True)
    

    if args.overfit:
        test_dset = copy.deepcopy(train_dset)
    else:
        test_dset = ETHDataset(
        cfg=cfg,
        training=False,
        data_dir=args.data_dir,
        subset=args.subset,
        rotate_time_frame=args.rotate_time_frame,
        imle=False)
        
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
    model = ETHIMLETransformer(
        model_config=cfg.MODEL,
        logger=logger,
        config=cfg,
    )

    if args.load_pretrained:
        state_dict = torch.load(args.ckpt_path, map_location='cpu')['model']
        for key, val in state_dict.items():
            if 'context_encoder' in key:
                if key.startswith('model.'):
                    key = key.replace('model.', '')
                assert key in model.state_dict(), f"Key {key} not in model state dict."
                model.state_dict()[key].copy_(val)

            if 'motion_decoder' in key:
                if key.startswith('model.'):
                    key = key.replace('model.', '')
                if 'adaLN' in key:
                    continue
                assert key in model.state_dict(), f"Key {key} not in model state dict."
                model.state_dict()[key].copy_(val)

        logger.info("The pretrained encoder and decoder have been loaded.")

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

    """Train the model"""
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
        ) ### grid search

    trainer.train()


if __name__ == "__main__":
    main()
