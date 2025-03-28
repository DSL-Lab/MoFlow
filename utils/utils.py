import os
import git 
import logging
import shutil
import torch
import numpy as np
import random
import glob
import git
from torch import nn
from pathlib import Path
from pathlib import Path
from easydict import EasyDict
from scipy.stats import gaussian_kde


def back_up_code_git(cfg, logger):
    # save version control information
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    logger.info("git hash: {}".format(sha))

    # backup code
    code_backup_dir = Path(cfg.cfg_dir) / 'code_backup'
    shutil.rmtree(code_backup_dir, ignore_errors=True)
    code_backup_dir.mkdir(parents=True, exist_ok=True)
    dirs_to_save = ['cfg', 'models', 'trainer']
    [shutil.copytree(os.path.join(cfg.ROOT, this_dir), os.path.join(code_backup_dir, this_dir)) for this_dir in dirs_to_save]
    ### find all the python files under ROOT and copy them under code_backup_dir
    all_py_files = glob.glob(os.path.join(cfg.ROOT, '*.py'), recursive=True)
    [shutil.copy2(py_file, os.path.join(code_backup_dir, os.path.relpath(py_file, cfg.ROOT))) for py_file in all_py_files]
    logger.info("Code is backedup to {}".format(code_backup_dir))


def log_config_to_file(cfg, pre='cfg_yml', logger=None):
    logger.info("{} Config {} details below {}".format("="*20, pre, "="*20))
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('--- %s.%s = edict() ---' % (pre, key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))
    logger.info("{} Config {} details above {}".format("-"*20, pre, "-"*20))


def compute_kde_nll(pred_trajs, gt_traj):
    kde_ll = 0.0
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[1]
    num_batches = pred_trajs.shape[0]
    kde_ll_time = np.zeros(num_timesteps)

    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = gaussian_kde(pred_trajs[batch_num, :, timestep].T)
                pdf = np.clip(
                    kde.logpdf(gt_traj[batch_num, timestep]),
                    a_min=log_pdf_lower_bound,
                    a_max=None,
                )[0]
                kde_ll += pdf / (num_timesteps)
                kde_ll_time[timestep] += pdf
            except np.linalg.LinAlgError:
                kde_ll = np.nan

    return -kde_ll, -kde_ll_time


def rotate_trajs_x_direction(past, future, past_abs, agent_of_interest=11): 
    """
    Define the rotation function to align the last segment in `past` of ball agent only to the x-direction
    """
    # Shape of past is [B, A, F, D] where F = number of frames and D = 2 (for 2D points)
    past_diff = past[:, agent_of_interest-1, -1] - past[:, agent_of_interest-1, -2]  # Difference between the last two points of ball trajectory

    # Calculate the rotation angle theta for alignment of ball's last segment
    past_theta = torch.atan2(past_diff[:, 1], past_diff[:, 0] + 1e-5)[:, None].repeat(1, past.size(1))  # Shape [B, A]
    # past_theta = torch.where((past_diff[:, 0] < 0), past_theta + math.pi, past_theta)  # Adjust for negative x-direction

    # Create a batch of rotation matrices for each agent in the batch
    rotate_matrix = torch.zeros((past_theta.size(0), past_theta.size(1), 2, 2)).to(past_theta.device)  # Shape [B, A, 2, 2]
    rotate_matrix[:, :, 0, 0] = torch.cos(past_theta)
    rotate_matrix[:, :, 0, 1] = torch.sin(past_theta)
    rotate_matrix[:, :, 1, 0] = -torch.sin(past_theta)
    rotate_matrix[:, :, 1, 1] = torch.cos(past_theta)

    # Apply the rotation to the `past`, `future`, and `past_abs` trajectories
    past_after = torch.matmul(rotate_matrix, past.transpose(-1, -2)).transpose(-1, -2)  # Shape [B, A, F, D]
    future_after = torch.matmul(rotate_matrix, future.transpose(-1, -2)).transpose(-1, -2)  # Shape [B, A, F, D]
    past_abs = torch.matmul(rotate_matrix, past_abs.transpose(-1, -2)).transpose(-1, -2)  # Shape [B, A, F, D]

    return past_after, future_after, past_abs


def apply_mask(input_tensor, mask, sample_dim=False):
    '''
    Apply mask to the input tensor
    mask: [B, A]
    input_tensor: [B, A, F, D], [B, A, D], [B, K, A, F, D]
    sample_dim: Whether dim=1 is the number of samples or not
    '''
    extend_dims = len(input_tensor.shape) - len(mask.shape)
    if sample_dim:
        mask = mask.unsqueeze(1)
        mask = mask[(..., ) + (None, ) * (extend_dims-1)]
    else:
        mask = mask[(..., ) + (None, ) * extend_dims]
    return input_tensor.masked_fill(mask, 0.)


def set_random_seed(rand_seed):
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def print_log(print_str, log, same_line=False, display=True):
    '''
    print a string to a log file

    parameters:
        print_str:          a string to print
        log:                a opened file to save the log
        same_line:          True if we want to print the string without a new next line
        display:            False if we want to disable to print the string onto the terminal
    '''
    if display:
        if same_line: print('{}'.format(print_str), end='')
        else: print('{}'.format(print_str))

    if same_line: log.write('{}'.format(print_str))
    else: log.write('{}\n'.format(print_str))
    log.flush()


class LossBuffer:
    def __init__(self, t_min, t_max, num_time_steps):
        """
        Initialize the LossBuffer with the specified number of denoising levels.
        """
        self.t_min = t_min
        self.t_max = t_max
        self.num_time_steps = num_time_steps
        self.t_interval = np.linspace(t_min, t_max, num_time_steps)
        self.loss_data = [[] for _ in range(self.num_time_steps)]
        self.last_epoch = -1

    def record_loss(self, t, loss, epoch_id):
        """
        Record the loss for a specific denoising level.
        @param t:       [B] the denoising level.
        @param loss:    [B] the loss value.    
        """

        flag_reset = False
        if epoch_id != self.last_epoch:
            self.last_epoch = epoch_id
            self.reset()
            flag_reset = epoch_id > 0
        
        if isinstance(t, torch.Tensor):
            t = t.cpu().numpy()
        if isinstance(loss, torch.Tensor):
            loss = loss.cpu().numpy()
        
        idx = np.digitize(t, self.t_interval) - 1
        for i, l in zip(idx, loss):
            self.loss_data[i].append(l)

        return flag_reset

    def reset(self):
        """
        Reset the loss data for a new epoch.
        """
        self.loss_data = [[] for _ in range(self.num_time_steps)]

    def get_average_loss(self):
        """
        Plot a histogram of denoising level vs. average loss for the last epoch.
        """
        avg_loss_per_level = [np.mean(l) if len(l) > 0 else 0.0 for l in self.loss_data]
        dict_loss_per_level = {t: l for t, l in zip(self.t_interval, avg_loss_per_level)}
        return dict_loss_per_level

