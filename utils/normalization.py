import torch

### Helper functions for linear normalization and unnormalization
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def normalize_min_max(t, min_val, max_val, a, b, identity=False):
    '''
    Normalize t to [a, b] range
    Args:
    t: input tensor
    min_val: minimum value of t
    max_val: maximum value of t
    a: minimum value of the output range
    b: maximum value of the output range
    '''
    if identity:
        return t
    else:
        return (b - a) * (t - min_val)/(max_val - min_val) + a 

def unnormalize_min_max(t, min_val, max_val, a, b, identity=False):
    '''
    Unnormalize t from [a, b] range back to [min_val, max_val] range
    Args:
    t: input tensor
    min_val: minimum value of t
    max_val: maximum value of t
    a: minimum value of the input range
    b: maximum value of the input range
    '''
    if identity:
        return t
    else:
        return (t - a) * (max_val - min_val)/(b - a) + min_val


def normalize_sqrt(traj_data, a, b):
    '''
    Normalize input tensor to [-1, 1] using square root.
    @param traj_data: [*, 2]
    @param a: [2]
    @param b: [2]
    '''
    traj_data = torch.abs(traj_data).sqrt() * torch.sign(traj_data)
    traj_data = traj_data / a.reshape(*([1] * (traj_data.dim() - 1)), -1) + b.reshape(*([1] * (traj_data.dim() - 1)), -1)
    return traj_data

def unnormalize_sqrt(traj_data, a, b):
    '''
    Unnormalize input tensor from [-1, 1] using square root.
    @param traj_data: [*, 2]
    @param a: [2]
    @param b: [2]
    '''
    traj_data = (traj_data - b.reshape(*([1] * (traj_data.dim() - 1)), -1)) * a.reshape(*([1] * (traj_data.dim() - 1)), -1)
    traj_data = torch.sign(traj_data) * traj_data ** 2
    return traj_data
