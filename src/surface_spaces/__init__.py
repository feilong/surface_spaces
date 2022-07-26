import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


def choose_geometry_for_space(space):
    if space.startswith('fsavg-'):
        return 'fsavg'
    if space.startswith('onavg-'):
        return 'on1031'
    raise ValueError(f'{space} not recognized.')


def get_cortical_mask(lr, space, geometry=None):
    if geometry is None:
        geometry = choose_geometry_for_space(space)
    mask_fn = os.path.join(DATA_DIR, 'cortical_masks', space, geometry, f'{lr}h.npy')
    mask = np.load(mask_fn)
    return mask
