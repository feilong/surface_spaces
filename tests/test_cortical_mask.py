import importlib
import pytest
import numpy as np
from surface_spaces import get_cortical_mask


def test_cortical_mask():
    spaces = [a + b for a in ['onavg-', 'fsavg-'] for b in ['ico32', 'ico64', 'ico128']]
    for space in spaces:
        ico = int(space.split('-ico')[1])
        nv = ico**2 * 10 + 2
        for geometry in ['on1031', 'fsavg', 'fsavg5', 'fsavg6']:
            for lr in 'lr':
                mask = get_cortical_mask(lr, space, geometry)
                assert mask.shape == (nv, )
                assert mask.dtype == bool


def test_comparison_with_searchlights():
    if importlib.util.find_spec('searchlights') is None:
        pytest.skip("Skip comparison of masks with the `searchlights` package.")
    from searchlights import get_mask
    spaces = [a + b for a in ['fsavg-'] for b in ['ico32', 'ico64', 'ico128']]
    for space, icoorder in zip(spaces, (5, 6, 7)):
        nv1 = 4**icoorder * 10 + 2
        for geometry in ['fsavg', 'fsavg5', 'fsavg6']:
            icoorder2 = {'fsavg': 7, 'fsavg5': 5, 'fsavg6': 6}[geometry]
            nv2 = 4**icoorder2 * 10 + 2
            for lr in 'lr':
                mask1 = get_mask(lr, mask_space=geometry.replace('fsavg', 'fsaverage'), icoorder=icoorder)
                mask2 = get_cortical_mask(lr, space, geometry)
                print(space, geometry, icoorder, mask1.shape, mask2.shape)
                assert mask1.shape == (min(nv1, nv2), )
                assert mask2.shape == (nv1, )
                if nv2 < nv1:
                    np.testing.assert_array_equal(mask1, mask2[:nv2])
                else:
                    np.testing.assert_array_equal(mask1, mask2)


# def test_number_of_cortical_vertices():
#     for space in ['']