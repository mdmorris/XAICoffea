import numpy as np


def coord_rotation(a, toaxis):
    '''
    Get the axis and angle of rotation for a coordinate transformation
    where vector "a" is rotated to be along axis "toaxis". 
    '''
    alpha = np.arccos( a.dot(toaxis)/( abs(a) * abs(toaxis) ))
    axis_perp = a.cross(toaxis).unit
    return axis_perp, alpha