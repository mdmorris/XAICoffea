import numpy as np

def geometric_cross(A,B):
    result = A.empty_like()
    result["fX"], result["fY"], result["fZ"] = A._cross(B)
    return result

def array_rotate_axis(A, u, alpha):
    result = A.empty_like()
    result["fX"], result["fY"], result["fZ"] = A.rotate_axis(u, alpha)
    result["fE"] = A["fE"]
    return result

def coord_rotation(a, toaxis):
    '''
    Get the axis and angle of rotation for a coordinate transformation
    where vector "a" is rotated to be along axis "toaxis". 
    '''
    alpha = a.angle(toaxis)
    axis_perp = geometric_cross(a, toaxis).unit
    return axis_perp, alpha