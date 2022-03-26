from typing import Tuple
import numpy as np


def stencil_weights(d: int, i_min: int, i_max: int)->Tuple[np.ndarray, int]:
    """
    Generates weights c_k for the finite difference stencil.

    Args:
        d (int): order of the derivative
        i_min (int): number of stencil points to the left of the point of the derivation
        i_max (int): number of stencil points to the right of the point of the derivation

    Returns:
        Tuple[np.ndarray, int]: returns a tuple of the stencli weight vector c_k (1d numpy array) and the largest of the two values i_min and i _max
    """    
    A = []
    b = []
    di = i_max-i_min + 1
    for n in range(di):
        A.append(np.array([1 if (i == 0) & (n == 0) else i**n for i in range(i_min, i_max + 1)]))
        b.append(0 if n != d else 1)

    A = np.array(A)
    b = np.array(b)
    i0 = np.max(np.abs([i_max, i_min]))
    return np.linalg.solve(A, b)*np.math.factorial(d), i0


def stencil_matrix_pure(c: np.ndarray, i0: int, y_axes=0)-> np.ndarray:
    """
    Generates a matrix that corresponds to a FDM stencil of a derivative in only one direction. This matrix can be convolved with the target field to take the derivative.
    Args:
        c (np.ndarray): stencil vector
        i0 (int): number of points on the largest side of the stencil
        y_axes (int, optional): Transposes the stencil matrix if the derivation is i y direction. Stencil vectors for x and y axes are the same but their stencil matrices are transposedversions of each other. Defaults to 0.

    Returns:
        np.ndarray: _description_
    """    
    n = c.size
    o = np.zeros(n)
    o[i0] = 1
    M = np.outer(o, c)

    if y_axes:
        return M.transpose()
    else:
        return M


def stencil_matrix_mixed(*c_s: np.ndarray):
    return np.outer(*c_s)
