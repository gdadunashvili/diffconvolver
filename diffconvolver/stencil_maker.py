import numpy as np


def stencil_weights(d: int, i_min: int, i_max: int):
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


def stencil_matrix_pure(c: np.ndarray, i0: int, y_axes=0):
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


if __name__ == '__main__':
    c, i0 = stencil_weights(1, -2, 1)
    Mxx = stencil_matrix_pure(c, i0, y_axes=0)
    Myy = stencil_matrix_pure(c, i0, y_axes=1)
    Mxx_yy = Mxx + Myy
    print(c)
    print(Mxx)
    print(Mxx_yy)
