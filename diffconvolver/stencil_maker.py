import numpy as np


def del_maker(ng, make_6th=False):
    c2, i02 = sm.stencil_weights(2, -ng, ng)
    c4, i04 = sm.stencil_weights(4, -ng, ng)
    c6, i06 = sm.stencil_weights(6, -ng, ng)

    dxx = sm.stencil_matrix_pure(c2, i02, y_axes=0)
    dyy = sm.stencil_matrix_pure(c2, i02, y_axes=1)

    del2 = dxx + dyy

    dx4 = sm.stencil_matrix_pure(c4, i04, y_axes=0)
    dy4 = sm.stencil_matrix_pure(c4, i04, y_axes=1)
    dx2y2 = sm.stencil_matrix_mixed(c2, c2)

    del4 = dx4 + dy4 + 2 * dx2y2
    if make_6th:
        dx6 = sm.stencil_matrix_pure(c6, i04, y_axes=0)
        dy6 = sm.stencil_matrix_pure(c6, i04, y_axes=0)
        dx2y4 = sm.stencil_matrix_mixed(c2, c4)
        dx4y2 = sm.stencil_matrix_mixed(c4, c2)

        del6 = dx6 + 3*(dx4y2 + dx2y4) + dy6

        return del2, del4, del6
    else:
        return del2, del4


class Grid2D:
    def __init__(self, x_min: int = 0, x_max: int = 50, y_min: int = 0, y_max: int = 2*np.pi,
                 nx: int = 1_00, n_g: int = 2):

        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min
        self.nx = nx
        self.n_g = n_g

        self.dx = (self.x_max - self.x_min) / float(self.nx - 2*self.n_g)
        self.ny = int((self.y_max - self.y_min) / self.dx)
        self.dy = (self.y_max - self.y_min) / float(self.ny - 2*self.n_g)

        self.x = np.linspace(self.x_min, self.x_max, self.nx - 2*self.n_g)
        self.y = np.linspace(self.y_min, self.y_max, self.ny - 2*self.n_g)
        self.x_gr, self.y_gr = np.meshgrid(self.x, self.y)

        self.lg = (slice(0, None), slice(0, self.n_g))
        self.lb = lambda k=0: (slice(self.n_g, -self.n_g), self.n_g + k)
        self.lg_mirror = (slice(0, None), slice(-2 * self.n_g, -self.n_g))

        self.rg = (slice(0, None), slice(-self.n_g, None))
        self.rb = lambda k=0: (slice(self.n_g, -self.n_g), -self.n_g - 1 + k)
        self.rg_mirror = (slice(0, None), slice(self.n_g, 2 * self.n_g))

        self.bg = (slice(0, self.n_g), slice(0, None))
        self.bb = lambda k=0: (self.n_g + k, slice(self.n_g, -self.n_g))
        self.bg_mirror = (slice(-2 * self.n_g, -self.n_g), slice(0, None))

        self.ug = (slice(-self.n_g, None), slice(0, None))
        self.ub = lambda k=0: (-self.n_g - 1 + k, slice(self.n_g, -self.n_g))
        self.ug_mirror = (slice(self.n_g, 2 * self.n_g), slice(0, None))

        self.bulk = (slice(self.n_g, -self.n_g), slice(self.n_g, -self.n_g))


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
