import diffconvolver.stencil_maker as sm
import numpy as np
from scipy import signal as sg


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

class Parameters:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.1, t_u: float = 1, t_psi: float = 1, sigma_eff=0.):
        self.sigma_eff = sigma_eff
        self.alpha = alpha
        self.gamma = gamma
        self.t_u = t_u
        self.t_psi = t_psi

        self.ap1 = self.alpha + 1
        self.tr = self.t_u/self.t_psi


class BcUpdate:
    def __init__(self, grid: Grid2D):
        self.grid = grid
        self.ghost_layers = [self.grid.lg, self.grid.rg, self.grid.ug, self.grid.bg]
        self.mirrors = [self.grid.lg_mirror, self.grid.rg_mirror, self.grid.ug_mirror, self.grid.bg_mirror]

    def periodic_x(self, u: np.ndarray, psi: np.ndarray):

        for ghost_layer, mirror in zip(self.ghost_layers[:2], self.mirrors[:2]):
            u[ghost_layer] = u[mirror]
            psi[ghost_layer] = psi[mirror]

    def periodic_y(self, u: np.ndarray, psi: np.ndarray):

        for ghost_layer, mirror in zip(self.ghost_layers[2:], self.mirrors[2:]):
            u[ghost_layer] = u[mirror]
            psi[ghost_layer] = psi[mirror]

    def double_periodic(self, u: np.ndarray, psi: np.ndarray):
        self.periodic_x(u, psi)
        self.periodic_y(u, psi)

    def fixed_boundary_in_x(self, u: np.ndarray, psi: np.ndarray,
                            u0l: np.ndarray, u0r: np.ndarray,
                            psi0l: np.ndarray, psi0r: np.ndarray):

        self.periodic_y(u, psi)

        u[self.grid.lb()] = u0l
        u[self.grid.rb()] = u0r

        psi[self.grid.lb()] = psi0l
        psi[self.grid.rb()] = psi0r

    def fixed_flux_in_x(self, u: np.ndarray, psi: np.ndarray,
                        du0l: np.ndarray, du0r: np.ndarray,
                        dpsi0l: np.ndarray, dpsi0r: np.ndarray):

        self.periodic_y(u, psi)

        u[self.grid.lb(-1)] = u[self.grid.lb(1)] - 2*du0l
        u[self.grid.rb(1)] = u[self.grid.rb(-1)] + 2*du0r

        psi[self.grid.lb(-1)] = psi[self.grid.lb(1)] - 2*dpsi0l
        psi[self.grid.rb(1)] = psi[self.grid.rb(-1)] + 2*dpsi0r

    def fixed_boundary_and_flux_in_x(self, u: np.ndarray, psi: np.ndarray,
                                     u0l: np.ndarray, u0r: np.ndarray, du0l: np.ndarray, du0r: np.ndarray,
                                     psi0l: np.ndarray, psi0r: np.ndarray, dpsi0l: np.ndarray, dpsi0r: np.ndarray):

        self.periodic_y(u, psi)

        self.fixed_boundary_in_x(u, psi, u0l, u0r, psi0l, psi0r)
        self.fixed_flux_in_x(u, psi, du0l, du0r, dpsi0l, dpsi0r)

    def free_hinge_in_x(self, u: np.ndarray, psi: np.ndarray,
                        u0l: np.ndarray, u0r: np.ndarray, psi0l: np.ndarray, psi0r: np.ndarray):

        self.periodic_y(u, psi)

        self.fixed_boundary_in_x(u, psi, u0l, u0r, psi0l, psi0r)

        u[self.grid.lb(-1)] = - u[self.grid.lb(1)] + 2*u[self.grid.lb()]
        u[self.grid.rb(1)] = - u[self.grid.rb(-1)] + 2*u[self.grid.rb()]

        psi[self.grid.lb(-1)] = - psi[self.grid.lb(1)] + 2*psi[self.grid.lb()]
        psi[self.grid.rb(1)] = - psi[self.grid.rb(-1)] + 2*psi[self.grid.rb()]


def bulk_update_flat(u: np.ndarray, psi: np.ndarray,
                     dt: float, d2: np.ndarray, d4: np.ndarray, grid: Grid2D, par: Parameters):

    u2 = sg.fftconvolve(u, d2, mode='valid')
    u4 = sg.fftconvolve(u, d4, mode='valid')

    psi2 = sg.fftconvolve(psi, d2, mode='valid')
    if par.gamma != 0:
        psi4 = sg.fftconvolve(psi, d4, mode='valid')
    else:
        psi4 = 0

    u[grid.bulk] -= dt*(u4 - (par.sigma_eff + 0.5)*u2 + psi2)

    psi[grid.bulk] += par.tr*dt*(-u4 + par.ap1*psi2 - par.gamma*psi4)


def bulk_update_tube(u: np.ndarray, psi: np.ndarray,
                     dt: float, d2: np.ndarray, d4: np.ndarray, grid: Grid2D, par: Parameters):

    u2 = sg.fftconvolve(u, d2, mode='valid')
    u4 = sg.fftconvolve(u, d4, mode='valid')
    # u6 = sg.fftconvolve(u, d6, mode='valid')

    psi2 = sg.fftconvolve(psi, d2, mode='valid')
    psi4 = sg.fftconvolve(psi, d4, mode='valid')
    # psi6 = sg.fftconvolve(psi, d6, mode='valid')

    u[grid.bulk] -= dt*(u4 + (2-par.sigma_eff)*u2 + (1-par.sigma_eff)*u[grid.bulk] + psi2 + psi[grid.bulk])
    psi[grid.bulk] += par.tr*dt*(u4 + u2 + par.ap1*psi2 - par.gamma*psi4)


def bulk_update_u_averaged_tube(psi: np.ndarray,
                                dt: float, d2: np.ndarray, d4: np.ndarray, d6: np.ndarray,
                                grid: Grid2D, par: Parameters):
    psi2 = sg.fftconvolve(psi, d2, mode='valid')
    psi4 = sg.fftconvolve(psi, d4, mode='valid')
    psi6 = sg.fftconvolve(psi, d6, mode='valid')
    psi3d2 = sg.fftconvolve(psi*psi*psi, d2, mode='valid')

    psi[grid.bulk] += dt * (par.alpha * psi2 - (par.gamma+2)*psi4 - psi6 + 20*psi3d2)
