import diffconvolver.stencil_maker as sm
import numpy as np
from scipy import signal as sg


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
