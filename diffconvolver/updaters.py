import diffconvolver.stencil_maker as sm
import numpy as np
from scipy import signal as sg
from Grid2D import Grid2D



def del_maker(ng, make_6th=False):
    c1, i01 = sm.stencil_weights(1, -ng, ng)
    c2, i02 = sm.stencil_weights(2, -ng, ng)
    c4, i04 = sm.stencil_weights(4, -ng, ng)
    c6, i06 = sm.stencil_weights(6, -ng, ng)

    dx = sm.stencil_matrix_pure(c1, i01, y_axes=0)
    dy = sm.stencil_matrix_pure(c1, i01, y_axes=1)
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
        return dx, dy, dxx, dyy, del2, del4

class Parameters:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.1, t_u: float = 1, t_psi: float = 1,
                 sigma: float=0., lmbda: float=0.1):
        self.sigma = sigma
        self.lmbda = lmbda
        self.alpha = alpha
        self.gamma = gamma
        self.t_u = t_u
        self.t_psi = t_psi

        self.ap1 = self.alpha + 1
        self.sigma_eff = sigma
        self.gamma_eff = self.gamma/(self.lmbda**2)
        self.two_sigma_plus_1 = 2*self.sigma + 1
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

    u[grid.bulk] -= dt*(u4 - 0.5*u2 + psi2)

    psi[grid.bulk] += par.tr*dt*(-u4 + par.two_sigma_plus_1*psi2 - par.gamma_eff*psi4)


def bulk_update_tube(u: np.ndarray, psi: np.ndarray,
                     dt: float, d2: np.ndarray, d4: np.ndarray, grid: Grid2D, par: Parameters):

    u2 = sg.fftconvolve(u, d2, mode='valid')
    u4 = sg.fftconvolve(u, d4, mode='valid')
    # u6 = sg.fftconvolve(u, d6, mode='valid')

    psi2 = sg.fftconvolve(psi, d2, mode='valid')
    # psi4 = sg.fftconvolve(psi, d4, mode='valid')

    if par.gamma != 0:
        psi4 = sg.fftconvolve(psi, d4, mode='valid')
    else:
        psi4 = 0

    # u[grid.bulk] -= dt*(u4 + (2-par.sigma_eff)*u2 + (1-par.sigma_eff)*u[grid.bulk] + psi2 + psi[grid.bulk])
    u[grid.bulk] -= dt*(u4 + 2*u2 + u[grid.bulk] + psi2 + psi[grid.bulk])
    psi[grid.bulk] += par.tr*dt*(u4 + u2 + par.two_sigma_plus_1 *psi2 - (par.gamma/par.lmbda**2)*psi4)
    # psi[grid.bulk] += par.tr*dt*(u4 + u2 + par.ap1*psi2 - par.gamma*psi4)


def nonlinear_bulk_update_tube(u: np.ndarray, psi: np.ndarray,
                     dt: float,
                     dx: np.ndarray, dy: np.ndarray, dxx: np.ndarray, dyy: np.ndarray,
                     d2: np.ndarray, d4: np.ndarray, grid: Grid2D, par: Parameters):

    ux = sg.fftconvolve(u, dx, mode='valid')
    uy = sg.fftconvolve(u, dy, mode='valid')
    uxx = sg.fftconvolve(u, dxx, mode='valid')

    uyy = sg.fftconvolve(u, dyy, mode='valid')
    u2 = sg.fftconvolve(u, d2, mode='valid')
    # print(uxx.shape, u2.shape)
    u4 = sg.fftconvolve(u, d4, mode='valid')
    # u6 = sg.fftconvolve(u, d6, mode='valid')

    psi2 = sg.fftconvolve(psi, d2, mode='valid')
    psi4 = sg.fftconvolve(psi, d4, mode='valid')


    psi_lin :np.ndarray = (u4 + u2 + par.ap1 * psi2)
    # print(psi_lin.shape)
    psi_underived = par.ap1 * psi[grid.bulk]*u[grid.bulk] + 0.5*(uxx - 3*uyy)*u[grid.bulk]
    # print(psi_underived.shape)
    psi_derived_x = sg.fftconvolve(psi_underived, dx, mode='same')
    # print('psi_derived_x:',psi_derived_x.shape)
    psi_derived_yy = sg.fftconvolve(psi_underived, dyy, mode='same')
    psi_derived0 = sg.fftconvolve(psi_underived, d2, mode='same')

    psi_derived = psi_derived0 + 2*(ux * psi_derived_x - u[grid.bulk] *psi_derived_yy)

    u[grid.bulk] -= dt * (u4 + 2*u2 + u[grid.bulk] + psi[grid.bulk]
                          + 1.5*u[grid.bulk]**2 + u[grid.bulk]*u2 +2*(u[grid.bulk] + uxx)*uyy + uy**2
                          -0.5*(par.ap1)*psi[grid.bulk]**2 + (uyy-uxx)*psi[grid.bulk])
    psi[grid.bulk] += par.tr * dt * (psi_lin + psi_derived)

def bulk_update_u_averaged_tube(psi: np.ndarray,
                                dt: float, d2: np.ndarray, d4: np.ndarray, d6: np.ndarray,
                                grid: Grid2D, par: Parameters):
    psi2 = sg.fftconvolve(psi, d2, mode='valid')
    psi4 = sg.fftconvolve(psi, d4, mode='valid')
    psi6 = sg.fftconvolve(psi, d6, mode='valid')
    psi3d2 = sg.fftconvolve(psi*psi*psi, d2, mode='valid')

    psi[grid.bulk] += dt * (par.alpha * psi2 - (par.gamma+2)*psi4 - psi6 + 20*psi3d2)
