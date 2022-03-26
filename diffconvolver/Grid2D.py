import numpy as np
class Grid2D:
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float,
                 nx: int, n_g: int):
        """
        Initiate a 2D grid on which the finite differences will be calculated.

        Args:
            x_min (float): _description_
            x_max (int): _description_
            y_min (int): _description_
            y_max (int): _description_
            nx (int, optional): number of grid point in x direction.
            n_b (int, optional): number of ghost points, i.e. number of points dedicated to the boundary layer, at each boundary.
        """  
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