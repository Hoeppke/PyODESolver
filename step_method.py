import numpy as np


class StepMethod(object):

    """Docstring for step_method.
        Time stepping method for ODE's
    """

    def __init__(self, N, y0, domain, func):
        """TODO: to be defined1. """
        self.N = N
        self.y0 = y0
        self.domain = domain
        self.func = func

    def setGrid(self, domain, N):
        self.domain = domain
        self.N = N

    def setFunc(self, func):
        self.func = func

    def generate(self):
        Tstart, Tend = self.interval
        Tgrid = np.linspace(Tstart, Tend, self.N)
        (time, uvec) = (Tstart, self.y0)
        yield (time, uvec)
        for tIdx in range(1, len(Tgrid)):
            time = Tgrid[tIdx]
            uvec = self.step(self.func, uvec)


    def newton_iteration():
        pass


    def step(self, f, u, t, h):
        raise NotImplementedError
