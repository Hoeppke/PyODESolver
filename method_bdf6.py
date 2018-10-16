from method_explicit_euler import ExplicitEuler
from rhs_function import RHSFunction
from scipy import optimize
from step_method import StepMethod
from method_RK4 import RK4 as RK4
import numpy as np
import scipy.sparse as sparse

class BDF6Method(StepMethod):

    past_data = []
    starterMethod = None
    starterSolver = None

    def __init__(self, N, y0, domain, func):
        StepMethod.__init__(self, N, y0, domain, func)

        # Start appending data to past data
        t0, tend = domain
        self.past_data.append([y0, t0])

        # Define RK4 as the starter method:
        self.starterMethod = RK4(N, y0, domain, func)
        self.starterSolver = self.starterMethod.generate()

    def step(self, f, u, t, h, tol=10**(-10), maxiter=10):
        """Implements the step method for the BDF6 method
        :returns: generator for the step values
        """

        # For BDF6 to work we need 6 pieces of old data.
        # figure out if the past data is filled well:
        if(len(self.past_data) >= 6):
            # Implementation of the BDF6 method
            # Guess the y_new using RK4
            y_new = next(self.starterSolver)
            t_new = t + h
            N = len(u)

            # define internal functions for the newton method:
            def myF(y_new):
                val = y_new
                val -= (360.0 / 147.0) * self.past_data[-1][0]
                val += (450.0 / 147.0) * self.past_data[-2][0]
                val -= (400.0 / 147.0) * self.past_data[-3][0]
                val += (225.0 / 147.0) * self.past_data[-4][0]
                val -= (72.0 / 147.0) * self.past_data[-5][0]
                val += (10.0 / 147.0) * self.past_data[-6][0]
                val -= h * (60.0 / 147.0) * f.eval(y_new, t_new)
                return val

            # define the internal jacobinal for the newton method:
            def myJacF(y_new):
                val1 = sparse.eye(N)
                val2 = -1.0 * (60.0/147.0) * h * f.jacobian(y_new, t_new)
                return val1 + val2

            itercount = 0
            err = 1
            while err > tol and itercount < maxiter:
                Jac = myJacF(y_new)
                Fval = myF(y_new)
                y_update = sparse.linalg.spsolve(Jac, Fval)
                y_new = y_new - y_update
                itercount += 1
                err = np.max(np.abs(myF(y_new)))

            # Append to the list of past solutions:
            self.past_data.append([y_new, t_new])
            return y_new

        else:
            # Use RK4 for the first 6 timesteps:
            y_new = next(self.starterSolver)
            t_new = t + h
            self.past_data.append([y_new, t_new])
            return y_new


