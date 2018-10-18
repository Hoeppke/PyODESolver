from step_method import StepMethod
from rhs_function import RHSFunction
from scipy import optimize
import scipy.sparse as sparse
import numpy as np


class MidPointRule(StepMethod):

    """
        This implements MPR
    """

    def step(self, func, uvec, time, steplen, tolerance=10**(-10), maxiter=10):
        t_old = time
        t_new = time + steplen
        y_old = np.copy(uvec)
        y_new = np.copy(uvec) + (steplen/2) * func.eval(y_old, t_old)
        err = 1
        numit = 0
        sp_I = sparse.eye(len(uvec))
        func_old = func.eval(y_old, t_old)

        def myF(y_new):
            val1 = (func.eval(y_new, t_new) + func_old)
            val2 = y_new - y_old - (steplen/2) * val1
            return val2

        def myJacF(y_new):
            A = (sp_I - steplen/2.0 * func.jacobian(y_new, y_new))
            return sparse.csr_matrix(A)

        while err > tolerance and numit < maxiter:
            Jac = myJacF(y_new)
            Fval = myF(y_new)
            y_update = sparse.linalg.spsolve(Jac, Fval)
            y_new = y_new - y_update
            numit += 1
            err = np.max(np.abs(myF(y_new)))
        return y_new
