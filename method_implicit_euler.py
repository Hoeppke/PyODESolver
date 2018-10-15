from step_method import StepMethod
from method_explicit_euler import ExplicitEuler
from rhs_function import RHSFunction
import scipy.sparse as sparse
import scipy.optimize.newton as newton


class ImplicitEuler(StepMethod):

    """This Class implements the implicit Euler time step method
       for solving ode problems."""

    def __init__(self, N, y0, domain, func):
        StepMethod.__init__(self, N, y0, domain, func)

    def step(self, f, u, t, h, tol=10**(-8), maxiter=100):
        # Create a copy of u
        y_old = np.copy(u)
        # Guess the new y_new using explicit Euler method
        y_new = ExplicitEuler.step(f, u, t, h)
        t_new = t + h
        t_old = t

        # Compute the function required for Newton iteration.
        def myF(y_new):
            val = y_new - y_old - 0.5*h*(f.eval(y_new, t_new) + f.eval(y_old, t_old))
            return val

        def myJacF(y_new):
            val1 = sparse.eye(N)
            val2 = -1.0*(h/2)*f.jacobian(y_new, t_new)
            return val1 + val2

        itercount = 1
        y_ans = newton(myF, y_new, fprime=myJacF)
        return y_ans
