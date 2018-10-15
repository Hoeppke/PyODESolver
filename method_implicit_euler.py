from step_method import StepMethod
from method_explicit_euler import ExplicitEuler
from rhs_function import RHSFunction

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
            y_new - y_old - 0.5*h*(f.eval(y_new, t_new) + f.eval(y_old, t_old))
        itercount = 1
        err = abs(myF(y_new))
        while(err > tol and itercount < maxiter):
            # Newton iteration in here!
