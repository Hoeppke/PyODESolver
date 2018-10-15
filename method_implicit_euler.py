from step_method import StepMethod


class ImplicitEuler(StepMethod):

    """This Class implements the implicit Euler time step method
       for solving ode problems."""

    def __init__(self, N, y0, domain, func):
        StepMethod.__init__(self, N, y0, domain, func)

    def step(self, f, u, t, h):
        # Perform the explicit evaluation.
