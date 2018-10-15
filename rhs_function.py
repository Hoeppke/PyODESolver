import scipy.sparse as sparse


class RHSFunction(object):

    """Docstring for RHSFunction. """

    def __init__(self):
        pass

    def eval(self, y_vec, time):
        """TODO: Docstring for eval.

        :y_vec: The vector of the current y value
        :time: The time as double
        :returns: The evaluation f(y_vec, time)

        """
        raise NotImplementedError

    def jacobian(self, y_vec, time):
        """TODO: Docstring for jacobian.

        :y_vec: The vector of the current y value
        :time: The time as double
        :returns: The evaluation of the jacobina grad_y f(y_vec, time)

        """
        raise NotImplementedError

class ExampleFunc01(RHSFunction):

    """Docstring for ExampleFunc01. """

    def __init__(self):
        """TODO: to be defined1. """
        RHSFunction.__init__(self)

    def eval(self, y_vec, time):
        """TODO: Docstring for eval.

        :y_vec: The vector of the current y value
        :time: The time as double
        :returns: The evaluation f(y_vec, time)

        """
        eval_vec = -2.0 * y_vec
        return eval_vec

    def jacobian(self, y_vec, time):
        """TODO: Docstring for jacobian.

        :y_vec: The vector of the current y value
        :time: The time as double
        :returns: The evaluation f(y_vec, time)

        """
        N = len(y_vec)
        jac = sparse.diags([-2.0], [0], shape=(N, N))
        return jac
