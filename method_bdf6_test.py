from method_bdf6 import BDF6Method
from rhs_function import ExampleFunc01
from rhs_function import ExampleFunc01_solution
import matplotlib.pyplot as plt
import numpy as np
import unittest


class TestBDF6Example(unittest.TestCase):

    def test_accuracy01(self):
        N = 10**3
        t = np.linspace(0, 1, num=N)
        # y0 = np.sin(np.linspace(-1.0*np.pi, 1.0*np.pi))
        y0 = np.array([np.pi])
        exactSol = ExampleFunc01_solution(y0, t).T

        # Compute numerical solution:
        bdf6_solver = BDF6Method(N, y0, [0, 1], ExampleFunc01())
        solution = bdf6_solver.generate()
        numericSol = np.zeros_like(exactSol)
        idx = 0
        for (time, val) in solution:
            numericSol[idx] = val
            idx += 1

        err = np.max(np.abs(exactSol - numericSol))
        # BDF6 does not work currently :(
        self.assertTrue(err < 10**(-7))


if __name__ == "__main__":
    unittest.main()
