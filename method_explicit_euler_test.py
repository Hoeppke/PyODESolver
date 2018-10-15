from method_explicit_euler import ExplicitEuler
from rhs_function import ExampleFunc01
from rhs_function import ExampleFunc01_solution
import numpy as np
import unittest
import matplotlib.pyplot as plt


class TestExplicitEulerMethod(unittest.TestCase):

    def test_accuracy01(self):
        N = 10**5
        t = np.linspace(0, 1, num=N)
        # y0 = np.sin(np.linspace(-1.0*np.pi, 1.0*np.pi))
        y0 = np.pi
        exactSol = ExampleFunc01_solution(y0, t).T

        # Compute numerical solution:
        ee_solver = ExplicitEuler(N, y0, [0, 1], ExampleFunc01())
        solution = ee_solver.generate()
        numericSol = np.zeros_like(exactSol)
        idx = 0
        for (time, val) in solution:
            numericSol[idx] = val
            idx+=1

        err = np.max(np.abs(exactSol - numericSol))
        print(err)
        self.assertTrue(err < 1.2*10**(-5))


if __name__ == "__main__":
    unittest.main()
