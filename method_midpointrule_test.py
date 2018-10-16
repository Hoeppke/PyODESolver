from method_midpointrule import MidPointRule
from rhs_function import ExampleFunc01
from rhs_function import ExampleFunc01_solution
import matplotlib.pyplot as plt
import numpy as np
import unittest

class TestMidPointRule(unittest.TestCase):

    def test_accuracy01(self):
        N = 2**10
        t = np.linspace(0, 1, num=N)
        y0 = np.array([np.pi])
        exactSol = ExampleFunc01_solution(y0, t).T

        #  Compute numerical sol:
        mpr_solver = MidPointRule(N, y0, [0, 1], ExampleFunc01())
        solution = mpr_solver.generate()
        numericSol = np.zeros_like(exactSol)
        idx = 0
        for (time, val) in solution:
            numericSol[idx] = val
            idx += 1

        err = np.max(np.abs(exactSol-numericSol))
        print(err)
        self.assertTrue(err < 4.0*10**(-7))


if __name__ == "__main__":
    unittest.main()
