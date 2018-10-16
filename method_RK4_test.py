import numpy as np
import matplotlib.pyplot as plt
import unittest
from method_RK4 import RK4
from rhs_function import ExampleFunc01
from rhs_function import ExampleFunc01_solution

class TestRK4(unittest.TestCase):

        def test_accuracy01(self):
            N=10**3
            t= np.linspace(0,1,num=N)
            y0=np.array([np.pi])
            exactSol=ExampleFunc01_solution(y0,t).T

            #  Compute numerical solution:
            RK4_solver=RK4(N,y0,[0, 1], ExampleFunc01())
            solution=RK4_solver.generate()
            numericSol=np.zeros_like(exactSol)
            idx = 0
            for (time,val) in solution:
                numericSol[idx] = val
                idx+=1

            err=np.max(np.abs(exactSol-numericSol))
            print(err)
            self.assertTrue(err < 4.0*10**(-7))

if __name__ == "__main__":
    unittest.main()

