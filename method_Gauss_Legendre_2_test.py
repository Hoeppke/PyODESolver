from method_Gauss_Legendre_2 import GaussLegendre
from rhs_function import ExampleFunc01
from rhs_function import ExampleFunc01_solution
import matplotlib.pyplot as plt
import numpy as np
import unittest

class TestGaussLegendre(unittest.TestCase):
        
        def test_accuracy01(self):
            N=2**5
            t=np.linspace(0,1,num=N)
            y0=np.array([np.pi])
            exactSol = ExampleFunc01_solution(y0,t).T

            #  Compute Numerical Solution:
            GaussLegendre_solver=GaussLegendre(N, y0, [0,1],ExampleFunc01())
            solution= GaussLegendre_solver.generate()
            numericSol= np.zeros_like(exactSol)
            idx = 0
            for (time,val) in solution:
                    numericSol[idx] = val
                    idx += 1

            err=np.max(np.abs(exactSol-numericSol))
            for i in range(N):
                print(np.abs(exactSol[i]-numericSol[i]))
            self.assertTrue(err<4*10**(-7))

if __name__ == "__main__":
    unittest.main()
