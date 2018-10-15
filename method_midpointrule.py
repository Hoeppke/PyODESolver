from step_method import StepMethod
from rhs_function import RHSFunction
from scipy import optimize
import scipy.sparse as sparse
import numpy as np

class MidPointRule(StepMethod):

    """ 
        This implements MPR
    """

    def step(self, func, uvec, time, steplen, tolerance=10**(-8)):
        y=np.copy(uvec)
        yprev=np.copy(uvec)
        err=1
        numit=0
        I=sparse.eye(len(uvec))
        while err>tolerance and numit<5:
            yprev=np.copy(y)
            A=(I-steplen/2.0*func.jacobian(yprev,time))
            F=yprev-uvec-steplen/2*(func.eval(yprev,time)+func.eval(uvec,time))
            b=-np.dot(I,F)+np.dot(A,yprev)
            y=sparse.linalg.spsolve(A,b)
            numit += 1
        return y
