from step_method import StepMethod
import scipy.sparse as sparse

class MidPointRule(StepMethod):

    """ 
        This implements MPR
    """

    def step(self,func,uvec,time,steplen,tolerance=10**(-8))
        y=np.copy(uvec)
        yprev=np.copy(uvec)
        err=1
        numit=0
        while err>tolerance&numit<5
            yprev=np.copy(y)
            A=(np.eye(len(uvec))-steplen/2.0*func.jacobian(yprev,time))
            b=yprev+steplen/2*func.eval(yprev,time)+np.dot(A,yprev)
            y=sparse.linalg.spsolve(A,b)
        return y
