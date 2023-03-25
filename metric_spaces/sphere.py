import autograd.numpy as anp
import numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
from .metric_space import MetricSpace

class Sphere(MetricSpace):
    def __init__(self, dim):
        assert dim==1
        self.dim = dim

    def _d(self, x, y):
        return np.sum(np.square(np.arccos(np.dot(x,y.T))), axis=0)
    
    def _frechet_mean(self, y, w):
        manifold = pymanopt.manifolds.Sphere(2)

        def _d(x, y): return anp.sum(anp.square(anp.arccos(anp.dot(x,y.T))), axis=0)
        
        @pymanopt.function.autograd(manifold)
        def cost(om): return anp.dot(w, _d(om.reshape((1,2)), y))

        problem = pymanopt.Problem(manifold, cost)
        optimizer = pymanopt.optimizers.SteepestDescent(verbosity=0)
        result = optimizer.run(problem)
        return result.point

    def __str__(self):
        return f'Sphere(dim={self.dim})'
 
def r2_to_angle(x):
    angles = np.arctan2(x[:,1], x[:,0])
    angles[angles < 0] = angles[angles < 0] + 2 * np.pi
    return angles