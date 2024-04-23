import numpy as np
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.product_manifold import ProductManifold, NFoldManifold
from geomstats.geometry.hypersphere import Hypersphere
from .metric_space import MetricSpace
from .sphere import Sphere

class Torus(MetricSpace):
    def __init__(self, dim):
        self.dim=dim
        # self.manifold=ProductManifold(manifolds=[Hypersphere(dim=1) for _ in range(self.dim)],
        #                        default_point_type='vector')
        self.manifold=NFoldManifold(base_manifold=Hypersphere(dim=1),
                                    default_point_type='matrix',
                                    default_coords_type='intrinsic',
                                    n_copies=self.dim)

    def _d(self, x, y):
        return self.manifold.metric.dist(x,y)
    
    def _frechet_mean(self, y, w):
        mean = FrechetMean(metric=self.manifold.metric)
        mean.fit(y, weights=w)
        return mean.estimate_
    
    def __str__(self) -> str:
        return f'Torus(dim={self.dim})'
 


