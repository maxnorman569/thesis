# numerical imports
import torch

# probability imports
import torch.distributions as dist

# typing imports
from typing import Tuple


class InducingPoints:

    def __init__(self, 
                 num_inducing_points : int = 100,
                 latitude_range : Tuple[float, float] = (-90., 90.),
                 longitude_range : Tuple[float, float] = (-180., 180.),
                 method : str = 'uniform') -> 'InducingPoints':
        """
        Class for generating inducing points for sparse GP uniformly in a range.

        Arguments:
            num_inducing_points (int)               : number of inducing points
            latitude_range (Tuple[float, float])    : latitude range
            longitude_range (Tuple[float, float])   : longitude range
            inducing_points (torch.Tensor)          : inducing points for sparse GP
            method (str)                            : method for generating inducing points (uniform or random)

        Returns:
            none
        """
        self.num_inducing_points = num_inducing_points
        self.latitude_range = latitude_range
        self.longitude_range = longitude_range
        self.inducing_points = self._generate(num_inducing_points = num_inducing_points, method = method)
    

    def _generate(self, 
                 num_inducing_points : int, 
                 method : str) -> torch.Tensor:
        """
        Generates inducing points for sparse GP using provided method

        Arguments:
            num_inducing_points (int)   : number of inducing points
            method (str)                : method for generating inducing points

        Returns:
            inducing_points (torch.Tensor)  : inducing points for sparse GP
        """
        # generate inducing points
        if method == 'uniform':
            inducing_lat = torch.linspace(self.latitude_range[0], self.latitude_range[1], num_inducing_points)
            inducing_lon = torch.linspace(self.longitude_range[0], self.longitude_range[1], num_inducing_points)
            inducing_points = torch.stack([inducing_lon, inducing_lat], dim = 1)
        else:
            inducing_lat = dist.Uniform(self.latitude_range[0], self.latitude_range[1]).sample((num_inducing_points,))
            inducing_lon = dist.Uniform(self.longitude_range[0], self.longitude_range[1]).sample((num_inducing_points,))
            inducing_points = torch.stack([inducing_lon, inducing_lat], dim = 1)

        return inducing_points


        
        