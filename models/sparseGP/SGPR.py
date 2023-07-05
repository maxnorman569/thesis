# numerical imports
import torch

# gp imports
import gpytorch as gp
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal


class SGPRModel(gp.models.ExactGP):
    """ Sparse Gaussian Process Regression Model """

    def __init__(self, 
                 train_x : torch.Tensor, 
                 train_y : torch.Tensor, 
                 likelihood : gp.likelihoods,
                 inducing_points : torch.Tensor,
                 ) -> 'SGPRModel':
        """
        Class for Sparse Gaussian Process Regression Model.

        Arguments:
            train_x (torch.Tensor)              : training data input
            train_y (torch.Tensor)              : training data output
            likelihood (gpytorch.likelihoods)   : likelihood function
            inducing_points (torch.Tensor)      : inducing points for sparse GP

        Returns:
            None
        """
        super(SGPRModel, self).__init__(train_x, train_y, likelihood)
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = likelihood
        self.inducing_points = inducing_points
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points = inducing_points, likelihood = self.likelihood)


    def initialise(self, 
                   x_std: torch.Tensor, 
                   y_std: torch.Tensor, 
                   lmbda: float, 
                   kappa: float) -> None:
        """
        Initialises the model by setting the hyperparameters.

        Arguments:
            x_std (torch.Tensor)    : standard deviation of the input data
            y_std (torch.Tensor)    : standard deviation of the output data
            lmbda (float)           : lengthscale hyperparameter
            kappa (float)           : noise hyperparameter
        
        Returns:
            None
        """
        self.covar_module.outputscale = y_std ** 2
        self.likelihood.noise = self.covar_module.outputscale / (kappa ** 2)
        self.covar_module.base_kernel.lengthscale = x_std / lmbda


    def forward(self, 
                x : torch.Tensor) -> MultivariateNormal:
        """ 
        Takes input data and returns the predicted mean and covariance of the Gaussian process at those input points. 

        Arguments:
            x (torch.Tensor)    : input data

        Returns:
            MultivariateNormal  : multivariate normal conditioned on training data at input points
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return MultivariateNormal(mean_x, covar_x)
    

    def parameter_learn(self, 
              training_iterations : int, 
              learning_rate : float = 0.1) -> None:
        """ 
        Performs parameter learning given a number of training iterations by minimizing the negative log marginal likelihood.

        Arguments:
            training_iterations (int)   : number of training iterations
            lr (float)                  : learning rate

        Returns:
            None
        """
        # set model and likelihood into training mode
        self.train()
        self.likelihood.train()

        # define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)

        # define the mll
        mll = gp.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        # define the training loop
        print('='*((5*15) + 4))
        print(f"{'Iteration':^15}|{'Loss':^15}|{'Noise':^15}|{'Lengthscale':^15}|{'Outputscale':^15}")
        print('='*((5*15) + 4))
        for i in range(training_iterations):
            # zero gradients from previous iteration
            optimizer.zero_grad()

            # output from model
            output = self(self.train_x)

            # calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()

            # print progress
            if (i+1) % 10 == 0:
                print(f"{i+1:^15}|{loss.item():^15.3f}|{self.likelihood.noise.item():^15.3f}|{self.base_covar_module.base_kernel.lengthscale.item():^15.3f}|{self.base_covar_module.outputscale.item():^15.3f}")

            # take step
            optimizer.step()


    def predict(self, 
                text_x : torch.Tensor) -> MultivariateNormal:
        """
        Returns predictive distribution of the model over given test points.

        Arguments:
            test_x (torch.Tensor)   : test points
        
        Returns:
            MultivariateNormal      : predictive distribution
        """
        # set model and likelihood into eval mode
        self.eval()
        self.likelihood.eval()

        # make predictions
        with torch.no_grad(), gp.settings.fast_pred_var():
            predictions = self.likelihood(self(text_x))

        return predictions