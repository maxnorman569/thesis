# numerical import
import torch

# gp imports
import gpytorch as gp
from gpytorch.distributions import MultivariateNormal

class GPModel(gp.models.ExactGP):

    """ Exact Gaussian Process Regression Model """

    def __init__(self, 
                 train_x, 
                 train_y, 
                 likelihood) -> 'GPModel':
        """
        Class for Exact Gaussian Process Regression Model.

        Arguments:
            train_x (torch.Tensor)              : training data input
            train_y (torch.Tensor)              : training data output
            likelihood (gpytorch.likelihoods)   : likelihood function

        Returns:
            None
        """
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = likelihood
        self.mean_module = gp.means.ConstantMean()
        self.covar_module = gp.kernels.ScaleKernel(gp.kernels.RBFKernel())

    def non_informative_initialise(self, 
                   lmbda: float, 
                   kappa: float) -> None:
        """
        Initialises the model by setting the hyperparameters.

        Arguments:
            x_std (torch.Tensor)    : standard deviation of the input data
            y_std (torch.Tensor)    : standard deviation of the output data
            lmbda (float)           : lengthscale hyperparameter
            kappa (float)           : noise hyperparameter (we expect kappa to be in the range of [2, 100])
        
        Returns:
            None
        """
        self.covar_module.outputscale = self.train_y.var()
        self.likelihood.noise = self.covar_module.outputscale / (kappa ** 2)
        self.covar_module.base_kernel.lengthscale = (self.train_x.std() / lmbda)

    
    def informative_initialise(self,
                                prior_amplitude : float,
                                lmbda: float,) -> None:
        """
        Initialises the model hyperparameters based on prior knolwedge of the plausible function amplitudes.

        Arguments:
            prior_amplitude (float) : amplitude of the function prior
            lmbda (float)           : lengthscale hyperparameter (lambda ~ 1 -> favours linear function, lambda ~ 10 -> favours highly non-linear functions)

        Returns:
            None
        """
        self.covar_module.outputscale = (torch.tensor(prior_amplitude) / 2) ** 2
        self.likelihood.noise = self.train_y.var() - self.covar_module.outputscale
        self.covar_module.base_kernel.lengthscale = (self.train_x.std() / lmbda)
    

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

        return gp.distributions.MultivariateNormal(mean_x, covar_x)
    

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
            if (i+1) % (training_iterations / 10) == 0:
                print(f"{i+1:^15}|{loss.item():^15.3f}|{self.likelihood.noise.item():^15.3f}|{self.covar_module.base_kernel.lengthscale.item():^15.3f}|{self.covar_module.outputscale.item():^15.3f}")

            # take step
            optimizer.step()

        return None


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