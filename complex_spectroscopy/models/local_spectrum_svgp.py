from typing import Tuple

import numpy as np
import tensorflow as tf
import gpflow

from gpflow.base import Parameter
from ..inducing_variables import LaplacianDirichletFeatures
#  special conditional registry for local spectrum needed because dispatch doesn't make it easy to
# distinguish svgp from a local spectrum svgp. Separating them with different registries feels cleanest
from ..conditionals import local_spectrum_conditional as conditional
# however, the kl (and inducing space) is shared with a regular svgp, so we can use the normal kl registry
from gpflow import kullback_leiblers as kl
from ..kullback_leiblers import *  # trigger registration of LDF features case to kl registry
from gpflow.config import default_float
from gpflow.utilities import positive, triangular
from gpflow.models.model import GPModel, InputData, MeanAndVariance, RegressionData
from gpflow.models.training_mixins import ExternalDataTrainingLossMixin
from gpflow.models.util import inducingpoint_wrapper


class Local_Spectrum_SVGP(GPModel, ExternalDataTrainingLossMixin):
    """
    This is the Sparse Variational GP (SVGP) with concepts from Tobar's Paper Bayesian Non-parametric Spectral
    Estimation. The key references are
    ::
      @inproceedings{hensman2014scalable,
        title={Scalable Variational Gaussian Process Classification},
        author={Hensman, James and Matthews, Alexander G. de G. and Ghahramani, Zoubin},
        booktitle={Proceedings of AISTATS},
        year={2015}
      }

      @journal{tobar2018,
        title={Bayesian Non-parametric Spectral Estimation},
        author={Tobar, Filipe},
        booktitle={NeurlIPS},
        year={2018}
      }

    """

    def __init__(
        self,
        kernel,
        inducing_variable,
        noise_scale,
        *,
        alpha=None,
        mean_function=None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        q_mu=None,
        q_sqrt=None,
        whiten: bool = True,
        num_data=None,
    ):
        """
        - kernel, inducing_variables, mean_function are appropriate
          GPflow objects
        - noise_scale is the estimated std_dev of observational noise (for a gaussian likelihood)
        - num_latent_gps is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # init the super class, accept args
        likelihood = gpflow.likelihoods.Gaussian(variance=noise_scale ** 2)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.num_data = int(num_data)
        self.q_diag = q_diag
        self.whiten = whiten
        # we require inducing variable of type LaplacianDirichletFeature
        self.inducing_variable = inducing_variable
        assert type(self.inducing_variable) is LaplacianDirichletFeatures
        if alpha is None:
            self.alpha = tf.reshape(tf.constant(0.1, gpflow.default_float()),(-1))
        else:
            self.alpha = alpha
        # init variational parameters
        num_inducing = self.inducing_variable.num_inducing
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.
        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.
        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros((num_inducing, self.num_latent_gps)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]

        if q_sqrt is None:
            if self.q_diag:
                ones = np.ones((num_inducing, self.num_latent_gps), dtype=default_float())
                self.q_sqrt = Parameter(ones, transform=positive())  # [M, P]
            else:
                q_sqrt = [
                    np.eye(num_inducing, dtype=default_float()) for _ in range(self.num_latent_gps)
                ]
                q_sqrt = np.array(q_sqrt)
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [P, M, M]
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent_gps = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=positive())  # [M, L|P]
            else:
                assert q_sqrt.ndim == 3
                self.num_latent_gps = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [L|P, M, M]

    def prior_kl(self) -> tf.Tensor:
        return kl.prior_kl(
            self.inducing_variable, self.kernel, self.q_mu, self.q_sqrt, whiten=self.whiten
        )

    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        return self.elbo(data)

    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        f_mean_real, f_mean_imag = tf.split(f_mean,2,axis=-2)
        f_var_real, f_var_imag = tf.split(f_var,2,axis=-2)
        var_exp = (self.likelihood.variational_expectations(f_mean_real,
                                                            f_var_real, tf.math.real(Y)) +
                   self.likelihood.variational_expectations(f_mean_imag,
                                                            f_var_imag, tf.math.imag(Y)))
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        objective = tf.reduce_sum(var_exp) * scale - kl
        return objective

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        mu, var = conditional(
            Xnew,
            self.inducing_variable,
            self.kernel,
            q_mu,
            alpha=self.alpha,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        # tf.debugging.assert_positive(var)  # We really should make the tests pass with this here
        return mu + self.mean_function(tf.concat([Xnew, Xnew],-2)), var