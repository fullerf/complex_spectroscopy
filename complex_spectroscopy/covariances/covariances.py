# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gpflow
from gpflow.inducing_variables import InducingVariables
from gpflow.utilities import to_default_float
from gpflow.base import TensorLike, Parameter
from gpflow import covariances as cov
from gpflow import kullback_leiblers as kl

from gpflow.conditionals import conditional
from gpflow.config import default_float
from gpflow.utilities import positive, triangular

Diag = tf.linalg.LinearOperatorDiag
# -


