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

from gpflow import covariances as cov

from gpflow.utilities.multipledispatch import Dispatcher
# -
__all__ = ['TimeDomainKuf', 'FreqDomainKuf', 'KufContext']

TimeDomainKuf = Dispatcher('TimeDomainKuf')
FreqDomainKuf = Dispatcher('TimeDomainKuf')


class KufContext(object):
    def __init__(self, namespace):
        self._old_Kuf = cov.Kuf.funcs.copy()
        self.namespace = namespace

    def __enter__(self):
        cov.Kuf.funcs = {**self._old_Kuf, **self.namespace.funcs}

    def __exit__(self, *args):
        cov.Kuf.funcs = {**self._old_Kuf}



