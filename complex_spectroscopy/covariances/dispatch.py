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
from gpflow.utilities.multipledispatch import Dispatcher
# -
__all__ = ['TimeDomainKuf', 'FreqDomainKuf']

TimeDomainKuf = Dispatcher('TimeDomainKuf')
FreqDomainKuf = Dispatcher('TimeDomainKuf')


