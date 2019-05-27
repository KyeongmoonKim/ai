import numpy as np
from hmmlearn import hmm

np.random.seed(42)

lr = hmm.GaussianHMM(n_components=3)
X, Z = lr.sample(100)
