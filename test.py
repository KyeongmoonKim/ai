import numpy as np
from hmmlearn import hmm

np.random.seed(42)

lr = hmm.GaussianHMM(n_components=3, covariance_type="diag", init_params="cm", params="cmt")
lr.startprob_ = np.array([1.0, 0.0, 0.0])
lr.transmat_ = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
X, Z = lr.sample(100)
