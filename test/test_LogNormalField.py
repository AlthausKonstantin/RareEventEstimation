import rareeventestimation as ree
import numpy as np


def test_LogNormalField():
    # given
    m = 0.1
    s = 0.4
    corr_len = 0.3
    num_terms = 20
    field = ree.LogNormalField(m, s, corr_len, num_terms, np.random.default_rng(1))
    x = np.array([0.2, 0.5])
    # when
    num_samples = 5000
    samples = np.zeros((len(x), num_samples))
    for i in range(num_samples):
        samples[..., i] = field(x)
    # then
    cross_cov = np.exp(-np.abs(np.diff(x)) / corr_len)
    true_cov = np.ones((2, 2))
    true_cov[0, 1] = cross_cov
    true_cov[1, 0] = cross_cov
    true_cov *= s
    assert np.linalg.norm(np.average(np.log(samples), axis=1) - m) < 0.01
    assert (
        np.linalg.norm(np.cov(np.log(samples), rowvar=True) - true_cov, ord=np.inf)
        < 0.05
    ), f"Error in Cov too large, {np.cov(np.log(samples), rowvar=True)} vs {true_cov}"
