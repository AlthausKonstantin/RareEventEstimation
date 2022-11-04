import rareeventestimation as ree
import numpy as np


def test_LogNormalField():
    # given
    m = 0.4
    s = 0.2
    l = 0.3
    num_terms = 100
    field = ree.LogNormalField(
        m, s, l, num_terms, np.random.default_rng(123))
    x = np.array([0.4, 0.6])
    # when
    num_samples = 10000
    samples = np.zeros((len(x), num_samples))
    for i in range(num_samples):
        samples[..., i] = field(x)
    # then
    cross_cov = np.exp(-np.diff(x)/l)
    true_cov = np.ones((2, 2))
    true_cov[0, 1] = cross_cov
    true_cov[1, 0] = cross_cov
    true_cov *= s**2
    assert np.linalg.norm(np.average(np.log(samples), axis=1)-m) < 0.005
    assert np.linalg.norm(
        np.cov(np.log(samples), rowvar=True)-true_cov) < 0.005


test_LogNormalField()
