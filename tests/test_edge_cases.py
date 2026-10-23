"""Structural correctness, stress tests, and explicit no-guarantee cases."""
import os,sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','src'))
import numpy as np
import pytest
from math import erf,sqrt
from proxyshift import certificate, confusion, generate, pooled

def gaussian_mixture_cdf(t, target_b):
    if t == float('inf'): return 1.
    return (1-target_b)*erf(t/sqrt(2))+target_b*erf(t/(3*sqrt(2)))

@pytest.mark.parametrize('noise',[0.,.05,.15,.25])
@pytest.mark.parametrize('target_b',[.2,.5,.8])
def test_gaussian_population_audit(noise,target_b):
    channel=confusion(noise)
    for seed in [71,171,271,371]:
        rng=np.random.default_rng(seed)
        scores,proxy,_=generate(8000,.2,channel,rng)
        q,pen=certificate(scores,proxy,np.array([.8,.2]),channel,np.array([1-target_b,target_b]))
        assert pen>=0
        assert gaussian_mixture_cdf(q,target_b)>=.90-1e-12, (noise,target_b,seed,q)

def test_completely_uninformative_proxy_channel_is_singular():
    channel=confusion(.5)
    rng=np.random.default_rng(2)
    s,z,_=generate(1000,.2,channel,rng)
    with pytest.raises(np.linalg.LinAlgError):
        certificate(s,z,np.array([.8,.2]),channel,np.array([.2,.8]))

def test_extreme_proxy_noise_can_be_vacuous():
    rng=np.random.default_rng(13)
    channel=confusion(.4)
    s,z,_=generate(16000,.2,channel,rng)
    q,pen=certificate(s,z,np.array([.8,.2]),channel,np.array([.2,.8]))
    assert q==float('inf') and pen>.1

def test_no_drift_pooled_is_finite():
    rng=np.random.default_rng(11)
    s,_,_=generate(5000,.2,confusion(.15),rng)
    assert np.isfinite(pooled(s))
