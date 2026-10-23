import os, sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','src'))
import numpy as np
from proxyshift import certificate, confusion, generate, pooled, latent_oracle

def test_identity():
    rng=np.random.default_rng(43)
    s,z,g=generate(5000,.2,confusion(0),rng)
    q,e=certificate(s,z,np.array([.8,.2]),confusion(0),np.array([.8,.2]), certified=False)
    assert abs(q-pooled(s))<.09

def test_validity_known_generator():
    rng=np.random.default_rng(9)
    m=confusion(.15)
    s,z,g=generate(18000,.2,m,rng)
    q,pen=certificate(s,z,np.array([.8,.2]),m,np.array([.2,.8]))
    assert np.isfinite(q) and pen > 0
    t,_,_=generate(100000,.8,m,rng)
    assert (t<=q).mean()>.9

def test_proxy_noise_not_group_labels():
    rng=np.random.default_rng(12)
    m=confusion(.2)
    _,z,g=generate(10000,.8,m,rng)
    assert 0.16 < (z!=g).mean() < .24
