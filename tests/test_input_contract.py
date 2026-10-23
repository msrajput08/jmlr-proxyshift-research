import os,sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','src'))
import pytest
import numpy as np
from proxyshift import certificate,confusion

@pytest.mark.parametrize('alpha,delta',[(0,.1),(.1,1),(1,.1),(-.2,.1)])
def test_invalid_levels_rejected(alpha,delta):
    with pytest.raises(ValueError):
        certificate([.1,.2],[0,1],[.8,.2],confusion(.15),[.2,.8],alpha,delta)

def test_invalid_prior_rejected():
    with pytest.raises(ValueError):
        certificate([.1,.2],[0,1],[.9,.2],confusion(.15),[.2,.8])

def test_invalid_channel_rejected():
    with pytest.raises(ValueError):
        certificate([.1,.2],[0,1],[.8,.2],np.array([[.8,.2],[.4,.8]]),[.2,.8])

def test_invalid_proxy_index_rejected():
    with pytest.raises(ValueError):
        certificate([.1,.2],[0,2],[.8,.2],confusion(.15),[.2,.8])

def test_negative_nonconformity_score_rejected():
    with pytest.raises(ValueError):
        certificate([-.1,.2],[0,1],[.8,.2],confusion(.15),[.2,.8])
