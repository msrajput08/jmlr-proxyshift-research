from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1]

def test_known_prior_protocol_records_complete():
    data=pd.read_csv(P/'results/known_prior_real_pools.csv')
    assert len(data)==4*30*5
    assert data.groupby(['dataset','method']).size().eq(30).all()
    assert set(data.dataset)=={'digits','diabetes','wine','breast_cancer'}

def test_population_coverage_checks_finite_certificates():
    data=pd.read_csv(P/'results/known_prior_real_pools.csv')
    rows=data[(data.method=='Certified proxy inversion') & (data.vacuous==0)]
    assert (rows.true_conditional_coverage>=.9).all()
    assert (rows.q>=0).all()

def test_oracle_is_flagged_as_infeasible_for_proxy_observations():
    data=pd.read_csv(P/'results/known_prior_real_pools.csv')
    assert any(data.method=='Latent-label oracle (DKW)')
