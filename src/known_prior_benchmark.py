"""Known-prior finite-support population experiment on public ML datasets.

Scores are sampled with replacement from a fixed, held-out empirical population
pool. Results are conditional on that pool, not an external deployment study.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_diabetes, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from proxyshift import certificate, pooled, latent_oracle, confusion
ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(20261023)
ALPHA, DELTA, NC, NT, REPS = .10, .10, 16000, 10000, 30
rows=[]
for name,loader in [('digits',load_digits),('diabetes',load_diabetes),('wine',load_wine),('breast_cancer',load_breast_cancer)]:
  dt=loader(); X=np.asarray(dt.data); Y=np.asarray(dt.target)
  idx=np.arange(len(Y)); train,pool=train_test_split(idx,train_size=.45,random_state=9021)
  if name=='diabetes':
    group=(X[:,0] > np.median(X[train,0])).astype(int)
    model=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
  elif name=='digits':
    group=(Y>=5).astype(int)
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=600))
  elif name=='wine':
    group=(Y==2).astype(int)
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=600))
  else:
    group=Y.astype(int)
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=600))
  model.fit(X[train],Y[train])
  if name=='diabetes': scores=np.abs(Y-model.predict(X))
  else:
    prob=model.predict_proba(X); scores=1.-prob[np.arange(len(Y)),Y]
  pools=[np.asarray(scores[pool][group[pool]==i],float) for i in (0,1)]
  assert min(map(len,pools))>=15
  M=confusion(.15); pi=np.array([.8,.2]); tau=np.array([.2,.8])
  def draw(n, mixture, generator):
    g=(generator.random(n)<mixture[1]).astype(int)
    v=np.empty(n)
    for j in (0,1):
      mask=(g==j); v[mask]=generator.choice(pools[j],size=mask.sum(),replace=True)
    z=(generator.random(n)<M[1,g]).astype(int)
    return v,z,g
  for rep in range(REPS):
    c,z,g=draw(NC,pi,RNG); test,_,trueg=draw(NT,tau,RNG)
    def score_true_cdf(q):
      return sum(tau[j]*(1. if np.isinf(q) else np.mean(pools[j]<=q)) for j in (0,1))
    candidates=[('Pooled',pooled(c,ALPHA)),
      ('Uncorrected proxy inversion',certificate(c,z,pi,M,tau,ALPHA,DELTA,False)[0]),
      ('Certified proxy inversion',certificate(c,z,pi,M,tau,ALPHA,DELTA,True)[0]),
      ('Latent-label oracle (uncertified)',latent_oracle(c,g,pi,tau,ALPHA)),
      ('Latent-label oracle (DKW)',certificate(c,g,pi,np.eye(2),tau,ALPHA,DELTA,True)[0])]
    for method,q in candidates:
      rows.append({'dataset':name,'rep':rep,'method':method,'coverage':float(np.mean(test<=q)),
        'true_conditional_coverage':float(score_true_cdf(q)),
        'q':q,'vacuous':int(not np.isfinite(q)),
        'group0_count':len(pools[0]),'group1_count':len(pools[1])})
  print(name, [len(v) for v in pools],flush=True)
r=pd.DataFrame(rows)
(rout:=ROOT/'results').mkdir(exist_ok=True)
r.to_csv(rout/'known_prior_real_pools.csv',index=False)
a=r.groupby(['dataset','method'],observed=True).agg(
    coverage=('coverage','mean'),coverage_sd=('coverage','std'),
    conditional_coverage=('true_conditional_coverage','mean'),
    vacuous_rate=('vacuous','mean'),half_width=('q',lambda x:float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else np.inf)
).reset_index()
a.to_csv(rout/'known_prior_real_summary.csv',index=False)
print(a.to_string(index=False))
