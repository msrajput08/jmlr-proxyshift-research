"""Exploratory real-data splits: estimated source priors violate theorem assumptions."""
import sys, pathlib
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from proxyshift import certificate, confusion, pooled,latent_oracle
ROOT=pathlib.Path(__file__).resolve().parent.parent
rng0=np.random.default_rng(124)
rows=[]
for dataset in ['digits','diabetes']:
  if dataset=='digits':
    dt=load_digits();x=np.asarray(dt.data);y=np.asarray(dt.target)
    truegroup=(y>=5).astype(int)
  else:
    dt=load_diabetes();x=np.asarray(dt.data);y=np.asarray(dt.target)
    truegroup=None
  for rep in range(25):
    seed=20261022+rep
    ids=np.arange(len(y))
    train,remain=train_test_split(ids,train_size=.45,random_state=seed)
    calib,test=train_test_split(remain,train_size=.55,random_state=seed+100)
    if dataset=='diabetes':
      threshold=float(np.median(x[train,0])); group=(x[:,0]>threshold).astype(int)
    else:
      group=truegroup
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500)) if dataset=='digits' else make_pipeline(StandardScaler(),Ridge(alpha=10))
    model.fit(x[train],y[train])
    if dataset=='digits':
      probs=model.predict_proba(x);scores=1-probs[np.arange(len(y)),y]
    else:
      scores=np.abs(y-model.predict(x))
    cscore=scores[calib];cgroup=group[calib]
    pi=np.bincount(cgroup,minlength=2)/len(calib)
    if np.min(pi)<.02: continue
    chan=confusion(.15)
    z=(rng0.random(len(calib))<chan[1,cgroup]).astype(int)
    for target in [.2,.5,.8]:
      pri=np.array([1-target,target])
      vals={}
      vals['Pooled']=pooled(cscore)
      vals['Proxy deconvolution (uncorrected)']=certificate(cscore,z,pi,chan,pri,certified=False)[0]
      vals['Certified proxy deconvolution']=certificate(cscore,z,pi,chan,pri,certified=True)[0]
      vals['Oracle latent-label mixture']=latent_oracle(cscore,cgroup,pi,pri)
      for label,q in vals.items():
        testgroup=group[test];testscore=scores[test]
        conditional=[(testscore[testgroup==g]<=q).mean() if (testgroup==g).any() else np.nan for g in (0,1)]
        cov=pri[0]*conditional[0]+pri[1]*conditional[1]
        rows.append({'dataset':dataset,'rep':rep,'target_b':target,'method':label,'coverage':cov,'half_width':q,
          'vacuous':int(not np.isfinite(q)),'calib_size':len(calib)})
raw=pd.DataFrame(rows)
raw.to_csv(ROOT/'results'/'real_data.csv',index=False)
agg=raw.groupby(['dataset','target_b','method'],observed=True).agg(coverage=('coverage','mean'),half_width=('half_width','mean'),vacuous_rate=('vacuous','mean'),calib_size=('calib_size','mean')).reset_index()
agg.to_csv(ROOT/'results'/'real_data_summary.csv',index=False)
print(agg[(agg.target_b==.8)].to_string(index=False))
