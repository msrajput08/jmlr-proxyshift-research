"""Reproduce the Gaussian proxy-shift experiments and numerical summaries.

Run from project root: python src/experiment.py
"""
import sys, json, time, pathlib
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from proxyshift import certificate, confusion, generate, pooled, latent_oracle

ROOT=pathlib.Path(__file__).resolve().parent.parent
(ROOT/'results').mkdir(exist_ok=True)
(ROOT/'figures').mkdir(exist_ok=True)
SEED=20261022
N=16000
N_TEST=10000
REPS=35
ALPHA=.1
DELTA=.1
p_source=.2
shifts=[.2,.5,.8]
noise_levels=[.05,.15,.25,.40]
rows=[]
t0=time.time()
for ei,noise in enumerate(noise_levels):
  m=confusion(noise)
  for r in range(REPS):
    rng=np.random.default_rng(SEED+ei*10000+r)
    scores,proxy,g=generate(N,p_source,m,rng)
    q_pool=pooled(scores,ALPHA)
    for target in shifts:
      prior=np.array([1-target,target])
      q_cert,penalty=certificate(scores,proxy,np.array([.8,.2]),m,prior,ALPHA,DELTA,True)
      q_plug,_=certificate(scores,proxy,np.array([.8,.2]),m,prior,ALPHA,DELTA,False)
      q_oracle=latent_oracle(scores,g,np.array([.8,.2]),prior,ALPHA)
      test_scores,test_proxy,test_g=generate(N_TEST,target,m,rng)
      for method,q in [('Pooled',q_pool),('Proxy deconvolution (uncorrected)',q_plug),('Oracle latent-label mixture',q_oracle),('Certified proxy deconvolution',q_cert)]:
        rows.append({'rep':r,'noise':noise,'target_b':target,'method':method,
          'coverage':float(np.mean(test_scores<=q)),
          'group_a_coverage':float(np.mean(test_scores[test_g==0]<=q)) if (test_g==0).any() else np.nan,
          'group_b_coverage':float(np.mean(test_scores[test_g==1]<=q)) if (test_g==1).any() else np.nan,
          'q':q,'half_width':q,'vacuous':int(not np.isfinite(q)),
          'penalty':penalty})
  print('noise',noise,'completed',int(time.time()-t0),'seconds',flush=True)
raw=pd.DataFrame(rows)
raw.to_csv(ROOT/'results'/'monte_carlo.csv',index=False)
agg=(raw.groupby(['noise','target_b','method'],observed=True)
      .agg(coverage=('coverage','mean'),coverage_sd=('coverage','std'),
      half_width=('half_width','mean'),vacuous_rate=('vacuous','mean'),
      penalty=('penalty','mean')) .reset_index())
agg.to_csv(ROOT/'results'/'summary.csv',index=False)
main=agg[(agg.noise==.15)&(agg.target_b==.8)].copy()
main.to_csv(ROOT/'results'/'main_table.csv',index=False)

fig,ax=plt.subplots(figsize=(7.1,4.3))
for name,sub in agg[agg.noise==.15].groupby('method',sort=False):
 sub=sub.sort_values('target_b')
 ax.plot(sub.target_b,sub.coverage,'o-',label=name)
ax.axhline(.9,color='black',linestyle='--',linewidth=1,label='Nominal 90%')
ax.set(xlabel='Target high-variance group fraction',ylabel='Observed marginal coverage',ylim=(.48,1.04),title='Effect of population-mixture shift')
ax.legend(fontsize=7,loc='lower left');fig.tight_layout()
fig.savefig(ROOT/'figures'/'coverage_shift.pdf',bbox_inches='tight');fig.savefig(ROOT/'figures'/'coverage_shift.png',dpi=230,bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(7.1,4.3))
for name,sub in agg[agg.target_b==.8].groupby('method',sort=False):
 sub=sub.sort_values('noise')
 ax.plot(sub.noise,sub.half_width,'o-',label=name)
ax.set(xlabel='Proxy-label flip probability',ylabel='Interval half-width (in score units)',title='Cost of noisy proxy information')
ax.legend(fontsize=7,loc='upper left');fig.tight_layout()
fig.savefig(ROOT/'figures'/'width_noise.pdf',bbox_inches='tight');fig.savefig(ROOT/'figures'/'width_noise.png',dpi=230,bbox_inches='tight');plt.close(fig)

pen=agg[(agg.target_b==.8)&(agg.method=='Certified proxy deconvolution')].sort_values('noise')
fig,ax=plt.subplots(figsize=(6.8,3.5))
ax.plot(pen.noise,pen.penalty,'s-',label='Finite-sample CDF error correction')
ax.set(xlabel='Proxy error probability',ylabel='Upper bound on target CDF error',title='Proxy inversion penalty grows with label noise')
fig.tight_layout();fig.savefig(ROOT/'figures'/'penalty_noise.pdf',bbox_inches='tight');fig.savefig(ROOT/'figures'/'penalty_noise.png',dpi=230,bbox_inches='tight');plt.close(fig)
manifest={'seed':SEED,'n_calibration':N,'n_test_per_rep':N_TEST,'repetitions':REPS,'alpha':ALPHA,'delta':DELTA,'source_group_b':p_source,
 'target_groups':shifts,'proxy_errors':noise_levels,'seconds':round(time.time()-t0,2),'real_data_used':False,
 'note':'Simulated conditional residual laws; no real-world external validation.'}
(ROOT/'results'/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(main[['method','coverage','coverage_sd','half_width','vacuous_rate','penalty']].to_string(index=False))
print('Total',round(time.time()-t0,2))
