import pandas as pd, pathlib
P=pathlib.Path(__file__).resolve().parent.parent
x=pd.read_csv(P/'results/main_table.csv'); order=['Pooled','Proxy deconvolution (uncorrected)','Oracle latent-label mixture','Certified proxy deconvolution']
x['method']=pd.Categorical(x['method'],categories=order,ordered=True);x=x.sort_values('method')
tex=['\\begin{tabular}{lrrrr}','\\hline','Calibration rule & Coverage & SD & Half-width & Vacuous \\\\','\\hline']
for _,r in x.iterrows():
 name={'Pooled':'Pooled split conformal', 'Proxy deconvolution (uncorrected)':'Uncorrected proxy inversion', 'Oracle latent-label mixture':'Latent-group oracle', 'Certified proxy deconvolution':'ProxyShift certificate'}[str(r['method'])]
 tex.append(f"{name} & {r['coverage']:.3f} & {r['coverage_sd']:.3f} & {r['half_width']:.3f} & {100*r['vacuous_rate']:.1f}\\% \\\\")
tex.append('\\hline\\end{tabular}')
(P/'paper'/'main_table.tex').write_text('\n'.join(tex)+'\n')
raw=pd.read_csv(P/'results/real_data_summary.csv')
real=['\\begin{tabular}{llrr}','\\hline','Dataset & Calibration rule & Coverage & Vacuous \\\\','\\hline']
for d in ['digits','diabetes']:
 for m in ['Pooled','Proxy deconvolution (uncorrected)','Certified proxy deconvolution']:
  r=raw[(raw.dataset==d)&(raw.target_b==.8)&(raw.method==m)].iloc[0]
  short={'Pooled':'Pooled','Proxy deconvolution (uncorrected)':'Uncorrected','Certified proxy deconvolution':'Certified'}[m]
  real.append(f"{d.title()} & {short} & {r['coverage']:.3f} & {100*r['vacuous_rate']:.0f}\\% \\\\")
real.append('\\hline\\end{tabular}')
(P/'paper'/'real_table.tex').write_text('\n'.join(real)+'\n')
summary_path=P/'results'/'known_prior_real_summary.csv'
if summary_path.exists():
 s=pd.read_csv(summary_path)
 labels={'Pooled':'Pooled','Uncorrected proxy inversion':'Proxy plug-in','Certified proxy inversion':'Proxy certificate','Latent-label oracle (DKW)':'Latent oracle + DKW'}
 lines=['\\begin{tabular}{llrr}','\\hline','Data set & Method & Coverage & Mean half-width \\\\','\\hline']
 for ds in ['digits','diabetes','wine','breast_cancer']:
  for i,m in enumerate(labels):
   r=s[(s.dataset==ds)&(s.method==m)].iloc[0]
   display={'digits':'Digits','diabetes':'Diabetes','wine':'Wine','breast_cancer':'Breast Cancer'}[ds] if i==0 else ''
   lines.append(f"{display} & {labels[m]} & {r.coverage:.3f} & {r.half_width:.3f} \\\\" )
  lines.append('\\hline')
 lines.append('\\end{tabular}')
 (P/'paper'/'known_prior_table.tex').write_text('\n'.join(lines)+'\n')
