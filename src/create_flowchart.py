from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
p=Path(__file__).resolve().parent.parent/'figures'
p.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(9.1,2.7));ax.set_xlim(0,10);ax.set_ylim(0,3);ax.axis('off')
entries=[(.1,1.08,1.75,.95,'Source scores\nand noisy proxy'),(2.25,1.08,1.75,.95,'Stratified\nempirical CDFs'),(4.40,1.08,1.75,.95,'Known channel\n+ prior inversion'),(6.55,1.08,1.75,.95,'DKW-adjusted\ntarget lower CDF'),(8.7,1.08,1.18,.95,'Certified\nthreshold')]
for x,y,w,h,t in entries:
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07',linewidth=1.05,edgecolor='#334155',facecolor='#E2E8F0'))
 ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8)
for x in [1.94,4.09,6.24,8.39]:
 ax.add_patch(FancyArrowPatch((x,1.56),(x+.28,1.56),arrowstyle='-|>',mutation_scale=13,lw=1.1))
ax.text(5,2.55,'ProxyShift: calibration workflow and certificate',ha='center',fontsize=12,fontweight='medium')
ax.text(5,.55,'Assumptions: independent calibration observations; score invariance within latent groups; known invertible proxy channel',ha='center',fontsize=8)
fig.savefig(p/'method_flowchart.pdf',bbox_inches='tight');fig.savefig(p/'method_flowchart.png',dpi=250,bbox_inches='tight');plt.close(fig)
