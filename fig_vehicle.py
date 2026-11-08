import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import bluerov2_heavy_dataset as b
B=b.allocation_matrix(); s=np.sqrt(.5)
pos=np.array([[.156,.111],[.156,-.111],[-.156,.111],[-.156,-.111],[.120,.218],[.120,-.218],[-.120,.218],[-.120,-.218]])
dirs=[(s,-s),(s,s),(s,s),(s,-s)]
fig,ax=plt.subplots(1,2,figsize=(10.5,4.6),gridspec_kw={'width_ratios':[1.1,1]})
a=ax[0]
# plot with y horizontal (starboard right), x up (forward)
a.add_patch(Rectangle((-0.169,-0.2285),0.338,0.457,fill=False,ls='--',ec='gray',lw=1))
a.text(0,-0.265,'approx. frame envelope 457 x 338 mm',ha='center',fontsize=8,color='gray')
for i in range(4):
    x,y=pos[i]; dx,dy=dirs[i]
    a.annotate('',xy=(y+dy*0.07,x+dx*0.07),xytext=(y,x),arrowprops=dict(arrowstyle='-|>',color='#1f5fa8',lw=2))
    a.plot(y,x,'o',color='#1f5fa8',ms=8); a.text(y+0.03*np.sign(y),x+(0.022 if x>0 else -0.045),f'T{i+1}',fontsize=10,color='#1f5fa8',ha='center')
for i in range(4,8):
    x,y=pos[i]; a.add_patch(Circle((y,x),0.03,fc='#f2c14e',ec='k',lw=1)); a.plot(y,x,'k.',ms=5)
    a.text(y,x+0.045,f'T{i+1}',fontsize=10,ha='center')
a.annotate('',xy=(0,0.3),xytext=(0,0),arrowprops=dict(arrowstyle='-|>',color='k')); a.text(0.01,0.3,'x (forward)',fontsize=9)
a.annotate('',xy=(0.3,0),xytext=(0,0),arrowprops=dict(arrowstyle='-|>',color='k')); a.text(0.3,0.012,'y (starboard)',fontsize=9)
a.plot(0,0,'k+',ms=10)
a.set_xlim(-0.34,0.45); a.set_ylim(-0.30,0.34); a.set_aspect('equal'); a.set_xlabel('y (m)'); a.set_ylabel('x (m)')
a.set_title('Top view: horizontal (blue, vectored 45°) and vertical (yellow, thrust along z) thrusters',fontsize=8.5)
a=ax[1]
im=a.imshow(B,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
a.set_xticks(range(8)); a.set_xticklabels([f'T{i+1}' for i in range(8)])
a.set_yticks(range(6)); a.set_yticklabels(['X','Y','Z','K (roll)','M (pitch)','N (yaw)'])
for i in range(6):
    for j in range(8):
        v=B[i,j]
        if abs(v)>1e-6: a.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8)
a.set_title('Allocation matrix B (force N/N, moment N m/N)',fontsize=9)
plt.tight_layout(); plt.savefig('paper/fig_vehicle.png',dpi=200); plt.savefig('paper/fig_vehicle.pdf')
