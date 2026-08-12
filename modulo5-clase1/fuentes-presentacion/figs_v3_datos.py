#!/usr/bin/env python3
"""Figuras APLICADAS de la Clase 1 · v3 — todas sobre la tasa normalizada."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm, matplotlib.colors as mcolors
import numpy as np, pandas as pd, pathlib
for f in ['FiraSans-Regular','FiraSans-Medium','FiraSans-SemiBold','FiraSans-Bold']:
    p=pathlib.Path(f'/tmp/fonts/{f}.ttf')
    if p.exists(): fm.fontManager.addfont(str(p))
RED='#C82B40'; DARK='#6B1525'; BLUE='#2563EB'; AMBER='#E69F00'; GREEN='#1B7F4B'
GRAY='#9CA3AF'; LGRAY='#E5E7EB'; INK='#2D2D2D'; MUTED='#6B7280'
plt.rcParams.update({'font.family':'Fira Sans','font.size':16,'axes.labelsize':16,
 'axes.labelcolor':MUTED,'text.color':INK,'xtick.color':MUTED,'ytick.color':MUTED,
 'xtick.labelsize':14,'ytick.labelsize':14,'axes.edgecolor':LGRAY,'axes.linewidth':1,
 'axes.grid':True,'grid.color':LGRAY,'grid.linewidth':.7,'grid.alpha':.8,
 'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white',
 'figure.dpi':200,'savefig.bbox':'tight','savefig.pad_inches':.15,'legend.frameon':False})
D=pathlib.Path('fig_v3'); D.mkdir(exist_ok=True)
def save(fig,n): fig.savefig(D/f'{n}.png'); plt.close(fig); print('  ✓',n)

CSV='/mnt/user-data/uploads/slb-diplomado/datos/volve_produccion.csv'
df=pd.read_csv(CSV,parse_dates=['fecha'])
f14=df[df.pozo=='15/9-F-14'].sort_values('fecha').set_index('fecha')
f12=df[df.pozo=='15/9-F-12'].sort_values('fecha').set_index('fecha')
def tasa(w):
    x=w[(w.horas>0)&(w.oil>0)].copy(); x['tasa']=x.oil*24/x.horas
    x['wc']=x.agua/(x.agua+x.oil); return x
d14=tasa(f14); m14=d14.tasa.resample('ME').median().dropna()
ORIGEN=pd.Timestamp('2014-01-31')

# ── A1 · el encargo ───────────────────────────────────────────────────────
tr=m14[m14.index<=ORIGEN]; fin=ORIGEN+pd.DateOffset(months=6)
fig,ax=plt.subplots(figsize=(11.0,4.4))
ax.plot(tr.index,tr.values,lw=2.6,color=INK)
ax.axvline(ORIGEN,color=RED,lw=1.8,ls=(0,(4,3)))
ax.axvspan(ORIGEN,fin,color=RED,alpha=.08,lw=0)
ax.text(ORIGEN-pd.Timedelta(days=40),4300,'hoy\n31-ene-2014',ha='right',fontsize=15,
        color=RED,weight='semibold',linespacing=1.35)
ax.text(ORIGEN+pd.Timedelta(days=92),1900,'?',fontsize=76,color=RED,weight='bold',ha='center')
ax.set_xlim(m14.index[0],fin+pd.Timedelta(days=50))
ax.set_ylabel('tasa mediana mensual  [bbl/d]')
save(fig,'a1_encargo')

# ── A2 · la serie diaria cruda ────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(11.0,4.2))
ax.plot(f14.index,f14.oil,lw=.9,color=INK,alpha=.8)
ax.set_ylabel('barriles producidos por día')
save(fig,'a2_cruda')

# ── A3 · los ceros son cierres ────────────────────────────────────────────
s=f14.oil.asfreq('D'); closed=(s.fillna(-1)==0)
runs,st=[],None
for t,v in closed.items():
    if v and st is None: st=t
    if not v and st is not None: runs.append((st,t)); st=None
if st is not None: runs.append((st,closed.index[-1]))
fig,ax=plt.subplots(figsize=(11.0,4.2))
for a,b in runs: ax.axvspan(a,b,color=RED,alpha=.22,lw=0,zorder=1)
ax.plot(f14.index,f14.oil,lw=1.1,color=INK,alpha=.75,zorder=3)
ax.set_ylabel('barriles por día')
ax.text(.985,.95,f'{int((f14.oil==0).sum())} días con oil = 0\n'
                 f'{int((f14.horas==0).sum())} días con horas = 0',
        transform=ax.transAxes,ha='right',va='top',fontsize=16,color=RED,
        weight='semibold',linespacing=1.6)
save(fig,'a3_ceros')

# ── A4 · disponibilidad ───────────────────────────────────────────────────
grid=np.full((9,366),np.nan)
for ts,v in f14.horas.items():
    if 2008<=ts.year<=2016: grid[ts.year-2008,ts.dayofyear-1]=v
fig,ax=plt.subplots(figsize=(11.5,3.9))
cmap=mcolors.LinearSegmentedColormap.from_list('u',['#FDF0F1','#F0AEB7','#D9707F','#C82B40','#8E1C2C'])
cmap.set_bad('#C9CDD2')
im=ax.imshow(np.ma.masked_invalid(grid),aspect='auto',cmap=cmap,vmin=0,vmax=24,
             extent=[0,366,8.5,-0.5],interpolation='nearest')
ax.set_yticks(range(9)); ax.set_yticklabels(range(2008,2017),fontsize=13)
ax.set_xlabel('día del año'); ax.grid(False)
for sp in ax.spines.values(): sp.set_visible(False)
cb=fig.colorbar(im,ax=ax,pad=.012,fraction=.03); cb.outline.set_visible(False)
cb.set_label('horas / día',color=MUTED,fontsize=13); cb.ax.tick_params(labelsize=12)
save(fig,'a4_disponibilidad')

# ── A5 · escalas aplicadas ────────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(11.5,4.0))
for a,log in zip(axes,[False,True]):
    a.plot(m14.index,m14.values,lw=2.8,color=RED)
    if log: a.set_yscale('log')
    a.text(.03,1.03,'escala normal' if not log else 'escala logarítmica',
           transform=a.transAxes,fontsize=15.5,weight='semibold',color=INK)
    a.tick_params(labelsize=12)
axes[0].set_ylabel('tasa  [bbl/d]')
axes[1].plot([m14.index[6],m14.index[-1]],[m14.iloc[6],m14.iloc[-1]],lw=1.6,
             color=INK,ls=(0,(4,3)),zorder=1)
save(fig,'a5_escalas')

# ── A6 · media móvil: ventanas ────────────────────────────────────────────
w=d14.tasa.asfreq('D').loc['2012-06':'2014-06']
fig,ax=plt.subplots(figsize=(11.0,4.3))
ax.plot(w.index,w.values,lw=.9,color=LGRAY,label='día a día')
for kk,col,lw_ in [(7,AMBER,2.0),(30,BLUE,2.4),(90,RED,2.9)]:
    ax.plot(w.index,w.rolling(kk,min_periods=kk//2).mean(),lw=lw_,color=col,
            label=f'media móvil {kk} días')
ax.legend(fontsize=14,loc='upper right',ncols=2); ax.set_ylabel('tasa  [bbl/d]')
save(fig,'a6_ventanas')

# ── A7 · media vs mediana ─────────────────────────────────────────────────
w2=f14.oil.loc['2013-01':'2014-06']
fig,ax=plt.subplots(figsize=(11.0,4.2))
ax.plot(w2.index,w2.values,lw=.9,color=LGRAY,label='día a día (con cierres)')
ax.plot(w2.index,w2.rolling(15).mean(),lw=2.7,color=RED,label='media móvil 15 d')
ax.plot(w2.index,w2.rolling(15).median(),lw=2.7,color=BLUE,label='mediana móvil 15 d')
ax.legend(fontsize=14,loc='lower left'); ax.set_ylabel('barriles por día')
save(fig,'a7_mediana')

# ── A8 · correlograma ─────────────────────────────────────────────────────
from statsmodels.tsa.stattools import acf
sr=d14.tasa.asfreq('D').interpolate(limit=5).dropna()
a=acf(sr,nlags=180)
fig,ax=plt.subplots(figsize=(11.0,4.2))
ax.vlines(range(len(a)),0,a,color=RED,lw=1.9); ax.axhline(0,color=INK,lw=1)
ci=1.96/np.sqrt(len(sr)); ax.fill_between([0,180],-ci,ci,color=GRAY,alpha=.18,lw=0)
ax.set_xlim(-1,180); ax.set_ylim(-.1,1.05)
ax.set_xlabel('rezago  [días]'); ax.set_ylabel('correlación')
ax.annotate(f'a 6 meses todavía vale {a[180]:.2f}',xy=(178,a[178]),xytext=(96,.9),
            fontsize=15.5,color=RED,weight='semibold',
            arrowprops=dict(arrowstyle='-',color=RED,lw=1.3))
save(fig,'a8_acf')

# ── A9 · el eje: cambio de horario ────────────────────────────────────────
dd=df.dropna(subset=['horas'])
fig,ax=plt.subplots(figsize=(11.0,4.3))
ax.scatter(dd.fecha,dd.horas,s=7,color=GRAY,alpha=.28,linewidths=0,zorder=2)
hi=dd[dd.horas>24]; lo=dd[(dd.horas>22.9)&(dd.horas<23.1)]
ax.scatter(hi.fecha,hi.horas,s=80,color=RED,zorder=4,edgecolor='white',linewidth=1.2)
ax.scatter(lo.fecha,lo.horas,s=80,color=BLUE,zorder=4,edgecolor='white',linewidth=1.2)
ax.axhline(24,color=INK,lw=1.3,ls=(0,(4,3)),zorder=3)
ax.text(pd.Timestamp('2007-10-10'),24.45,'24 h',fontsize=14,color=INK,weight='semibold')
ax.annotate('días de 25 h — último domingo de octubre',xy=(hi.fecha.iloc[0],25),
    xytext=(pd.Timestamp('2009-03-01'),27.5),fontsize=15,color=RED,weight='semibold',
    arrowprops=dict(arrowstyle='-',color=RED,lw=1.4,shrinkA=0,shrinkB=5))
ax.annotate('días de 23 h — último domingo de marzo',xy=(pd.Timestamp('2013-03-31'),23),
    xytext=(pd.Timestamp('2011-11-01'),17.4),fontsize=15,color=BLUE,weight='semibold',
    arrowprops=dict(arrowstyle='-',color=BLUE,lw=1.4,shrinkA=0,shrinkB=5))
ax.set_ylim(14.5,29); ax.set_ylabel('horas de operación')
save(fig,'a9_dst')

# ── A10 · el evento F-12 ──────────────────────────────────────────────────
g=tasa(f12).loc['2014-06':'2015-09']
gm=g[['tasa','wc']].resample('ME').median()
fig,axes=plt.subplots(2,1,figsize=(9.4,5.4),sharex=True,gridspec_kw={'hspace':.15})
axes[0].plot(gm.index,gm.tasa,lw=2.8,color=RED,marker='o',ms=8,mfc='white',mew=2.2)
axes[0].set_ylabel('tasa  [bbl/d]',fontsize=14)
axes[1].plot(gm.index,gm.wc*100,lw=2.8,color=BLUE,marker='o',ms=8,mfc='white',mew=2.2)
axes[1].set_ylabel('corte de agua  [%]',fontsize=14)
for a2 in axes:
    a2.axvspan(pd.Timestamp('2014-12-01'),pd.Timestamp('2014-12-31'),color=DARK,alpha=.13,lw=0)
    a2.tick_params(labelsize=12)
axes[0].text(pd.Timestamp('2014-12-16'),gm.tasa.max()*.45,'cerrado\ntodo el mes',
             ha='center',fontsize=14,color=DARK,weight='semibold',linespacing=1.35)
save(fig,'a10_evento')

# ── A11 · K-means regímenes ───────────────────────────────────────────────
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
m=d14[['tasa','wc']].resample('ME').median().dropna()
X=StandardScaler().fit_transform(m)
m['c']=KMeans(n_clusters=3,n_init=10,random_state=0).fit_predict(X)
orden=m.groupby('c').tasa.mean().sort_values(ascending=False).index
nom={orden[0]:'plateau — mucha producción, poca agua',
     orden[1]:'transición — el agua avanza',
     orden[2]:'cola — poca producción, agua dominante'}
col={orden[0]:GREEN,orden[1]:AMBER,orden[2]:RED}
fig,axes=plt.subplots(1,2,figsize=(11.8,4.3),gridspec_kw={'width_ratios':[1,1.5]})
for c in orden:
    g2=m[m.c==c]; axes[0].scatter(g2.wc*100,g2.tasa,s=52,color=col[c],alpha=.9,linewidths=0)
axes[0].set_xlabel('corte de agua  [%]'); axes[0].set_ylabel('tasa mediana  [bbl/d]')
axes[0].text(.03,1.03,'los meses agrupados',transform=axes[0].transAxes,fontsize=15,
             weight='semibold',color=INK)
axes[1].plot(m.index,m.tasa,lw=1.0,color=LGRAY,zorder=0)
for c in orden:
    g2=m[m.c==c]; axes[1].scatter(g2.index,g2.tasa,s=44,color=col[c],label=nom[c],linewidths=0)
axes[1].legend(fontsize=12.5,loc='upper right'); axes[1].set_ylabel('tasa  [bbl/d]')
axes[1].text(.03,1.03,'los mismos grupos, en el tiempo',transform=axes[1].transAxes,
             fontsize=15,weight='semibold',color=INK)
save(fig,'a11_regimenes')

# ── A12 · DBSCAN anomalías ────────────────────────────────────────────────
dd2=d14[d14.horas>=20].copy(); dd2['dlog']=np.log(dd2.tasa).diff()
feats=dd2[['tasa','wc','dlog']].dropna()
lab=DBSCAN(eps=.6,min_samples=12).fit_predict(StandardScaler().fit_transform(feats))
anom=feats[lab==-1]
fig,ax=plt.subplots(figsize=(11.5,4.2))
ax.plot(feats.index,feats.tasa,lw=1.0,color=GRAY,zorder=1)
ax.scatter(anom.index,anom.tasa,s=52,color=RED,zorder=3,linewidths=0,
           label=f'{len(anom)} días señalados como raros  ({100*len(anom)/len(feats):.1f} %)')
ax.legend(fontsize=14,loc='upper right'); ax.set_ylabel('tasa  [bbl/d]')
save(fig,'a12_anomalias')

# ── A13 · error por horizonte ─────────────────────────────────────────────
maes=[(m14-m14.shift(h)).abs().dropna().mean() for h in range(1,13)]
fig,ax=plt.subplots(figsize=(9.8,4.3))
ax.plot(range(1,13),maes,marker='o',ms=10,mfc='white',mew=2.6,lw=3,color=RED)
ax.set_xlabel('¿cuántos meses adelante pronosticamos?'); ax.set_ylabel('error típico  [bbl/d]')
ax.set_xticks(range(1,13))
ax.annotate(f'{maes[0]:.0f}',xy=(1,maes[0]),xytext=(1.25,maes[0]+60),fontsize=16,weight='bold',color=INK)
ax.annotate(f'{maes[-1]:.0f}',xy=(12,maes[-1]),xytext=(11.1,maes[-1]+45),fontsize=16,weight='bold',color=RED)
save(fig,'a13_horizonte')

# ── A14 · MAE vs RMSE ─────────────────────────────────────────────────────
rng=np.random.default_rng(4); real=m14.loc['2013':'2014'].dropna().values; n=len(real)
pA=real+rng.normal(0,55,n); pB=real+rng.normal(0,18,n); pB[6]+=480
fig,axes=plt.subplots(1,2,figsize=(11.5,4.1),sharey=True)
for ax,p,t in [(axes[0],pA,'Modelo A — se equivoca poquito todos los meses'),
               (axes[1],pB,'Modelo B — casi perfecto, con un mes desastroso')]:
    e=real-p; mae=np.mean(np.abs(e)); rmse=np.sqrt(np.mean(e**2)); x=np.arange(n)
    ax.plot(x,real,lw=2.6,color=INK,label='real')
    ax.plot(x,p,lw=2.4,color=RED,ls=(0,(4,2)),label='pronóstico')
    ax.text(.03,1.03,t,transform=ax.transAxes,fontsize=13.5,weight='semibold',color=INK)
    ax.text(.03,.07,f'MAE {mae:.0f}     RMSE {rmse:.0f}',transform=ax.transAxes,
            fontsize=16.5,weight='bold',color=DARK)
    ax.set_xlabel('mes'); ax.tick_params(labelsize=12)
axes[0].set_ylabel('tasa  [bbl/d]'); axes[0].legend(fontsize=13,loc='upper right')
save(fig,'a14_metricas')

# ── A15 · costo asimétrico ────────────────────────────────────────────────
err=np.linspace(-400,400,300); costo=np.where(err<0,-err*70*30,err*70*30*1.6)/1000
fig,ax=plt.subplots(figsize=(9.8,4.3))
ax.plot(err,costo,lw=3.2,color=RED); ax.fill_between(err,costo,color=RED,alpha=.07,lw=0)
ax.axvline(0,color=INK,lw=1.2)
ax.set_xlabel('error del pronóstico  [bbl/d]'); ax.set_ylabel('costo del mes  [miles USD]')
ax.text(-385,760,'me quedé corto\nfacilidades chicas,\nproducción diferida',fontsize=14,
        color=MUTED,linespacing=1.5,va='top')
ax.text(115,760,'me pasé\ncompromisos incumplidos,\npresupuesto no alcanzado',fontsize=14,
        color=RED,weight='semibold',linespacing=1.5,va='top')
save(fig,'a15_costo')

# ── A16 · el modelo ───────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
te=m14[(m14.index>ORIGEN)&(m14.index<=ORIGEN+pd.DateOffset(months=6))]
def loglin(se,nf):
    t=np.arange(len(se)).reshape(-1,1); lr=LinearRegression().fit(t,np.log(se.values))
    return np.exp(lr.predict(np.arange(len(se),len(se)+nf).reshape(-1,1)))
p_all=loglin(tr,len(te)); p_12=loglin(tr.iloc[-12:],len(te))
mae=lambda p: np.mean(np.abs(te.values-p))
zoom=m14[m14.index>=pd.Timestamp('2012-01-31')]; h=zoom[zoom.index<=ORIGEN]
fig,ax=plt.subplots(figsize=(11.2,4.5))
ax.plot(h.index,h.values,lw=2.4,color=GRAY,label='historia conocida')
ax.plot(te.index,te.values,'o',ms=11,color=INK,label='lo que realmente pasó',zorder=5)
ax.plot(te.index,np.repeat(tr.iloc[-1],len(te)),lw=2.4,ls=(0,(4,3)),color=AMBER,
        label=f'«igual que este mes» — se equivoca {mae(np.repeat(tr.iloc[-1],len(te))):.0f}')
ax.plot(te.index,p_all,lw=2.6,ls=(0,(1,1.6)),color=BLUE,
        label=f'recta con toda la historia — se equivoca {mae(p_all):.0f}')
ax.plot(te.index,p_12,lw=3.0,color=RED,
        label=f'recta con el último año — se equivoca {mae(p_12):.0f}')
ax.axvline(ORIGEN,color=INK,lw=1.4,ls=(0,(4,3)))
ax.legend(fontsize=12.5,loc='lower left'); ax.set_ylabel('tasa  [bbl/d]')
ax.set_xlim(zoom.index[0],te.index[-1]+pd.Timedelta(days=30))
save(fig,'a16_modelo')
print('   MAE ingenuo %.0f | todo %.0f | 12m %.0f'%(mae(np.repeat(tr.iloc[-1],len(te))),mae(p_all),mae(p_12)))

# ── A17 · resultado en dólares ────────────────────────────────────────────
res=[('Recta con el último año',mae(p_12),RED),
     ('«Igual que este mes»',mae(np.repeat(tr.iloc[-1],len(te))),AMBER),
     ('Recta con toda la historia',mae(p_all),BLUE)]
fig,ax=plt.subplots(figsize=(10.8,3.5))
ys=np.arange(len(res))[::-1]
for y0,(nom2,mm,c) in zip(ys,res):
    usd=mm*30*70/1000
    ax.barh(y0,usd,height=.55,color=c,alpha=.92)
    ax.text(usd+7,y0,f'{usd:.0f} mil USD al mes',va='center',fontsize=15,color=INK,weight='semibold')
    ax.text(-7,y0,nom2,va='center',ha='right',fontsize=15,color=INK)
ax.set_yticks([]); ax.set_xlim(0,400); ax.grid(axis='y',visible=False)
ax.spines['left'].set_visible(False)
ax.set_xlabel('cuánto cuesta equivocarse, al mes  ·  crudo a 70 USD/bbl')
save(fig,'a17_resultado')
print('listo')
