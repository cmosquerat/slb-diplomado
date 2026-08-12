#!/usr/bin/env python3
"""Figuras EXPLICATIVAS: cada herramienta se muestra antes de aplicarse."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
import numpy as np, pandas as pd, pathlib
for f in ['FiraSans-Regular','FiraSans-Medium','FiraSans-SemiBold','FiraSans-Bold']:
    p = pathlib.Path(f'/tmp/fonts/{f}.ttf')
    if p.exists(): fm.fontManager.addfont(str(p))
RED='#C82B40'; DARK='#6B1525'; BLUE='#2563EB'; AMBER='#E69F00'; GREEN='#1B7F4B'
GRAY='#9CA3AF'; LGRAY='#E5E7EB'; INK='#2D2D2D'; MUTED='#6B7280'
plt.rcParams.update({'font.family':'Fira Sans','font.size':16,'axes.labelsize':16,
 'axes.labelcolor':MUTED,'text.color':INK,'xtick.color':MUTED,'ytick.color':MUTED,
 'xtick.labelsize':14,'ytick.labelsize':14,'axes.edgecolor':LGRAY,'axes.linewidth':1,
 'axes.grid':True,'grid.color':LGRAY,'grid.linewidth':.7,'grid.alpha':.8,
 'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white',
 'figure.dpi':200,'savefig.bbox':'tight','savefig.pad_inches':.15,'legend.frameon':False})
D = pathlib.Path('fig_v3')
def save(fig,n): fig.savefig(D/f'{n}.png'); plt.close(fig); print('  ✓',n)

CSV='/mnt/user-data/uploads/slb-diplomado/datos/volve_produccion.csv'
df = pd.read_csv(CSV, parse_dates=['fecha'])
f14 = df[df.pozo=='15/9-F-14'].sort_values('fecha').set_index('fecha')
f12 = df[df.pozo=='15/9-F-12'].sort_values('fecha').set_index('fecha')

# ══ C1 · la ventana de 5 días: horas vs volumen ═══════════════════════════
w = f14.loc['2010-09-28':'2010-10-02']
lab = [d.strftime('%d-%b') for d in w.index]
fig, axes = plt.subplots(1,2, figsize=(11.5,4.2))
c = [GRAY,GRAY,RED,GRAY,GRAY]
axes[0].bar(lab, w.horas, color=c, width=.62)
axes[0].set_ylabel('horas que operó'); axes[0].set_ylim(0,27)
axes[0].text(2, w.horas.iloc[2]+1.2, '9.5 h', ha='center', fontsize=16, weight='bold', color=RED)
axes[0].text(.02,1.04,'columna  horas', transform=axes[0].transAxes, fontsize=15, weight='semibold', color=INK)
axes[1].bar(lab, w.oil, color=c, width=.62)
axes[1].set_ylabel('barriles producidos'); axes[1].set_ylim(0,3400)
axes[1].text(2, w.oil.iloc[2]+130, '1 122', ha='center', fontsize=16, weight='bold', color=RED)
axes[1].text(.02,1.04,'columna  oil', transform=axes[1].transAxes, fontsize=15, weight='semibold', color=INK)
for a in axes: a.tick_params(axis='x', labelsize=13)
save(fig,'c1_ventana')

# ══ C2 · la normalización corregida ═══════════════════════════════════════
fig, ax = plt.subplots(figsize=(10.5,4.2))
norm = w.oil.copy(); norm.iloc[2] = w.oil.iloc[2]*24/w.horas.iloc[2]
ax.bar(lab, w.oil, color=[GRAY]*5, width=.6, label='barriles reportados')
ax.bar(lab[2], norm.iloc[2], color='none', width=.6, edgecolor=RED, lw=3,
       hatch='///', label='normalizado a 24 h')
ax.plot(lab, [w.oil.iloc[0],w.oil.iloc[1],norm.iloc[2],w.oil.iloc[3],w.oil.iloc[4]],
        'o--', color=RED, ms=10, lw=2, mfc='white', mew=2.5, zorder=5)
ax.annotate('1 122 × 24 / 9.5  =  2 834', xy=(2, norm.iloc[2]), xytext=(2.55, 3250),
            fontsize=16, weight='bold', color=RED,
            arrowprops=dict(arrowstyle='-', color=RED, lw=1.6))
ax.set_ylim(0,3700); ax.set_ylabel('barriles')
ax.legend(fontsize=14, loc='lower left')
ax.tick_params(axis='x', labelsize=13)
save(fig,'c2_normalizado')

# ══ C3 · QUÉ ES una escala logarítmica ════════════════════════════════════
v = np.array([100,200,400,800,1600])
fig, axes = plt.subplots(1,2, figsize=(11.5,3.6))
for ax, log in zip(axes,[False,True]):
    ax.scatter(np.zeros(len(v)), v, s=150, color=RED, zorder=3)
    for x in v: ax.text(0.14, x, f'{x}', va='center', fontsize=16, color=INK)
    if log: ax.set_yscale('log')
    ax.set_xlim(-.5,1.2); ax.set_xticks([]); ax.grid(axis='x', visible=False)
    ax.text(.5,1.06,'escala normal' if not log else 'escala logarítmica',
            transform=ax.transAxes, fontsize=16, weight='semibold', color=INK, ha='center')
    ax.set_yticks(v if log else [0,400,800,1200,1600])
    ax.set_yticklabels([str(x) for x in (v if log else [0,400,800,1200,1600])])
axes[0].text(.5,-.16,'los saltos se ven cada vez más grandes',
             transform=axes[0].transAxes, ha='center', fontsize=14, color=MUTED)
axes[1].text(.5,-.16,'cada “×2” ocupa lo mismo → quedan equiespaciados',
             transform=axes[1].transAxes, ha='center', fontsize=14, color=RED, weight='semibold')
save(fig,'c3_que_es_log')

# ══ C4 · QUÉ ES una media móvil ═══════════════════════════════════════════
np.random.seed(3)
y = np.array([820,760,910,700,845,780,690,800,720,760,830,700])
k = 3
ma = pd.Series(y).rolling(k).mean().values
fig, ax = plt.subplots(figsize=(11.0,4.0))
x = np.arange(len(y))
ax.bar(x, y, color=LGRAY, width=.55, zorder=1)
ax.plot(x, ma, 'o-', color=RED, lw=3, ms=9, mfc='white', mew=2.5, zorder=4)
for i in [4]:
    ax.add_patch(plt.Rectangle((i-k+0.55, 0), k-0.1, 1000, color=AMBER, alpha=.22, zorder=0))
    ax.annotate(f'({y[i-2]}+{y[i-1]}+{y[i]}) / 3  =  {ma[i]:.0f}',
                xy=(i, ma[i]), xytext=(i+0.4, 1050), fontsize=16, weight='bold', color=RED,
                arrowprops=dict(arrowstyle='-', color=RED, lw=1.5))
ax.set_ylim(0,1220); ax.set_xticks(x); ax.set_xticklabels([f'd{i+1}' for i in x], fontsize=13)
ax.set_ylabel('valor diario')
ax.text(.01,1.02,'la ventana se desliza y promedia los últimos 3 días',
        transform=ax.transAxes, fontsize=15, color=MUTED)
save(fig,'c4_que_es_ma')

# ══ C5 · QUÉ ES la autocorrelación ════════════════════════════════════════
s = f14.oil.replace(0,np.nan).asfreq('D').interpolate(limit=3).dropna().loc['2011':'2012']
k = 30
fig, axes = plt.subplots(1,2, figsize=(11.5,4.0), gridspec_kw={'width_ratios':[1.5,1]})
ax = axes[0]
ax.plot(s.index, s.values, lw=2, color=INK, label='la serie')
ax.plot(s.index, s.shift(k).values, lw=2, color=RED, ls=(0,(4,2)), label=f'la misma, corrida {k} días')
ax.legend(fontsize=13.5, loc='upper right'); ax.set_ylabel('barriles')
ax.tick_params(labelsize=12)
ax = axes[1]
ax.scatter(s.shift(k).values, s.values, s=14, color=RED, alpha=.35, linewidths=0)
ax.set_xlabel(f'valor hace {k} días'); ax.set_ylabel('valor de hoy')
r = pd.Series(s.values).corr(pd.Series(s.shift(k).values))
ax.text(.05,.92,f'correlación = {r:.2f}', transform=ax.transAxes, fontsize=17,
        weight='bold', color=RED)
ax.tick_params(labelsize=12)
save(fig,'c5_que_es_acf')

# ══ C6 · barajar destruye ═════════════════════════════════════════════════
mm = f14.oil.replace(0,np.nan).resample('ME').mean().dropna().values
rng = np.random.default_rng(7); sh = rng.permutation(mm)
fig, axes = plt.subplots(1,2, figsize=(11.5,3.6), sharey=True)
axes[0].plot(mm, lw=2.6, color=RED); axes[0].text(.03,1.04,'los datos en su orden',
    transform=axes[0].transAxes, fontsize=15.5, weight='semibold', color=INK)
axes[1].plot(sh, lw=2.6, color=GRAY); axes[1].text(.03,1.04,'las mismas filas, barajadas',
    transform=axes[1].transAxes, fontsize=15.5, weight='semibold', color=INK)
axes[0].set_ylabel('barriles / día'); [a.set_xlabel('mes') for a in axes]
save(fig,'c6_barajar')

# ══ C7 · QUÉ ES agrupar: de tabla a puntos (K-means) ══════════════════════
d = f14[(f14.horas>0)&(f14.oil>0)].copy()
d['tasa']=d.oil*24/d.horas; d['wc']=d.agua/(d.agua+d.oil)
m = d[['tasa','wc']].resample('ME').median().dropna()
fig, ax = plt.subplots(figsize=(7.6,4.6))
ax.scatter(m.wc*100, m.tasa, s=70, color=GRAY, alpha=.85, linewidths=0)
ej = m.iloc[[3, 40, 80]]
for (i,row),lab2 in zip(ej.iterrows(), ['un mes joven','un mes intermedio','un mes viejo']):
    ax.scatter([row.wc*100],[row.tasa], s=170, color=RED, zorder=5, edgecolor='white', lw=2)
    ax.annotate(f'{i.strftime("%b-%Y")}\n({row.wc*100:.0f} % agua, {row.tasa:.0f} bbl/d)',
                xy=(row.wc*100,row.tasa), xytext=(row.wc*100+9, row.tasa+520),
                fontsize=12.5, color=RED, weight='semibold', linespacing=1.35,
                arrowprops=dict(arrowstyle='-', color=RED, lw=1.2))
ax.set_xlabel('corte de agua  [%]'); ax.set_ylabel('tasa mediana  [bbl/d]')
ax.set_ylim(-200, 6000)
save(fig,'c7_atributos')
print('conceptos listos')
