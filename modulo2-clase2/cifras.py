"""Imprime todas las cifras que aparecen en la presentación y el cuaderno de la clase 2."""
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
df = pd.read_csv(RAIZ / "datos" / "volve_pozos_diario.csv", parse_dates=["fecha"])
ab = df[df["horas"] > 0]          # días con el pozo abierto
p = ab["petroleo"]

print("filas", len(df), "pozos", df.pozo.nunique(), df.fecha.min().date(), df.fecha.max().date())
print("días cerrados: %d (%.1f %%)" % ((df.horas == 0).sum(), 100 * (df.horas == 0).mean()))
print("\nPETRÓLEO, días abiertos (%d)" % len(ab))
print("  media %.0f  mediana %.0f  desv %.0f  asimetría %.2f" % (p.mean(), p.median(), p.std(), p.skew()))
print("  P10 %.0f P25 %.0f P50 %.0f P75 %.0f P90 %.0f" % tuple(p.quantile([.1, .25, .5, .75, .9])))
print("  días bajo la media: %.1f %%   bajo el P10: %.1f %%" % (100 * (p < p.mean()).mean(), 100 * (p < p.quantile(.1)).mean()))
print("  media - 1 desv: %.0f" % (p.mean() - p.std()))
print("  con días cerrados: media %.0f mediana %.0f" % (df.petroleo.mean(), df.petroleo.median()))
print("\nHORAS (todos los días): media %.1f mediana %.1f asimetría %.2f; =24: %.1f %%; entre 0 y 24: %.1f %%"
      % (df.horas.mean(), df.horas.median(), df.horas.skew(), 100 * (df.horas == 24).mean(), 100 * ((df.horas > 0) & (df.horas < 24)).mean()))
print("asimetría gas %.2f agua %.2f (días abiertos)" % (ab.gas.skew(), ab.agua.skew()))
print("\nSUCIEDAD para la clase 3: agua negativa %d filas; horas > 24: %d filas; días cerrados %d"
      % ((df.agua < 0).sum(), (df.horas > 24).sum(), (df.horas == 0).sum()))
print("\nPOR POZO (días abiertos)")
for w, g in ab.groupby("pozo"):
    print("  %s días %4d media %6.0f mediana %6.0f asimetría %5.2f bajo la media %.0f %% | r días-pet %.2f spearman %.2f | r horas-pet %.2f"
          % (w, len(g), g.petroleo.mean(), g.petroleo.median(), g.petroleo.skew(), 100 * (g.petroleo < g.petroleo.mean()).mean(),
             g[["dias", "petroleo"]].corr().iloc[0, 1], g[["dias", "petroleo"]].corr("spearman").iloc[0, 1], g[["horas", "petroleo"]].corr().iloc[0, 1]))
print("\nRELACIONES (días abiertos)")
print("  gas-petróleo r %.3f; gas por barril: %.0f pies cúbicos" % (ab[["gas", "petroleo"]].corr().iloc[0, 1], 1000 * (ab.gas.sum() / ab.petroleo.sum())))
print("  horas-petróleo r %.2f; agua-petróleo r %.2f; días-petróleo r %.2f" % (ab[["horas", "petroleo"]].corr().iloc[0, 1], ab[["agua", "petroleo"]].corr().iloc[0, 1], ab[["dias", "petroleo"]].corr().iloc[0, 1]))
print("\nmatriz:\n", ab[["horas", "dias", "petroleo", "gas", "agua"]].corr().round(2).to_string())
h = ab["horas"]
print("\nHORAS, solo días en que el pozo trabajó: media %.1f mediana %.1f asimetría %.2f; =24: %.1f %%"
      % (h.mean(), h.median(), h.skew(), 100 * (h == 24).mean()))
