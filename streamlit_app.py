# -*- coding: utf-8 -*-
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Opcionales (el app se adapta si no están)
HAVE_PROPHET = True
try:
    from prophet import Prophet
except Exception:
    HAVE_PROPHET = False

HAVE_STATSM = True
try:
    import statsmodels.api as sm
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:
    HAVE_STATSM = False

import plotly.graph_objects as go

# -------------------- UI --------------------
st.set_page_config(page_title="Proyección mercado/inscritos – shares dinámicos", layout="wide")
st.title("📈 Proyección de mercado e inscritos (shares dinámicos)")
st.caption("Carga tu Excel o usa el DEMO. Compara métodos y ajusta shares por anclas.")

# -------------------- Utilidades --------------------
FUTURE_YEARS = np.arange(2025, 2031)

def compute_cagr(start_val: float, end_val: float, n_years: int) -> float:
    if start_val <= 0 or end_val <= 0 or n_years <= 0:
        return 0.0
    return (end_val / start_val) ** (1 / n_years) - 1

def project_market(df_hist: pd.DataFrame, method: str, future_years: np.ndarray) -> pd.Series:
    y = df_hist["market_size"].values
    x = df_hist["year"].values.reshape(-1, 1)

    if method == "CAGR":
        tm_start, tm_end = float(y[0]), float(y[-1])
        n_years = int(df_hist["year"].iloc[-1] - df_hist["year"].iloc[0])
        cagr = compute_cagr(tm_start, tm_end, n_years)
        preds = [tm_end * ((1 + cagr) ** i) for i in range(1, len(future_years) + 1)]
        return pd.Series(preds, index=future_years, name="market")

    if method == "Lineal":
        lr = LinearRegression().fit(x, y)
        preds = lr.predict(future_years.reshape(-1, 1))
        return pd.Series(preds, index=future_years, name="market")

    if method == "Prophet" and HAVE_PROPHET:
        pdf = df_hist[["year", "market_size"]].rename(columns={"year": "ds", "market_size": "y"}).copy()
        pdf["ds"] = pd.to_datetime(pdf["ds"], format="%Y")
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False, interval_width=0.8)
        m.fit(pdf)
        f = m.make_future_dataframe(periods=len(future_years), freq="Y")
        fc = m.predict(f).tail(len(future_years))
        return pd.Series(fc["yhat"].values, index=future_years, name="market")

    if method == "ARIMA" and HAVE_STATSM:
        model = sm.tsa.arima.ARIMA(df_hist["market_size"], order=(1, 1, 1)).fit()
        preds = model.forecast(steps=len(future_years)).values
        return pd.Series(preds, index=future_years, name="market")

    if method == "Holt-Winters" and HAVE_STATSM:
        hw = ExponentialSmoothing(df_hist["market_size"], trend="add").fit()
        preds = hw.forecast(len(future_years)).values
        return pd.Series(preds, index=future_years, name="market")

    # Fallback
    lr = LinearRegression().fit(x, y)
    preds = lr.predict(future_years.reshape(-1, 1))
    return pd.Series(preds, index=future_years, name="market")

def interpolate_between_anchors(years, anchor_years, anchor_values):
    shares = {}
    ays = sorted(anchor_years)
    for i in range(len(ays) - 1):
        y0, y1 = ays[i], ays[i + 1]
        v0, v1 = anchor_values[y0], anchor_values[y1]
        step = (v1 - v0) / (y1 - y0)
        for y in range(y0, y1 + 1):
            shares[y] = float(v0 + step * (y - y0))
    for y in shares:
        shares[y] = max(0.0, min(1.0, shares[y]))
    return pd.Series([shares[y] for y in years], index=years)

# -------------------- Datos: demo o uploader --------------------
APP_DIR = Path(__file__).parent
DEMO_PATH = APP_DIR / "data" / "Prueba_dentista.xlsx"

st.subheader("Fuente de datos")
use_demo = False
if DEMO_PATH.exists():
    use_demo = st.toggle("Usar archivo DEMO incluido", value=True,
                         help="Carga automáticamente data/Prueba_dentista.xlsx")
    with open(DEMO_PATH, "rb") as f:
        st.download_button("⬇️ Descargar ejemplo (Dentista)", f.read(),
                           file_name="Prueba_dentista.xlsx")

up = st.file_uploader("📎 O sube tu Excel/CSV (Año | TamañoMercado | Inscritos | ShareObjetivo)",
                      type=["xlsx", "xls", "csv"], accept_multiple_files=False)

if use_demo:
    raw = pd.read_excel(DEMO_PATH, sheet_name=0)
elif up is not None:
    suffix = Path(up.name).suffix.lower()
    raw = pd.read_excel(up, sheet_name=0) if suffix in (".xlsx", ".xls") else pd.read_csv(up)
else:
    st.info("Activa **Usar archivo DEMO** o sube un archivo propio para continuar.")
    st.stop()

# Normalizar
df = raw.rename(columns={
    "Año": "year",
    "TamañoMercado": "market_size",
    "Inscritos": "enrolled",
    "ShareObjetivo": "share_obj",
    "Escuela": "school",
    "Clasificacion": "classification",
}).sort_values("year").reset_index(drop=True)

df["share_hist"] = np.where(
    (df.get("enrolled").notna()) & (df.get("market_size").notna()) & (df.get("market_size") > 0),
    df["enrolled"] / df["market_size"], np.nan
)

st.subheader("Datos cargados")
st.dataframe(df, use_container_width=True)

# -------------------- Modelo de mercado --------------------
st.markdown("---")
st.subheader("Proyección de Tamaño de Mercado")

avail_methods = ["CAGR", "Lineal"]
if HAVE_PROPHET: avail_methods.append("Prophet")
if HAVE_STATSM:  avail_methods += ["ARIMA", "Holt-Winters"]

method = st.selectbox("Método base", options=avail_methods, index=1)

hist = df[(df["market_size"].notna()) & (df["year"] <= 2024)][["year", "market_size"]].copy()
if hist.empty or len(hist) < 3:
    st.error("Necesito al menos 3 años de Tamaño de Mercado histórico (<=2024) para proyectar.")
    st.stop()

market_proj = project_market(hist, method, FUTURE_YEARS)

# -------------------- Shares dinámicos --------------------
st.markdown("---")
st.subheader("Shares objetivo dinámicos (2025–2030) por anclas")

base_shares = {int(r.year): float(r.share_obj) for _, r in df[df["year"].isin(FUTURE_YEARS)][["year", "share_obj"]].fillna(np.nan).iterrows()}
for y in FUTURE_YEARS:
    if y not in base_shares or pd.isna(base_shares[y]):
        base_shares[y] = 0.15 + (0.30 - 0.15) * ((y - FUTURE_YEARS[0]) / (FUTURE_YEARS[-1] - FUTURE_YEARS[0]))

cols = st.columns(len(FUTURE_YEARS))
anchors, values = {}, {}
for i, y in enumerate(FUTURE_YEARS):
    with cols[i]:
        anchors[y] = st.checkbox(f"Ancla {y}", value=(y in (FUTURE_YEARS[0], FUTURE_YEARS[-1])))
        values[y] = st.number_input(f"Share {y}", min_value=0.0, max_value=1.0,
                                    value=float(base_shares[y]), step=0.005, key=f"share_{y}")

# asegurar extremos
anchors[FUTURE_YEARS[0]] = True
anchors[FUTURE_YEARS[-1]] = True

anchor_years = [y for y in FUTURE_YEARS if anchors[y]]
anchor_vals  = {y: values[y] for y in anchor_years}
share_series = interpolate_between_anchors(list(FUTURE_YEARS), anchor_years, anchor_vals)

c1, c2, c3 = st.columns(3)
with c1:
    enforce_monotone = st.checkbox("Forzar shares no-decrecientes", value=True)
with c2:
    round_3 = st.checkbox("Redondear shares a 3 decimales", value=True)
with c3:
    cap_30 = st.checkbox("Cap share máx. 0.30 (ejemplo)", value=False)

if enforce_monotone:
    for i in range(1, len(share_series)):
        if share_series.iloc[i] < share_series.iloc[i - 1]:
            share_series.iloc[i] = share_series.iloc[i - 1]
if cap_30:
    share_series = share_series.clip(upper=0.30)
if round_3:
    share_series = (share_series * 1000).round().astype(int) / 1000.0

st.dataframe(pd.DataFrame({"year": FUTURE_YEARS, "share_obj_dyn": share_series.values}),
             use_container_width=True)

# -------------------- Resultados (método seleccionado) --------------------
results = pd.DataFrame({"year": FUTURE_YEARS})
results["market_proj"] = market_proj.values
results["share_obj_dyn"] = share_series.values
results["enrolled_proj"] = results["market_proj"] * results["share_obj_dyn"]

st.markdown("---")
st.subheader("Resultados – método seleccionado")
st.dataframe(results, use_container_width=True)

buf1 = io.BytesIO()
results.to_csv(buf1, index=False)
st.download_button("💾 Descargar CSV – método seleccionado", data=buf1.getvalue(),
                   file_name="proyeccion_resultados.csv", mime="text/csv")

# -------------------- Benchmark (tablas + CURVAS comparativas) --------------------
st.markdown("---")
st.subheader("Benchmark: comparación de modelos (tablas y curvas)")

bench_methods = ["CAGR", "Lineal"]
if HAVE_PROPHET: bench_methods.append("Prophet")
if HAVE_STATSM:  bench_methods += ["ARIMA", "Holt-Winters"]

if st.checkbox("Calcular benchmark de todos los modelos", value=True):
    bench_df = pd.DataFrame({"year": FUTURE_YEARS})
    for mth in bench_methods:
        bench_df[mth] = project_market(hist, mth, FUTURE_YEARS).values

    # inscritos por modelo (usa los shares dinámicos resultantes)
    shares_used = share_series.values
    for mth in bench_methods:
        bench_df[f"Inscritos_{mth}"] = bench_df[mth] * shares_used

    st.write("**Tabla de mercado 2025–2030 por modelo**")
    st.dataframe(bench_df[["year"] + bench_df.columns[1:len(bench_methods)+1].tolist()],
                 use_container_width=True)

    st.write("**Tabla de inscritos 2025–2030 por modelo**")
    st.dataframe(bench_df[["year", "Inscritos_CAGR", "Inscritos_Lineal"] +
                          ([ "Inscritos_Prophet"] if "Prophet" in bench_methods else []) +
                          ([ "Inscritos_ARIMA"] if "ARIMA" in bench_methods else []) +
                          ([ "Inscritos_Holt-Winters"] if "Holt-Winters" in bench_methods else [])],
                 use_container_width=True)

    # --------- NUEVO: GRÁFICAS DE CURVAS COMPARATIVAS (Plotly) ----------
    st.write("### Curvas comparativas – Tamaño de mercado")
    fig_m = go.Figure()
    for mth in bench_methods:
        fig_m.add_trace(go.Scatter(x=bench_df["year"], y=bench_df[mth],
                                   mode="lines+markers", name=mth))
    fig_m.update_layout(xaxis_title="Año", yaxis_title="Tamaño de mercado",
                        hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig_m, use_container_width=True)

    st.write("### Curvas comparativas – Inscritos proyectados (usa shares dinámicos)")
    fig_e = go.Figure()
    for mth in bench_methods:
        fig_e.add_trace(go.Scatter(x=bench_df["year"], y=bench_df[f"Inscritos_{mth}"],
                                   mode="lines+markers", name=f"Inscritos {mth}"))
    fig_e.update_layout(xaxis_title="Año", yaxis_title="Inscritos proyectados",
                        hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig_e, use_container_width=True)

    buf2 = io.BytesIO()
    bench_df.to_csv(buf2, index=False)
    st.download_button("💾 Descargar CSV – benchmark (todos los modelos)",
                       data=buf2.getvalue(), file_name="benchmark_modelos.csv", mime="text/csv")

