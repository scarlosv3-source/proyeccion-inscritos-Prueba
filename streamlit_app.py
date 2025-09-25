# streamlit_app.py
# -------------------------------------------------------------
# Proyección de Tamaño de Mercado (2025–2030) y Inscritos
# con shares OBJETIVO dinámicos (anclajes + interpolación)
# Modelos comparados: CAGR | Lineal | Prophet* | ARIMA | Holt-Winters
# * Prophet es opcional; el app sigue funcionando si no está instalado.
# -------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st

# Modelos
from sklearn.linear_model import LinearRegression

# Imports opcionales con manejo de errores
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

st.set_page_config(page_title="Proyección Mercado/Inscritos – Shares dinámicos", layout="wide")
st.title("📈 Proyección mercado e inscritos con shares dinámicos")
st.caption("Sube tu archivo Excel con columnas: Año | TamañoMercado | Inscritos | ShareObjetivo. Opcionalmente agrega Escuela/Clasificacion si deseas.")

# ------------------------------
# Utilidades
# ------------------------------

def compute_cagr(start_val: float, end_val: float, n_years: int) -> float:
    if start_val <= 0 or end_val <= 0 or n_years <= 0:
        return 0.0
    return (end_val / start_val) ** (1 / n_years) - 1


def project_market(df_hist: pd.DataFrame, method: str, future_years: np.ndarray) -> pd.Series:
    """Devuelve Serie con proyección de mercado para future_years según method."""
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
        yhat = fc["yhat"].values
        return pd.Series(yhat, index=future_years, name="market")

    if method == "ARIMA" and HAVE_STATSM:
        model = sm.tsa.arima.ARIMA(df_hist["market_size"], order=(1, 1, 1)).fit()
        preds = model.forecast(steps=len(future_years)).values
        return pd.Series(preds, index=future_years, name="market")

    if method == "Holt-Winters" and HAVE_STATSM:
        hw = ExponentialSmoothing(df_hist["market_size"], trend="add").fit()
        preds = hw.forecast(len(future_years)).values
        return pd.Series(preds, index=future_years, name="market")

    # Fallback a Lineal si el método no está disponible
    lr = LinearRegression().fit(x, y)
    preds = lr.predict(future_years.reshape(-1, 1))
    return pd.Series(preds, index=future_years, name="market")


def interpolate_between_anchors(years, anchor_years, anchor_values):
    """Interpolación lineal para años entre anclas.
    years: lista ordenada (p.ej. [2025..2030])
    anchor_years: lista con años anclados, subset de years (debe incluir extremos deseados)
    anchor_values: dict {year: value} para anchors
    """
    shares = {}
    # recorrer tramos entre anchors
    ays = sorted(anchor_years)
    for i in range(len(ays) - 1):
        y0, y1 = ays[i], ays[i + 1]
        v0, v1 = anchor_values[y0], anchor_values[y1]
        step = (v1 - v0) / (y1 - y0)
        for y in range(y0, y1 + 1):
            shares[y] = float(v0 + step * (y - y0))
    # Clamp 0..1
    for y in shares:
        shares[y] = max(0.0, min(1.0, shares[y]))
    return pd.Series([shares[y] for y in years], index=years)


# ------------------------------
# Entrada de datos
# ------------------------------

up = st.file_uploader("📎 Sube el Excel (una hoja con: Año | TamañoMercado | Inscritos | ShareObjetivo)", type=["xlsx", "xls"]) 

# Parámetros
future_years = np.arange(2025, 2031)

if up is not None:
    raw = pd.read_excel(up, sheet_name=0)
    # Normalizar columnas
    colmap = {
        "Año": "year",
        "TamañoMercado": "market_size",
        "Inscritos": "enrolled",
        "ShareObjetivo": "share_obj",
        "Escuela": "school",
        "Clasificacion": "classification",
    }
    df = raw.rename(columns={k: v for k, v in colmap.items() if k in raw.columns}).copy()

    # Si faltan columnas mínimas, avisar
    needed = {"year", "market_size", "enrolled", "share_obj"}
    if not needed.intersection(df.columns):
        st.error("No se encontraron columnas mínimas. Asegúrate de incluir: Año, TamañoMercado, Inscritos, ShareObjetivo")
        st.stop()

    # Filtrar una sola carrera (si viene columna school)
    if "school" in df.columns:
        schools = sorted(df["school"].dropna().unique())
        chosen = st.selectbox("Selecciona escuela/carrera", options=["(todas)"] + list(schools), index=0)
        if chosen != "(todas)":
            df = df[df["school"] == chosen].copy()

    df = df.sort_values("year").reset_index(drop=True)

    # Share histórico implícito
    df["share_hist"] = np.where(
        (df.get("enrolled").notna()) & (df.get("market_size").notna()) & (df.get("market_size") > 0),
        df["enrolled"] / df["market_size"], np.nan
    )

    st.subheader("Datos cargados (histórico y objetivos)")
    st.dataframe(df, use_container_width=True)

    # Selección de método de mercado
    avail_methods = ["CAGR", "Lineal"]
    if HAVE_PROPHET:
        avail_methods.append("Prophet")
    if HAVE_STATSM:
        avail_methods += ["ARIMA", "Holt-Winters"]

    st.markdown("---")
    st.subheader("Proyección de Tamaño de Mercado – Selecciona modelo")
    method = st.selectbox("Método base de proyección", options=avail_methods, index=1)

    # Entrena con histórico hasta 2024
    hist = df[(df["market_size"].notna()) & (df["year"] <= 2024)][["year", "market_size"]].copy()
    if hist.empty or len(hist) < 3:
        st.error("Necesito al menos 3 años de Tamaño de Mercado histórico (<=2024) para proyectar.")
        st.stop()

    market_proj = project_market(hist.rename(columns={"year": "year", "market_size": "market_size"}), method, future_years)

    # ------------------------------
    # Panel de Shares dinámicos (anclajes)
    # ------------------------------
    st.markdown("---")
    st.subheader("Shares objetivo dinámicos (2025–2030)")
    st.caption("Marca años como 'ancla' y define su share. Los años entre anclas se interpolan automáticamente. Los valores se limitan a [0,1].")

    # Valores base: si viene share_obj en el archivo, los usamos. Si no, inicializamos con 0.15 y 0.30 al 2030.
    base_shares = {int(r.year): float(r.share_obj) for _, r in df[df["year"].isin(future_years)][["year", "share_obj"]].fillna(np.nan).iterrows()}
    for y in future_years:
        if y not in base_shares or pd.isna(base_shares[y]):
            # default suave lineal 0.15 → 0.30
            base_shares[y] = 0.15 + (0.30 - 0.15) * ((y - future_years[0]) / (future_years[-1] - future_years[0]))

    cols = st.columns(len(future_years))
    anchors = {}
    values = {}

    for i, y in enumerate(future_years):
        with cols[i]:
            anchors[y] = st.checkbox(f"Ancla {y}", value=(y in (future_years[0], future_years[-1])))
            values[y] = st.number_input(f"Share {y}", min_value=0.0, max_value=1.0, value=float(base_shares[y]), step=0.005, key=f"share_{y}")

    # Siempre garantizar que haya al menos dos anclas (extremos)
    if not anchors[future_years[0]]:
        anchors[future_years[0]] = True
    if not anchors[future_years[-1]]:
        anchors[future_years[-1]] = True

    anchor_years = [y for y in future_years if anchors[y]]
    anchor_values = {y: values[y] for y in anchor_years}

    # Interpolar shares finales a usar
    share_series = interpolate_between_anchors(list(future_years), anchor_years, anchor_values)

    # Opciones extra
    c1, c2, c3 = st.columns(3)
    with c1:
        enforce_monotone = st.checkbox("Forzar que los shares sean no-decrecientes", value=True)
    with c2:
        round_3 = st.checkbox("Redondear shares a 3 decimales", value=True)
    with c3:
        cap_30 = st.checkbox("Cap share máx. en 0.30 (ejemplo)", value=False)

    # Aplicar constraints
    if enforce_monotone:
        for i in range(1, len(share_series)):
            if share_series.iloc[i] < share_series.iloc[i - 1]:
                share_series.iloc[i] = share_series.iloc[i - 1]
    if cap_30:
        share_series = share_series.clip(upper=0.30)
    if round_3:
        share_series = (share_series * 1000).round().astype(int) / 1000.0

    st.markdown("**Shares objetivo resultantes (post-reglas):**")
    st.dataframe(pd.DataFrame({"year": future_years, "share_obj_dyn": share_series.values}), use_container_width=True)

    # ------------------------------
    # Cálculo de inscritos con el método seleccionado
    # ------------------------------
    results = pd.DataFrame({"year": future_years})
    results["market_proj"] = market_proj.values
    results["share_obj_dyn"] = share_series.values
    results["enrolled_proj"] = results["market_proj"] * results["share_obj_dyn"]

    st.markdown("---")
    st.subheader("Resultados (método seleccionado)")
    st.dataframe(results, use_container_width=True)

    # Descargar CSV del método seleccionado
    out1 = io.BytesIO()
    results.to_csv(out1, index=False)
    st.download_button("💾 Descargar CSV – método seleccionado", data=out1.getvalue(), file_name="proyeccion_resultados.csv", mime="text/csv")

    # ------------------------------
    # Benchmark opcional (los 5 modelos a la vez) – SIN GRÁFICAS
    # ------------------------------
    st.markdown("---")
    st.subheader("Benchmark (opcional): 5 modelos de mercado sin gráficas")

    bench_methods = ["CAGR", "Lineal"]
    if HAVE_PROPHET:
        bench_methods.append("Prophet")
    if HAVE_STATSM:
        bench_methods += ["ARIMA", "Holt-Winters"]

    if st.checkbox("Calcular benchmark de todos los modelos", value=False):
        bench_df = pd.DataFrame({"year": future_years})
        for mth in bench_methods:
            bench_df[mth] = project_market(hist, mth, future_years).values
        # Inscritos por cada modelo
        shares_used = share_series.values
        for mth in bench_methods:
            bench_df[f"Inscritos_{mth}"] = bench_df[mth] * shares_used

        st.dataframe(bench_df, use_container_width=True)

        out2 = io.BytesIO()
        bench_df.to_csv(out2, index=False)
        st.download_button("💾 Descargar CSV – benchmark (todos los modelos)", data=out2.getvalue(), file_name="benchmark_modelos.csv", mime="text/csv")

else:
    st.info("Sube tu archivo para comenzar. Ejemplo de columnas: Año | TamañoMercado | Inscritos | ShareObjetivo")
