# markov.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, detrend
from sklearn.cluster import KMeans

st.set_page_config(page_title="TUG Semi-Markov", layout="wide")
st.title("📱 Segmentação do TUG com Threshold e Cadeias Semi-Markov")

# ============================================================
# IO & PREPROCESSAMENTO
# ============================================================
def read_table_any(file):
    try:
        return pd.read_csv(file, sep=";")
    except Exception:
        file.seek(0)
        return pd.read_csv(file, sep=None, engine="python")

def lowpass(x, fs, fc, order=4):
    wn = fc / (fs / 2)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x)

def resample(t, y, fs_target):
    t, y = np.asarray(t), np.asarray(y)
    idx = np.argsort(t)
    t, y = t[idx], y[idx]
    dt = 1 / fs_target
    t_new = np.arange(t[0], t[-1], dt)
    y_new = np.interp(t_new, t, y)
    return t_new, y_new

def preprocess_xyz(t_ms, x, y, z, fs_target, detr=True, fc=1.5):
    t_s = t_ms / 1000
    t_s, x = resample(t_s, x, fs_target)
    _, y = resample(t_s, y, fs_target)
    _, z = resample(t_s, z, fs_target)

    if detr:
        x, y, z = detrend(x), detrend(y), detrend(z)

    x, y, z = lowpass(x, fs_target, fc), lowpass(y, fs_target, fc), lowpass(z, fs_target, fc)

    norma = np.sqrt(x**2 + y**2 + z**2)
    abs_dxdt = np.abs(np.gradient(norma, t_s))

    return {
        "tempo_ms": t_s * 1000,
        "t_s": t_s,
        "x_filt": x,
        "y_filt": y,
        "z_filt": z,
        "norma_filt": norma,
        "abs_dxdt": abs_dxdt
    }

# ============================================================
# SEMI-MARKOV HELPERS
# ============================================================
def merge_short_runs(states, min_len):
    states = states.copy()
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(states):
            j = i
            while j < len(states) and states[j] == states[i]:
                j += 1
            if j - i < min_len:
                if i > 0:
                    states[i:j] = states[i - 1]
                elif j < len(states):
                    states[i:j] = states[j]
                changed = True
            i = j
    return states

def detect_seq(states, base_states, n_base, n_out):
    is_base = np.isin(states, base_states)
    N = len(states)
    onset, offset = None, None

    for i in range(N - n_base - n_out):
        if is_base[i:i+n_base].all() and (~is_base[i+n_base:i+n_base+n_out]).all():
            onset = i + n_base
            break

    for i in range(N - n_base - n_out):
        if (~is_base[i:i+n_out]).all() and is_base[i+n_out:i+n_out+n_base].all():
            offset = i + n_out - 1

    return onset, offset

def transition_matrix(states, K):
    P = np.zeros((K, K))
    for i in range(len(states) - 1):
        P[states[i], states[i+1]] += 1
    P = P / np.maximum(P.sum(axis=1, keepdims=True), 1)
    return P

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Arquivo")
    file = st.file_uploader("CSV / TXT", type=["csv", "txt"])

    st.header("Colunas")
    col_t = st.text_input("Tempo (ms)", "DURACAO")
    col_x = st.text_input("Eixo X", "AVL EIXO X")
    col_y = st.text_input("Eixo Y", "AVL EIXO Y")
    col_z = st.text_input("Eixo Z", "AVL EIXO Z")

    st.header("Pré-processamento")
    fs_target = st.number_input("Reamostragem (Hz)", 20, 500, 100)
    fc = st.number_input("Filtro LP (Hz)", 0.5, 10.0, 1.5)
    detr = st.checkbox("Detrend", True)

    st.header("Segmentação")
    baseline_s = st.number_input("Baseline (s)", 0.5, 10.0, 2.0)
    run_len = st.number_input("Duração mínima (amostras)", 1, 50, 5)

    n_base = st.number_input("N baseline", 1, 50, 5)
    n_out = st.number_input("N fora", 1, 50, 5)

    kmeans_k = st.multiselect("K-means (K)", [3,4,5,6], [4,5])

    run = st.button("▶ Rodar")

# ============================================================
# EXECUÇÃO
# ============================================================
if not file:
    st.stop()

df = read_table_any(file)
df.columns = df.columns.str.strip()

raw_df = df[[col_t, col_x, col_y, col_z]].copy()

data = preprocess_xyz(
    df[col_t].values.astype(float),
    df[col_x].values.astype(float),
    df[col_y].values.astype(float),
    df[col_z].values.astype(float),
    fs_target,
    detr,
    fc
)

t_s = data["t_s"]
norma = data["norma_filt"]
abs_dxdt = data["abs_dxdt"]

base_mask = t_s <= baseline_s

if not run:
    st.line_chart(norma)
    st.stop()

results = {}

# =========================
# K-MEANS SEMI-MARKOV
# =========================
for K in kmeans_k:
    km = KMeans(n_clusters=K, n_init=20, random_state=0)
    z = (norma - norma.mean()) / norma.std()
    states = km.fit_predict(z.reshape(-1,1))

    order = np.argsort(km.cluster_centers_.flatten())
    inv = np.zeros_like(order)
    inv[order] = np.arange(K)
    states = inv[states]

    states = merge_short_runs(states, run_len)
    base_states = pd.Series(states[base_mask]).value_counts().index[:1].values

    on, off = detect_seq(states, base_states, n_base, n_out)
    P = transition_matrix(states, K)

    results[f"KMeans_K{K}"] = {
        "states": states,
        "on": on,
        "off": off,
        "P": P
    }

# ============================================================
# VISUALIZAÇÃO
# ============================================================
st.subheader("Segmentação")

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(t_s, norma, label="Norma")
for name, r in results.items():
    if r["on"] is not None:
        ax.axvline(t_s[r["on"]], linestyle="--", label=f"{name} início")
    if r["off"] is not None:
        ax.axvline(t_s[r["off"]], linestyle=":", label=f"{name} fim")
ax.legend()
ax.grid()
st.pyplot(fig)

# ============================================================
# MATRIZES
# ============================================================
st.subheader("Matrizes de Transição")
for name, r in results.items():
    st.markdown(f"### {name}")
    st.dataframe(pd.DataFrame(r["P"]).round(3))

# ============================================================
# DOWNLOADS
# ============================================================
st.subheader("Downloads")

proc_df = pd.DataFrame(data)
st.download_button(
    "📥 Dados processados (100 Hz)",
    proc_df.to_csv(index=False).encode(),
    "tug_processado.csv"
)

st.download_button(
    "📥 Dados brutos",
    raw_df.to_csv(index=False).encode(),
    "tug_raw.csv"
)
