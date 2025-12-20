# markov.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, detrend
from sklearn.cluster import KMeans

st.set_page_config(page_title="TUG Semi-Markov", layout="wide")
st.title("📱 Segmentação do TUG com Threshold, Bins e Semi-Markov (K-means)")

# ============================================================
# IO & PREPROCESSAMENTO
# ============================================================
def read_table_any(file):
    """Lê CSV/TXT tentando ';' e depois separador automático."""
    try:
        return pd.read_csv(file, sep=";")
    except Exception:
        file.seek(0)
        return pd.read_csv(file, sep=None, engine="python")

def lowpass(x, fs, fc, order=4):
    wn = fc / (fs / 2)
    if wn <= 0 or wn >= 1:
        raise ValueError("Cutoff inválido (normalizado fora de (0,1)).")
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x)

def resample_xyz(t_s, x, y, z, fs_target):
    """
    Interpola x, y, z para a mesma grade temporal (fs_target).
    """
    t_s = np.asarray(t_s, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    # ordena por tempo
    idx = np.argsort(t_s)
    t_s = t_s[idx]
    x, y, z = x[idx], y[idx], z[idx]

    # remove tempos duplicados
    keep = np.diff(t_s, prepend=t_s[0] - 1) > 0
    t_s = t_s[keep]
    x, y, z = x[keep], y[keep], z[keep]

    dt = 1.0 / fs_target
    t_new = np.arange(t_s[0], t_s[-1] + 1e-12, dt)

    x_new = np.interp(t_new, t_s, x)
    y_new = np.interp(t_new, t_s, y)
    z_new = np.interp(t_new, t_s, z)

    return t_new, x_new, y_new, z_new

def preprocess_xyz(t_ms, x, y, z, fs_target, do_detrend=True, fc=1.5, order=4):
    """
    Pipeline:
    - tempo(ms)->s
    - interpolação para fs_target (ex. 100 Hz)
    - detrend opcional
    - passa-baixa (fc=1.5 Hz) em cada eixo
    - norma
    - |d(norma)/dt|
    """
    t_s_raw = np.asarray(t_ms, dtype=float) / 1000.0
    t_s, x_i, y_i, z_i = resample_xyz(t_s_raw, x, y, z, fs_target)

    if do_detrend:
        x_i = detrend(x_i, type="linear")
        y_i = detrend(y_i, type="linear")
        z_i = detrend(z_i, type="linear")

    x_f = lowpass(x_i, fs_target, fc, order=order)
    y_f = lowpass(y_i, fs_target, fc, order=order)
    z_f = lowpass(z_i, fs_target, fc, order=order)

    norma = np.sqrt(x_f**2 + y_f**2 + z_f**2)
    abs_dxdt = np.abs(np.gradient(norma, t_s))

    return {
        "tempo_ms": t_s * 1000.0,
        "t_s": t_s,
        "x_interp": x_i,
        "y_interp": y_i,
        "z_interp": z_i,
        "x_filt": x_f,
        "y_filt": y_f,
        "z_filt": z_f,
        "norma_filt": norma,
        "abs_dxdt": abs_dxdt,
        "fs": float(fs_target),
    }

# ============================================================
# ESTADOS / SEMI-MARKOV HELPERS
# ============================================================
def merge_short_runs(states, min_len):
    """Funde runs curtos (<min_len) no vizinho (suavização semi-Markov hard)."""
    states = np.asarray(states).copy()
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(states):
            j = i
            while j < len(states) and states[j] == states[i]:
                j += 1
            if (j - i) < min_len:
                # escolhe vizinho
                if i > 0:
                    states[i:j] = states[i - 1]
                elif j < len(states):
                    states[i:j] = states[j]
                changed = True
            i = j
    return states

def dominant_states_in_window(states, mask, top_m=1):
    vc = pd.Series(states[mask]).value_counts()
    return vc.index[:top_m].to_numpy()

def detect_seq(states, base_states, n_base, n_out):
    """
    onset: n_base baseline -> n_out não-baseline
    offset: n_out não-baseline -> n_base baseline (última ocorrência)
    Retorna índices (onset, offset).
    """
    states = np.asarray(states)
    base_states = np.atleast_1d(base_states)
    is_base = np.isin(states, base_states)

    N = len(states)
    win = n_base + n_out
    onset = None
    for i in range(0, N - win + 1):
        if is_base[i:i+n_base].all() and (~is_base[i+n_base:i+win]).all():
            onset = i + n_base
            break

    offset = None
    for i in range(0, N - win + 1):
        if (~is_base[i:i+n_out]).all() and is_base[i+n_out:i+win].all():
            offset = i + n_out - 1
    return onset, offset

def transition_matrix(states, K):
    """Matriz de transição 1 passo (probabilidades)."""
    states = np.asarray(states)
    C = np.zeros((K, K), dtype=int)
    for i in range(len(states) - 1):
        C[states[i], states[i+1]] += 1
    row = C.sum(axis=1, keepdims=True)
    P = np.divide(C, row, out=np.zeros_like(C, dtype=float), where=row != 0)
    return C, P

def p_change_table(P):
    diag = np.diag(P)
    return pd.DataFrame({"P(permanecer)": diag, "P(mudar)": 1.0 - diag})

def empirical_p_event(states, base_states, n_base, n_out):
    """Probabilidade empírica do padrão onset/offset por janelas."""
    states = np.asarray(states)
    base_states = set(np.atleast_1d(base_states).tolist())
    is_base = np.isin(states, list(base_states))
    N = len(states)
    win = n_base + n_out

    opp_on = ev_on = opp_off = ev_off = 0
    for i in range(0, N - win + 1):
        if np.all(is_base[i:i+n_base]):
            opp_on += 1
            if np.all(~is_base[i+n_base:i+win]):
                ev_on += 1
        if np.all(~is_base[i:i+n_out]):
            opp_off += 1
            if np.all(is_base[i+n_out:i+win]):
                ev_off += 1

    return pd.DataFrame(
        {
            "Oportunidades": [opp_on, opp_off],
            "Eventos": [ev_on, ev_off],
            "Probabilidade empírica": [
                ev_on / opp_on if opp_on else np.nan,
                ev_off / opp_off if opp_off else np.nan,
            ],
        },
        index=[f"Onset: base({n_base})→out({n_out})", f"Offset: out({n_out})→base({n_base})"],
    )

def discretize_bins(x, K):
    """Discretização por quantis em K estados (0..K-1)."""
    x = np.asarray(x, dtype=float)
    qs = np.linspace(0, 1, K + 1)
    edges = np.unique(np.quantile(x, qs))
    if len(edges) < 3:
        edges = np.linspace(np.min(x), np.max(x), K + 1)
    s = np.digitize(x, edges[1:-1], right=False)
    return s, edges

def idx_to_time(t_s, idx):
    return float(t_s[idx]) if idx is not None else np.nan

def plot_signal(t_s, y, overlays, title, ylabel="Norma (filtrada)"):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(t_s, y, label="sinal")
    for name, d in overlays.items():
        on_s = d.get("on_s", np.nan)
        off_s = d.get("off_s", np.nan)
        style = d.get("style", "--")
        if np.isfinite(on_s):
            ax.axvline(on_s, linestyle=style, label=f"início ({name})")
        if np.isfinite(off_s):
            ax.axvline(off_s, linestyle=style, label=f"fim ({name})")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend(loc="upper right")
    return fig

def plot_states_band(t_s, states_dict, title):
    fig, ax = plt.subplots(figsize=(12, 2.4))
    for name, d in states_dict.items():
        stt = d["states"]
        K = d["K"]
        y = stt / max(1, (K - 1))
        ax.plot(t_s, y, linewidth=1, label=name)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Estado (norm.)")
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend(loc="upper right")
    return fig

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Arquivo")
    file = st.file_uploader("CSV / TXT", type=["csv", "txt"])

    st.header("Formato de entrada")
    input_mode = st.selectbox(
        "Tipo de arquivo",
        ["Tenho eixos (tempo + X + Y + Z)", "Já tenho norma (tempo + norma)"],
        index=0
    )

    st.header("Colunas")
    if input_mode == "Tenho eixos (tempo + X + Y + Z)":
        col_t = st.text_input("Tempo (ms)", "DURACAO")
        col_x = st.text_input("Eixo X", "AVL EIXO X")
        col_y = st.text_input("Eixo Y", "AVL EIXO Y")
        col_z = st.text_input("Eixo Z", "AVL EIXO Z")
    else:
        col_t = st.text_input("Tempo (ms)", "tempo")
        col_norm = st.text_input("Norma", "norma")

    st.header("Pré-processamento")
    fs_target = st.number_input("Reamostragem alvo (Hz)", 20, 500, 100, 10)
    do_detr = st.checkbox("Detrend", True)

    st.header("Filtro")
    fc = st.number_input("Passa-baixa (Hz)", 0.1, 20.0, 1.5, 0.1)
    filt_order = st.number_input("Ordem Butterworth", 2, 8, 4, 1)

    st.header("Baseline e sequência")
    baseline_s = st.number_input("Janela baseline (s)", 0.2, 10.0, 2.0, 0.1)
    run_len = st.number_input("Duração mínima run (amostras)", 1, 200, 5, 1)
    n_base = st.number_input("N baseline consecutivos", 1, 1000, 5, 1)
    n_out = st.number_input("N não-baseline consecutivos", 1, 1000, 5, 1)

    st.header("Threshold binário")
    k_std_amp = st.number_input("k_amp (μ + k·σ)", 0.1, 20.0, 3.0, 0.5)
    k_std_der = st.number_input("k_der (μ + k·σ)", 0.1, 20.0, 3.0, 0.5)

    st.header("Bins (quantis)")
    use_bins = st.checkbox("Ativar bins", value=True)
    K_bins_amp = st.number_input("K bins (amplitude)", 3, 15, 5, 1)
    K_bins_der = st.number_input("K bins (|dx/dt|)", 3, 15, 5, 1)
    top_m_bins = st.number_input("Top-m baseline (bins)", 1, 5, 1, 1)

    st.header("K-means (semi-Markov)")
    Ks = st.multiselect("K candidatos", options=[3, 4, 5, 6, 7, 8], default=[4, 5])
    top_m_km = st.number_input("Top-m baseline (K-means)", 1, 5, 1, 1)

    st.header("Mostrar")
    show_thr_amp = st.checkbox("Threshold amplitude", True)
    show_thr_der = st.checkbox("Threshold derivada", True)
    show_bins_amp = st.checkbox("Bins amplitude", True)
    show_bins_der = st.checkbox("Bins derivada", True)
    show_kmeans = st.checkbox("K-means", True)

    run_btn = st.button("▶ Rodar", type="primary")

# ============================================================
# LOAD
# ============================================================
if not file:
    st.info("Envie um arquivo para começar.")
    st.stop()

df = read_table_any(file)
df.columns = df.columns.str.strip()

raw_df = None

if input_mode == "Tenho eixos (tempo + X + Y + Z)":
    for c in [col_t, col_x, col_y, col_z]:
        if c not in df.columns:
            st.error(f"Coluna '{c}' não encontrada. Disponíveis: {list(df.columns)}")
            st.stop()

    time_ms_raw = df[col_t].values.astype(float)
    x_raw = df[col_x].values.astype(float)
    y_raw = df[col_y].values.astype(float)
    z_raw = df[col_z].values.astype(float)

    raw_df = pd.DataFrame({
        "tempo_ms_raw": time_ms_raw,
        "x_raw": x_raw,
        "y_raw": y_raw,
        "z_raw": z_raw,
    })

    data = preprocess_xyz(
        t_ms=time_ms_raw,
        x=x_raw, y=y_raw, z=z_raw,
        fs_target=float(fs_target),
        do_detrend=bool(do_detr),
        fc=float(fc),
        order=int(filt_order),
    )

    t_s = data["t_s"]
    norma_filt = data["norma_filt"]
    abs_dxdt = data["abs_dxdt"]
    fs = data["fs"]

    # base_cols (TODOS com o mesmo tamanho do reamostrado)
    out_base_cols = {
        "tempo_ms": data["tempo_ms"],
        "t_s": t_s,
        "x_interp": data["x_interp"],
        "y_interp": data["y_interp"],
        "z_interp": data["z_interp"],
        "x_filt": data["x_filt"],
        "y_filt": data["y_filt"],
        "z_filt": data["z_filt"],
        "norma_filt": norma_filt,
        "abs_dxdt": abs_dxdt
    }

else:
    if col_t not in df.columns or col_norm not in df.columns:
        st.error(f"Colunas não encontradas. Disponíveis: {list(df.columns)}")
        st.stop()

    time_ms = df[col_t].values.astype(float)
    norma_raw = df[col_norm].values.astype(float)

    # infere fs e filtra
    dt_ms = np.median(np.diff(time_ms))
    if dt_ms <= 0:
        st.error("Tempo inválido (dt<=0).")
        st.stop()

    fs = 1000.0 / dt_ms
    t_s = time_ms / 1000.0
    norma_filt = lowpass(norma_raw, fs, float(fc), order=int(filt_order))
    abs_dxdt = np.abs(np.gradient(norma_filt, t_s))

    out_base_cols = {
        "tempo_ms": time_ms,
        "t_s": t_s,
        "norma_raw": norma_raw,
        "norma_filt": norma_filt,
        "abs_dxdt": abs_dxdt
    }

st.caption(f"fs final: **{fs:.2f} Hz** | N: **{len(t_s)}**")

base_mask = t_s <= float(baseline_s)
if base_mask.sum() < max(10, 2 * int(run_len)):
    st.warning("Poucas amostras na baseline (janela curta ou fs baixo).")

if not run_btn:
    st.pyplot(plot_signal(t_s, norma_filt, overlays={}, title="Pré-visualização: norma filtrada"))
    st.stop()

# ============================================================
# MÉTODOS
# ============================================================
methods = {}

def add_method(key, label, states_sm, K, base_states):
    C, P = transition_matrix(states_sm, K)
    labels = [f"S{i+1}" for i in range(K)]
    on, off = detect_seq(states_sm, base_states, int(n_base), int(n_out))
    methods[key] = {
        "label": label,
        "states": states_sm,
        "K": K,
        "baseline_states": np.atleast_1d(base_states),
        "on": on,
        "off": off,
        "C_df": pd.DataFrame(C, index=labels, columns=labels),
        "P_df": pd.DataFrame(P, index=labels, columns=labels),
        "chg_df": p_change_table(P).set_index(pd.Index(labels)),
        "emp_df": empirical_p_event(states_sm, base_states, int(n_base), int(n_out)),
    }

# 1) Threshold amplitude (binário)
mu0 = float(np.mean(norma_filt[base_mask]))
sd0 = float(np.std(norma_filt[base_mask], ddof=0) + 1e-12)
thr_amp = mu0 + float(k_std_amp) * sd0
st_thr_amp = (norma_filt > thr_amp).astype(int)
st_thr_amp_sm = merge_short_runs(st_thr_amp, int(run_len))
add_method("thr_amp", "Threshold amplitude (binário)", st_thr_amp_sm, 2, base_states=[0])

# 2) Threshold derivada (binário)
muD = float(np.mean(abs_dxdt[base_mask]))
sdD = float(np.std(abs_dxdt[base_mask], ddof=0) + 1e-12)
thr_der = muD + float(k_std_der) * sdD
st_thr_der = (abs_dxdt > thr_der).astype(int)
st_thr_der_sm = merge_short_runs(st_thr_der, int(run_len))
add_method("thr_der", "Threshold derivada |dx/dt| (binário)", st_thr_der_sm, 2, base_states=[0])

# 3) Bins multi-estados
if use_bins:
    # amplitude bins
    Kb = int(K_bins_amp)
    st_bins_amp, _ = discretize_bins(norma_filt, Kb)
    st_bins_amp_sm = merge_short_runs(st_bins_amp, int(run_len))
    base_states_bins_amp = dominant_states_in_window(st_bins_amp_sm, base_mask, top_m=int(top_m_bins))
    add_method("bins_amp", f"Bins amplitude (K={Kb})", st_bins_amp_sm, Kb, base_states_bins_amp)

    # derivada bins
    Kd = int(K_bins_der)
    st_bins_der, _ = discretize_bins(abs_dxdt, Kd)
    st_bins_der_sm = merge_short_runs(st_bins_der, int(run_len))
    base_states_bins_der = dominant_states_in_window(st_bins_der_sm, base_mask, top_m=int(top_m_bins))
    add_method("bins_der", f"Bins |dx/dt| (K={Kd})", st_bins_der_sm, Kd, base_states_bins_der)

# 4) K-means semi-Markov
if Ks:
    z = (norma_filt - np.mean(norma_filt)) / (np.std(norma_filt, ddof=0) + 1e-12)
    for K in Ks:
        K = int(K)
        km = KMeans(n_clusters=K, n_init=20, random_state=7)
        lab = km.fit_predict(z.reshape(-1, 1))

        centers = km.cluster_centers_.flatten()
        order = np.argsort(centers)  # S1 < S2 < ...
        inv = np.zeros_like(order)
        inv[order] = np.arange(K)
        states = inv[lab]

        states_sm = merge_short_runs(states, int(run_len))
        base_states_km = dominant_states_in_window(states_sm, base_mask, top_m=int(top_m_km))
        add_method(f"kmeans_K{K}", f"Semi-Markov K-means (K={K})", states_sm, K, base_states_km)

# ============================================================
# FILTRO DE EXIBIÇÃO
# ============================================================
display_keys = []
if show_thr_amp:
    display_keys.append("thr_amp")
if show_thr_der:
    display_keys.append("thr_der")
if use_bins and show_bins_amp and "bins_amp" in methods:
    display_keys.append("bins_amp")
if use_bins and show_bins_der and "bins_der" in methods:
    display_keys.append("bins_der")
if show_kmeans:
    display_keys += [k for k in methods.keys() if k.startswith("kmeans_K")]

display_keys = [k for k in display_keys if k in methods]

# ============================================================
# OUTPUT CSV (sem conflito de tamanhos)
# ============================================================
out = pd.DataFrame(out_base_cols)
for key, m in methods.items():
    out[f"state_{key}"] = m["states"]
    out[f"onset_{key}"] = 0
    out[f"offset_{key}"] = 0
    if m["on"] is not None:
        out.loc[m["on"], f"onset_{key}"] = 1
    if m["off"] is not None:
        out.loc[m["off"], f"offset_{key}"] = 1

# ============================================================
# UI
# ============================================================
tab0, tab1, tab2, tab3 = st.tabs(["🧪 Pré-processamento", "📈 Segmentação", "📊 Probabilidades", "📥 Downloads"])

with tab0:
    st.subheader("Checagem")
    st.pyplot(plot_signal(t_s, norma_filt, overlays={}, title="Norma filtrada (pronta para segmentação)"))
    if input_mode == "Tenho eixos (tempo + X + Y + Z)":
        fig, ax = plt.subplots(figsize=(12, 3.8))
        ax.plot(t_s, out["x_filt"], label="X filt")
        ax.plot(t_s, out["y_filt"], label="Y filt")
        ax.plot(t_s, out["z_filt"], label="Z filt")
        ax.set_title("Eixos filtrados (após interp + detrend + LP)")
        ax.set_xlabel("Tempo (s)")
        ax.grid(True, which="both")
        ax.legend(loc="upper right")
        st.pyplot(fig)

with tab1:
    st.subheader("Resumo")
    rows = []
    for key in display_keys:
        m = methods[key]
        rows.append({
            "Método": m["label"],
            "K": m["K"],
            "Baseline estados": ", ".join(map(str, m["baseline_states"])),
            "Início (s)": idx_to_time(t_s, m["on"]),
            "Fim (s)": idx_to_time(t_s, m["off"]),
            "Duração (s)": (idx_to_time(t_s, m["off"]) - idx_to_time(t_s, m["on"])) if (m["on"] is not None and m["off"] is not None) else np.nan,
            "#transições": int(np.sum(m["states"][1:] != m["states"][:-1])),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    overlays = {}
    for key in display_keys:
        m = methods[key]
        overlays[m["label"]] = {
            "on_s": idx_to_time(t_s, m["on"]),
            "off_s": idx_to_time(t_s, m["off"]),
            "style": "--" if "Threshold" in m["label"] else ":"
        }
    st.pyplot(plot_signal(t_s, norma_filt, overlays, "Norma + início/fim (métodos selecionados)"))

    st_states = {methods[k]["label"]: {"states": methods[k]["states"], "K": methods[k]["K"]} for k in display_keys}
    st.pyplot(plot_states_band(t_s, st_states, "Estados (normalizados)"))

with tab2:
    st.subheader("Probabilidades")
    for key in display_keys:
        m = methods[key]
        st.markdown(f"### {m['label']}")
        c1, c2, c3 = st.columns([1.6, 1.0, 1.2])
        with c1:
            st.caption("Matriz de contagens")
            st.dataframe(m["C_df"], use_container_width=True)
            st.caption("Matriz de transição (1 passo)")
            st.dataframe(m["P_df"].round(3), use_container_width=True)
        with c2:
            st.caption("P(mudar)")
            dfchg = m["chg_df"].copy()
            dfchg.index = m["P_df"].index
            st.dataframe(dfchg.round(3), use_container_width=True)
        with c3:
            st.caption("Evento por sequência (empírico)")
            st.dataframe(m["emp_df"].round(4), use_container_width=True)

with tab3:
    st.subheader("Downloads")

    st.download_button(
        "📥 Baixar CSV (processado + estados)",
        out.to_csv(index=False).encode("utf-8"),
        file_name="registro_segmentacao_completo.csv",
        mime="text/csv",
    )

    if raw_df is not None:
        st.download_button(
            "📥 Baixar CSV (bruto XYZ)",
            raw_df.to_csv(index=False).encode("utf-8"),
            file_name="registro_bruto_xyz.csv",
            mime="text/csv",
        )
