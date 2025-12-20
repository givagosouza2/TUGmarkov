# app.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, detrend
from sklearn.cluster import KMeans


st.set_page_config(page_title="iTUG Segmentação: Threshold + Semi-Markov", layout="wide")
st.title("📱 iTUG: Threshold (amplitude/derivada) + Multi-estados (bins) + Semi-Markov (K-means)")

# -----------------------------
# Helpers (IO / preprocess)
# -----------------------------
def read_table_any(file) -> pd.DataFrame:
    """Lê CSV/TXT tentando ; e separador automático."""
    try:
        return pd.read_csv(file, sep=";")
    except Exception:
        file.seek(0)
        try:
            return pd.read_csv(file, sep=None, engine="python")
        except Exception:
            file.seek(0)
            return pd.read_csv(file)

def infer_fs_from_time_ms(time_ms: np.ndarray) -> float:
    dt_ms = np.median(np.diff(time_ms.astype(float)))
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("Não foi possível inferir fs: verifique a coluna de tempo (ms).")
    return 1000.0 / dt_ms

def lowpass_filter(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    wn = cutoff_hz / (fs / 2)
    if wn <= 0 or wn >= 1:
        raise ValueError("Cutoff inválido (normalizado fora de (0,1)).")
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x.astype(float))

def resample_to_fs(t_s: np.ndarray, y: np.ndarray, fs_target: float):
    """Interpolação linear para grade uniforme."""
    t_s = np.asarray(t_s, dtype=float)
    y = np.asarray(y, dtype=float)

    order = np.argsort(t_s)
    t_s = t_s[order]
    y = y[order]

    # remove tempos duplicados
    uniq = np.diff(t_s, prepend=t_s[0] - 1) > 0
    t_s = t_s[uniq]
    y = y[uniq]

    dt = 1.0 / fs_target
    t_new = np.arange(t_s[0], t_s[-1] + 1e-12, dt)
    y_new = np.interp(t_new, t_s, y)
    return t_new, y_new

def preprocess_axes_to_norm(
    t_ms: np.ndarray,
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    fs_target: float,
    do_detrend: bool,
    cutoff_hz: float,
    filt_order: int
):
    """
    Pipeline:
    1) tempo (ms) -> t_s
    2) interpolação para fs_target (100 Hz por padrão)
    3) detrend (opcional)
    4) filtro LP 1.5 Hz (ou o que você escolher) em cada eixo
    5) norma = sqrt(x^2 + y^2 + z^2)
    """
    t_s = np.asarray(t_ms, dtype=float) / 1000.0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    t_s_new, x_i = resample_to_fs(t_s, x, fs_target)
    _, y_i = resample_to_fs(t_s, y, fs_target)
    _, z_i = resample_to_fs(t_s, z, fs_target)

    if do_detrend:
        x_i = detrend(x_i, type="linear")
        y_i = detrend(y_i, type="linear")
        z_i = detrend(z_i, type="linear")

    x_f = lowpass_filter(x_i, fs=fs_target, cutoff_hz=cutoff_hz, order=filt_order)
    y_f = lowpass_filter(y_i, fs=fs_target, cutoff_hz=cutoff_hz, order=filt_order)
    z_f = lowpass_filter(z_i, fs=fs_target, cutoff_hz=cutoff_hz, order=filt_order)

    norma = np.sqrt(x_f**2 + y_f**2 + z_f**2)
    dxdt = np.gradient(norma, t_s_new)
    abs_dxdt = np.abs(dxdt)

    time_ms_new = t_s_new * 1000.0
    return time_ms_new, t_s_new, norma, abs_dxdt, x_i, y_i, z_i, x_f, y_f, z_f


# -----------------------------
# Helpers (states / semi-markov)
# -----------------------------
def runs_from_labels(labels: np.ndarray):
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    starts = [0]
    ends = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            ends.append(i - 1)
            starts.append(i)
    ends.append(len(labels) - 1)
    return [(s, e, labels[s]) for s, e in zip(starts, ends)]

def merge_short_runs(labels: np.ndarray, min_len: int = 5) -> np.ndarray:
    """Funde runs curtos (<min_len) ao vizinho mais longo (suavização tipo semi-Markov)."""
    labels = np.asarray(labels).copy()
    if len(labels) == 0:
        return labels
    changed = True
    while changed:
        changed = False
        runs = runs_from_labels(labels)
        if len(runs) <= 1:
            break
        for idx, (s, e, lab) in enumerate(runs):
            L = e - s + 1
            if L >= min_len:
                continue
            left = runs[idx - 1] if idx - 1 >= 0 else None
            right = runs[idx + 1] if idx + 1 < len(runs) else None

            if left is None and right is None:
                continue
            if left is None:
                new_lab = right[2]
            elif right is None:
                new_lab = left[2]
            else:
                left_len = left[1] - left[0] + 1
                right_len = right[1] - right[0] + 1
                new_lab = left[2] if left_len >= right_len else right[2]

            labels[s : e + 1] = new_lab
            changed = True
            break
    return labels

def dominant_states_in_window(states: np.ndarray, mask: np.ndarray, top_m: int = 1) -> np.ndarray:
    vc = pd.Series(states[mask]).value_counts()
    return vc.index[:top_m].to_numpy()

def detect_onset_offset_seq(states: np.ndarray, baseline_states, n_base: int = 5, n_out: int = 5):
    """
    onset:  n_base baseline consecutivos -> n_out não-baseline consecutivos
    offset: n_out não-baseline consecutivos -> n_base baseline consecutivos (última ocorrência)
    """
    states = np.asarray(states)
    baseline_states = set(np.atleast_1d(baseline_states).tolist())
    is_base = np.isin(states, list(baseline_states))

    N = len(states)
    win = n_base + n_out

    onset_idx = None
    for i in range(0, N - win + 1):
        if np.all(is_base[i:i+n_base]) and np.all(~is_base[i+n_base:i+win]):
            onset_idx = i + n_base
            break

    offset_idx = None
    for i in range(0, N - win + 1):
        if np.all(~is_base[i:i+n_out]) and np.all(is_base[i+n_out:i+win]):
            offset_idx = i + n_out - 1

    return onset_idx, offset_idx

def transition_matrix(states: np.ndarray, n_states: int):
    states = np.asarray(states)
    counts = np.zeros((n_states, n_states), dtype=int)
    for i in range(len(states) - 1):
        counts[states[i], states[i + 1]] += 1
    row = counts.sum(axis=1, keepdims=True)
    P = np.divide(counts, row, out=np.zeros_like(counts, dtype=float), where=row != 0)
    return counts, P

def p_change_table(P: np.ndarray, labels):
    diag = np.diag(P)
    return pd.DataFrame({"P(permanecer)": diag, "P(mudar)": 1 - diag}, index=labels)

def empirical_p_event(states: np.ndarray, baseline_states, n_base: int, n_out: int) -> pd.DataFrame:
    """Probabilidade empírica do evento n_base + n_out."""
    states = np.asarray(states)
    baseline_states = set(np.atleast_1d(baseline_states).tolist())
    is_base = np.isin(states, list(baseline_states))
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

def discretize_bins(xv: np.ndarray, n_states: int):
    """Discretização por quantis (robusta)."""
    qs = np.linspace(0, 1, n_states + 1)
    edges = np.quantile(xv, qs)
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(np.min(xv), np.max(xv), n_states + 1)
    states = np.digitize(xv, edges[1:-1], right=False)  # 0..K-1
    return states, edges

def idx_to_time(t_s: np.ndarray, idx):
    return float(t_s[idx]) if idx is not None else np.nan

def n_transitions(states: np.ndarray) -> int:
    s = np.asarray(states)
    if len(s) < 2:
        return 0
    return int(np.sum(s[1:] != s[:-1]))

def activity_duration_seconds(t_s: np.ndarray, onset_idx, offset_idx) -> float:
    if onset_idx is None or offset_idx is None:
        return np.nan
    if offset_idx <= onset_idx:
        return np.nan
    return float(t_s[offset_idx] - t_s[onset_idx])

def state_entropy(P: np.ndarray) -> float:
    eps = 1e-12
    Hs = []
    for i in range(P.shape[0]):
        row = P[i]
        s = row.sum()
        if s <= 0:
            continue
        p = row / s
        Hs.append(float(-np.sum(p * np.log2(p + eps))))
    return float(np.mean(Hs)) if Hs else np.nan

# -----------------------------
# Plot helpers
# -----------------------------
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
        states = d["states"]
        K = d["K"]
        y = states / max(1, (K - 1))
        ax.plot(t_s, y, linewidth=1, label=name)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Estado (norm.)")
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend(loc="upper right")
    return fig

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Entrada")
    file = st.file_uploader("Upload CSV/TXT", type=["csv", "txt"])

    st.header("Formato de entrada")
    input_mode = st.selectbox(
        "Tipo de arquivo",
        ["Já tenho norma (tempo + norma)", "Tenho eixos (tempo + X + Y + Z)"],
        index=1
    )

    st.header("Colunas")
    if input_mode == "Já tenho norma (tempo + norma)":
        col_time = st.text_input("Tempo (ms)", value="tempo")
        col_norm = st.text_input("Norma", value="norma")
    else:
        col_time = st.text_input("Tempo (ms)", value="DURACAO")
        col_x = st.text_input("Eixo X", value="AVL EIXO X")
        col_y = st.text_input("Eixo Y", value="AVL EIXO Y")
        col_z = st.text_input("Eixo Z", value="AVL EIXO Z")

    st.header("Reamostragem / Pré-processamento")
    target_fs = st.number_input("Frequência alvo (Hz)", 10, 500, 100, 10)
    do_detr = st.checkbox("Aplicar detrend (remover tendência)", value=True)

    st.header("Filtro")
    cutoff_hz = st.number_input("Filtro passa-baixa (Hz)", 0.1, 20.0, 1.5, 0.1)
    filt_order = st.number_input("Ordem Butterworth", 2, 8, 4, 1)

    st.header("Baseline e sequência (detecção)")
    baseline_window_s = st.number_input("Janela baseline (s)", 0.2, 10.0, 2.0, 0.1)

    st.subheader("Limpeza semi-Markov (runs curtos)")
    run_len = st.number_input("Duração mínima de run (amostras)", 1, 500, 5, 1)

    st.subheader("Critério de detecção por sequência")
    n_base = st.number_input("Onset: N baseline consecutivos", 1, 1000, 5, 1)
    n_out = st.number_input("Onset: N não-baseline consecutivos", 1, 1000, 5, 1)
    use_diff_for_offset = st.checkbox("Usar critérios diferentes para FIM (offset)", value=False)
    if use_diff_for_offset:
        n_out_off = st.number_input("Offset: N não-baseline consecutivos", 1, 1000, int(n_out), 1)
        n_base_off = st.number_input("Offset: N baseline consecutivos", 1, 1000, int(n_base), 1)
    else:
        n_out_off, n_base_off = int(n_out), int(n_base)

    st.header("Threshold (binário)")
    k_std_amp = st.number_input("k_amp (μ + k·σ)", 0.1, 20.0, 3.0, 0.5)
    k_std_der = st.number_input("k_der (μ + k·σ)", 0.1, 20.0, 3.0, 0.5)

    st.header("Bins (multi-estados)")
    use_bins = st.checkbox("Ativar bins", value=True)
    n_bins_amp = st.number_input("K bins (amplitude)", 3, 15, 5, 1)
    n_bins_der = st.number_input("K bins (|dx/dt|)", 3, 15, 5, 1)
    baseline_top_m_bins = st.number_input("Top-m baseline (bins)", 1, 5, 1, 1)

    st.header("K-means (semi-Markov)")
    ks = st.multiselect("K candidatos", options=[3, 4, 5, 6, 7, 8], default=[4, 5])
    baseline_top_m_km = st.number_input("Top-m baseline (K-means)", 1, 5, 1, 1)

    st.header("O que mostrar")
    show_thr_amp = st.checkbox("Mostrar: threshold amplitude", value=True)
    show_thr_der = st.checkbox("Mostrar: threshold derivada", value=True)
    show_bins_amp = st.checkbox("Mostrar: bins amplitude", value=True)
    show_bins_der = st.checkbox("Mostrar: bins derivada", value=True)
    show_kmeans = st.checkbox("Mostrar: K-means", value=True)

    run_btn = st.button("▶️ Rodar", type="primary")


# -----------------------------
# Load & build (tempo_ms, t_s, norma_filt, abs_dxdt)
# -----------------------------
if not file:
    st.info("Faça upload de um arquivo CSV/TXT para começar.")
    st.stop()

df = read_table_any(file)
df.columns = [c.strip() for c in df.columns]

# Pré-processamento conforme modo de entrada
if input_mode == "Já tenho norma (tempo + norma)":
    if col_time not in df.columns or col_norm not in df.columns:
        st.error(f"Colunas não encontradas. Disponíveis: {list(df.columns)}")
        st.stop()

    time_ms = df[col_time].values.astype(float)
    norma_raw = df[col_norm].values.astype(float)

    fs = infer_fs_from_time_ms(time_ms)
    t_s = time_ms / 1000.0

    norma_filt = lowpass_filter(norma_raw, fs=fs, cutoff_hz=float(cutoff_hz), order=int(filt_order))
    abs_dxdt = np.abs(np.gradient(norma_filt, t_s))

    debug_axes = None
    out_base_cols = {
        "tempo_ms": time_ms,
        "t_s": t_s,
        "norma_raw": norma_raw,
        "norma_filt": norma_filt,
        "abs_dxdt": abs_dxdt
    }

else:
    for c in [col_time, col_x, col_y, col_z]:
        if c not in df.columns:
            st.error(f"Coluna '{c}' não encontrada. Disponíveis: {list(df.columns)}")
            st.stop()

    time_ms_raw = df[col_time].values.astype(float)
    x_raw = df[col_x].values.astype(float)
    y_raw = df[col_y].values.astype(float)
    z_raw = df[col_z].values.astype(float)

    # interp -> detrend -> LP -> norma
    time_ms, t_s, norma_filt, abs_dxdt, x_i, y_i, z_i, x_f, y_f, z_f = preprocess_axes_to_norm(
        t_ms=time_ms_raw,
        x=x_raw, y=y_raw, z=z_raw,
        fs_target=float(target_fs),
        do_detrend=bool(do_detr),
        cutoff_hz=float(cutoff_hz),
        filt_order=int(filt_order),
    )
    fs = float(target_fs)

    debug_axes = {
        "t_s": t_s,
        "x_i": x_i, "y_i": y_i, "z_i": z_i,
        "x_f": x_f, "y_f": y_f, "z_f": z_f,
    }

    out_base_cols = {
        "tempo_ms": time_ms,
        "t_s": t_s,
        "x_raw": x_raw,
        "y_raw": y_raw,
        "z_raw": z_raw,
        "x_interp": x_i,
        "y_interp": y_i,
        "z_interp": z_i,
        "x_filt": x_f,
        "y_filt": y_f,
        "z_filt": z_f,
        "norma_filt": norma_filt,
        "abs_dxdt": abs_dxdt
    }

st.caption(f"fs usada: **{fs:.2f} Hz** | N: **{len(t_s)}**")

base_mask = t_s <= float(baseline_window_s)
if base_mask.sum() < max(10, 2 * int(run_len)):
    st.warning("Poucas amostras na baseline. Considere aumentar a janela baseline.")

if not run_btn:
    st.pyplot(plot_signal(t_s, norma_filt, overlays={}, title="Pré-visualização: norma filtrada"))
    if debug_axes is not None:
        st.info("Você está no modo eixos: rode para ver segmentações. (Você pode também checar os eixos em um gráfico depois.)")
    st.stop()


# -----------------------------
# Métodos
# -----------------------------
methods = {}

def compute_on_off(states_sm, base_states):
    on, _ = detect_onset_offset_seq(states_sm, base_states, n_base=int(n_base), n_out=int(n_out))
    _, off = detect_onset_offset_seq(states_sm, base_states, n_base=int(n_base_off), n_out=int(n_out_off))
    return on, off

def add_method(key, label, states_sm, K, base_states):
    _, P = transition_matrix(states_sm, K)
    labels = [f"S{i+1}" for i in range(K)]
    methods[key] = dict(
        label=label,
        states=states_sm,
        K=K,
        baseline_states=np.atleast_1d(base_states),
        on=None,
        off=None,
        P_df=pd.DataFrame(P, index=labels, columns=labels),
        chg_df=p_change_table(P, labels),
        emp_df=empirical_p_event(states_sm, base_states, int(n_base), int(n_out)),
    )
    on, off = compute_on_off(states_sm, base_states)
    methods[key]["on"] = on
    methods[key]["off"] = off

# Threshold amplitude (binário)
mu0 = norma_filt[base_mask].mean()
sd0 = norma_filt[base_mask].std(ddof=0) + 1e-12
thr_amp = mu0 + float(k_std_amp) * sd0
st_amp = (norma_filt > thr_amp).astype(int)
st_amp_sm = merge_short_runs(st_amp, min_len=int(run_len))
add_method("thr_amp_bin", "Threshold amplitude (binário)", st_amp_sm, 2, base_states=[0])

# Threshold derivada (binário)
muD = abs_dxdt[base_mask].mean()
sdD = abs_dxdt[base_mask].std(ddof=0) + 1e-12
thr_der = muD + float(k_std_der) * sdD
st_der = (abs_dxdt > thr_der).astype(int)
st_der_sm = merge_short_runs(st_der, min_len=int(run_len))
add_method("thr_der_bin", "Threshold derivada |dx/dt| (binário)", st_der_sm, 2, base_states=[0])

# Bins (multi-estados)
if use_bins:
    # amplitude bins
    Kb = int(n_bins_amp)
    st_bins_amp, _ = discretize_bins(norma_filt, Kb)
    st_bins_amp_sm = merge_short_runs(st_bins_amp, min_len=int(run_len))
    base_states_amp = dominant_states_in_window(st_bins_amp_sm, base_mask, top_m=int(baseline_top_m_bins))
    add_method("bins_amp", f"Bins amplitude (K={Kb})", st_bins_amp_sm, Kb, base_states=base_states_amp)

    # derivative bins
    Kd = int(n_bins_der)
    st_bins_der, _ = discretize_bins(abs_dxdt, Kd)
    st_bins_der_sm = merge_short_runs(st_bins_der, min_len=int(run_len))
    base_states_der = dominant_states_in_window(st_bins_der_sm, base_mask, top_m=int(baseline_top_m_bins))
    add_method("bins_der", f"Bins |dx/dt| (K={Kd})", st_bins_der_sm, Kd, base_states=base_states_der)

# K-means semi-Markov
kmeans_keys = []
if ks:
    z = (norma_filt - norma_filt.mean()) / (norma_filt.std(ddof=0) + 1e-12)
    for K in ks:
        K = int(K)
        km = KMeans(n_clusters=K, n_init=20, random_state=7)
        lab = km.fit_predict(z.reshape(-1, 1))

        centers = km.cluster_centers_.flatten()
        order = np.argsort(centers)  # garante S1 < S2 < ...
        inv = np.zeros_like(order)
        inv[order] = np.arange(K)
        states = inv[lab]

        states_sm = merge_short_runs(states, min_len=int(run_len))
        base_states = dominant_states_in_window(states_sm, base_mask, top_m=int(baseline_top_m_km))

        key = f"semi_kmeans_K{K}"
        kmeans_keys.append(key)
        add_method(key, f"Semi-Markov K-means (K={K})", states_sm, K, base_states=base_states)


# -----------------------------
# O que mostrar
# -----------------------------
display_keys = []
if show_thr_amp:
    display_keys.append("thr_amp_bin")
if show_thr_der:
    display_keys.append("thr_der_bin")
if use_bins and show_bins_amp and "bins_amp" in methods:
    display_keys.append("bins_amp")
if use_bins and show_bins_der and "bins_der" in methods:
    display_keys.append("bins_der")
if show_kmeans:
    display_keys += [k for k in methods.keys() if k.startswith("semi_kmeans_K")]

display_keys = [k for k in display_keys if k in methods]


# -----------------------------
# Registro para download
# -----------------------------
out = pd.DataFrame(out_base_cols)
for key, m in methods.items():
    out[f"state_{key}"] = m["states"]
    out[f"onset_{key}"] = 0
    out[f"offset_{key}"] = 0
    if m["on"] is not None:
        out.loc[m["on"], f"onset_{key}"] = 1
    if m["off"] is not None:
        out.loc[m["off"], f"offset_{key}"] = 1


# -----------------------------
# Tabs
# -----------------------------
tab0, tab1, tab2, tab3 = st.tabs(["🧪 Pré-processamento", "📈 Segmentação", "📊 Probabilidades", "🧮 Comparação"])

with tab0:
    st.subheader("Checagem do pré-processamento")
    st.write(
        f"- fs final: **{fs:.2f} Hz** | LP: **{cutoff_hz:.2f} Hz** | detrend: **{do_detr}** | alvo: **{target_fs} Hz** (se modo eixos)"
    )
    st.pyplot(plot_signal(t_s, norma_filt, overlays={}, title="Norma filtrada (pronta para segmentação)"))

    if debug_axes is not None:
        fig, ax = plt.subplots(figsize=(12, 3.8))
        ax.plot(debug_axes["t_s"], debug_axes["x_f"], label="X filtrado")
        ax.plot(debug_axes["t_s"], debug_axes["y_f"], label="Y filtrado")
        ax.plot(debug_axes["t_s"], debug_axes["z_f"], label="Z filtrado")
        ax.set_title("Eixos filtrados (após interp + detrend + LP)")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Unidade do sensor")
        ax.grid(True, which="both")
        ax.legend(loc="upper right")
        st.pyplot(fig)

with tab1:
    st.subheader("Resumo (métodos selecionados)")
    rows = []
    for key in display_keys:
        m = methods[key]
        rows.append({
            "Método": m["label"],
            "K": m["K"],
            "Baseline estados": ", ".join([str(s) for s in np.atleast_1d(m["baseline_states"])]),
            "Início (s)": idx_to_time(t_s, m["on"]),
            "Fim (s)": idx_to_time(t_s, m["off"]),
            "Duração (s)": activity_duration_seconds(t_s, m["on"], m["off"]),
            "#transições": n_transitions(m["states"]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    overlays = {}
    for key in display_keys:
        m = methods[key]
        overlays[m["label"]] = dict(
            on_s=idx_to_time(t_s, m["on"]),
            off_s=idx_to_time(t_s, m["off"]),
            style="--" if "Threshold" in m["label"] else ":",
        )

    st.pyplot(plot_signal(t_s, norma_filt, overlays, "Norma filtrada + onset/offset (métodos selecionados)"))

    states_for_band = {methods[k]["label"]: {"states": methods[k]["states"], "K": methods[k]["K"]} for k in display_keys}
    st.pyplot(plot_states_band(t_s, states_for_band, "Sequência de estados (normalizada) — comparação visual"))

    with st.expander("Parâmetros usados"):
        st.write({
            "baseline_window_s": float(baseline_window_s),
            "run_len (limpeza runs)": int(run_len),
            "onset n_base": int(n_base),
            "onset n_out": int(n_out),
            "offset n_out": int(n_out_off),
            "offset n_base": int(n_base_off),
            "k_std_amp": float(k_std_amp),
            "k_std_der": float(k_std_der),
            "bins_K_amp": int(n_bins_amp),
            "bins_K_der": int(n_bins_der),
            "bins_top_m_baseline": int(baseline_top_m_bins),
            "kmeans_Ks": ks,
            "kmeans_top_m_baseline": int(baseline_top_m_km),
        })

with tab2:
    st.subheader("Probabilidades (métodos selecionados)")
    for key in display_keys:
        m = methods[key]
        st.markdown(f"### {m['label']}")
        c1, c2, c3 = st.columns([1.4, 1.0, 1.3])
        with c1:
            st.caption("Matriz de transição (1 passo)")
            st.dataframe(m["P_df"].round(3), use_container_width=True)
        with c2:
            st.caption("P(mudar) por estado")
            st.dataframe(m["chg_df"].round(3), use_container_width=True)
        with c3:
            st.caption("Evento por sequência (sustentado)")
            st.dataframe(m["emp_df"].round(4), use_container_width=True)

with tab3:
    st.subheader("Comparação quantitativa (métodos selecionados)")
    metrics = []
    for key in display_keys:
        m = methods[key]
        P = m["P_df"].to_numpy(dtype=float)
        metrics.append({
            "Método": m["label"],
            "K": m["K"],
            "Duração (s)": activity_duration_seconds(t_s, m["on"], m["off"]),
            "#transições": n_transitions(m["states"]),
            "Entropia (bits)": state_entropy(P),
            "P_on(seq)": float(m["emp_df"]["Probabilidade empírica"].iloc[0]),
            "P_off(seq)": float(m["emp_df"]["Probabilidade empírica"].iloc[1]),
        })
    st.dataframe(pd.DataFrame(metrics), use_container_width=True)


# -----------------------------
# Download
# -----------------------------
st.subheader("📥 Download do registro (pré-processado + estados + marcadores)")
csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button(
    "Baixar CSV",
    data=csv_bytes,
    file_name="registro_segmentacao_completo.csv",
    mime="text/csv",
)
