# markov_xyz_preprocess_peaks.py
# Streamlit app: preprocess (interp 100 Hz + detrend + LP 1.5 Hz) + states (K-means/quantis)
# + onset/offset global + 2 maiores picos entre onset..offset
# + delimitação de cada componente (início/fim) via busca retrógrada/anterógrada até baseline (estados Markov)

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.signal import butter, filtfilt, detrend, find_peaks

# ============================================================
# PAGE
# ============================================================
st.set_page_config(page_title="Markov – Transientes por picos", layout="wide")
st.title("📌 Markov / Semi-Markov — 2 transientes (picos) + limites por baseline")

# ============================================================
# IO
# ============================================================
def read_table_any(file):
    """Lê CSV/TXT tentando ';' e depois separador automático."""
    try:
        return pd.read_csv(file, sep=";")
    except Exception:
        file.seek(0)
        return pd.read_csv(file, sep=None, engine="python")

def find_first_existing(cols, candidates):
    cols_set = set([c.strip() for c in cols])
    for c in candidates:
        if c in cols_set:
            return c
    return None

# ============================================================
# PREPROCESS: interp(fs_target) + detrend + lowpass(fc)
# ============================================================
def lowpass(x, fs, fc, order=4):
    wn = fc / (fs / 2)
    if not (0 < wn < 1):
        raise ValueError("Cutoff inválido (normalizado fora de (0,1)).")
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x)

def resample_xyz(t_s, x, y, z, fs_target):
    t_s = np.asarray(t_s, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

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

def preprocess_xyz(t_ms, x, y, z, fs_target=100.0, do_detrend=True, fc=1.5, order=4):
    # tempo ms -> s
    t_s_raw = np.asarray(t_ms, dtype=float) / 1000.0

    # interp para fs_target
    t_s, x_i, y_i, z_i = resample_xyz(t_s_raw, x, y, z, float(fs_target))

    # detrend
    if do_detrend:
        x_i = detrend(x_i, type="linear")
        y_i = detrend(y_i, type="linear")
        z_i = detrend(z_i, type="linear")

    # lowpass
    x_f = lowpass(x_i, float(fs_target), float(fc), order=int(order))
    y_f = lowpass(y_i, float(fs_target), float(fc), order=int(order))
    z_f = lowpass(z_i, float(fs_target), float(fc), order=int(order))

    norma = np.sqrt(x_f * x_f + y_f * y_f + z_f * z_f)

    return {
        "t_s": t_s,
        "fs": float(fs_target),
        "x_f": x_f,
        "y_f": y_f,
        "z_f": z_f,
        "norma": norma,
    }

# ============================================================
# SEMI-MARKOV / MARKOV HELPERS
# ============================================================
def merge_short_runs(states, min_len):
    """Funde runs curtos (<min_len) no vizinho (semi-Markov hard)."""
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
                if i > 0:
                    states[i:j] = states[i - 1]
                elif j < len(states):
                    states[i:j] = states[j]
                changed = True
            i = j
    return states

def transition_matrix(states, K):
    states = np.asarray(states)
    C = np.zeros((K, K), dtype=int)
    for i in range(len(states) - 1):
        C[states[i], states[i + 1]] += 1
    row = C.sum(axis=1, keepdims=True)
    P = np.divide(C, row, out=np.zeros_like(C, dtype=float), where=row != 0)
    return C, P

def p_change_table(P):
    diag = np.diag(P)
    return pd.DataFrame({"P(permanecer)": diag, "P(mudar)": 1.0 - diag})

def dominant_states_in_window(states, mask, top_m=1):
    vc = pd.Series(states[mask]).value_counts()
    return vc.index[:top_m].to_numpy()

def detect_seq(states, base_states, n_base, n_out):
    """
    onset: n_base baseline -> n_out não-baseline
    offset: n_out não-baseline -> n_base baseline (última ocorrência)
    Retorna (onset_idx, offset_idx).
    """
    states = np.asarray(states)
    base_states = np.atleast_1d(base_states)
    is_base = np.isin(states, base_states)

    N = len(states)
    win = n_base + n_out

    onset = None
    for i in range(0, N - win + 1):
        if is_base[i : i + n_base].all() and (~is_base[i + n_base : i + win]).all():
            onset = i + n_base
            break

    offset = None
    for i in range(0, N - win + 1):
        if (~is_base[i : i + n_out]).all() and is_base[i + n_out : i + win].all():
            offset = i + n_out - 1

    return onset, offset

def empirical_p_event(states, base_states, n_base, n_out):
    """Probabilidade empírica do padrão onset/offset por janelas."""
    states = np.asarray(states)
    base_states = set(np.atleast_1d(base_states).tolist())
    is_base = np.isin(states, list(base_states))
    N = len(states)
    win = n_base + n_out

    opp_on = ev_on = opp_off = ev_off = 0
    for i in range(0, N - win + 1):
        if np.all(is_base[i : i + n_base]):
            opp_on += 1
            if np.all(~is_base[i + n_base : i + win]):
                ev_on += 1
        if np.all(~is_base[i : i + n_out]):
            opp_off += 1
            if np.all(is_base[i + n_out : i + win]):
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

# ============================================================
# DISCRETIZAÇÃO (estados)
# ============================================================
def discretize_quantile_bins(x, K):
    """K estados por quantis (0..K-1)."""
    x = np.asarray(x, dtype=float)
    qs = np.linspace(0, 1, K + 1)
    edges = np.unique(np.quantile(x, qs))
    if len(edges) < 3:
        edges = np.linspace(np.min(x), np.max(x), K + 1)
    s = np.digitize(x, edges[1:-1], right=False)
    return s, edges

def discretize_kmeans_1d(x, K, random_state=7):
    """K-means em 1D, reordenando estados por centro (S1 < S2 < ...)."""
    x = np.asarray(x, dtype=float)
    z = (x - np.mean(x)) / (np.std(x, ddof=0) + 1e-12)
    km = KMeans(n_clusters=K, n_init=20, random_state=random_state)
    lab = km.fit_predict(z.reshape(-1, 1))

    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)  # do menor centro para o maior
    inv = np.zeros_like(order)
    inv[order] = np.arange(K)
    states = inv[lab]
    return states, centers[order]

# ============================================================
# Picos + limites dos componentes via baseline (Markov)
# ============================================================
def two_largest_peaks(y, i0, i1, fs, min_dist_s=0.30, prominence=None):
    """
    Retorna índices dos 2 maiores picos (por altura) no intervalo [i0, i1].
    min_dist_s controla distância mínima entre picos.
    prominence (opcional) filtra picos pequenos.
    """
    if i0 is None or i1 is None or i1 <= i0 + 3:
        return []

    seg = y[i0 : i1 + 1]
    min_dist = max(1, int(min_dist_s * fs))

    peaks, _props = find_peaks(seg, distance=min_dist, prominence=prominence)
    if len(peaks) == 0:
        return []

    heights = seg[peaks]
    order = np.argsort(heights)[::-1]  # maiores primeiro
    top = peaks[order[:2]]
    top_abs = (top + i0).tolist()
    top_abs.sort()  # no tempo
    return top_abs

def find_baseline_run_backward(is_base, start_idx, n_base):
    """Procura para trás run baseline (n_base) terminando em j."""
    j = int(start_idx)
    n_base = int(n_base)
    while j >= n_base - 1:
        if is_base[j - n_base + 1 : j + 1].all():
            return j
        j -= 1
    return None

def find_baseline_run_forward(is_base, start_idx, n_base):
    """Procura para frente run baseline (n_base) começando em j."""
    N = len(is_base)
    j = int(start_idx)
    n_base = int(n_base)
    while j <= N - n_base:
        if is_base[j : j + n_base].all():
            return j
        j += 1
    return None

def component_bounds_from_peak(states_sm, base_states, peak_idx, n_base=5, clamp=(None, None)):
    """
    A partir de um pico, busca retrógrada/anterógrada por retorno à baseline.
    start_idx: primeiro índice após uma run baseline (n_base) antes do pico
    end_idx: último índice antes de uma run baseline (n_base) após o pico
    clamp=(i0,i1) restringe a busca.
    """
    states_sm = np.asarray(states_sm)
    base_states = np.atleast_1d(base_states)
    is_base = np.isin(states_sm, base_states)

    N = len(states_sm)
    i0, i1 = clamp
    if i0 is None:
        i0 = 0
    if i1 is None:
        i1 = N - 1

    peak_idx = int(np.clip(int(peak_idx), int(i0), int(i1)))

    j_back = find_baseline_run_backward(is_base, peak_idx, int(n_base))
    if j_back is None or j_back < i0:
        start_idx = int(i0)
    else:
        start_idx = int(min(i1, j_back + 1))

    j_fwd = find_baseline_run_forward(is_base, peak_idx, int(n_base))
    if j_fwd is None or j_fwd > i1:
        end_idx = int(i1)
    else:
        end_idx = int(max(i0, j_fwd - 1))

    if end_idx < start_idx:
        start_idx = peak_idx
        end_idx = peak_idx

    return start_idx, end_idx

# ============================================================
# PLOTS
# ============================================================
def idx_to_time(t_s, idx):
    return float(t_s[idx]) if idx is not None else np.nan

def plot_states(t_s, states, K, title="Estados (normalizados)"):
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.plot(t_s, states / max(1, (K - 1)), linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("estado (norm.)")
    ax.grid(True, which="both")
    return fig

def plot_signal_marks_components(t_s, y, onset_idx, offset_idx, peaks, comps, title, ylabel):
    """
    peaks: [p1, p2] (índices)
    comps: [(s1,e1,p1),(s2,e2,p2)] com índices
    """
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(t_s, y, linewidth=1, label="sinal")

    if onset_idx is not None:
        ax.axvline(t_s[onset_idx], linestyle="--", label="início global")
    if offset_idx is not None:
        ax.axvline(t_s[offset_idx], linestyle="--", label="fim global")

    for i, p in enumerate(peaks, start=1):
        ax.axvline(t_s[p], linestyle=":", linewidth=1.2, label=f"pico {i}")

    for i, (s, e, _p) in enumerate(comps, start=1):
        ax.axvline(t_s[s], linestyle="-.", linewidth=1.2, label=f"comp{i} início")
        ax.axvline(t_s[e], linestyle="-.", linewidth=1.2, label=f"comp{i} fim")

    ax.set_title(title)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both")
    ax.legend(loc="upper right")
    return fig

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Arquivo")
    file = st.file_uploader("TXT/CSV (tempo + X + Y + Z)", type=["txt", "csv"])

    st.header("Colunas (default do seu arquivo)")
    col_t = st.text_input("Tempo (ms)", "DURACAO")
    col_x = st.text_input("Eixo X", "AVL EIXO X")
    col_y = st.text_input("Eixo Y", "AVL EIXO Y")
    col_z = st.text_input("Eixo Z", "AVL EIXO Z")

    st.header("Pré-processamento (igual ao pipeline original)")
    fs_target = st.number_input("Reamostragem (Hz)", 20, 500, 100, 10)
    do_detr = st.checkbox("Detrend (linear)", True)
    fc = st.number_input("Passa-baixa (Hz)", 0.1, 20.0, 1.5, 0.1)
    filt_order = st.number_input("Ordem Butterworth", 2, 8, 4, 1)

    st.header("Sinal 1D para Markov")
    sig_choice = st.selectbox("Escolha", ["Norma", "Eixo X", "Eixo Y", "Eixo Z"], index=0)
    use_abs_der = st.checkbox("Usar |d(sinal)/dt|", False)

    st.header("Baseline / Semi-Markov / Sequência global")
    baseline_s = st.number_input("Janela baseline (s)", 0.05, 30.0, 2.0, 0.1)
    top_m_base = st.number_input("Top-m estados baseline", 1, 5, 1, 1)

    min_run = st.number_input("min_run (amostras)", 1, 500, 5, 1)
    n_base = st.number_input("n_base (consecutivos baseline)", 1, 2000, 5, 1)
    n_out = st.number_input("n_out (consecutivos fora baseline)", 1, 2000, 5, 1)

    st.header("Discretização")
    method = st.selectbox("Método", ["Quantis (bins)", "K-means"], index=1)
    Ks = st.multiselect("K para testar", options=list(range(2, 11)), default=[4, 5])

    st.header("2 Transientes (picos) dentro de onset..offset")
    min_peak_dist_s = st.number_input("Distância mínima entre picos (s)", 0.05, 5.0, 0.30, 0.05)
    use_prom = st.checkbox("Usar prominence (filtrar picos pequenos)", False)
    prom_factor = st.number_input(
        "Prominence (fração do range no intervalo)",
        0.0,
        1.0,
        0.10,
        0.01,
        disabled=not use_prom,
    )

    run_btn = st.button("▶ Rodar", type="primary")

# ============================================================
# LOAD + PREPROCESS
# ============================================================
if not file:
    st.info("Envie um arquivo para começar.")
    st.stop()

df = read_table_any(file)
df.columns = df.columns.str.strip()

# auto-detect leve
if col_t not in df.columns:
    auto_t = find_first_existing(df.columns, ["DURACAO", "tempo", "Tempo", "time", "TIME"])
    if auto_t:
        col_t = auto_t
if col_x not in df.columns:
    auto_x = find_first_existing(df.columns, ["AVL EIXO X", "X", "x", "gyro_x", "gyr_x"])
    if auto_x:
        col_x = auto_x
if col_y not in df.columns:
    auto_y = find_first_existing(df.columns, ["AVL EIXO Y", "Y", "y", "gyro_y", "gyr_y"])
    if auto_y:
        col_y = auto_y
if col_z not in df.columns:
    auto_z = find_first_existing(df.columns, ["AVL EIXO Z", "Z", "z", "gyro_z", "gyr_z"])
    if auto_z:
        col_z = auto_z

missing = [c for c in [col_t, col_x, col_y, col_z] if c not in df.columns]
if missing:
    st.error(f"Colunas ausentes: {missing}. Disponíveis: {list(df.columns)}")
    st.stop()

time_ms = df[col_t].to_numpy(dtype=float)
x_raw = df[col_x].to_numpy(dtype=float)
y_raw = df[col_y].to_numpy(dtype=float)
z_raw = df[col_z].to_numpy(dtype=float)

try:
    data = preprocess_xyz(
        t_ms=time_ms,
        x=x_raw,
        y=y_raw,
        z=z_raw,
        fs_target=float(fs_target),
        do_detrend=bool(do_detr),
        fc=float(fc),
        order=int(filt_order),
    )
except Exception as e:
    st.error(f"Erro no pré-processamento: {e}")
    st.stop()

t_s = data["t_s"]
fs = data["fs"]
x_f, y_f, z_f = data["x_f"], data["y_f"], data["z_f"]
norma = data["norma"]

if sig_choice == "Norma":
    sig0 = norma
elif sig_choice == "Eixo X":
    sig0 = x_f
elif sig_choice == "Eixo Y":
    sig0 = y_f
else:
    sig0 = z_f

sig = np.abs(np.gradient(sig0, t_s)) if use_abs_der else sig0
sig_name = ("abs_d(" + sig_choice + ")/dt") if use_abs_der else sig_choice

st.caption(
    f"✅ Preprocess: interp→{fs_target}Hz | detrend={do_detr} | LP(fc={fc}Hz, ordem={filt_order}) "
    f"| fs_final=**{fs:.2f}Hz** | N=**{len(t_s)}** | sinal=**{sig_name}**"
)

base_mask = t_s <= float(baseline_s)
if base_mask.sum() < max(10, 2 * int(min_run)):
    st.warning("Poucas amostras na baseline (aumente baseline_s ou reduza min_run).")

# prévia
if not run_btn:
    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.plot(t_s, sig, linewidth=1)
    ax.set_title("Prévia do sinal 1D (após preprocess)")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel(sig_name)
    ax.grid(True, which="both")
    st.pyplot(fig)
    st.stop()

# ============================================================
# PROCESSA POR K
# ============================================================
tab_titles = [f"K={K}" for K in Ks] + ["📌 Comparativo", "📥 Download"]
tabs = st.tabs(tab_titles)

results = []
out = pd.DataFrame(
    {
        "t_s": t_s,
        "x_f": x_f,
        "y_f": y_f,
        "z_f": z_f,
        "norma": norma,
        "sig_used": sig,
    }
)

for idx_tab, K in enumerate(Ks):
    K = int(K)

    # discretiza
    if method == "Quantis (bins)":
        states, edges = discretize_quantile_bins(sig, K)
        extra = f"edges≈{np.round(edges, 6)}"
    else:
        states, centers_sorted = discretize_kmeans_1d(sig, K)
        extra = f"centers≈{np.round(centers_sorted, 6)}"

    # semi-markov
    states_sm = merge_short_runs(states, int(min_run))

    # baseline states (top-m)
    base_states = dominant_states_in_window(states_sm, base_mask, top_m=int(top_m_base))

    # onset/offset global (sequência)
    on, off = detect_seq(states_sm, base_states, int(n_base), int(n_out))
    on_s = idx_to_time(t_s, on)
    off_s = idx_to_time(t_s, off)
    dur_s = (off_s - on_s) if (on is not None and off is not None) else np.nan

    # picos e componentes no intervalo onset..offset
    peaks = []
    comps = []  # [(s,e,p), ...]
    if on is not None and off is not None and off > on + max(10, int(0.3 * fs)):
        prom = None
        if use_prom:
            seg = sig[on : off + 1]
            prom = float(prom_factor) * float(np.max(seg) - np.min(seg) + 1e-12)

        peaks = two_largest_peaks(
            sig, on, off, fs,
            min_dist_s=float(min_peak_dist_s),
            prominence=prom
        )

        for p in peaks:
            s_i, e_i = component_bounds_from_peak(
                states_sm=states_sm,
                base_states=base_states,
                peak_idx=p,
                n_base=int(n_base),      # run baseline exigida (igual ao seu critério)
                clamp=(on, off)
            )
            comps.append((s_i, e_i, p))

        # garante ordem temporal dos componentes
        comps.sort(key=lambda x: x[2])

    # matrizes
    C, P = transition_matrix(states_sm, K)
    labels = [f"S{i+1}" for i in range(K)]
    C_df = pd.DataFrame(C, index=labels, columns=labels)
    P_df = pd.DataFrame(P, index=labels, columns=labels)
    chg_df = p_change_table(P)
    chg_df.index = labels
    emp_df = empirical_p_event(states_sm, base_states, int(n_base), int(n_out))

    # salva no out
    out[f"state_K{K}"] = states_sm
    out[f"onset_K{K}"] = 0
    out[f"offset_K{K}"] = 0
    if on is not None:
        out.loc[on, f"onset_K{K}"] = 1
    if off is not None:
        out.loc[off, f"offset_K{K}"] = 1

    # marca picos/componentes (2)
    for j in [1, 2]:
        out[f"peak{j}_K{K}"] = 0
        out[f"comp{j}_start_K{K}"] = 0
        out[f"comp{j}_end_K{K}"] = 0

    if len(comps) >= 1:
        s1, e1, p1 = comps[0]
        out.loc[p1, f"peak1_K{K}"] = 1
        out.loc[s1, f"comp1_start_K{K}"] = 1
        out.loc[e1, f"comp1_end_K{K}"] = 1

    if len(comps) >= 2:
        s2, e2, p2 = comps[1]
        out.loc[p2, f"peak2_K{K}"] = 1
        out.loc[s2, f"comp2_start_K{K}"] = 1
        out.loc[e2, f"comp2_end_K{K}"] = 1

    # resumo
    row = {
        "K": K,
        "Método": method,
        "Baseline states": ", ".join(map(str, base_states)),
        "Início global (s)": on_s,
        "Fim global (s)": off_s,
        "Duração global (s)": dur_s,
        "#transições": int(np.sum(states_sm[1:] != states_sm[:-1])),
        "Picos detectados": len(peaks),
    }

    # comp1/comp2 tempos
    if len(comps) >= 1:
        s1, e1, p1 = comps[0]
        row.update({
            "Pico1 (s)": float(t_s[p1]),
            "Comp1 início (s)": float(t_s[s1]),
            "Comp1 fim (s)": float(t_s[e1]),
            "Comp1 duração (s)": float(t_s[e1] - t_s[s1]),
        })
    else:
        row.update({"Pico1 (s)": np.nan, "Comp1 início (s)": np.nan, "Comp1 fim (s)": np.nan, "Comp1 duração (s)": np.nan})

    if len(comps) >= 2:
        s2, e2, p2 = comps[1]
        row.update({
            "Pico2 (s)": float(t_s[p2]),
            "Comp2 início (s)": float(t_s[s2]),
            "Comp2 fim (s)": float(t_s[e2]),
            "Comp2 duração (s)": float(t_s[e2] - t_s[s2]),
        })
    else:
        row.update({"Pico2 (s)": np.nan, "Comp2 início (s)": np.nan, "Comp2 fim (s)": np.nan, "Comp2 duração (s)": np.nan})

    row["Extra"] = extra
    results.append(row)

    # ============================================================
    # UI por K
    # ============================================================
    with tabs[idx_tab]:
        st.subheader(f"{method} | {extra}")
        st.write(
            f"**Baseline states (top-m):** {', '.join(map(str, base_states))}  \n"
            f"**Início global:** {on_s:.3f}s | **Fim global:** {off_s:.3f}s | **Duração:** {dur_s:.3f}s"
            if np.isfinite(on_s) and np.isfinite(off_s)
            else f"**Baseline states (top-m):** {', '.join(map(str, base_states))}  \n"
                 f"**Início/Fim global:** não detectado com (n_base={n_base}, n_out={n_out})"
        )

        st.pyplot(
            plot_signal_marks_components(
                t_s=t_s,
                y=sig,
                onset_idx=on,
                offset_idx=off,
                peaks=peaks,
                comps=comps,
                title="Sinal + onset/offset + 2 picos + limites dos componentes (baseline via estados)",
                ylabel=sig_name
            )
        )

        st.pyplot(plot_states(t_s, states_sm, K, title="Estados (semi-Markov)"))

        c1, c2, c3 = st.columns([1.6, 1.0, 1.2])
        with c1:
            st.caption("Matriz de contagens")
            st.dataframe(C_df, use_container_width=True)
            st.caption("Matriz de transição (1 passo)")
            st.dataframe(P_df.round(3), use_container_width=True)
        with c2:
            st.caption("P(mudar) por estado")
            st.dataframe(chg_df.round(3), use_container_width=True)
        with c3:
            st.caption("Evento global por sequência (empírico)")
            st.dataframe(emp_df.round(4), use_container_width=True)

        st.caption(
            "Componentes: após achar cada pico dentro de [onset..offset], "
            "busca retrógrada/anterógrada no vetor de estados (semi-Markov) até encontrar uma run de baseline "
            f"com tamanho n_base={n_base}. O início é após a run anterior; o fim é antes da run posterior."
        )

# ============================================================
# TAB: comparativo
# ============================================================
with tabs[len(Ks)]:
    st.subheader("Comparativo (todos os K)")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

# ============================================================
# TAB: download
# ============================================================
with tabs[len(Ks) + 1]:
    st.subheader("Download")
    st.download_button(
        "📥 Baixar CSV (preprocess + estados + onset/offset + picos + componentes)",
        out.to_csv(index=False).encode("utf-8"),
        file_name="markov_preprocess_peaks_components.csv",
        mime="text/csv",
    )
