import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.cluster import KMeans

st.set_page_config(page_title="iTUG Segmentação: Limiar + Semi-Markov", layout="wide")
st.title("📱 iTUG: Segmentação por Limiar (amplitude/derivada) e Semi-Markov (K-means + duração)")

# -----------------------------
# Helpers
# -----------------------------
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

def detect_onset_offset_5plus5(states: np.ndarray, baseline_states, run: int = 5):
    states = np.asarray(states)
    baseline_states = set(np.atleast_1d(baseline_states).tolist())
    is_base = np.isin(states, list(baseline_states))
    N = len(states)

    onset_idx = None
    for i in range(0, N - 2 * run + 1):
        if np.all(is_base[i : i + run]) and np.all(~is_base[i + run : i + 2 * run]):
            onset_idx = i + run
            break

    offset_idx = None
    for i in range(0, N - 2 * run + 1):
        if np.all(~is_base[i : i + run]) and np.all(is_base[i + run : i + 2 * run]):
            offset_idx = i + run - 1
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

def empirical_p_event(states: np.ndarray, baseline_states, run: int = 5) -> pd.DataFrame:
    states = np.asarray(states)
    baseline_states = set(np.atleast_1d(baseline_states).tolist())
    is_base = np.isin(states, list(baseline_states))
    N = len(states)

    opp_on = ev_on = opp_off = ev_off = 0
    for i in range(0, N - 2 * run + 1):
        if np.all(is_base[i : i + run]):
            opp_on += 1
            if np.all(~is_base[i + run : i + 2 * run]):
                ev_on += 1
        if np.all(~is_base[i : i + run]):
            opp_off += 1
            if np.all(is_base[i + run : i + 2 * run]):
                ev_off += 1

    return pd.DataFrame(
        {
            "Oportunidades": [opp_on, opp_off],
            "Eventos (5+5)": [ev_on, ev_off],
            "Probabilidade empírica": [
                ev_on / opp_on if opp_on else np.nan,
                ev_off / opp_off if opp_off else np.nan,
            ],
        },
        index=["Onset: baseline→não-baseline", "Offset: não-baseline→baseline"],
    )

def discretize_bins(xv: np.ndarray, n_states: int):
    qs = np.linspace(0, 1, n_states + 1)
    edges = np.quantile(xv, qs)
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(np.min(xv), np.max(xv), n_states + 1)
    states = np.digitize(xv, edges[1:-1], right=False)
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
        if row.sum() <= 0:
            continue
        p = row / row.sum()
        Hs.append(float(-np.sum(p * np.log2(p + eps))))
    return float(np.mean(Hs)) if Hs else np.nan

def plot_signal(t_s, x_filt, overlays, title):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(t_s, x_filt, label="norma filtrada")
    for name, d in overlays.items():
        on_s = d.get("on_s", np.nan)
        off_s = d.get("off_s", np.nan)
        style = d.get("style", "--")
        if np.isfinite(on_s):
            ax.axvline(on_s, linestyle=style, label=f"início ({name})")
        if np.isfinite(off_s):
            ax.axvline(off_s, linestyle=style, label=f"fim ({name})")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Norma (filtrada)")
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

def score_k_method(m, t_s, w_trans, w_ent, w_invalid, w_p55):
    """Score maior = melhor."""
    P = m["P_df"].to_numpy(dtype=float)
    ent = state_entropy(P)
    trans = n_transitions(m["states"])
    dur = activity_duration_seconds(t_s, m["on"], m["off"])
    invalid = 1.0 if (not np.isfinite(dur)) else 0.0
    p_on = float(m["emp_df"].loc["Onset: baseline→não-baseline", "Probabilidade empírica"])
    p_off = float(m["emp_df"].loc["Offset: não-baseline→baseline", "Probabilidade empírica"])
    p_on = 0.0 if not np.isfinite(p_on) else p_on
    p_off = 0.0 if not np.isfinite(p_off) else p_off

    # Normalizações simples (robustas o suficiente)
    trans_pen = trans
    ent_pen = 0.0 if not np.isfinite(ent) else ent

    score = (w_p55 * (p_on + p_off)) - (w_trans * trans_pen) - (w_ent * ent_pen) - (w_invalid * invalid)
    return score, dict(ent=ent, trans=trans, dur=dur, p_on=p_on, p_off=p_off, invalid=invalid)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Entrada")
    file = st.file_uploader("Upload CSV", type=["csv"])

    st.header("Colunas")
    col_time = st.text_input("Tempo (ms)", value="tempo")
    col_norm = st.text_input("Norma", value="norma")

    st.header("Pré-processamento")
    cutoff_hz = st.number_input("Filtro LP (Hz)", 0.1, 20.0, 1.5, 0.1)
    filt_order = st.number_input("Ordem Butterworth", 2, 8, 4, 1)
    baseline_window_s = st.number_input("Janela baseline (s)", 0.2, 10.0, 2.0, 0.1)

    st.header("Critério temporal")
    run_len = st.number_input("Duração mínima (amostras) / critério 5+5", 2, 50, 5, 1)

    st.header("Limiar (amplitude/derivada)")
    k_std_amp = st.number_input("k_amp", 0.5, 10.0, 3.0, 0.5)
    k_std_der = st.number_input("k_der", 0.5, 10.0, 3.0, 0.5)

    st.header("Threshold multi-estados (bins)")
    thr_multistate = st.checkbox("Ativar bins (multi-estados)", value=True)
    n_bins_thr = st.number_input("K bins (amplitude)", 3, 10, 5, 1)
    n_bins_der = st.number_input("K bins (|dx/dt|)", 3, 10, 5, 1)
    baseline_top_m_thr = st.number_input("Top-m baseline (bins)", 1, 5, 1, 1)

    st.header("Semi-Markov (K-means)")
    ks = st.multiselect("K (estados)", options=[3, 4, 5, 6, 7, 8], default=[4, 5])
    baseline_top_m_km = st.number_input("Top-m baseline (K-means)", 1, 5, 1, 1)

    st.header("O que mostrar")
    show_thr_amp_bin = st.checkbox("Mostrar: limiar amplitude (binário)", value=True)
    show_thr_der_bin = st.checkbox("Mostrar: limiar derivada (binário)", value=True)
    show_thr_bins_amp = st.checkbox("Mostrar: bins amplitude", value=True)
    show_thr_bins_der = st.checkbox("Mostrar: bins |dx/dt|", value=True)
    show_kmeans = st.checkbox("Mostrar: semi-Markov K-means", value=True)

    st.header("Modo automático (escolher melhor K)")
    auto_pick_k = st.checkbox("Ativar seleção automática do melhor K (K-means)", value=True)
    st.caption("Score = +w_p55*(P_on+P_off) - w_trans*(#transições) - w_ent*(entropia) - w_invalid*(inválido)")
    w_trans = st.number_input("w_trans", 0.0, 5.0, 0.05, 0.01)
    w_ent = st.number_input("w_ent", 0.0, 5.0, 0.20, 0.05)
    w_invalid = st.number_input("w_invalid", 0.0, 100.0, 10.0, 1.0)
    w_p55 = st.number_input("w_p55", 0.0, 10.0, 2.0, 0.5)

    run_btn = st.button("▶️ Rodar", type="primary")

# -----------------------------
# Load data
# -----------------------------
if not file:
    st.info("Faça upload de um CSV com colunas de tempo (ms) e norma.")
    st.stop()

try:
    df = pd.read_csv(file)
except Exception:
    file.seek(0)
    df = pd.read_csv(file, sep=None, engine="python")

if col_time not in df.columns or col_norm not in df.columns:
    st.error(f"Colunas não encontradas. Disponíveis: {list(df.columns)}")
    st.stop()

time_ms = df[col_time].values.astype(float)
x_raw = df[col_norm].values.astype(float)

fs = infer_fs_from_time_ms(time_ms)
t_s = time_ms / 1000.0

x_filt = lowpass_filter(x_raw, fs=fs, cutoff_hz=float(cutoff_hz), order=int(filt_order))
dxdt = np.gradient(x_filt, t_s)
abs_dxdt = np.abs(dxdt)

st.caption(f"fs estimada: **{fs:.2f} Hz** | N: **{len(df)}**")

base_mask = t_s <= float(baseline_window_s)
if base_mask.sum() < max(10, 2 * int(run_len)):
    st.warning("Poucas amostras na baseline. Considere aumentar a janela baseline.")

if not run_btn:
    st.pyplot(plot_signal(t_s, x_filt, overlays={}, title="Pré-visualização: norma filtrada"))
    st.stop()

# -----------------------------
# Run methods
# -----------------------------
methods = {}

# A1: threshold amplitude (binary)
mu0 = x_filt[base_mask].mean()
sd0 = x_filt[base_mask].std(ddof=0) + 1e-12
thr_amp = mu0 + float(k_std_amp) * sd0
st_amp_bin = (x_filt > thr_amp).astype(int)
st_amp_bin_sm = merge_short_runs(st_amp_bin, min_len=int(run_len))
on_amp_bin, off_amp_bin = detect_onset_offset_5plus5(st_amp_bin_sm, baseline_states=[0], run=int(run_len))
_, P = transition_matrix(st_amp_bin_sm, 2)
labels2 = ["Rest(0)", "Active(1)"]
methods["thr_amp_bin"] = dict(
    label="Limiar amplitude (binário)",
    states=st_amp_bin_sm, K=2,
    baseline_states=np.array([0]),
    on=on_amp_bin, off=off_amp_bin,
    P_df=pd.DataFrame(P, index=labels2, columns=labels2),
    chg_df=p_change_table(P, labels2),
    emp_df=empirical_p_event(st_amp_bin_sm, baseline_states=[0], run=int(run_len)),
)

# A2: threshold derivative (binary)
muD = abs_dxdt[base_mask].mean()
sdD = abs_dxdt[base_mask].std(ddof=0) + 1e-12
thr_der = muD + float(k_std_der) * sdD
st_der_bin = (abs_dxdt > thr_der).astype(int)
st_der_bin_sm = merge_short_runs(st_der_bin, min_len=int(run_len))
on_der_bin, off_der_bin = detect_onset_offset_5plus5(st_der_bin_sm, baseline_states=[0], run=int(run_len))
_, P = transition_matrix(st_der_bin_sm, 2)
methods["thr_der_bin"] = dict(
    label="Limiar derivada |dx/dt| (binário)",
    states=st_der_bin_sm, K=2,
    baseline_states=np.array([0]),
    on=on_der_bin, off=off_der_bin,
    P_df=pd.DataFrame(P, index=labels2, columns=labels2),
    chg_df=p_change_table(P, labels2),
    emp_df=empirical_p_event(st_der_bin_sm, baseline_states=[0], run=int(run_len)),
)

# threshold multi-state bins
if thr_multistate:
    Kb = int(n_bins_thr)
    st_amp_bins, _ = discretize_bins(x_filt, Kb)
    st_amp_bins_sm = merge_short_runs(st_amp_bins, min_len=int(run_len))
    base_states_amp_bins = dominant_states_in_window(st_amp_bins_sm, base_mask, top_m=int(baseline_top_m_thr))
    on_amp_bins, off_amp_bins = detect_onset_offset_5plus5(st_amp_bins_sm, base_states_amp_bins, run=int(run_len))
    _, P = transition_matrix(st_amp_bins_sm, Kb)
    labelsKb = [f"S{k+1}" for k in range(Kb)]
    methods["thr_amp_bins"] = dict(
        label=f"Bins amplitude (K={Kb})",
        states=st_amp_bins_sm, K=Kb,
        baseline_states=base_states_amp_bins,
        on=on_amp_bins, off=off_amp_bins,
        P_df=pd.DataFrame(P, index=labelsKb, columns=labelsKb),
        chg_df=p_change_table(P, labelsKb),
        emp_df=empirical_p_event(st_amp_bins_sm, base_states_amp_bins, run=int(run_len)),
    )

    Kd = int(n_bins_der)
    st_der_bins, _ = discretize_bins(abs_dxdt, Kd)
    st_der_bins_sm = merge_short_runs(st_der_bins, min_len=int(run_len))
    base_states_der_bins = dominant_states_in_window(st_der_bins_sm, base_mask, top_m=int(baseline_top_m_thr))
    on_der_bins, off_der_bins = detect_onset_offset_5plus5(st_der_bins_sm, base_states_der_bins, run=int(run_len))
    _, P = transition_matrix(st_der_bins_sm, Kd)
    labelsKd = [f"S{k+1}" for k in range(Kd)]
    methods["thr_der_bins"] = dict(
        label=f"Bins |dx/dt| (K={Kd})",
        states=st_der_bins_sm, K=Kd,
        baseline_states=base_states_der_bins,
        on=on_der_bins, off=off_der_bins,
        P_df=pd.DataFrame(P, index=labelsKd, columns=labelsKd),
        chg_df=p_change_table(P, labelsKd),
        emp_df=empirical_p_event(st_der_bins_sm, base_states_der_bins, run=int(run_len)),
    )

# semi-Markov K-means (candidates for auto-pick)
kmeans_keys = []
if ks:
    x_z = (x_filt - x_filt.mean()) / (x_filt.std(ddof=0) + 1e-12)
    for K in ks:
        K = int(K)
        km = KMeans(n_clusters=K, n_init=20, random_state=7)
        lab = km.fit_predict(x_z.reshape(-1, 1))

        centers = km.cluster_centers_.flatten()
        order = np.argsort(centers)
        inv = np.zeros_like(order)
        inv[order] = np.arange(K)
        states = inv[lab]

        states_sm = merge_short_runs(states, min_len=int(run_len))
        base_states = dominant_states_in_window(states_sm, base_mask, top_m=int(baseline_top_m_km))
        on_i, off_i = detect_onset_offset_5plus5(states_sm, base_states, run=int(run_len))

        _, P = transition_matrix(states_sm, n_states=K)
        labelsK = [f"S{k+1}" for k in range(K)]
        key = f"semi_kmeans_K{K}"
        kmeans_keys.append(key)
        methods[key] = dict(
            label=f"Semi-Markov K-means (K={K})",
            states=states_sm, K=K,
            baseline_states=base_states,
            on=on_i, off=off_i,
            P_df=pd.DataFrame(P, index=labelsK, columns=labelsK),
            chg_df=p_change_table(P, labelsK),
            emp_df=empirical_p_event(states_sm, base_states, run=int(run_len)),
        )

# auto-pick best K
best_k_key = None
score_rows = []
if auto_pick_k and kmeans_keys:
    for key in kmeans_keys:
        m = methods[key]
        sc, comps = score_k_method(m, t_s, w_trans, w_ent, w_invalid, w_p55)
        score_rows.append({
            "Método": m["label"],
            "Score": sc,
            "K": m["K"],
            "Entropia": comps["ent"],
            "#transições": comps["trans"],
            "Duração(s)": comps["dur"],
            "P_on(5+5)": comps["p_on"],
            "P_off(5+5)": comps["p_off"],
            "Inválido": comps["invalid"],
        })
    score_df = pd.DataFrame(score_rows).sort_values("Score", ascending=False)
    best_k_key = score_df.iloc[0]["Método"]
else:
    score_df = None

# -----------------------------
# Display filtering (checkboxes)
# -----------------------------
display_keys = []
if show_thr_amp_bin:
    display_keys.append("thr_amp_bin")
if show_thr_der_bin:
    display_keys.append("thr_der_bin")
if thr_multistate and show_thr_bins_amp and "thr_amp_bins" in methods:
    display_keys.append("thr_amp_bins")
if thr_multistate and show_thr_bins_der and "thr_der_bins" in methods:
    display_keys.append("thr_der_bins")
if show_kmeans:
    display_keys += [k for k in methods.keys() if k.startswith("semi_kmeans_K")]

display_keys = [k for k in display_keys if k in methods]

# -----------------------------
# Register for download
# -----------------------------
out = pd.DataFrame({
    "tempo_ms": time_ms,
    "t_s": t_s,
    "norma_raw": x_raw,
    "norma_filt": x_filt,
    "abs_dxdt": abs_dxdt,
})
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
tab1, tab2, tab3, tab4 = st.tabs(["📈 Segmentação", "📊 Probabilidades", "🧮 Comparação", "🤖 Auto K"])

with tab1:
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
            style="--" if "Limiar" in m["label"] else ":",
        )
    st.pyplot(plot_signal(t_s, x_filt, overlays, "Norma filtrada + onset/offset (métodos selecionados)"))

    states_for_band = {methods[k]["label"]: {"states": methods[k]["states"], "K": methods[k]["K"]} for k in display_keys}
    st.pyplot(plot_states_band(t_s, states_for_band, "Sequência de estados (normalizada)"))

with tab2:
    for key in display_keys:
        m = methods[key]
        st.markdown(f"### {m['label']}")
        c1, c2, c3 = st.columns([1.4, 1.0, 1.2])
        with c1:
            st.caption("Matriz de transição (1 passo)")
            st.dataframe(m["P_df"].round(3), use_container_width=True)
        with c2:
            st.caption("P(mudar) por estado")
            st.dataframe(m["chg_df"].round(3), use_container_width=True)
        with c3:
            st.caption("Prob. empírica do evento 5+5")
            st.dataframe(m["emp_df"].round(4), use_container_width=True)

with tab3:
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
            "P_on(5+5)": float(m["emp_df"].loc["Onset: baseline→não-baseline", "Probabilidade empírica"]),
            "P_off(5+5)": float(m["emp_df"].loc["Offset: não-baseline→baseline", "Probabilidade empírica"]),
        })
    st.dataframe(pd.DataFrame(metrics), use_container_width=True)

with tab4:
    st.write("Seleção automática considera apenas **Semi-Markov (K-means)**.")
    if score_df is None or score_df.empty:
        st.info("Ative K-means e escolha pelo menos um K para usar a seleção automática.")
    else:
        st.subheader("Ranking por score (maior = melhor)")
        st.dataframe(score_df, use_container_width=True)

        best_row = score_df.iloc[0]
        st.success(f"✅ Melhor K escolhido: **K={int(best_row['K'])}** ({best_row['Método']})")

# -----------------------------
# Download
# -----------------------------
st.subheader("📥 Download do registro (estados e marcadores)")
csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button(
    "Baixar CSV",
    data=csv_bytes,
    file_name="registro_segmentacao_completo.csv",
    mime="text/csv",
)
