# markov_simplificado_xyz.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

st.set_page_config(page_title="Markov – Segmentação (XYZ)", layout="wide")
st.title("📌 Markov / Semi-Markov — Segmentação a partir de arquivo XYZ (seu formato)")

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
        C[states[i], states[i+1]] += 1
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

def empirical_p_event(states, base_states, n_base, n_out):
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
        index=[
            f"Onset: base({n_base})→out({n_out})",
            f"Offset: out({n_out})→base({n_base})",
        ],
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
    order = np.argsort(centers)
    inv = np.zeros_like(order)
    inv[order] = np.arange(K)
    states = inv[lab]
    return states, centers[order]

def idx_to_time(t_s, idx):
    return float(t_s[idx]) if idx is not None else np.nan

def plot_signal_with_marks(t_s, y, on_s, off_s, title, ylabel="Sinal"):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(t_s, y, linewidth=1)
    if np.isfinite(on_s):
        ax.axvline(on_s, linestyle="--", label="início")
    if np.isfinite(off_s):
        ax.axvline(off_s, linestyle="--", label="fim")
    ax.set_title(title)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both")
    ax.legend(loc="upper right")
    return fig

def plot_states(t_s, states, K, title="Estados (normalizados)"):
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.plot(t_s, states / max(1, (K - 1)), linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("estado (norm.)")
    ax.grid(True, which="both")
    return fig

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Arquivo")
    file = st.file_uploader("TXT/CSV (seu modelo)", type=["txt", "csv"])

    st.header("Colunas (defaults compatíveis com seu arquivo)")
    col_t = st.text_input("Tempo (ms)", "DURACAO")
    col_x = st.text_input("Eixo X", "AVL EIXO X")
    col_y = st.text_input("Eixo Y", "AVL EIXO Y")
    col_z = st.text_input("Eixo Z", "AVL EIXO Z")

    st.header("Sinal 1D para discretizar")
    signal_mode = st.selectbox(
        "Escolha o sinal",
        ["Norma (sqrt(x^2+y^2+z^2))", "Eixo X", "Eixo Y", "Eixo Z", "|d(sinal)/dt| (da escolha acima)"],
        index=0,
    )
    base_signal_for_der = st.selectbox(
        "Se escolher |d/dt|, derivar de:",
        ["Norma", "X", "Y", "Z"],
        index=0,
        help="Só usado quando o modo for |d(sinal)/dt|",
    )

    st.header("Baseline")
    baseline_s = st.number_input("Janela baseline (s)", 0.05, 30.0, 2.0, 0.1)
    top_m_base = st.number_input("Top-m estados baseline", 1, 5, 1, 1)

    st.header("Semi-Markov")
    min_run = st.number_input("min_run (amostras)", 1, 500, 5, 1)

    st.header("Critério de sequência")
    n_base = st.number_input("n_base", 1, 2000, 5, 1)
    n_out = st.number_input("n_out", 1, 2000, 5, 1)

    st.header("Discretização")
    method = st.selectbox("Método", ["Quantis (bins)", "K-means"], index=0)
    Ks = st.multiselect("K para testar", options=list(range(2, 11)), default=[3, 4, 5])

    run_btn = st.button("▶ Rodar", type="primary")

# ============================================================
# LOAD + PREP DO SINAL 1D
# ============================================================
if not file:
    st.info("Envie um arquivo para começar.")
    st.stop()

df = read_table_any(file)
df.columns = df.columns.str.strip()

# tenta auto-detect se o usuário não tiver exatamente igual
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
x = df[col_x].to_numpy(dtype=float)
y = df[col_y].to_numpy(dtype=float)
z = df[col_z].to_numpy(dtype=float)

t_s = time_ms / 1000.0
dt = np.median(np.diff(t_s))
if dt <= 0:
    st.error("Tempo inválido (dt<=0).")
    st.stop()
fs = 1.0 / dt

norma = np.sqrt(x*x + y*y + z*z)

# escolhe sinal base
if signal_mode == "Norma (sqrt(x^2+y^2+z^2))":
    sig = norma
    sig_name = "norma"
elif signal_mode == "Eixo X":
    sig = x
    sig_name = "x"
elif signal_mode == "Eixo Y":
    sig = y
    sig_name = "y"
elif signal_mode == "Eixo Z":
    sig = z
    sig_name = "z"
else:
    # derivada: pega base escolhida
    base = {"Norma": norma, "X": x, "Y": y, "Z": z}[base_signal_for_der]
    sig = np.abs(np.gradient(base, t_s))
    sig_name = f"abs_d({base_signal_for_der})/dt"

st.caption(f"Arquivo OK | fs≈**{fs:.2f} Hz** | N=**{len(t_s)}** | sinal=**{sig_name}**")

base_mask = t_s <= float(baseline_s)
if base_mask.sum() < max(10, 2 * int(min_run)):
    st.warning("Poucas amostras na baseline (aumente baseline_s ou reduza min_run).")

if not run_btn:
    st.pyplot(plot_signal_with_marks(t_s, sig, np.nan, np.nan, "Prévia do sinal 1D", ylabel=sig_name))
    st.stop()

# ============================================================
# RODA PARA CADA K
# ============================================================
results = []
tabs = st.tabs([f"K={K}" for K in Ks] + ["📥 Download"])

out = pd.DataFrame(
    {
        "tempo_ms": time_ms,
        "t_s": t_s,
        "x": x,
        "y": y,
        "z": z,
        "norma": norma,
        "sig_used": sig,
    }
)

for idx_tab, K in enumerate(Ks):
    K = int(K)

    if method == "Quantis (bins)":
        states, edges = discretize_quantile_bins(sig, K)
        extra = f"edges≈{np.round(edges, 6)}"
    else:
        states, centers_sorted = discretize_kmeans_1d(sig, K)
        extra = f"centers≈{np.round(centers_sorted, 6)}"

    states_sm = merge_short_runs(states, int(min_run))
    base_states = dominant_states_in_window(states_sm, base_mask, top_m=int(top_m_base))

    on, off = detect_seq(states_sm, base_states, int(n_base), int(n_out))
    on_s = idx_to_time(t_s, on)
    off_s = idx_to_time(t_s, off)
    dur = (off_s - on_s) if (on is not None and off is not None) else np.nan

    C, P = transition_matrix(states_sm, K)
    labels = [f"S{i+1}" for i in range(K)]
    C_df = pd.DataFrame(C, index=labels, columns=labels)
    P_df = pd.DataFrame(P, index=labels, columns=labels)
    chg_df = p_change_table(P)
    chg_df.index = labels
    emp_df = empirical_p_event(states_sm, base_states, int(n_base), int(n_out))

    results.append(
        {
            "K": K,
            "baseline_states": ", ".join(map(str, base_states)),
            "onset_s": on_s,
            "offset_s": off_s,
            "dur_s": dur,
            "n_transitions": int(np.sum(states_sm[1:] != states_sm[:-1])),
            "extra": extra,
        }
    )

    out[f"state_K{K}"] = states_sm
    out[f"onset_K{K}"] = 0
    out[f"offset_K{K}"] = 0
    if on is not None:
        out.loc[on, f"onset_K{K}"] = 1
    if off is not None:
        out.loc[off, f"offset_K{K}"] = 1

    with tabs[idx_tab]:
        st.subheader(f"{method} | {extra}")
        st.write(
            f"**Baseline states (top-m):** {', '.join(map(str, base_states))}  \n"
            f"**Início:** {on_s:.3f}s | **Fim:** {off_s:.3f}s | **Duração:** {dur:.3f}s"
            if np.isfinite(on_s) and np.isfinite(off_s)
            else f"**Baseline states (top-m):** {', '.join(map(str, base_states))}  \n"
                 f"**Início/Fim:** não detectado com (n_base={n_base}, n_out={n_out})"
        )

        st.pyplot(plot_signal_with_marks(t_s, sig, on_s, off_s, "Sinal + marcas", ylabel=sig_name))
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
            st.caption("Evento por sequência (empírico)")
            st.dataframe(emp_df.round(4), use_container_width=True)

with tabs[-1]:
    st.subheader("Resumo comparativo")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.download_button(
        "📥 Baixar CSV (XYZ + sinal + estados + onset/offset)",
        out.to_csv(index=False).encode("utf-8"),
        file_name="markov_xyz_estados_eventos.csv",
        mime="text/csv",
    )
