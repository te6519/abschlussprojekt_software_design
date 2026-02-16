import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import io
from graph4 import System, create_mbb_beam, plot_structure

st.set_page_config(page_title="Topology Optimizer", layout="wide")
st.title("Topologieoptimierung")

if "loaded_data" not in st.session_state:
    st.session_state.loaded_data = None


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:

    st.header("1. Gitter-Einstellungen")
    width  = st.slider("Breite (Knoten)",  min_value=5, max_value=50, value=20)
    height = st.slider("Höhe (Knoten)",    min_value=3, max_value=20, value=5)

    st.header("📂 Laden")
    uploaded_file = st.file_uploader("Struktur laden (.json)", type="json")
    if uploaded_file is not None:
        st.session_state.loaded_data = json.load(uploaded_file)
        st.success("Datei geladen! Drücke 'Simulation Starten'.")

    st.header("2. Randbedingungen")
    st.caption("Knoten-ID = Spalte × Höhe + Zeile  (ab 0)")

    festlager_input = st.text_input(
        "Festlager-Knoten (x+z fest)", value="0",
        help="Kommagetrennte Knoten-IDs, z.B. '0, 5'")

    default_rollenlager = str((width - 1) * height)
    rollenlager_input = st.text_input(
        "Rollenlager-Knoten (nur z fest)", value=default_rollenlager,
        help="Kommagetrennte Knoten-IDs")

    st.header("3. Externe Kräfte")
    default_force_node = str((width // 2) * height + (height - 1))
    kraefte_input = st.text_area(
        "Kräfte (Knoten, Fx, Fz)",
        value=f"{default_force_node}, 0, -10",
        help="Pro Zeile: Knoten-ID, Fx, Fz\nBeispiel:\n54, 0, -10\n27, 5, 0")

    st.header("4. Optimierung")
    remove_pct = st.slider("Masse entfernen (%)", 0, 90, 50)

    st.header("5. Visualisierung")
    show_labels      = st.checkbox("Knoten-IDs anzeigen",     value=False)
    show_deformation = st.checkbox("Verformung anzeigen",     value=False)
    deformation_scale = 0.0
    if show_deformation:
        deformation_scale = st.slider("Verformungsskalierung", 1.0, 200.0, 50.0)
    show_intermediate = st.checkbox("Zwischenschritte anzeigen", value=True)

    start_btn = st.button("Simulation Starten", type="primary")


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def parse_node_ids(text):
    ids = []
    for part in text.split(","):
        part = part.strip()
        if part:
            try:
                ids.append(int(part))
            except ValueError:
                pass
    return ids


def parse_forces(text, dim):
    F = np.zeros(dim)
    for line in text.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                node_id = int(parts[0])
                fx = float(parts[1])
                fz = float(parts[2])
                if 2 * node_id + 1 < dim:
                    F[2 * node_id]     += fx
                    F[2 * node_id + 1] += fz
            except ValueError:
                pass
    return F


def figure_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ── Hauptlogik ───────────────────────────────────────────────────────────────

if start_btn:

    # Struktur erstellen oder aus Datei laden
    if st.session_state.loaded_data is not None:
        system = System.load_from_dict(st.session_state.loaded_data)
        nodes  = system.nodes
        st.info(f"Struktur aus Datei geladen: {len(nodes)} Knoten, {len(system.springs)} Federn")
    else:
        nodes, springs = create_mbb_beam(width, height)
        system = System(nodes, springs)

        festlager_ids   = parse_node_ids(festlager_input)
        rollenlager_ids = parse_node_ids(rollenlager_input)

        u_fixed_idx = []
        for nid in festlager_ids:
            u_fixed_idx.append(2 * nid)
            u_fixed_idx.append(2 * nid + 1)
        for nid in rollenlager_ids:
            u_fixed_idx.append(2 * nid + 1)

        max_id = max(nodes.keys())
        dim = 2 * (max_id + 1)
        F = parse_forces(kraefte_input, dim)

        system.set_boundary_conditions(F, u_fixed_idx)

    # Ausgangszustand berechnen und anzeigen
    st.subheader("Ausgangszustand")
    system.assemble_global_stiffness()
    system.solve()
    st.pyplot(plot_structure(system, "Vollständige Struktur", show_labels, "jet", deformation_scale))

    # Optimierung starten
    to_delete = int(len(nodes) * (remove_pct / 100))
    progress_bar = st.progress(0, text=f"Lösche {to_delete} Knoten...")
    intermediate_placeholder = st.empty()

    def optimization_callback(j, total, sys):
        pct = int((j / total) * 100)
        progress_bar.progress(pct, text=f"Optimierung: {j}/{total} Knoten gelöscht...")
        if show_intermediate:
            sys.assemble_global_stiffness()
            sys.solve()
            fig = plot_structure(sys, f"Zwischenschritt: {j}/{total} gelöscht",
                                show_labels, "jet", deformation_scale)
            intermediate_placeholder.pyplot(fig)
            plt.close(fig)

    try:
        remaining = system.reduce_mass(to_delete, callback=optimization_callback)
        progress_bar.progress(100, text="Fertig!")
        intermediate_placeholder.empty()

        # Finales Ergebnis berechnen und anzeigen
        system.assemble_global_stiffness()
        system.solve()

        st.subheader("Optimiertes Ergebnis")
        st.success(f"Verbleibende Masse: {remaining} Knoten (-{remove_pct}%)")

        result_fig = plot_structure(system, "Optimierte Topologie",
                                    show_labels, "jet", deformation_scale)
        st.pyplot(result_fig)

        # Speichern-Bereich
        st.subheader("💾 Speichern")

        st.download_button(
            label="📷 Optimierte Topologie als Bild herunterladen",
            data=figure_to_png_bytes(result_fig),
            file_name="optimierte_topologie.png",
            mime="image/png"
        )

        save_data = system.save_to_dict()
        st.download_button(
            label="💾 Struktur speichern und später fortsetzen",
            data=json.dumps(save_data, indent=2),
            file_name="topologie_struktur.json",
            mime="application/json"
        )

    except Exception as e:
        st.error(f"Fehler während der Optimierung: {e}")