import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import io
from PIL import Image
from graph4 import System, create_mbb_beam, create_from_image, plot_structure

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

    st.header("📷 Bild als Struktur")
    uploaded_img = st.file_uploader("Bild laden (.png, .jpg)", type=["png", "jpg", "jpeg"])
    if uploaded_img is not None:
        preview_img = Image.open(uploaded_img).convert("L")
        st.image(preview_img, caption="Vorschau (Grayscale)", use_container_width=True)

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
    deformation_scale = 0.0
    show_intermediate = st.checkbox("Zwischenschritte anzeigen", value=True)
    gif_every_n = st.slider("GIF: Jeden n-ten Schritt", 1, 20, 5,
                            help="Jeder n-te Zwischenschritt wird als Frame gespeichert")

    start_btn = st.button("Optimierung Starten", type="primary")


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


# ── Struktur erstellen & Vorschau ──────────────────────────
# Struktur erstellen: Bild > JSON > Gitter
if uploaded_img is not None:
    img = Image.open(uploaded_img).convert("L")
    if max(img.size) > 30:
        img.thumbnail((30, 30))
    img_array = np.array(img)
    nodes, springs, width, height = create_from_image(img_array, 128)
    system = System(nodes, springs)
    st.info(f"Struktur aus Bild: {len(nodes)} Knoten, {len(springs)} Federn ({img_array.shape[1]}×{img_array.shape[0]} px)")

elif st.session_state.loaded_data is not None:
    system = System.load_from_dict(st.session_state.loaded_data)
    nodes  = system.nodes
    st.info(f"Struktur aus Datei geladen: {len(nodes)} Knoten, {len(system.springs)} Federn")
else:
    nodes, springs = create_mbb_beam(width, height)
    system = System(nodes, springs)

# Randbedingungen setzen
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

# ── Optimierung bei clicken ─────────────────────────────────────────────
if start_btn:
    to_delete = int(len(nodes) * (remove_pct / 100))
    progress_bar = st.progress(0, text=f"Lösche {to_delete} Knoten...")
    intermediate_placeholder = st.empty()
    gif_frames = []  # Frames für GIF sammeln

    def optimization_callback(j, total, sys):
        pct = int((j / total) * 100)
        progress_bar.progress(pct, text=f"Optimierung: {j}/{total} Knoten gelöscht...")
        needs_plot = show_intermediate or (j % gif_every_n == 0)
        if needs_plot:
            sys.assemble_global_stiffness()
            sys.solve()
            fig = plot_structure(sys, f"Zwischenschritt: {j}/{total} gelöscht",
                                show_labels, "jet", deformation_scale)
            if show_intermediate:
                intermediate_placeholder.pyplot(fig)
            if j % gif_every_n == 0:
                gif_frames.append(figure_to_png_bytes(fig))
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

        # GIF erstellen
        if gif_frames:
            # Letzten Frame hinzufügen
            gif_frames.append(figure_to_png_bytes(result_fig))

            pil_frames = [Image.open(io.BytesIO(f)).convert("RGBA") for f in gif_frames]
            gif_buf = io.BytesIO()
            # Erster und letzter Frame 1s, Rest 300ms
            durations = [1000] + [300] * (len(pil_frames) - 2) + [1000]
            pil_frames[0].save(
                gif_buf, format="GIF", save_all=True,
                append_images=pil_frames[1:],
                duration=durations, loop=0
            )
            gif_buf.seek(0)

            st.download_button(
                label="🎞️ Animation als GIF herunterladen",
                data=gif_buf.getvalue(),
                file_name="topologie_animation.gif",
                mime="image/gif"
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