import numpy as np
import numpy.typing as npt
import networkx as nx
import matplotlib.pyplot as plt
import json

class Node:
    def __init__(self, id: int, x: float, z: float):
        self.id = id
        self.x = x
        self.z = z

class Spring:
    def __init__(self, node_i: Node, node_j: Node):
        self.node_i = node_i
        self.node_j = node_j
        self.K_o_n = None  # local stiffness matrix

    def get_single_stiffnesses(self):
        
        x_ji=self.node_j.x - self.node_i.x
        z_ji=self.node_j.z - self.node_i.z

        e_n = np.array([x_ji, z_ji])
        e_n = e_n / np.linalg.norm(e_n)

        laenge = np.sqrt(x_ji**2 + z_ji**2)

        if np.isclose(laenge, 1.0): #gleich wie ==1.0, aber mit Toleranz
            k=1.0
        else:
            k=1.0/np.sqrt(2)    #diagonal spring stiffness
        
        K = k * np.array([[1.0, -1.0], [-1.0, 1.0]])

        O_n = np.outer(e_n, e_n)

        self.Ko_n = np.kron(K, O_n)
       
        return self.Ko_n
    
    def get_d_o_free(self):

        i = self.node_i.id
        j = self.node_j.id

        d_o_free = [2*i, 2*i+1, 2*j, 2*j+1]

        return d_o_free
    
    def calc_weighting(self, u: npt.NDArray[np.float64]) -> float:
        # Berechnung der Gewichtung für die Topologieoptimierung
        d_o_free = self.get_d_o_free()
        u_local = u[d_o_free]

        c= 0.5* u_local.T @ self.Ko_n @ u_local

        return c
    

class System:
    def __init__(self, nodes: dict[int, Node], springs: list[Spring]):
        self.nodes = nodes
        self.springs = springs
        self.Kg=None # global stiffness matrix
        self.u=None  # global displacement vector
        self.graph_structure = nx.Graph()
        self.mass=None
        self.ids_sorted=None
        self.F=None
        self.u_fixed_idx=None

    def set_boundary_conditions(self, F: npt.NDArray[np.float64], u_fixed_idx: list[int]):
        self.F = F
        self.u_fixed_idx = u_fixed_idx
        
    
    def assemble_global_stiffness(self):
        if not self.nodes:
            max_id = 0
        else :
            max_id = max(self.nodes.keys())

        dim = 2 * (max_id+1)
        self.Kg = np.zeros((dim, dim))

        for spring in self.springs:
            #nur Federn die existieren verwenden
            if spring.node_i.id in self.nodes and spring.node_j.id in self.nodes:
                Ko_n = spring.get_single_stiffnesses()
                d_o_free = spring.get_d_o_free()
                self.Kg[np.ix_(d_o_free, d_o_free)] += Ko_n  # Eintragen in die globale Steifigkeitsmatrix am richtigen Ort
                #ist das gleiche wie zuvor die beiden for-schleifen, nur effizienter

        #Singularität bei gelöschten zeilen verhindern 
        for i in range(dim):
            if self.Kg[i, i] == 0.0 and np.all(self.Kg[i, :] == 0.0):
                self.Kg[i, i] = 1.0
        return self.Kg
    
    def solve(self, eps=1e-9) -> npt.NDArray[np.float64] | None:
        dim = self.Kg.shape[0]
        if self.F.shape[0] != dim:
            F_new = np.zeros(dim)
            # Kopiere so viel wie möglich (das kleinere von beiden)
            min_len = min(self.F.shape[0], dim)
            F_new[:min_len] = self.F[:min_len]
            self.F = F_new

        assert self.Kg.shape[0] == self.Kg.shape[1], "Stiffness matrix K must be square."
        assert self.Kg.shape[0] == self.F.shape[0], "Force vector F must have the same size as K."

        K_calc = self.Kg.copy()
        F_calc = self.F.copy()

        # Nur gültige Indizes für die aktuelle Matrixgröße verwenden
        current_dim = K_calc.shape[0]
        valid_fixed_idx = [idx for idx in self.u_fixed_idx if idx < current_dim]

        for d in valid_fixed_idx:       #Randbedingungen einbauen
            K_calc[d, :] = 0.0
            K_calc[:, d] = 0.0
            K_calc[d, d] = 1.0
            F_calc[d] = 0.0

        try:
            self.u = np.linalg.solve(K_calc, F_calc) # solve the linear system Ku = F
            self.u[valid_fixed_idx] = 0.0

            return self.u
        
        except np.linalg.LinAlgError:
            # If the stiffness matrix is singular we can try a small regularization to stilll get a solution
            K_calc += np.eye(K_calc.shape[0]) * eps

            try:
                self.u = np.linalg.solve(K_calc, F_calc) # solve the linear system Ku = F
                self.u[valid_fixed_idx] = 0.0
                return self.u
            
            except np.linalg.LinAlgError:
                return None

    def create_graph_structure(self, Nodes):

        weight=None

        if Nodes is None:       #diese if else statement wird gebraucht um einerseits über main aufgerufen bzw. erstmalig aufgerufen werden zu können,
            #und einen Graphen mit allen Knoten zu erstellen, andererseits aber auch über reduce_mass aufgerufen werden zu können, 
            # um einen neuen kopierten (prüf)Graphen mit den reduzierten Knoten zu erstellen
            Nodes = self.nodes
            graph_structure = self.graph_structure
        else:
            graph_structure = nx.Graph() #neuer Graph für die Topologieoptimierung, damit der ursprüngliche Graph mit allen Knoten erhalten bleibt

        for spring in self.springs:
            i = spring.node_i.id
            j = spring.node_j.id

            #Nur Federn hinzufügen, wenn BEIDE Knoten noch existieren!
            if i in Nodes and j in Nodes:
                weight = spring.calc_weighting(self.u)
                graph_structure.add_edge(i, j, weight=weight)
        
        for node in Nodes.values():
            graph_structure.add_node(node.id)

        return graph_structure        

    def sort_nodes_by_relevance(self)-> list [int]:

        dict_to_sort = dict()

        for node in self.nodes.values():    #für alle Nodes in nodes(dict)
            total_work_for_node = 0.0

            for edge in self.graph_structure.edges(node.id, data=True):
                u, v, data = edge
                single_weight = data['weight']
                
                # Sicherstellung, dass die "wichtigsten" Knoten (fixe und Knoten mit Kräften) nie rausgelöscht werden
                if node.id in self.u_fixed_idx or self.F[2*node.id] != 0.0 or self.F[2*node.id+1] != 0.0:
                    total_work_for_node = float('inf')  # fixe oder Knoten wo F wirkt nach unten sortieren
                    #damit sie nie aufgrund von Topologieoptimierung rausgelöscht werden
                else:
                    total_work_for_node += single_weight/2 # da jede Kante zu 2 Knoten gehört-->/2

            dict_to_sort[node.id] = total_work_for_node

        sorted_dict = dict(sorted(dict_to_sort.items(), key=lambda item: item[1]))
        self.ids_sorted = list(sorted_dict.keys())
        print(f"{self.ids_sorted=}")

        return self.ids_sorted
    
    def save_to_dict(self) -> dict:
        """Serialisiert das System in ein dict für JSON-Export."""
        nodes_data = {str(nid): {"x": n.x, "z": n.z} for nid, n in self.nodes.items()}
        springs_data = [[s.node_i.id, s.node_j.id] for s in self.springs]
        return {
            "nodes": nodes_data,
            "springs": springs_data,
            "F": self.F.tolist() if self.F is not None else [],
            "u_fixed_idx": self.u_fixed_idx if self.u_fixed_idx is not None else []
        }

    @classmethod
    def load_from_dict(cls, data: dict) -> 'System':
        """Rekonstruiert ein System aus einem gespeicherten dict."""
        nodes = {}
        for nid_str, ndata in data["nodes"].items():
            nid = int(nid_str)
            nodes[nid] = Node(nid, ndata["x"], ndata["z"])

        springs = []
        for i_id, j_id in data["springs"]:
            if i_id in nodes and j_id in nodes:
                springs.append(Spring(nodes[i_id], nodes[j_id]))

        system = cls(nodes, springs)
        if data.get("F"):
            system.F = np.array(data["F"])
        if data.get("u_fixed_idx"):
            system.u_fixed_idx = data["u_fixed_idx"]
        return system

    def reduce_mass(self, del_amount: int, callback=None):
        self.mass = len(self.nodes)
        i = 0
        j = 0
        force_nodes = set()
        fixed_nodes = set()
        #schaut auf welchem Knoten eine Kraft wirkt und sie in force_nodes
        while i < len(self.F):
            if self.F[i] != 0.0 or self.F[i+1] != 0.0:
                force_nodes.add(i // 2)
            #schaut ob der Knoten ein fester Knoten ist
            if i in self.u_fixed_idx or i+1 in self.u_fixed_idx:
                fixed_nodes.add(i // 2)
            i += 2

        # Initialisierung: Falls noch keine Sortierung existiert, erstelle sie
        if self.ids_sorted is None:
             self.assemble_global_stiffness()
             self.solve()
             self.create_graph_structure(None)
             self.sort_nodes_by_relevance()

        #Kopie der Liste
        candidates = self.ids_sorted.copy() 

        # Wir laufen so lange, wie wir noch löschen müssen (j < del_amount)
        # UND solange wir noch Kandidaten in der Liste haben (len(candidates) > 0)
        while j < del_amount :

            self.assemble_global_stiffness()
            self.solve() 
            self.create_graph_structure(None)
            self.sort_nodes_by_relevance()
            candidates = self.ids_sorted.copy()

            if not candidates:
                print("Keine Kandidaten mehr zum Löschen übrig.")
                break
            
            while len(candidates) > 0:
                node_to_delete = candidates.pop(0) 

                # Arbeitskopie erstellen für diesen Versuch
                nodes_try = self.nodes.copy()
                
                # Versuchen zu löschen
                if node_to_delete in nodes_try:
                    nodes_try.pop(node_to_delete)
                else:
                    continue # Falls Knoten schon weg ist

                # Graphen bauen für den Check
                new_graph_w_reduced_nodes = self.create_graph_structure(nodes_try)

                # Prüfen ob der Pfad noch existiert
                does_path_exist = True
                
                # Wenn es gar keine Force-Knoten oder Fixed-Knoten gibt, ist der Pfad egal/nicht prüfbar
                if not force_nodes or not fixed_nodes:
                    does_path_exist = True 
                else:
                    for force_node in force_nodes:
                        for fixed_node in fixed_nodes:
                            # Prüfen ob die Knoten noch im Graphen sind
                            if force_node in new_graph_w_reduced_nodes and fixed_node in new_graph_w_reduced_nodes:
                                if nx.has_path(new_graph_w_reduced_nodes, force_node, fixed_node):
                                    does_path_exist = True
                                    break 
                                else:
                                    does_path_exist = False
                            else:
                                # Wenn ein wichtiger Start/Endknoten fehlt -> Pfad kaputt
                                does_path_exist = False 
                        
                        if does_path_exist: 
                            break

                if does_path_exist:
                    self.nodes = nodes_try # Den echten Zustand aktualisieren
                    self.graph_structure = new_graph_w_reduced_nodes
                    j += 1
                    print(f"Knoten {node_to_delete} erfolgreich gelöscht. ({j}/{del_amount})")
                    # Callback für Zwischenschritt-Visualisierung
                    if callback is not None:
                        callback(j, del_amount, self)
                    break 
                else:
                    # Wir gehen einfach in die nächste Runde und nehmen den nächsten Kandidaten
                    print(f"Knoten {node_to_delete} konnte nicht gelöscht werden (Pfad unterbrochen).")

        reduced_mass = len(self.nodes)
        return reduced_mass

def create_mbb_beam(width: int, height: int) -> tuple[dict[int, Node], list[Spring]]:
    nodes = dict()
    springs = []

    # Knoten erstellen
    for x in range(width):
        for z in range(height):
            node_id = x * height + z
            nodes[node_id] = Node(node_id, float(x), float(z))

    # Federn erstellen 
    for x in range(width):
        for z in range(height):
            u = x * height + z

            # Nach rechts
            if x < width - 1:
                v = (x + 1) * height + z
                springs.append(Spring(nodes[u], nodes[v]))

                # Diagonal nach rechts oben
                if z < height - 1:
                    v_diag = (x + 1) * height + (z + 1)
                    springs.append(Spring(nodes[u], nodes[v_diag]))

            # Nach oben
            if z < height - 1:
                v = x * height + (z + 1)
                springs.append(Spring(nodes[u], nodes[v]))

                # Diagonal nach links oben
                if x > 0:
                    v_diag_left = (x - 1) * height + (z + 1)
                    springs.append(Spring(nodes[u], nodes[v_diag_left]))
    
    return nodes, springs


def create_from_image(image_array, threshold: int = 128):
    #Erstellt Knoten & Federn aus einem Grayscale-Bild.
    img_h, img_w = image_array.shape
    nodes = dict()
    springs = []

    # Knoten nur dort erstellen, wo Pixel dunkel genug ist
    for x in range(img_w):
        for z in range(img_h):
            if image_array[z, x] < threshold:
                z_s = img_h - 1 - z          
                node_id = x * img_h + z_s    
                nodes[node_id] = Node(node_id, float(x), float(z_s))

    # Federn zwischen benachbarten existierenden Knoten
    for x in range(img_w):
        for z_s in range(img_h):
            u = x * img_h + z_s
            if u not in nodes:
                continue

            # Nach rechts
            if x < img_w - 1:
                v = (x + 1) * img_h + z_s
                if v in nodes:
                    springs.append(Spring(nodes[u], nodes[v]))

                # Diagonal nach rechts oben
                if z_s < img_h - 1:
                    v_diag = (x + 1) * img_h + (z_s + 1)
                    if v_diag in nodes:
                        springs.append(Spring(nodes[u], nodes[v_diag]))

            # Nach oben
            if z_s < img_h - 1:
                v = x * img_h + (z_s + 1)
                if v in nodes:
                    springs.append(Spring(nodes[u], nodes[v]))

                # Diagonal nach links oben
                if x > 0:
                    v_diag_left = (x - 1) * img_h + (z_s + 1)
                    if v_diag_left in nodes:
                        springs.append(Spring(nodes[u], nodes[v_diag_left]))

    return nodes, springs, img_w, img_h


def plot_structure(system: System, title: str = "Struktur", show_labels: bool = False, colormap: str = "viridis", deformation_scale: float = 0.0) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))

    #verschobene Koordinaten berechnen
    def get_pos(node):
        x, z = node.x, node.z
        if deformation_scale > 0 and system.u is not None:
            idx = node.id
            if 2*idx+1 < len(system.u):
                x += deformation_scale * system.u[2*idx]
                z += deformation_scale * system.u[2*idx+1]
        return x, z

    # Koordinaten sammeln
    x_vals = []
    z_vals = []
    for node in system.nodes.values():
        px, pz = get_pos(node)
        x_vals.append(px)
        z_vals.append(pz)
    
    ax.scatter(x_vals, z_vals, c='black', s=10, zorder=5)

    # Energie berechnen für Heatmap
    valid_springs = []
    energies = []
    if system.u is not None:
        for spring in system.springs:
            if spring.node_i.id in system.nodes and spring.node_j.id in system.nodes:
                val = spring.calc_weighting(system.u)
                energies.append(val)
                valid_springs.append(spring)

    colors = []
    weights = []
    
    if energies:
        max_e = max(energies)
        # Logarithmische Skalierung für bessere Sichtbarkeit von kleinen Unterschieden
        log_energies = np.log1p(energies)
        max_log = max(log_energies) if max(log_energies) > 0 else 1.0
        
        norm_energies = log_energies / max_log
    else:
        norm_energies = [0] * len(system.springs)

    # Schleife fürs Zeichnen
    cmap = plt.get_cmap(colormap)
    #Heatmap
    for idx, spring in enumerate(valid_springs):
        val_norm = norm_energies[idx]
        color = cmap(val_norm)
        # Dicke Linien für wichtige Elemente
        weight = 1.0 + 2.0 * val_norm 

        x_i, z_i = get_pos(spring.node_i)
        x_j, z_j = get_pos(spring.node_j)
        ax.plot([x_i, x_j], [z_i, z_j], 
                color=color, linewidth=weight, alpha=0.8)

    if show_labels:
        for node in system.nodes.values():
            px, pz = get_pos(node)
            ax.text(px, pz, str(node.id), fontsize=6, color='red')

    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

if __name__ == "__main__":

    # 1. Gitter erzeugen
    width = 40   # Knoten horizontal
    height = 10  # Knoten vertikal
    nodes, springs = create_mbb_beam(width, height)    

    system = System(nodes, springs)

    # 2. Randbedingungen
    # Links unten (x=0, z=0): FESTLAGER (x und z fixiert)
    bottom_left = 0 * height + 0  # = 0
    
    # Rechts unten (x=width-1, z=0): ROLLENLAGER (nur z fixiert)
    bottom_right = (width - 1) * height + 0

    u_fixed_idx = [
        2 * bottom_left,       # x-Richtung fest (Festlager)
        2 * bottom_left + 1,   # z-Richtung fest (Festlager)
        2 * bottom_right + 1   # z-Richtung fest (Rollenlager, x ist frei!)
    ]

    # 3. Kraft F oben in der Mitte, nach unten
    F = np.zeros(2 * len(nodes))
    top_center = (width // 2) * height + (height - 1)  # Knoten oben mitte
    F[2 * top_center + 1] = -10.0  # Kraft nach UNTEN

    print(f"Gitter: {width}x{height} = {len(nodes)} Knoten, {len(springs)} Federn")
    print(f"Festlager: Knoten {bottom_left} (links unten)")
    print(f"Rollenlager: Knoten {bottom_right} (rechts unten)")
    print(f"Kraft auf Knoten {top_center} (oben mitte)")

    # 4. Berechnen
    system.set_boundary_conditions(F, u_fixed_idx)
    system.assemble_global_stiffness()
    system.solve()

    system.create_graph_structure(None)
    system.sort_nodes_by_relevance()

    print(f"\nStartmasse: {len(system.nodes)} Knoten")

    # 5. Optimieren! Versuche ~40% der Knoten zu löschen
    to_delete = int(len(nodes) * 0.4)
    print(f"Versuche {to_delete} Knoten zu löschen...\n")
    system.reduce_mass(to_delete)

    print(f"\nEndmasse: {len(system.nodes)} Knoten")

    # 6. Ergebnis zeichnen
    pos = {node.id: (node.x, node.z) for node in system.nodes.values()}
    
    plt.figure(figsize=(14, 5))
    nx.draw(system.graph_structure, pos=pos, with_labels=True, 
            node_color='lightblue', edge_color='gray', node_size=200, font_size=7)
    plt.title("MBB-Balken nach Topologieoptimierung")
    plt.axis('equal')
    plt.show()