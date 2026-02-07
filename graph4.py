import numpy as np
import numpy.typing as npt
import networkx as nx
import matplotlib.pyplot as plt

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
        if self.F.shape[0] < dim:
            F_new = np.zeros(dim)
            F_new[:self.F.shape[0]] = self.F
            self.F = F_new

        assert self.Kg.shape[0] == self.Kg.shape[1], "Stiffness matrix K must be square."
        assert self.Kg.shape[0] == self.F.shape[0], "Force vector F must have the same size as K."

        K_calc = self.Kg.copy()
        F_calc = self.F.copy()

        for d in self.u_fixed_idx:       #Randbedingungen einbauen
            K_calc[d, :] = 0.0
            K_calc[:, d] = 0.0
            K_calc[d, d] = 1.0
            F_calc[d]=0.0

        try:
            self.u = np.linalg.solve(K_calc, F_calc) # solve the linear system Ku = F
            self.u[self.u_fixed_idx] = 0.0

            return self.u
        
        except np.linalg.LinAlgError:
            # If the stiffness matrix is singular we can try a small regularization to stilll get a solution
            K_calc += np.eye(K_calc.shape[0]) * eps

            try:
                self.u = np.linalg.solve(K_calc, F_calc) # solve the linear system Ku = F
                self.u[self.u_fixed_idx] = 0.0
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
    
    def reduce_mass(self, del_amount: int):
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
                    continue # Falls Knoten schon weg ist (sollte nicht passieren), weiter

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
                            # Wichtig: Prüfen ob die Knoten noch im Graphen sind
                            if force_node in new_graph_w_reduced_nodes and fixed_node in new_graph_w_reduced_nodes:
                                if nx.has_path(new_graph_w_reduced_nodes, force_node, fixed_node):
                                    does_path_exist = True
                                    break # Ein Pfad reicht uns (meistens)
                                else:
                                    does_path_exist = False
                            else:
                                # Wenn ein wichtiger Start/Endknoten fehlt -> Pfad kaputt
                                does_path_exist = False 
                        
                        if does_path_exist: 
                            break

                if does_path_exist:
                    # ERFOLG: Wir übernehmen die Änderung
                    self.nodes = nodes_try # Den echten Zustand aktualisieren
                    self.graph_structure = new_graph_w_reduced_nodes
                    j += 1 # Ein Ziel erreicht!
                    print(f"Knoten {node_to_delete} erfolgreich gelöscht. ({j}/{del_amount})")
                    break # Wir gehen zurück in die äußere Schleife, um den nächsten Knoten zu löschen (und nicht sofort den nächsten Kandidaten nehmen)
                else:
                    # FEHLSCHLAG: Wir machen NICHTS am System
                    # Wir erhöhen j NICHT, weil wir ja nichts gelöscht haben.
                    # Wir gehen einfach in die nächste Runde und nehmen den nächsten Kandidaten aus 'candidates'.
                    print(f"Knoten {node_to_delete} konnte nicht gelöscht werden (Pfad unterbrochen).")

        reduced_mass = len(self.nodes)
        return reduced_mass

        

if __name__ == "__main__":

    nodes=dict()
    nodes={
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (1.0, 1.0),
        3: (0.0, 1.0)
    }

    springs_loc = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]

    springs = []
    nodes_uebergabe = dict()

    for n, (x, z) in nodes.items():
        nodes_uebergabe[n] = Node(n, x, z)

    for i, j in springs_loc:
        springs.append(Spring(nodes_uebergabe[i], nodes_uebergabe[j]))

    system=System(nodes_uebergabe, springs)

    Kg=system.assemble_global_stiffness()

    u_fixed_idx = [0, 1] # fix node

    #anstatt: F = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # apply force at node j in x-direction
    #besser, weil flexibler:
    F = np.zeros(2*len(nodes_uebergabe))
    F[2] = 10.0  # apply force at node 1 in x-direction

    system.set_boundary_conditions(F, u_fixed_idx)  #Randbedingungen setzen
    u = system.solve()   
    print(f"{u=}")

    graph_structure = system.create_graph_structure(None)
    system.sort_nodes_by_relevance()
    system.reduce_mass(1)
    nx.draw(system.graph_structure, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.show()