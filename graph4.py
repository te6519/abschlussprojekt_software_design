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

        self.Kg = np.zeros((2*len(self.nodes), 2*len(self.nodes)))

        for spring in self.springs:
            Ko_n = spring.get_single_stiffnesses()

            d_o_free = spring.get_d_o_free()

            self.Kg[np.ix_(d_o_free, d_o_free)] += Ko_n  # Eintragen in die globale Steifigkeitsmatrix am richtigen Ort
            #ist das gleiche wie zuvor die beiden for-schleifen, nur effizienter
        
        return self.Kg
    
    def solve(self, eps=1e-9) -> npt.NDArray[np.float64] | None:

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
                # If it is still singular we give up
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

        sorted_dict = dict(sorted(dict_to_sort.items()))
        self.ids_sorted = list(sorted_dict.keys())
        print(f"{self.ids_sorted=}")

        return self.ids_sorted

    def reduce_mass(self, del_amount: int):

        self.mass = len(self.nodes) #weil jeder Knoten 1 kg wiegt
        i = 0
        j = 0
        nodes_copied = self.nodes.copy()
        new_list_of_ids_sorted = None
        force_nodes = set() #Knoten wo Kräfte wirken
        fixed_nodes = set() #fixe Knoten
        # Vorher wurde es mit einer liste gemacht, aber bei beretis einige zusätzlcihe Knoten, dauerte es bereits Minuten zum Ergebnsi
        #deshalb wird jetzt anstelle von einer lsite mit set() gearbeitet
        #vorteil set: schnell und keine doppelten einträge möglich
        does_path_exist = False

        while i < len(self.F):

            if self.F[i] != 0.0 or self.F[i+1] != 0.0:
                force_nodes.add(i//2)
            
            if i in self.u_fixed_idx or i+1 in self.u_fixed_idx:
                fixed_nodes.add(i//2)

            i += 2
        
        new_list_of_ids_sorted=self.ids_sorted[0]#braucht man für die folgenden while schleife
        
        while j < del_amount:

            node_to_delete = new_list_of_ids_sorted[0] #der "unwichtigste" Knoten, der gelöscht werden soll, ist immer der erste in der Liste ids_sorted, da sie nach Relevanz sortiert ist
            new_list_of_ids_sorted = new_list_of_ids_sorted.pop(0)
            nodes_copied.pop(node_to_delete)
            
            j += 1

            new_graph_w_reduced_nodes=self.create_graph_structure(nodes_copied)

            for force_node in force_nodes:

                for fixed_node in fixed_nodes:
                    if nx.has_path(new_graph_w_reduced_nodes, self.nodes[force_node], self.nodes[fixed_node]):
                        does_path_exist = True
                        break
                    else:
                        does_path_exist = False
                    
        
            if not does_path_exist:
                nodes_copied=self.nodes.copy() #wenn kein Pfad existiert, dann wird der ursprüngliche Graph mit allen Knoten wiederhergestellt
                new_graph_w_reduced_nodes=None

            else:
                self.nodes = nodes_copied
                self.graph_structure = new_graph_w_reduced_nodes
        
        reduced_mass = len(self.nodes)

        return (reduced_mass)

        

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

    u = system.solve(F, u_fixed_idx)

    print(f"{u=}")

    graph_structure = system.create_graph_structure()
    
    system.sort_nodes_by_relevance(u_fixed_idx, F)

    nx.draw(graph_structure, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.show()

    calc.mass(zielmasse=3)