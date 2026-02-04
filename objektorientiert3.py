import numpy as np
import numpy.typing as npt

class Node:
    def __init__(self, id: int, x: float, z: float):
        self.id = id
        self.x = x
        self.z = z

class Spring:
    def __init__(self, node_i: Node, node_j: Node):
        self.node_i = node_i
        self.node_j = node_j

    def get_single_stiffnesses(self):
        
        K_o_dict = {}

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

        Ko_n = np.kron(K, O_n)
       
        return Ko_n
    
    def get_d_o_free(self):

        i = self.node_i.id
        j = self.node_j.id

        d_o_free = [2*i, 2*i+1, 2*j, 2*j+1]

        return d_o_free


class System:
    def __init__(self, nodes: dict[int, Node], springs: list[Spring]):
        self.nodes = nodes
        self.springs = springs
        self.Kg=None # global stiffness matrix
    
    def assemble_global_stiffness(self):

        self.Kg = np.zeros((2*len(self.nodes), 2*len(self.nodes)))
        for spring in self.springs:
            Ko_n = spring.get_single_stiffnesses()

            d_o_free = spring.get_d_o_free()

            self.Kg[np.ix_(d_o_free, d_o_free)] += Ko_n  # Eintragen in die globale Steifigkeitsmatrix am richtigen Ort
            #ist das gleiche wie zuvor die beiden for-schleifen, nur effizienter
        
        return self.Kg
    
    def solve(self, F: npt.NDArray[np.float64], u_fixed_idx: list[int], eps=1e-9) -> npt.NDArray[np.float64] | None:

        assert self.Kg.shape[0] == self.Kg.shape[1], "Stiffness matrix K must be square."
        assert self.Kg.shape[0] == F.shape[0], "Force vector F must have the same size as K."

        K_calc = self.Kg.copy()
        F_calc = F.copy()

        for d in u_fixed_idx:       #Randbedingungen einbauen
            K_calc[d, :] = 0.0
            K_calc[:, d] = 0.0
            K_calc[d, d] = 1.0
            F_calc[d]=0.0

        try:
            u = np.linalg.solve(K_calc, F_calc) # solve the linear system Ku = F
            u[u_fixed_idx] = 0.0

            return u
        
        except np.linalg.LinAlgError:
            # If the stiffness matrix is singular we can try a small regularization to still get a solution
            K_calc += np.eye(K_calc.shape[0]) * eps

            try:
                u = np.linalg.solve(K_calc, F_calc) # solve the linear system Ku = F
                u[u_fixed_idx] = 0.0

                return u
            
            except np.linalg.LinAlgError:
                # If it is still singular we give up
                return None



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

    u_fixed_idx = [0, 1] # fix node i in both directions

    #anstatt: F = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # apply force at node j in x-direction
    #besser, weil flexibler:
    F = np.zeros(2*len(nodes_uebergabe))
    F[2] = 10.0  # apply force at node 1 in x-direction

    u = system.solve(F, u_fixed_idx)

    print(f"{u=}")