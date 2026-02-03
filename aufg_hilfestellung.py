import numpy as np
import numpy.typing as npt

def solve(K: npt.NDArray[np.float64], F: npt.NDArray[np.float64], u_fixed_idx: list[int], eps=1e-9) -> npt.NDArray[np.float64] | None:
    """Solve the linear system Ku = F with fixed boundary conditions.

    Parameters
    ----------
    K : npt.NDArray[np.float64]
        Stiffness matrix.   
    F : npt.NDArray[np.float64]
        Force vector.
    u_fixed_idx : list[int]
        List of indices where the displacement is fixed (Dirichlet boundary conditions).
    eps : float, optional
        Regularization parameter to avoid singular matrix, by default 1e-9

    Returns
    -------
    npt.NDArray[np.float64] | None
        Displacement vector or None if the system is unsolvable.
    """

    assert K.shape[0] == K.shape[1], "Stiffness matrix K must be square."
    assert K.shape[0] == F.shape[0], "Force vector F must have the same size as K."

    for d in u_fixed_idx:
        K[d, :] = 0.0
        K[:, d] = 0.0
        K[d, d] = 1.0

    try:
        u = np.linalg.solve(K, F) # solve the linear system Ku = F
        u[u_fixed_idx] = 0.0

        return u
    
    except np.linalg.LinAlgError:
        # If the stiffness matrix is singular we can try a small regularization to still get a solution
        K += np.eye(K.shape[0]) * eps

        try:
            u = np.linalg.solve(K, F) # solve the linear system Ku = F
            u[u_fixed_idx] = 0.0

            return u
        
        except np.linalg.LinAlgError:
            # If it is still singular we give up
            return None

def calc_single_stiffnesses(nodes: dict[int, tuple[float, float]]):
    
    for n, (x, y) in nodes.items():
        if n == 0:
            x0, y0, n_0 = x, y, n
        elif n == 1:
            x1, y1, n_1 = x, y, n
        elif n == 2:
            x2, y2, n_2 = x, y, n
        elif n == 3:
            x3, y3, n_3 = x, y, n

    springs = [ ((x0, y0, n_0), (x1, y1, n_1)),   #n=node_number
                ((x1, y1, n_1), (x2, y2, n_2)),
                ((x2, y2, n_2), (x3, y3, n_3)),
                ((x3, y3, n_3), (x0, y0, n_0)),
                ((x0, y0, n_0), (x2, y2, n_2)),
                ((x1, y1, n_1), (x3, y3, n_3))]
    
    K_o_dict = {}
    
    for i, j in springs:

        x_ji=j[0]-i[0]
        y_ji=j[1]-i[1]

        e_n = np.array([x_ji, y_ji])
        e_n = e_n / np.linalg.norm(e_n)

        laenge = np.sqrt(x_ji**2 + y_ji**2)

        if laenge == 1:
            k=1.0
        else:
            k=1.0/np.sqrt(2)    #diagonal spring stiffness
        
        K = k * np.array([[1.0, -1.0], [-1.0, 1.0]])

        O_n = np.outer(e_n, e_n)

        Ko_n = np.kron(K, O_n)

        K_o_dict[(i[2], j[2])] = Ko_n
    
    return K_o_dict


def spring(nodes: dict[int, tuple[float, float]]) -> None:
    """Example of a spring system with 4 nodes and 6 spring elements.
    """
    Kg = np.zeros((8, 8))

    K_o_dict = calc_single_stiffnesses(nodes)

    for (i, j), Ko_n in K_o_dict.items():
        # Die 4 relevanten Indizes (Freiheitsgrade) bestimmen [cite: 153]
        dofs = [2*i, 2*i+1, 2*j, 2*j+1]
        
        # Die Werte an die berechneten Adressen in Kg addieren (Superposition) [cite: 150, 151]
        # Ohne np.ix_ nutzt du hier die manuelle Zuweisung oder Slicing
        for m in range(4):
            for n in range(4):
                Kg[dofs[m], dofs[n]] += Ko_n[m, n]

    
    print(f"{Kg=}")

    u_fixed_idx = [0, 1] # fix node i in both directions

    F = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]) # apply force at node j in x-direction

    u = solve(Kg, F, u_fixed_idx)
    print(f"{u=}")


if __name__ == "__main__":

    nodes=dict()
    nodes={
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (1.0, 1.0),
        3: (0.0, 1.0)
    }
    
    spring(nodes)