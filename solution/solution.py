"""
Qbit corner team, 2021
Alex Vigneron
Faiza ...
Heba Hussein
Jacopo De Santis
"""

from typing import List, Tuple, Optional, Union

import numpy as np
import cirq


def matrix_to_sycamore_operations(
    target_qubits: List[cirq.GridQubit], matrix: np.ndarray
) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    """A method to convert a unitary matrix to a list of Sycamore operations.

    This method will return a list of `cirq.Operation`s using the qubits and (optionally) ancilla
    qubits to implement the unitary matrix `matrix` on the target qubits `qubits`.
    The operations are also supported by `cirq.google.gate_sets.SYC_GATESET`.

    Args:
        target_qubits: list of qubits the returned operations will act on. The qubit order defined by the list
            is assumed to be used by the operations to implement `matrix`.
        matrix: a matrix that is guaranteed to be unitary and of size (2**len(qs), 2**len(qs)).
    Returns:
        A tuple of operations and ancilla qubits allocated.
            Operations: In case the matrix is supported, a list of operations `ops` is returned.
                `ops` acts on `qs` qubits and for which `cirq.unitary(ops)` is equal to `matrix` up
                 to certain tolerance. In case the matrix is not supported, it might return NotImplemented to
                 reduce the noise in the judge output.
            Ancilla qubits: In case ancilla qubits are allocated a list of ancilla qubits. Otherwise
                an empty list.
        .
    """
    if is_supported(matrix): #probably needs to change xD
        return NotImplemented, []
    else:
        pass
        ops_list, qbs = build_op_tree_from_op_mat(matrix, target_qubits)
        ops_tree = build_op_tree_from_op_list(ops_list, qbs)
        return ops_tree





def build_op_tree_from_op_list(l:List[cirq.ops.Operation], qbs:List[cirq.GridQubit]) -> cirq.OP_TREE:
    """iterate over the list, starting from the last element, each iteration
    build the tree from there.
    like:
    1st_elem = l[-1]
    last_op(1st_elem, corresponding_qbs)
    2nd_elem = l[-2]
    ...
    """
    pass

# def build_op_from_gateqb(gatesList[cirq.ops.Operation],List[cirq.GridQubit]) -> List[cirq.OP_TREE]:
#     """ """
#


def is_supported(matrix: np.ndarray) -> bool:
    key=unitary_to_key(matrix)
    try:
        cirq.google.Sycamore.validate_operation(EXISTING_GATES(key))
    except:
        return False
    return True
    """Returns True if the current unitary is supported by Sycamore,
    False otherwise."""
    pass


def unitary_to_key(u:np.ndarray) -> Optional[string]:
    for i in range(len(EXISTING_GATES.values())):
        gate = EXISTING_GATES.values()[i]
        unitary_gate=cirq.unitary(gate)
        if np.array_equal(u,unitary_gate):
            return EXISTING_GATES.keys()[i]
    return None
#     """If the unitary describes an existing gate, matches it to its string name,
#     i.e. "X", "CNOT", etc. If doesn't exist as a primary unitary, returns None.
#     """
#     pass


def simple_decomposition(matrix: np.ndarray
                ) -> Tuple[Optional[List[cirq.ops.Operation]],List[cirq.GridQubit]]:
    """ Tries to match the existing unitary with EXISTING_GATES, returns the
    matching operation if it's there, None otherwise.
    use unitary_to_key to check the EXISTING_GATES dict
    """
    pass



# def build_op_list_from_mat(matrix:nd.array, qbs) -> List[cirq.OP_TREE]:
#     """ uses decompose and then build_op_from_gateqb to return a list of
#     operators"""


def decompose(matrix: np.ndarray, qbs_l:List[cirq.GridQubit]
                ) -> Tuple[Optional[List[cirq.ops.Operation]],List[cirq.GridQubit]]:
    """ Tries to decompose with simple gate, if not goes for the bruteforce one"""
    simple_op, qbs = simple_decomposition(matrix)
    if simple_op:
        pass #return this simple op and its qb(s)
    else:
        true_decompose(matr, qbs_l)
        pass #here's the hardwork, if the matrix is not simple to decompose


def true_decompose(matrix: np.ndarray, qbs_l:List[cirq.GridQubit]
                ) -> Tuple[Optional[List[cirq.ops.Operation]],List[cirq.GridQubit]]:
    """
    Brute-force idea:
    try each and every existing gate on each and every existing qubit until
    you reach Identity
    store the "path" as you go. if you reach Identity, that's a correct path,
    in that case use the INVERSES dict to make a list of all the inverse
    operators,
    which are therefore the operators you'd need to apply to get from Identity
    to your desired matrix.
    cons: n^3 complexity if I'm not mistaken... pro: that's only thing I could
    come up with!
    """





    pass


def find_path(matrix: np.ndarray, path, max_path):
    """
    To be used by true_decompose:
    checks if there is a recursive path that can give us the identity
    Args:
        matrix: the input unitary
        path: list of operations
        max_path: recursion limit to stop searching

    Returns:
        list of operations if found, or -1 if not

    """
    #     print(matrix)
    #     print(path)
    #     print(max_path)
    if (matrix.shape[0] == matrix.shape[1]) and (matrix == np.eye(matrix.shape[0])).all():  # if matrix is identity
        return path

    if max_path == 0:
        return -1

    for key, value in EXISTING_GATES.items():
        path.append(key)

        find_path(matrix @ get_inverse(value), path, max_path - 1)
    return -1


def get_unitary(gate) -> np.ndarray:
    """
    To be applied on the gates dictionary
    Args:
        gate: input gate

    Returns:
        equivalent unitary matrix
    """
    a = cirq.NamedQubit("a") #dummy qubit
    return cirq.unitary(gate(a))


def get_inverse(gate:np.ndarray)-> np.ndarray:
    return np.linalg.pinv(get_unitary(gate))

def test():
    matrix = np.array([[0, 1], [1, 0]])
    print(find_path(matrix, [], 3))

#with keys as name of the gate, ex CNOT, X... and value the gate objects

#covered constant (with no variable inputs) gates here https://quantumai.google/cirq/gates
EXISTING_GATES = dict({'X':cirq.X,'Y':cirq.Y,'Z':cirq.Z,'H':cirq.H ,'S':cirq.S,'T':cirq.T})#,'CZ':cirq.CZ,'CNOT':cirq.CNOT,'SWAP':cirq.SWAP,'ISWAP':cirq.ISWAP,'CCNOT':cirq.CCNOT,
  #     "CCZ":cirq.CCZ,'CSWAP':cirq.CSWAP})


"""keys are names of gates, value is the inverse of that gate.
ex: paulis are involutory (their own inverses) so INVERSES[X] = cirq.X etc for 
the rest that's data that is known so that's findable on the web I believe.
"""
#INVERSES = dict({'X':cirq.X,'Y':cirq.Y,'Z':cirq.Z,'H': cirq.H })

test()




