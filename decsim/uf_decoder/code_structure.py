from scipy.sparse import csr_matrix


class CodeStructure:
    def __init__(self, H_x, H_z, logicals_x, logicals_z, L, repetitions=1, cluster_list_decoding=False, peeling_list_decoding=False, efficient_decoding=False, hardware_goldenmodel=False):
        """initialize code structure
        Args:
            H_x: X stabilizer check matrix
            H_z: Z stabilizer check matrix
            logicals_x: X logical operator matrix
            logicals_z: Z logical operator matrix
            L: lattice size
        """
        # ensure input parameters are valid
        if not isinstance(L, int) or L <= 0:
            raise ValueError("L must be a positive integer")

        # convert input matrices to csr_matrix
        self.H_x = csr_matrix(H_x)
        self.H_z = csr_matrix(H_z)
        self.logicals_x = csr_matrix(logicals_x)
        self.logicals_z = csr_matrix(logicals_z)

        # infer code parameters from matrix dimensions
        self.num_stabs_x, self.num_qubits = H_x.shape
        self.num_stabs_z, _ = H_z.shape

        # set lattice size
        self.L = L
        self.repetitions = repetitions
        self.cluster_list_decoding = cluster_list_decoding
        self.peeling_list_decoding = peeling_list_decoding
        self.efficient_decoding = efficient_decoding
        self.hardware_goldenmodel = hardware_goldenmodel

        # infer code type
        self._infer_code_type()


    def _infer_code_type(self):
        X = self.num_stabs_x
        Z = self.num_stabs_z
        L = self.L

        if self.num_qubits == L:
            self.code_type = 'repetition'
        elif X==L*L and Z==L*L:
            self.code_type = 'toric'
        elif X==L*(L-1) and Z==L*(L-1):
            self.code_type = 'planar'
        else:
            self.code_type = 'rotated'
        # else:
        #     raise ValueError(f"Unrecognized stabilizer counts X={X}, Z={Z} for L={L}")
        self.periodic = (self.code_type in ('toric', 'repetition'))
        # if self.code_type == 'planar':
        #     self.num_qubits = L*L + (L-1)**2
        # elif self.code_type == 'rotated':
        #     self.num_qubits = L*L
        # else:
        #     self.num_qubits = 2*L*L
