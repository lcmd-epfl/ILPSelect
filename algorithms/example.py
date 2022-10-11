import fragments

"""
M=fragments.model("../representations/database_global_vector.npz", "../representations/target_global_vector.npz", "global_vector")
M.setup(1e6)
M.optimize()
M.output()

M=fragments.model("../representations/database_local_vector.npz", "../representations/target_local_vector.npz", "local_vector")
M.setup(1e6)
M.optimize()
M.output()
"""
M=fragments.model("../representations/database_local_matrix.npz", "../representations/target_local_matrix.npz", "local_matrix")
M.setup(1e6)
M.optimize()
M.output()
