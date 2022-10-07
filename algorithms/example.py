import fragments

M=fragments.model("../representations/database.npz", "../representations/target.npz")
M.setup("global_vector",1e6)
M.optimize()
M.output()

