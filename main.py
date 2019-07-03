from big_functions import minimizer, painter, SGD_step_by_step_minimization, overlearning_paint

qubits = 1  #integer, number of qubits
layers = 1 #integer, number of layers (time we reupload data)
chi = 'fidelity_chi' #Cost function; choose between ['fidelity_chi', 'weighted_fidelity_chi']
problem='sphere' #name of the problem, choose among ['circle', 'wavy circle', '3 circles', 'wavy lines', 'sphere', 'non convex', 'crown']
entanglement = 'y' #entanglement y/n
method = 'L-BFGS-B' #minimization methods, scipy methods or 'SGD'
name = 'run' #However you want to name your files
seed = 30 #random seed
#epochs=3000 #number of epochs, only for SGD methods

#SGD_step_by_step_minimization(problem, qubits, entanglement, layers, name)
minimizer(chi, problem, qubits, entanglement, layers, method, name, seed = seed)
painter(chi, problem, qubits, entanglement, layers, method, name, standard_test=True, seed=seed)
