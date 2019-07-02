from big_functions import minimizer, painter, SGD_step_by_step_minimization, overlearning_paint

qubits = 1  #integer, number of qubits
#layers = 10 #integer, number of layers (time we reupload data)
#figur of merit, choose between ['fidelity_chi', 'weighted_fidelity_chi']

problem='wavy lines' #name of the problem, choose among ['circle', 'wavy circle', '3 circles', 'wavy lines', 'sphere', 'non convex', 'crown']
entanglement = 'y' #entanglement y/n
method = 'L-BFGS-B' #minimization methods, scipy methods or 'SGD'
name = 'test_WF' #However you want to name your files
seed = 30 #random seed
epochs=3000 #number of epochs, only for SGD methods
err=False


#SGD_step_by_step_minimization(problem, qubits, entanglement, layers, name, epochs=epochs, eta = .1)
#overlearning_paint(chi, problem, qubits, entanglement, layers, method, name)

#minimizer(chi, problem, qubits, entanglement, layers, method, name, 
                  #epochs=epochs)

#painter(chi, problem, qubits, entanglement, layers, method, name, 
               #standard_test = True)


if err==True:
    name += '_error'

LAYERS = [1]
for layers in LAYERS:
    chi = 'weighted_fidelity_chi'
    print(chi)
    print(problem)
    print(qubits)
    print(entanglement)
    print(layers)
    print(method)
    print(name)
    print('\n\n')
    #SGD_step_by_step_minimization(problem, qubits, entanglement, layers, name, epochs=epochs)
    #overlearning_paint(chi, problem, qubits, entanglement, layers, method, name)
    #worldmap_painter(chi, problem, qubits, entanglement, layers, method, name, standard_test = True, err=err, seed = seed)
    minimizer(chi, problem, qubits, entanglement, layers, method, name, epochs=epochs, err=err, seed = seed)
    painter(chi, problem, qubits, entanglement, layers, method, name, standard_test=True, seed=seed)


    
    
