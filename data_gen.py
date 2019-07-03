##########################################################################
#Quantum classifier
#Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil, J. Ignacio Latorre
#Code by APS
#Code-checks by ACL
#June 3rd 2019


#Universitat de Barcelona / Barcelona Supercomputing Center/Institut de Ciències del Cosmos

###########################################################################

## This file creates the data points for the different problems to be tackled by the quantum classifier 



import numpy as np

problems = ['circle', '3 circles', 'wavy circle', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']

def data_generator(problem, samples=None):
    """
    This function generates the data for a problem
    INPUT: 
        -problem: Name of the problem, one of: 'circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines'
        -samples Number of samples for the data
    OUTPUT:
        -data: set of training and test data
        -settings: things needed for drawing
    """
    problem = problem.lower()
    if problem not in problems:
        raise ValueError('problem must be one of {}'.format(problems))
    if samples == None:
        if problem == 'sphere': 
            samples = 4500
        elif problem == 'hypersphere':
            samples = 5000
        else: 
            samples = 4200
            
    if problem == 'circle':
        data, settings = _circle(samples)
        
    if problem == '3 circles':
        data, settings = _3_circles(samples)
        
    if problem == 'wavy lines':
        data, settings = _wavy_lines(samples)

    if problem == 'squares':
        data, settings = _squares(samples)
        
    if problem == 'sphere':
        data, settings = _sphere(samples)
        
    if problem == 'non convex':
        data, settings = _non_convex(samples)
        
    if problem == 'crown':
        data, settings = _crown(samples)
        
    if problem == 'tricrown':
        data, settings = _tricrown(samples)
        
    if problem == 'hypersphere':
        data, settings = _hypersphere(samples)
    
        
    return data, settings 

def _circle(samples):
    centers = np.array([[0, 0]])
    radii = np.array([np.sqrt(2/np.pi)])
    data=[]
    dim = 2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        y = 0
        for c, r in zip(centers, radii):  
            if np.linalg.norm(x - c) < r:
                y = 1 

        data.append([x, y])
            
    return data, (centers, radii)

def _3_circles(samples):
    centers = np.array([[-1, 1], [1, 0], [-.5, -.5]])
    radii = np.array([1, np.sqrt(6/np.pi - 1), 1/2]) 
    data=[]
    dim = 2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        y = 0
        for j, (c, r) in enumerate(zip(centers, radii)): 
            if np.linalg.norm(x - c) < r:
                y = j + 1 
                
        data.append([x, y])
                
    
    return data, (centers, radii)
    

def _wavy_lines(samples, freq = 1):
    def fun1(s):
        return s + np.sin(freq * np.pi * s)
    
    def fun2(s):
        return -s + np.sin(freq * np.pi * s)
    data=[]
    dim=2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        if x[1] < fun1(x[0]) and x[1] < fun2(x[0]): y = 0
        if x[1] < fun1(x[0]) and x[1] > fun2(x[0]): y = 1
        if x[1] > fun1(x[0]) and x[1] < fun2(x[0]): y = 2
        if x[1] > fun1(x[0]) and x[1] > fun2(x[0]): y = 3        
        data.append([x, y])

    return data, freq

def _squares(samples):
    data=[]
    dim=2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        if x[0] < 0 and x[1] < 0: y = 0
        if x[0] < 0 and x[1] > 0: y = 1
        if x[0] > 0 and x[1] < 0: y = 2
        if x[0] > 0 and x[1] > 0: y = 3        
        data.append([x, y])
    
    return data, None


def _non_convex(samples, freq = 1, x_val = 2, sin_val = 1.5):
    def fun(s):
        return -x_val * s + sin_val * np.sin(freq * np.pi * s)
    
    data = []
    dim = 2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        if x[1] < fun(x[0]): y = 0
        if x[1] > fun(x[0]): y = 1
        data.append([x, y])

    return data, (freq, x_val, sin_val)
            
def _crown(samples):
    c = [[0,0],[0,0]]
    r = [np.sqrt(.8), np.sqrt(.8 - 2/np.pi)]
    data = []
    dim = 2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        if np.linalg.norm(x - c[0]) < r[0] and np.linalg.norm(x - c[1]) > r[1]:
            y = 1
        else: 
            y=0
        data.append([x, y])

    return data, (c, r)


def _tricrown(samples):
    centers = [[0,0],[0,0]]
    radii = [np.sqrt(.8 - 2/np.pi), np.sqrt(.8)]
    data = []
    dim = 2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        y=0
        for j,(r,c) in enumerate(zip(radii, centers)):
            if np.linalg.norm(x - c) > r:
                y = j + 1
        data.append([x, y])

    return data, (centers, radii)

def _sphere(samples):
    centers = np.array([[0, 0, 0]]) 
    radii = np.array([(3/np.pi)**(1/3)]) 
    data=[]
    dim = 3
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        y = 0
        for c, r in zip(centers, radii): 
            if np.linalg.norm(x - c) < r:
                y = 1 

        data.append([x, y])
    
    return data, (centers, radii)

def _hypersphere(samples):
    centers = np.array([[0, 0, 0, 0]]) 
    radii = np.array([(2/np.pi)**(1/2)]) 
    data=[]
    dim = 4
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1 
        y = 0
        for c, r in zip(centers, radii): 
            if np.linalg.norm(x - c) < r:
                y = 1 

        data.append([x, y])
    
    return data, (centers, radii)


