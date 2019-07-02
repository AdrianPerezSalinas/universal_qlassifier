##########################################################################
#Quantum classifier
#Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil, J. Ignacio Latorre
#Code by APS
#Code-checks by ACL
#June 3rd 2019


#Universitat de Barcelona / Barcelona Supercomputing Center/Institut de Ciències del Cosmos

###########################################################################

## This file creates the data points for the different problems to be tackled by the quantum classifier 

## How this file is related to other files


import numpy as np
from sklearn.datasets import load_iris

problems = ['circle', 'wavy circle', '3 circles', 'wavy lines', 'sphere', 'non convex', 'crown', 'tricrown', 'hypersphere', 'iris', 'squares']

def data_generator(problem, samples=None, err = False):
    """
    This function generates the data for a problem
    INPUT: 
        -problem: Name of the problem, one of: 'Circle', 'Wavy Circle', '3 Circles', 'Wavy Lines', 'Sphere', 'non convex', 'crown', 'tricrown'
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
        data, settings = _circle(samples, err)
    
    if problem == 'wavy circle':
        data, settings = _wavy_circle(samples, err)
        
    if problem == '3 circles':
        data, settings = _3_circles(samples, err)
        
    if problem == 'wavy lines':
        data, settings = _wavy_lines(samples, err)

    if problem == 'squares':
        data, settings = _squares(samples, err)
        
    if problem == 'sphere':
        data, settings = _sphere(samples, err)
        
    if problem == 'non convex':
        data, settings = _non_convex(samples, err)
        
    if problem == 'crown':
        data, settings = _crown(samples, err)
        
    if problem == 'tricrown':
        data, settings = _tricrown(samples, err)
        
    if problem == 'hypersphere':
        data, settings = _hypersphere(samples, err)
    
    if problem == 'iris':
        data, settings = _iris()
        
    return data, settings 

#All of them are auxiliary functions for data_generator
def _circle(samples, err):
    centers = np.array([[0, 0]]) # center
    radii = np.array([np.sqrt(2/np.pi)]) # radius
# radius taken as surface circle/surface data = 0.5 
    data=[]
    dim = 2
    if err == False:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1 #Controls dimension of data
            y = 0
            for c, r in zip(centers, radii):  
                if np.linalg.norm(x - c) < r:
                    y = 1 #Labels for every circle

            data.append([x, y])
    if err == True:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1 #Controls dimension of data
            y = 0
            for c, r in zip(centers, radii):  
                if np.linalg.norm(x - c) < r + .1 * (2 * np.random.rand(1) - 1):
                    y = 1 #Labels for every circle

            data.append([x, y])
            
    return data, (centers, radii)

def _wavy_circle(samples, err, wave=0.3, freq=20):
# wave : amplitude oscillation
# freq: frquency of oscillation
    centers = np.array([[0, 0]]) # center
    radii = np.array([np.sqrt(2/np.pi)]) # radius
    waves = np.array([.3])
    freqs = np.array([20])
    data=[]
    dim = 2
    if err == False:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1 #Controls dimension of data
            y = 0
            for c, r, w, f in zip(centers, radii, waves, freqs):  
                if np.linalg.norm(x-c) < r*(1 + w * np.cos(f*np.arctan((x-c)[1] / (x-c)[0]))):
                    y = 1 #Labels for every circle

            data.append([x, y])
    
    if err == True:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1 #Controls dimension of data
            y = 0
            for c, r, w, f in zip(centers, radii, waves, freqs):  
                if np.linalg.norm(x-c) < r*(1 + w * np.cos(f*np.arctan((x-c)[1] / (x-c)[0]))) + .1 * (2 * np.random.rand(1) - 1):
                    y = 1 #Labels for every circle

            data.append([x, y])
    
    return data, (centers, radii, waves, freqs)


def _3_circles(samples, err):
    centers = np.array([[-1, 1], [1, 0], [-.5, -.5]]) # circles centers
    radii = np.array([1, np.sqrt(6/np.pi - 1), 1/2]) # circles radius
# some circles do not fit completely in data space
    data=[]
    dim = 2
    if err == False:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1 #Controls dimension of data
            y = 0
            for j, (c, r) in enumerate(zip(centers, radii)): 
                if np.linalg.norm(x - c) < r:
                    y = j + 1 #Labels for every circle
                    
            data.append([x, y])
                
    if err == True:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1 #Controls dimension of data
            y = 0
            for j, (c, r) in enumerate(zip(centers, radii)): 
                if np.linalg.norm(x - c) < r + .1 * (2 * np.random.rand(1) - 1):
                    y = j + 1 #Labels for every circle
                    
            data.append([x, y])
    
    return data, (centers, radii)
    

def _wavy_lines(samples, err, freq = 1):
    # frequency = 1
    def fun1(s):
        return s + np.sin(freq * np.pi * s)
    
    def fun2(s):
        return -s + np.sin(freq * np.pi * s)
    data=[]
    dim=2
    if err == False:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1
            if x[1] < fun1(x[0]) and x[1] < fun2(x[0]): y = 0
            if x[1] < fun1(x[0]) and x[1] > fun2(x[0]): y = 1
            if x[1] > fun1(x[0]) and x[1] < fun2(x[0]): y = 2
            if x[1] > fun1(x[0]) and x[1] > fun2(x[0]): y = 3        
            data.append([x, y])
            
    if err == True:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1 + .1 * (2 * np.random.rand(dim) - 1)
            if x[1] < fun1(x[0]) and x[1] < fun2(x[0]): y = 0
            if x[1] < fun1(x[0]) and x[1] > fun2(x[0]): y = 1
            if x[1] > fun1(x[0]) and x[1] < fun2(x[0]): y = 2
            if x[1] > fun1(x[0]) and x[1] > fun2(x[0]): y = 3        
            data.append([x, y])
    
    return data, freq

def _squares(samples, err):
    # frequency = 1
    def fun1(s):
        return s + np.sin(freq * np.pi * s)
    
    def fun2(s):
        return -s + np.sin(freq * np.pi * s)
    data=[]
    dim=2
    if err == False:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1
            if x[0] < 0 and x[1] < 0: y = 0
            if x[0] < 0 and x[1] > 0: y = 1
            if x[0] > 0 and x[1] < 0: y = 2
            if x[0] > 0 and x[1] > 0: y = 3        
            data.append([x, y])
            
    if err == True:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1 + .1 * (2 * np.random.rand(dim) - 1)
            if x[1] < 0 and x[1] < 0: y = 0
            if x[1] < 0 and x[1] > 0: y = 1
            if x[1] > 0 and x[1] < 0: y = 2
            if x[1] > 0 and x[1] > 0: y = 3   
       
            data.append([x, y])
    
    return data, None


def _non_convex(samples, err, freq = 1, x_val = 2, sin_val = 1.5):
    def fun(s):
        return -x_val * s + sin_val * np.sin(freq * np.pi * s)
    
    data = []
    dim = 2
    if err == False:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1
            if x[1] < fun(x[0]): y = 0
            if x[1] > fun(x[0]): y = 1
            data.append([x, y])

    if err == True:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1 + .1 * (2 * np.random.rand(dim) - 1)
            if x[1] < fun(x[0]): y = 0
            if x[1] > fun(x[0]): y = 1
            data.append([x, y])
            
    return data, (freq, x_val, sin_val)
            
def _crown(samples, err):
    c = [[0,0],[0,0]]
    r = [np.sqrt(.8), np.sqrt(.8 - 2/np.pi)]
    data = []
    dim = 2
    if err == False:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1
            if np.linalg.norm(x - c[0]) < r[0] and np.linalg.norm(x - c[1]) > r[1]:
                y = 1
            else: 
                y=0
            data.append([x, y])
    if err == True:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1
            if np.linalg.norm(x - c[0]) < r[0] + .1 * (2 * np.random.rand(1) - 1) and np.linalg.norm(x - c[1]) > r[1] + .1 * (2 * np.random.rand(1) - 1):
                y = 1
            else: 
                y=0
            data.append([x, y])


    return data, (c, r)


def _tricrown(samples, err):
    centers = [[0,0],[0,0]]
    radii = [np.sqrt(.8 - 2/np.pi), np.sqrt(.8)]
    data = []
    dim = 2
    if err == False:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1
            y=0
            for j,(r,c) in enumerate(zip(radii, centers)):
                if np.linalg.norm(x - c) > r:
                    y = j + 1
            data.append([x, y])

    if err == True:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1
            y=0
            for j,(r,c) in enumerate(zip(radii, centers)):
                if np.linalg.norm(x - c) > r + .1 * (2 * np.random.rand(1) - 1):
                    y = j + 1
            data.append([x, y])


    return data, (centers, radii)

def _sphere(samples, err):
    centers = np.array([[0, 0, 0]]) # center
    radii = np.array([(3/np.pi)**(1/3)]) # radius
# radius taken as volume sphere / volume data = 0.5
    data=[]
    dim = 3
    if err == False:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1 #Controls dimension of data
            y = 0
            for c, r in zip(centers, radii): 
                if np.linalg.norm(x - c) < r:
                    y = 1 #Labels for every circle

            data.append([x, y])
            
    if err == True:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1 #Controls dimension of data
            y = 0
            for c, r in zip(centers, radii): 
                if np.linalg.norm(x - c) < r + .1 * (2 * np.random.rand(1) - 1):
                    y = 1 #Labels for every circle
                    
            data.append([x, y])
    
    return data, (centers, radii)

def _hypersphere(samples, err):
    centers = np.array([[0, 0, 0, 0]]) # center
    radii = np.array([(2/np.pi)**(1/2)]) # radius
# radius taken as volume sphere / volume data = 0.5
    data=[]
    dim = 4
    if err == False:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1 #Controls dimension of data
            y = 0
            for c, r in zip(centers, radii): 
                if np.linalg.norm(x - c) < r:
                    y = 1 #Labels for every circle

            data.append([x, y])
            
    if err == True:
        for i in range(samples):
            x = 2 * (np.random.rand(dim)) - 1 #Controls dimension of data
            y = 0
            for c, r in zip(centers, radii): 
                if np.linalg.norm(x - c) < r + .1 * (2 * np.random.rand(1) - 1):
                    y = 1 #Labels for every circle
                    
            data.append([x, y])
    
    return data, (centers, radii)

def _iris():
    data = []
    iris = load_iris()
    X = iris['data']
    Y = iris['target']
    X = X - np.min(X, axis = 0)
    X = X / np.max(X, axis = 0)
    for x, y in zip(X,Y):
        data.append([x,y])
        
    return data, None
    

