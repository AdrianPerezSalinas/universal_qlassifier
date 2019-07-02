##########################################################################
#Quantum classifier
#Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil, J. Ignacio Latorre
#Code by APS
#Code-checks by ACL
#June 3rd 2019


#Universitat de Barcelona / Barcelona Supercomputing Center/Institut de Ciències del Cosmos

###########################################################################


## This is an auxiliary file. It provides the tools needed for simulating quantum
# circuits.

import numpy as np
class QCircuit(object):
    def __init__(self,qubits):
        self.num_qubits = qubits
        self.psi = [0]*2**self.num_qubits
        self.psi[0] = 1
        self.E_x=0
        self.E_y=0
        self.E_z=0
        
    def Ry(self,i,theta):
        if i>=self.num_qubits: raise ValueError('There are not enough qubits')
        c = np.cos(theta/2)
        s = np.sin(theta/2)
        for k in range(2**(self.num_qubits-1)):
            S = k%(2**i) + 2*(k - k%(2**i))
            S_=S + 2**i
            a=c*self.psi[S] - s*self.psi[S_];
            b=s*self.psi[S] + c*self.psi[S_];
            self.psi[S]=a; self.psi[S_]=b;
            
    def Rx(self,i,theta):
        if i>=self.num_qubits: raise ValueError('There are not enough qubits')
        c = np.cos(theta/2)
        s = np.sin(theta/2)
        for k in range(2**(self.num_qubits-1)):
            S = k%(2**i) + 2*(k - k%(2**i))
            S_=S + 2**i
            a=c*self.psi[S] - 1j*s*self.psi[S_];
            b=-1j*s*self.psi[S] + c*self.psi[S_];
            self.psi[S]=a; self.psi[S_]=b;

    def U2(self,i,phi,lamb):
        if i >= self.num_qubits: raise ValueError('There are not enough qubits')
        f = np.exp(1j*phi)
        l = np.exp(-1j*lamb)
        for k in range(2**(self.num_qubits-1)):
            S = k%(2**i) + 2*(k - k%(2**i))
            S_=S + 2**i
            a=1/np.sqrt(2)*(self.psi[S] - l*self.psi[S_]);
            b=1/np.sqrt(2)*(f*self.psi[S] + f*l*self.psi[S_]);
            self.psi[S]=a; self.psi[S_]=b;

    def U3(self, i, theta3):
        if i >= self.num_qubits: raise ValueError('There are not enough qubits')
        c = np.cos(theta3[0] / 2)
        s = np.sin(theta3[0] / 2)
        e_phi = np.exp(1j * theta3[1] / 2)
        e_phi_s = np.conj(e_phi)
        e_lambda = np.exp(1j * theta3[2] / 2)
        e_lambda_s = np.conj(e_lambda)
        for k in range(2 ** (self.num_qubits - 1)):
            S = k % (2 ** i) + 2 * (k - k % (2 ** i))
            S_ = S + 2 ** i
            a = c * e_phi * e_lambda * self.psi[S] - s * e_phi * e_lambda_s * self.psi[S_];
            b = s * e_phi_s * e_lambda * self.psi[S] + c * e_phi_s * e_lambda_s * self.psi[S_];
            self.psi[S] = a;
            self.psi[S_] = b;
            
    def Rz(self,i,theta):
        if i>=self.num_qubits: raise ValueError('There are not enough qubits')
        ex = np.exp(1j*theta)
        for k in range(2**(self.num_qubits-1)):
            S = k%(2**i) + 2*(k - k%(2**i)) + 2**i
            self.psi[S]=ex*self.psi[S]; 
            
    def Hx(self,i):
        if i>=self.num_qubits: raise ValueError('There are not enough qubits')
        for k in range(2**(self.num_qubits-1)):
            S = k%(2**i) + 2*(k - k%(2**i))
            S_=S + 2**i
            a=1/np.sqrt(2)*self.psi[S] + 1/np.sqrt(2)*self.psi[S_];
            b=1/np.sqrt(2)*self.psi[S] - 1/np.sqrt(2)*self.psi[S_];
            self.psi[S] = a
            self.psi[S_] = b
            
    def Hy(self,i):
        if i>=self.num_qubits: raise ValueError('There are not enough qubits')
        for k in range(2**(self.num_qubits-1)):
            S = k%(2**i) + 2*(k - k%(2**i))
            S_=S + 2**i
            a =1/np.sqrt(2)*self.psi[S] -1j/np.sqrt(2)*self.psi[S_];
            b =-1j/np.sqrt(2)*self.psi[S] + 1/np.sqrt(2)*self.psi[S_];
            self.psi[S] = a
            self.psi[S_] = b
            
    def HyT(self,i):
        if i>=self.num_qubits: raise ValueError('There are not enough qubits')
        for k in range(2**(self.num_qubits-1)):
            S = k%(2**i) + 2*(k - k%(2**i))
            S_=S + 2**i
            a=1/np.sqrt(2)*self.psi[S] +1j/np.sqrt(2)*self.psi[S_];
            b=1j/np.sqrt(2)*self.psi[S] + 1/np.sqrt(2)*self.psi[S_];
            self.psi[S]=a; self.psi[S_]=b;
            
    def Cz(self,i,j):
        if i>=self.num_qubits: raise ValueError('There are not enough qubits')
        if j>=self.num_qubits: raise ValueError('There are not enough qubits')
        if i==j: raise ValueError('Control and target qubits are the same')
        if j<i: a=i; i=j; j=a;
        for k in range(2**(self.num_qubits-2)):
            S = k%2**i + (
               ( k - k%2**i)*2)%2**j + 2*(
                      (k-k%2**i)*2-((2*(k-k%2**i))%2**j)) + 2**i + 2**j;
            self.psi[S]=-self.psi[S]
     
    def SWAP(self,i,j):
        if i>=self.num_qubits: raise ValueError('There are not enough qubits')
        if j>=self.num_qubits: raise ValueError('There are not enough qubits')
        if i==j: raise ValueError('Control and target qubits are the same')
        for k in range(2**(self.num_qubits-2)):
            S = k%2**i + (
               ( k - k%2**i)*2)%2**j + 2*(
                      (k-k%2**i)*2-((2*(k-k%2**i))%2**j)) + 2**j;
            S_ = S + 2**i - 2**j
            a=self.psi[S_]
            self.psi[S_] = self.psi[S]
            self.psi[S] = a
    
    
    def Cx(self,i,j):
        #i = control
        #j = target
        if i>=self.num_qubits: raise ValueError('There are not enough qubits')
        if j>=self.num_qubits: raise ValueError('There are not enough qubits')
        if i==j: raise ValueError('Control and target qubits are the same')
        for k in range(2**(self.num_qubits-2)):
            S = k%2**i + (
               ( k - k%2**i)*2)%2**j + 2*(
                      (k-k%2**i)*2-((2*(k-k%2**i))%2**j)) + 2**i;
            S_ = S + 2**j
            '''
            a=self.psi[S_]
            self.psi[S_] = self.psi[S]
            self.psi[S] = a
            '''
            self.psi[S],self.psi[S_] = self.psi[S_],self.psi[S]
    def Cy(self,i,j):
        if i>=self.num_qubits: raise ValueError('There are not enough qubits')
        if j>=self.num_qubits: raise ValueError('There are not enough qubits')
        if i==j: raise ValueError('Control and target qubits are the same')
        for k in range(2**(self.num_qubits-2)):
            S = k%2**i + (
               ( k - k%2**i)*2)%2**j + 2*(
                      (k-k%2**i)*2-((2*(k-k%2**i))%2**j)) + 2**i;
            S_ = S + 2**j
            self.psi[S],self.psi[S_] = 1j*self.psi[S_],-1j*self.psi[S]

    def MeasureZ(self):
        self.E_z = 0;
        for h in range(2 ** self.num_qubits):
            s = np.binary_repr(h, width=self.num_qubits)
            self.E_z += np.abs(self.psi[h])**2*(s.count('1')-s.count('0'))

    def MeasureX(self):
        self.E_x = 0;
        for i in range(self.num_qubits):
            self.Hx(i);
        for h in range(2 ** self.num_qubits):
            s = np.binary_repr(h, width=self.num_qubits)
            self.E_x += np.abs(self.psi[h])**2*(s.count('1')-s.count('0'))
        for i in range(self.num_qubits):
            self.Hx(i);

    def MeasureY(self):
        self.E_y = 0;
        for i in range(self.num_qubits):
            self.Hy(i);
        for h in range(2 ** self.num_qubits):
            s = np.binary_repr(h, width=self.num_qubits)
            self.E_y += np.abs(self.psi[h])**2*(s.count('1')-s.count('0'))
        for i in range(self.num_qubits):
            self.HyT(i);

    def reduced_density_matrix(self, q):
        rho = np.zeros((2,2), dtype='complex')
        for i in range(2):
            for j in range(i + 1):
                for k in range(2**(self.num_qubits-1)):
                    S = k%(2**q) + 2*(k - k%(2**q))
                    rho[i,j] += self.psi[S + i*2**q] * np.conj(self.psi[S + j*2**q])
                rho[j,i] = np.conj(rho[i,j])
        return rho


