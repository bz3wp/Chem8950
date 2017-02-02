import psi4
import numpy as np
from collections import namedtuple
import configparser

config = configparser.ConfigParser()
config.read('Options.ini')

mol = psi4.geometry(config['DEFAULT']['molecule'])
basis = psi4.core.BasisSet.build(mol, 'BASIS', 'cc-pvdz', puream=0)
# using namedtuple allows us to call x, y, z instead of tuple indices
RecursionResults = namedtuple('RecursionResults', ['x','y','z'])  

def int_recursion(PA, PB, alpha, AMa, AMb):
    
    if len(PA) != 3 or len(PB) !=3:
        raise ''

    # Allocate the x, y, z matrices
    x = np.zeros((AMa+2, AMb+2))
    y = np.zeros((AMa+2, AMb+2))
    z = np.zeros((AMa+2, AMb+2))
    
    x[0,0]=1
    y[0,0]=1
    z[0,0]=1
   # Performing recursion 
    
    for a in range(AMa):
        x[a +1, 0] = PA[0]*x[a,0] + 1/(2*alpha)*a*x[a-1,0]
        y[a +1, 0] = PA[1]*y[a,0] + 1/(2*alpha)*a*y[a-1,0]
        z[a +1, 0] = PA[2]*z[a,0] + 1/(2*alpha)*a*z[a-1,0]

    for b in range(AMb):
        x[0,b+1] = PB[0]*x[0,b] + 1/(2*alpha)*b*x[0,b-1] 
        y[0,b+1] = PB[1]*y[0,b] + 1/(2*alpha)*b*y[0,b-1] 
        z[0,b+1] = PB[2]*z[0,b] + 1/(2*alpha)*b*z[0,b-1] 

    for a in range(1,AMa+1):
        for b in range(AMb):
            x[a,b+1] = PB[0]*x[a,b] + 1/(2*alpha)*a*x[a-1,b] + 1/(2*alpha)*b*x[a,b-1]
            y[a,b+1] = PB[1]*y[a,b] + 1/(2*alpha)*a*y[a-1,b] + 1/(2*alpha)*b*y[a,b-1]
            z[a,b+1] = PB[2]*z[a,b] + 1/(2*alpha)*a*z[a-1,b] + 1/(2*alpha)*b*z[a,b-1]


    # Returning results of recursion 
    return RecursionResults(x, y, z)

def int_overlap(mol,basis):
    S = np.zeros((basis.nao(),basis.nao()))

    for i in range(basis.nshell()):
        for j in range(basis.nshell()):
            for p in range(basis.shell(i).nprimitive):
                for q in range(basis.shell(j).nprimitive): 
                    expp = basis.shell(i).exp(p)
                    expq = basis.shell(j).exp(q)
                    alpha = expp + expq
                    zeta = (expp*expq)/alpha
                    cp = basis.shell(i).coef(p)
                    cq = basis.shell(j).coef(q)
                    A = np.array([mol.x(basis.shell(i).ncenter), mol.y(basis.shell(i).ncenter), mol.z(basis.shell(i).ncenter)])   
                    B = np.array([mol.x(basis.shell(j).ncenter), mol.y(basis.shell(j).ncenter), mol.z(basis.shell(j).ncenter)])
                    P = (expp*A + expq*B)/alpha
                    PA = P - A
                    PB = P - B
                    AB = A - B
                    k = (np.pi/alpha)**(3/2)*np.exp(-zeta*(AB[0]**2 + AB[1]**2 + AB[2]**2))
                    AMa = basis.shell(i).am 
                    AMb = basis.shell(j).am
    
                    (I_x, I_y, I_z) = int_recursion(PA, PB, alpha, AMa+1, AMb+1)
                    counter1 = 0
                    for ii in range(AMa+1):
                        l1 = AMa - ii
                        for jj in range(ii+1):
                            m1 = ii - jj
                            n1 = jj
                            
                            counter2 = 0
                            for aa in range(AMb+1):
                                l2 = AMb - aa
                                for bb in range(aa + 1):                            
                                    m2 = aa - bb
                                    n2 = bb
                                    
                                    S[basis.shell(i).function_index+counter1, basis.shell(j).function_index+counter2] += k*cp*cq*I_x[l1,l2]*I_y[m1,m2]*I_z[n1,n2]
   
                                    counter2 += 1
                            counter1 += 1
    
    return(S)                                    

def int_kinetic(mol,basis):
    T = np.zeros((basis.nao(),basis.nao()))
    
    for i in range(basis.nshell()):
        for j in range(basis.nshell()):
            for p in range(basis.shell(i).nprimitive):
                for q in range(basis.shell(j).nprimitive): 
                    expp = basis.shell(i).exp(p)
                    expq = basis.shell(j).exp(q)
                    alpha = expp + expq
                    zeta = (expp*expq)/alpha
                    cp = basis.shell(i).coef(p)
                    cq = basis.shell(j).coef(q)
                    A = np.array([mol.x(basis.shell(i).ncenter), mol.y(basis.shell(i).ncenter), mol.z(basis.shell(i).ncenter)])   
                    B = np.array([mol.x(basis.shell(j).ncenter), mol.y(basis.shell(j).ncenter), mol.z(basis.shell(j).ncenter)])
                    P = (expp*A + expq*B)/alpha
                    PA = P - A
                    PB = P - B
                    AB = A - B
                    k = (np.pi/alpha)**(3/2)*np.exp(-zeta*(AB[0]**2 + AB[1]**2 + AB[2]**2))
                    AMa = basis.shell(i).am 
                    AMb = basis.shell(j).am
    
                    (x, y, z) = int_recursion(PA, PB, alpha, AMa+1, AMb+1)
                                    
                    counter1 = 0
                    for ii in range(AMa+1):
                        l1 = AMa - ii
                        for jj in range(ii+1):
                            m1 = ii - jj
                            n1 = jj
                            
                            counter2 = 0
                            for aa in range(AMb+1):
                                l2 = AMb - aa
                                for bb in range(aa + 1):                            
                                    m2 = aa - bb
                                    n2 = bb
                                    T_x = (1/2)*(l1*l2*x[l1-1,l2-1]+4*expp*expq*x[l1+1,l2+1]-2*expp*l2*x[l1+1,l2-1]-2*expq*l1*x[l1-1,l2+1])*y[m1,m2]*z[n1,n2]
                                    T_y = (1/2)*(m1*m2*y[m1-1,m2-1]+4*expp*expq*y[m1+1,m2+1]-2*expp*m2*y[m1+1,m2-1]-2*expq*m1*y[m1-1,m2+1])*x[l1,l2]*z[n1,n2]
                                    T_z = (1/2)*(n1*n2*z[n1-1,n2-1]+4*expp*expq*z[n1+1,n2+1]-2*expp*n2*z[n1+1,n2-1]-2*expq*n1*z[n1-1,n2+1])*x[l1,l2]*y[m1,m2]
                                
                                    T[basis.shell(i).function_index+counter1, basis.shell(j).function_index+counter2] += k*cp*cq*(T_x+T_y+T_z)
   
                                    counter2 += 1
                            counter1 += 1
    
    return T       


def int_dipole(mol,basis):
    dx = np.zeros((basis.nao(),basis.nao()))
    dy = np.zeros((basis.nao(),basis.nao()))
    dz = np.zeros((basis.nao(),basis.nao()))
    
    for i in range(basis.nshell()):
        for j in range(basis.nshell()):
            for p in range(basis.shell(i).nprimitive):
                for q in range(basis.shell(j).nprimitive): 
                    expp = basis.shell(i).exp(p)
                    expq = basis.shell(j).exp(q)
                    alpha = expp + expq
                    zeta = (expp*expq)/alpha
                    cp = basis.shell(i).coef(p)
                    cq = basis.shell(j).coef(q)
                    A = np.array([mol.x(basis.shell(i).ncenter), mol.y(basis.shell(i).ncenter), mol.z(basis.shell(i).ncenter)])   
                    B = np.array([mol.x(basis.shell(j).ncenter), mol.y(basis.shell(j).ncenter), mol.z(basis.shell(j).ncenter)])
                    P = (expp*A + expq*B)/alpha
                    PA = P - A
                    PB = P - B
                    AB = A - B
                    k = (np.pi/alpha)**(3/2)*np.exp(-zeta*(AB[0]**2 + AB[1]**2 + AB[2]**2))
                    AMa = basis.shell(i).am 
                    AMb = basis.shell(j).am
    
                    (x, y, z) = int_recursion(PA, PB, alpha, AMa+1, AMb+1)
                                    
                    counter1 = 0
                    for ii in range(AMa+1):
                        l1 = AMa - ii
                        for jj in range(ii+1):
                            m1 = ii - jj
                            n1 = jj
                            
                            counter2 = 0
                            for aa in range(AMb+1):
                                l2 = AMb - aa
                                for bb in range(aa + 1):                            
                                    m2 = aa - bb
                                    n2 = bb

                                    dx[basis.shell(i).function_index+counter1, basis.shell(j).function_index+counter2] += k*cp*cq*(x[l1+1,l2] + A[0]*x[l1,l2])*y[m1,m2]*z[n1,n2]
                                    dy[basis.shell(i).function_index+counter1, basis.shell(j).function_index+counter2] += k*cp*cq*(y[m1+1,m2] + A[1]*y[m1,m2])*x[l1,l2]*z[n1,n2]
                                    dz[basis.shell(i).function_index+counter1, basis.shell(j).function_index+counter2] += k*cp*cq*(z[n1+1,n2] + A[2]*z[n1,n2])*y[m1,m2]*x[l1,l2]
                                    

                                    counter2 += 1
                            counter1 += 1
    
    return dx,dy,dz       



#int_overlap(mol,basis) 
#int_kinetic(mol,basis)
