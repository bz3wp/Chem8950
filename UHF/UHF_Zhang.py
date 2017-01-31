#############################################################################
# File name: UHF_Zhang.py
# Author: Boyi Zhang 
# Date: 1/20/17
# Class: CHEM8950
# Description: Unrestricted Hartree-Fock 
#############################################################################


# Loading Psi4, numpy, input file

import psi4
import numpy as np

import configparser 
config = configparser.ConfigParser() 
config.read('Options.ini')

# Setting up initial conditions 

# Defining geometry 
molecule = psi4.geometry(config['DEFAULT']['molecule'])

# Number of alpha electrons
nalpha = int(config['DEFAULT']['nalpha'])

# Number of beta electrons 
nbeta = int(config['DEFAULT']['nbeta'])

# Maximum number of SCF iterations 
SCF_MAX_ITER = int(config['SCF']['max_iter'])

# Constructing new molecule in C1 symmetry 
molecule.update_geometry()
print('The nuclear repulsion energy is %20.14f' % molecule.nuclear_repulsion_energy())

# Loading in basis set 
basis = psi4.core.BasisSet.build(molecule, "BASIS", config['DEFAULT']['basis'])

# Creating MintsHelper object to compute the integrals 
mints = psi4.core.MintsHelper(basis)

#Computing the needed integrals

# Overlap
S = mints.ao_overlap().to_array()

# Kinetic
T = mints.ao_kinetic().to_array()

# Potential
V = mints.ao_potential().to_array() 

# Two-electron repulsion 
I = mints.ao_eri().to_array() 

# Form the one-electron Hamiltonian (H) 
H = T + V
print('The one-electron Hamiltonian is:\n', H)

# Constructing orthogonalizer A=S^(-1/2) using psi4
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = A.to_array()

# Constructing initial density matrix 

# Form an initial transformed Ft matrix using H matrix 
Ft = A.dot(H).dot(A)

# Diagonalize the Ft matric using standard eigenvalue routine
e, C = np.linalg.eigh(Ft)

# Form SCF eigenvector matrix 
C = A.dot(C)

# Forming doubly occupied orbitals from eigenvector matrix
Ca = C[:, :nalpha]
Cb = C[:, :nbeta]

# Form the first density matrix(D) using doubly occupied orbitals from eigenvector matrix
Da = np.einsum('pi,qi->pq', Ca, Ca)
Db = np.einsum('pi,qi->pq', Cb, Cb)
print('Da:\n', Da)
print('Db:\n', Db)

# SCF iterations 

E = 0.0
Eold = 0.0

#Returns array of zeros with same type as D
Da_old = np.zeros_like(Da)
Db_old = np.zeros_like(Db) 

# Starting for-loop for scf iteration
for iteration in range(1, SCF_MAX_ITER+1):

    # Build Fock matrix
    Ja = np.einsum('pqrs, rs->pq', I, Da)
    Jb = np.einsum('pqrs, rs->pq', I, Db)
    Ka = np.einsum('prqs, rs->pq', I, Da)
    Kb = np.einsum('prqs, rs->pq', I, Db)
    Fa = H + Ja - Ka + Jb
    Fb = H + Jb - Kb + Ja

    # Calculate SCF energy 
    E_SCF = (1/2)*(np.einsum('pq,pq->', Fa + H, Da) + np.einsum('pq,pq->',Fb + H, Db)) + molecule.nuclear_repulsion_energy()
    print('UHF iteration %3d: energy %20.14f dE %1.5E' % (iteration, E_SCF, (E_SCF - Eold)))

    # Stops if converges under certain threshold
    if (abs(E_SCF - Eold) < 1.e-10):
        break 
    
    # Redefine values
    Eold = E_SCF
    Da_old = Da
    Db_old = Db

    # Transform the Fock matrix 
    Ft_a = A.dot(Fa).dot(A)
    Ft_b = A.dot(Fb).dot(A)

    # Diagonalize the Fock matrix
    ea, Calpha = np.linalg.eigh(Ft_a)
    eb, Cbeta = np.linalg.eigh(Ft_b)

    # Construct new SCF eigenvector matrix
    Calpha = A.dot(Calpha)
    Cbeta = A.dot(Cbeta)
    # Form the density matrix 
    Ca = Calpha[:, :nalpha]
    Cb = Cbeta[:, :nbeta]
    Da = np.einsum('pi,qi->pq', Ca, Ca)
    Db = np.einsum('pi,qi->pq', Cb, Cb)

    
