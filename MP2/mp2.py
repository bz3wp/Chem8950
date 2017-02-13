import psi4
import numpy as np
from uhf import UHF

class MP2:

    def __init__(self, uhf):
        self.e = np.sort(uhf.e)
        self.C = uhf.C[:,uhf.e.argsort()]
        self.nocc = uhf.nocc
        self.ntot = uhf.ntot
        g = spin_block_tei(uhf.I) 
        self.gao = g.transpose(0,2,1,3)-g.transpose(0,2,3,1) 

    def get_energy(self): 
        C, gao, nocc, ntot, e = self.C, self.gao, self.nocc, self.ntot, self.e
        # transform integrals
        gmo = int_trans_2(gao,C) 
        # get energy
        E = 0.0
        for i in range(nocc):
            for j in range(nocc):
                for a in range(nocc,ntot):
                    for b in range(nocc,ntot):
                        E += ((1/4)*gmo[i,j,a,b]**2)/(e[i]+e[j]-e[a]-e[b])
        print('The MP2 correlation energy is {:20.14f}'.format(E))
        return E 
                

def spin_block_tei(gao):
    I = np.eye(2)
    gao = np.kron(I, gao)
    return np.kron(I, gao.T)

def int_trans_1(gao, C):
    return np.einsum('pqrs, pP, qQ, rR, sS -> PQRS', gao, C, C, C, C)

def int_trans_2(gao, C):

    return np.einsum('pQRS, pP -> PQRS',
           np.einsum('pqRS, qQ -> pQRS',
           np.einsum('pqrS, rR -> pqRS',
           np.einsum('pqrs, sS -> pqrS', gao, C),C),C),C)


if __name__ == '__main__':
    uhf = UHF('Options.ini')
    uhf.get_energy()
    mp2 = MP2(uhf)
    mp2.get_energy()
    psi4.set_options({'basis':'sto-3g',
                        'scf_type': 'pk',
                        'MP2_type' : 'conv',
                        'puream' : False,
                        'reference': 'uhf',
                        'guess' : 'core',
                        'e_convergence' : 1e-10})
    psi4.energy('mp2')
