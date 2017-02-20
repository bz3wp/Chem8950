import psi4
import numpy as np
from uhf import UHF
import scipy.linalg as la
import time

class MP2:

    def __init__(self, uhf):
        self.e = np.sort(uhf.e)
        self.C = uhf.C[:,uhf.e.argsort()]
        self.nocc = uhf.nocc
        self.ntot = uhf.ntot

        df = uhf.df_basis
        basis = uhf.basis
        mints = psi4.core.MintsHelper(basis) 
        zero = psi4.core.BasisSet.zero_ao_basis_set()

        g = spin_block_tei(uhf.I) 
        self.gao = g.transpose(0,2,1,3)-g.transpose(0,2,3,1) 

        J = mints.ao_eri(df,zero,df,zero).to_array()
        J = np.squeeze(J)
        self.J_prime = la.inv(la.sqrtm(J))  #where J_prime = J**(-1/2)

        pqP = mints.ao_eri(basis,basis,zero,df).to_array()
        pqP = spin_block_tei_df(pqP)
        pqP = np.squeeze(pqP)
        self.b_pqP = np.einsum('pqP,QP->pqQ',pqP,self.J_prime)
    
        self.E = 0.0 
    def get_energy(self): 
        t0 = time.time()
        C, gao, nocc, ntot, e, E = self.C, self.gao, self.nocc, self.ntot, self.e, self.E
        # transform integrals
        gmo = int_trans_2(gao,C) 
        # get energy
        for i in range(nocc):
            e_i = e[i]
            for j in range(i,nocc):
                e_j = e[j]
                for a in range(nocc,ntot):
                    for b in range(a,ntot):
                        E += (gmo[i,j,a,b]**2)/(e_i+e_j-e[a]-e[b])
        t1 = time.time()
        print('The MP2 correlation energy is {:20.14f}'.format(E))
        print('MP2 took {:7.5f} seconds'.format(t1-t0))
        self.E = E
        return E 
               
    def get_energy_df(self): 
        t2 = time.time()
        E_df = 0.0
        C, nocc, ntot, e, E = self.C, self.nocc, self.ntot, self.e, self.E
        b_iaP = int_trans_df(self.b_pqP,C) 

        e_ab = e[nocc:]
        e_vv = e_ab.reshape(-1, 1) + e_ab

        for i in range(nocc):
            e_i = e[i]
            for j in range(i,nocc):
                e_j = e[j]
                
                e_denom = 1.0 / (e_i + e_j - e_vv)

                gmo_df_ab = np.einsum('aP, bP ->ab', b_iaP[i,nocc:,:],b_iaP[j,nocc:,:])
                E_df += np.einsum('ab,ab,ab->', gmo_df_ab, gmo_df_ab - gmo_df_ab.T, e_denom) 
        t3 = time.time()
        print('The DF-MP2 correlation energy is {:20.14f}'.format(E_df))
        print('DF error: {:20.14f}'.format(E-E_df))
        print('DF-MP2 took {:7.5f} seconds'.format(t3-t2))
def spin_block_tei(gao):
    I = np.eye(2)
    gao = np.kron(I, gao)
    return np.kron(I, gao.T)

def spin_block_tei_df(gao):
    I = np.eye(2)
    return np.kron(I,gao.T).T
    
def int_trans_1(gao, C):
    return np.einsum('pqrs, pP, qQ, rR, sS -> PQRS', gao, C, C, C, C)

def int_trans_2(gao, C):

    return np.einsum('pQRS, pP -> PQRS',
           np.einsum('pqRS, qQ -> pQRS',
           np.einsum('pqrS, rR -> pqRS',
           np.einsum('pqrs, sS -> pqrS', gao, C),C),C),C)

def int_trans_df(b_pqP, C):
    a = np.einsum('pqP,pi -> iqP',b_pqP,C)
    return np.einsum('iqP,qa -> iaP',a,C)
    
if __name__ == '__main__':
    uhf = UHF('Options.ini')
    uhf.get_energy()
    mp2 = MP2(uhf)
    mp2.get_energy()
    mp2.get_energy_df()
    psi4.set_options({'basis':'cc-pvdz',
                        'scf_type': 'pk',
                        'MP2_type' : 'conv',
                        'puream' : False,
                        'reference': 'uhf',
                        'guess' : 'core',
                        'e_convergence' : 1e-10})
    #psi4.energy('mp2')
