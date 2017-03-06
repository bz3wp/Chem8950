import psi4
import numpy as np 
from uhf import UHF
import scipy.linalg as la
import time

class CCD:

    def __init__(self, uhf):
        self.e = np.sort(uhf.e) 
        self.C = uhf.C[:,uhf.e.argsort()]
        self.nocc = uhf.nocc
        self.ntot = uhf.ntot
        self.nvir = self.ntot - self.nocc
        self.max_iter = uhf.ccd_max_iter
        self.den_fit = uhf.ccd_df
        df = uhf.df_basis
        basis = uhf.basis
        mints = psi4.core.MintsHelper(basis) 
        zero = psi4.core.BasisSet.zero_ao_basis_set()
        g = spin_block_tei(uhf.I)
        self.gao = g.transpose(0,2,1,3)-g.transpose(0,2,3,1)
        self.E = 0.0 
    def get_energy(self): 
        time_0 = time.time()
        C, gao, nocc, nvir, e,E= self.C, self.gao, self.nocc, self.nvir, self.e,self.E
        # transform integrals
        E_old = self.E
        t_old = np.zeros((nocc,nocc,nvir,nvir))
        o = slice(None,nocc)
        v = slice(nocc,None)
        x = np.newaxis
        e_ijab = 1./(e[o,x,x,x]+e[x,o,x,x]-e[x,x,v,x]-e[x,x,x,v])
        gmo2 = int_trans_2(gao,C[:,v],C[:,v],C[:,v],C[:,v]) 
        gmo1 = int_trans_2(gao,C[:,v],C[:,v],C[:,o],C[:,o])
        gmo3 = int_trans_2(gao,C[:,o],C[:,o],C[:,o],C[:,o])
        gmo4 = int_trans_2(gao,C[:,v],C[:,o],C[:,o],C[:,v])
        gmo5 = int_trans_2(gao,C[:,o],C[:,o],C[:,v],C[:,v])
        for iteration in range(self.max_iter):
            t4 = np.einsum('cjkb,ikac -> ijab', gmo4,t_old)
            t6 =(1/2)*np.einsum('cdkl,ikac,ljdb->ijab',gmo5,t_old,t_old)
            t7 =(1/2)*np.einsum('cdkl,klca,ijdb->ijab',gmo5,t_old,t_old)
            t8 = (1/2)*np.einsum('cdkl,kicd,ljab->ijab',gmo5,t_old,t_old)
            t = (gmo1+(1/2)*np.einsum('cdab,ijcd->ijab',gmo2,t_old)+(1/2)*np.einsum('ijkl,klab->ijab',gmo3,t_old)+
                t4-t4.transpose((1,0,2,3))-t4.transpose((0,1,3,2))+t4.transpose((1,0,3,2))+
                ((1/2)**2)*np.einsum('cdkl,klab,ijcd->ijab',gmo5,t_old,t_old) + t6 - t6.transpose((1,0,2,3)) - t6.transpose((0,1,3,2))+t6.transpose((1,0,3,2))-
                t7+t7.transpose((0,1,3,2)) - t8 + t8.transpose((1,0,2,3)))

            t = t*e_ijab

            E_CCD = (1/4)*np.einsum('ijab,ijab->',gmo1,t)
            t_norm = np.linalg.norm(t-t_old)
            print('UCCD iteration {:3d}: energy {:20.14f} dE {:2.5E} t_norm {:2.5E}'.format(iteration, E_CCD,(E_CCD-E_old),t_norm))

            if (abs(E_CCD - E_old))<1.e-10 and t_norm < 1.e-10:
                break 
            E_old = E_CCD
            t_old = t
        time_1 = time.time()
        print('The UCCD correlation energy is {:20.14f}'.format(E_CCD))
        print('The total UCCD is {:20.14f}'.format(uhf.E_SCF + E_CCD))
        print('UCCD took: {:7.5} seconds'.format(time_1-time_0))
        self.E = E_CCD
        return E_CCD
               
def spin_block_tei(gao):
    I = np.eye(2)
    gao = np.kron(I, gao)
    return np.kron(I, gao.T)

def spin_block_tei_df(gao):
    I = np.eye(2)
    return np.kron(I,gao.T).T
    
def int_trans_1(gao, C):
    return np.einsum('pqrs, pP, qQ, rR, sS -> PQRS', gao, C, C, C, C)

def int_trans_2(gao,C1,C2,C3,C4):

    return np.einsum('pQRS, pP -> PQRS',
           np.einsum('pqRS, qQ -> pQRS',
           np.einsum('pqrS, rR -> pqRS',
           np.einsum('pqrs, sS -> pqrS', gao, C1),C2),C3),C4)

def int_trans_df(b_pqP, C):
    a = np.einsum('pqP,pi -> iqP',b_pqP,C)
    return np.einsum('iqP,qa -> iaP',a,C)
    

if __name__ == '__main__':
    uhf = UHF('Options.ini')
    uhf.get_energy()
    ccd= CCD(uhf)
    ccd.get_energy()
