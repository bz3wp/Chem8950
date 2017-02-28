import psi4
import numpy as np 
from uhf import UHF
import scipy.linalg as la
import time

class CEPA0:

    def __init__(self, uhf):
        self.e = np.sort(uhf.e) 
        self.C = uhf.C[:,uhf.e.argsort()]
        self.nocc = uhf.nocc
        self.ntot = uhf.ntot
        self.nvir = self.ntot - self.nocc
        self.max_iter = uhf.cepa0_max_iter
        self.den_fit = uhf.cepa0_df
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
        time_0 = time.time()
        C, gao, nocc, nvir, e,E= self.C, self.gao, self.nocc, self.nvir, self.e,self.E
        # transform integrals
        gmo = int_trans_2(gao,C,C,C,C) 
        E_old = self.E
        t_old = np.zeros((nocc,nocc,nvir,nvir))
        o = slice(None,nocc)
        v = slice(nocc,None)
        x = np.newaxis
        e_ijab = 1./(e[o,x,x,x]+e[x,o,x,x]-e[x,x,v,x]-e[x,x,x,v])
        for iteration in range(self.max_iter):
            t4 = np.einsum('kbcj,ikac -> ijab', gmo[o,v,v,o],t_old)
            t = (gmo[o,o,v,v]+(1/2)*np.einsum('abcd,ijcd->ijab',gmo[v,v,v,v],t_old)+(1/2)*np.einsum('klij,klab->ijab',gmo[o,o,o,o],t_old)+
                t4.transpose((0,1,2,3))-t4.transpose((1,0,2,3))-t4.transpose((0,1,3,2))+t4.transpose((1,0,3,2)))

            t = t*e_ijab

            E_CEPA0 = (1/4)*np.einsum('ijab,ijab->',gmo[o,o,v,v],t)
            t_norm = np.linalg.norm(t-t_old)
            print('UCEPA0 iteration {:3d}: energy {:20.14f} dE {:2.5E} t_norm {:2.5E}'.format(iteration, E_CEPA0,(E_CEPA0-E_old),t_norm))

            if (abs(E_CEPA0 - E_old))<1.e-10 and t_norm < 1.e-10:
                break 
            E_old = E_CEPA0
            t_old = t
        time_1 = time.time()
        print('The UCEPA0 correlation energy is {:20.14f}'.format(E_CEPA0))
        print('The total UCEPA0 is {:20.14f}'.format(uhf.E_SCF - E_CEPA0))
        print('UCEPA0 took: {:7.5} seconds'.format(time_1-time_0))
        self.E = E_CEPA0
        return E_CEPA0
               
    def get_energy_df(self):
        time_df_0 = time.time() 
        C, gao, nocc, nvir,ntot, e= self.C, self.gao, self.nocc, self.nvir, self.ntot,self.e
        E_df_old = 0.0
        t_old = np.zeros((nocc,nocc,nvir,nvir))
        t2 = np.zeros((nocc,nocc,nvir,nvir))
        o = slice(None,nocc)
        v = slice(nocc,None)
        x = np.newaxis
        gmo_2 = np.zeros((nvir,nvir))
        gmo2 = np.zeros((nvir,nvir))
        gmo1 = int_trans_2(gao,C[:,v],C[:,v],C[:,o],C[:,o])
        gmo3 = int_trans_2(gao,C[:,o],C[:,o],C[:,o],C[:,o])
        gmo4 = int_trans_2(gao,C[:,o],C[:,v],C[:,v],C[:,o])
        gmo_caP = int_trans_df(self.b_pqP,C[:,v]) 
        e_ijab = 1./(e[o,x,x,x]+e[x,o,x,x]-e[x,x,v,x]-e[x,x,x,v])
        
        for iteration in range(self.max_iter):
            for a in range(nvir):
                for b in range(nvir):
                   gmo2 = np.einsum('cP,dP->cd',gmo_caP[a],gmo_caP[b])
                   t2[:,:,a,b] = np.einsum('cd,ijcd->ij',(gmo2-gmo2.T),t_old)
            t4 = np.einsum('kbcj,ikac -> ijab', gmo4,t_old)
            t = e_ijab*(gmo1 + (1/2)*t2+(1/2)*np.einsum('klij,klab->ijab',gmo3,t_old)+
                t4-t4.transpose((1,0,2,3))-t4.transpose((0,1,3,2))+t4.transpose((1,0,3,2)))

            E_CEPA0_df = (1/4)*np.einsum('ijab,ijab->',gmo1,t)
            t_norm = np.linalg.norm(t-t_old)
            #print('DF-UCEPA0 iteration {:3d}: energy {:20.14f} dE {:2.5E} t_norm {:2.5E}'.format(iteration, E_CEPA0_df,(E_CEPA0_df-E_df_old),t_norm))

            if (abs(E_CEPA0_df - E_df_old))<1.e-10 and t_norm < 1.e-10:
                break 
            E_df_old = E_CEPA0_df
            t_old = t
        time_df_1 = time.time()
        print('The DF-UCEPA0 correlation energy is {:20.14f}'.format(E_CEPA0_df))
        print('The total DF-UCEPA0 is {:20.14f}'.format(uhf.E_SCF - E_CEPA0_df))
        print('DF-UCEPA0 error: {:20.14f}'.format(self.E - E_CEPA0_df))
        print('DF-UCEPA0 took: {:7.5} seconds'.format(time_df_1-time_df_0))
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
    cepa0 = CEPA0(uhf)
    if cepa0.den_fit==1: 
        cepa0.get_energy()
        cepa0.get_energy_df()
    else:
        cepa0.get_energy()
