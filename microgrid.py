# -*- coding: utf-8 -*-
"""
CIGRE microgrid benchmark class
"""

import numpy as np
from matplotlib import pyplot as plt


def EV_Demand(nEV):
  ''' 
  Electric Vehicle stochastic demand generation.
  '''
  nEV_val = np.array([10.0, 20.0, 30.0]) #Number of Vehicles
  idx = np.where(nEV_val == nEV)[0][0]
  shape = [2.2784,3.71581,5.15059] #Gamma's shape parameter
  scale = [103.043,110.621,133.027] #Gamma's scale parameter
  return -np.random.gamma(shape[idx],scale=scale[idx])/10000 #EV demand in W

def Wind_Sce(Pnom, dist, K1, K2):
  ''' 
  Wind Generation scenario assumming a normal, Weibull or beta distributions, 
  with parameters K1 and K2.
  '''
  # TODO uptade paremeters according to footnote
  weibull = 1.0
  beta = 2.0
  normal = 3.0
  if (dist == weibull):    
    #v = (-K1*np.log(1.-np.random.rand()))**(1.0/K2)        
    v = K1*np.random.weibull(K2)
  
  if (dist == normal):
     v = (K1+np.random.randn()*K2)  
     
  P = 0.
  if (v<3.5 or v>20.):
      P = 0 
  elif (v<=14.5 and v>=3.5):
      P = Pnom*(v-3.5)/11.
  else:
      P = Pnom
  return P

def Solar_Sce(Pnom, dist, K1, K2):
  ''' 
  Solar Generation scenario assumming a normal, Weibull or beta distributions, 
  with parameters K1 and K2.  Nominal radiation scenario 1000 W/m^2  
  which represents a nomila power Pnom.
  '''
  # TODO buscar articulo con granjas de paneles para incluir en la simulacion
  weibull = 1.0
  beta = 2.0
  normal = 3.0
  P = 0.
  if (dist == weibull):    
    v = (-K1*np.log(1.-np.random.rand()))**(1.0/K2)    
    if (v<=12.):
        P = Pnom*(v/12.)**3
    else:
        P = Pnom

  if (dist == beta):
    A, eta = 7000, 15
    s = np.random.beta(K1,K2)
    P = (eta*A*s)/100000*Pnom

  if (dist == normal):
      P = (K1+np.random.randn()*K2)/1000*Pnom  
  return P


class mgrid(object):
    def __init__(self):
        Vnom = 400 # Line to Line voltage in volts
        wnom = 2.*np.pi*60. # Nominal angular velocity
        Pnom = 10000 # Nominal power in watts
        # N1 N2 L(m) type 
        Lines = np.array([[0,1,35,0],
                       [1,2,35,0],
                       [2,3,35,0],
                       [3,4,35,0],
                       [4,5,35,0],
                       [5,6,35,0],
                       [6,7,35,0],
                       [7,8,35,0],
                       [8,9,35,0],
                       [2,10,30,1],
                       [3,11,30,2],
                       [5,12,30,3],
                       [9,13,30,2],
                       [3,14,35,0],
                       [14,15,35,0],
                       [15,16,35,0],
                       [16,17,30,0],
                       [8,18,30,1]])
        # Rph Xph Ro Xo(ohm/km) Cap(uF/km)
        Impedances = np.array([[0.284,0.083,1.136,0.417,0.38], #(1) OL-Twisted cable 4x120 mm2
                               [3.690,0.094,13.64,0.472,0.05], #(2) SC - 4x6 mm2 Cu
                               [1.380,0.082,5.520,0.418,0.18], #(3) SC - 4x16 mm2 Cu
                               [0.871,0.081,3.480,0.409,0.22]])#(4) SC - 4x25 mm2 Cu   
        
        DemandL = np.array([10,12,13,17,18])
        DemandP = np.array([100.0,55000.0,47000.0,72000.0,15000.0])/10.0
        #node 10 EV
        #node number, type 0:normal 1:EV, params
        DemandType = np.array([1,0,0,0,0])
        solar = 1.0
        wind = 2.0
        EV = 3.0
        weibull = 1.0
        beta = 2.0
        normal = 3.0
        # node power Kp Kq Tau type dist K1 K2
        Converters = np.array([[11,Pnom,0.05,0.04,0.32E-3,solar,beta,900,40],
                            [12,Pnom,0.08,0.09,0.38E-3,solar,beta,900,40],
                            [13,2000,0.10,0.09,0.41E-3,wind,weibull,11,1.2],
                            [17,Pnom,0.09,0.10,0.31E-3,solar,beta,900,40],
                            [18,2000,0.08,0.08,0.34E-3,wind,weibull,11,1.2]])
        #Organizar la estructura
        NumN = np.max([np.max(Lines[:,0]),np.max(Lines[:,1])])+1
        NumL = np.size(Lines[:,0])
        NumD = np.size(DemandL) 
        NumC = np.size(Converters[:,0])
        Zbase = Vnom*Vnom/Pnom
        Y = 1j*np.zeros((NumN,NumN))
        Yc = 1j*np.zeros((NumN,1))
        mg = 1j*np.zeros((NumL,4))
        zlin = 1j*np.zeros((NumL,1))
        A = np.zeros((NumL,NumN))
        for k in range(NumL):
            n1 = Lines[k,0]
            n2 = Lines[k,1]
            t = Lines[k,3]
            z = (Impedances[t,0]+1j*Impedances[t,1])/1000/Zbase*Lines[k,2]
            b = 1j*Impedances[t,4]*wnom/1000*Lines[k,2]*Zbase*1E-6
            mg[k,0:4] = np.array([n1,n2,z,b])
            Y[n1,n1] = Y[n1,n1] + 1.0/z + b
            Y[n1,n2] = Y[n1,n2] - 1.0/z
            Y[n2,n1] = Y[n2,n1] - 1.0/z
            Y[n2,n2] = Y[n2,n2] + 1.0/z + b
            A[k,n1] =  1.0
            A[k,n2] = -1.0
            zlin[k] = z
            Yc[n1] = Yc[n1] + b
            Yc[n2] = Yc[n2] + b
            
        # Include the demand as constant impedances
        DemandP = DemandP/Pnom
        Yd = 1j*np.zeros((NumN,1))
        for k in range(NumD):
            n1 = np.int32(DemandL[k])
            g = DemandP[k]
            Y[n1,n1] = Y[n1,n1] + g
            Yd[n1] = Yd[n1] + g
            
        # Load flow in grid mode
        Converters[:,1] = Converters[:,1]/Pnom
        s = np.zeros((NumN,))
        for k in range(NumC):
            n1 = np.int32(Converters[k,0]) #node
            s[n1] = Converters[k,1]
            
        # Yn = csr_matrix(Y[1:,1:]) #Using scipy
        Yn = Y[1:,1:]
        Y0 = Y[1:,0]
        V  = 1j*np.ones((NumN,))
        er = 1.0
        while er>1E-8:
            V[1:] = np.linalg.solve(Yn,np.conj(s[1:]/V[1:])-Y0*V[0])
            In = Y@V
            sc = V*np.conj(In)
            er = np.linalg.norm(s[1:]-sc[1:])
            
        self.NumN = NumN  # Numbe of nodes
        self.NumL = NumL  # Number of lines
        self.NumD = NumD  # Number of loads
        self.NumC = NumC  # Number of converters
        self.lines = mg
        self.demandP = DemandP
        self.demandL = DemandL
        self.demandType = DemandType
        self.converters = Converters
        self.Ybus = Y
        self.V = V
        self.A = A
        self.zlin = zlin
        self.Yd = Yd
        self.Yc = Yc
        self.MC = lambda:0 # Easy way to create an object
        self.MC.nd = 100 # Number of samples by default
        self.savename = 'data'
        self.path = ''
        self.npar = 5

    def microgrid1(self):
        '''
        A Benchmark Test System for Networked Microgrids
        '''
        Vnom = 11000 # Line to Line voltage in volts
        wnom = 2.*np.pi*50. # Nominal angular velocity
        Pnom = 10000 # Nominal power in watts

        Lines = np.array([[0,1,3000,0],
                        [1,2,3000,0],
                        [1,3,2400,1],
                        [1,4,3000,0],
                        [1,5,1600,2],
                        [1,6,1600,2],
                        [2,3,2000,3],
                        [2,5,1500,4],
                        [3,4,2000,3],
                        [3,5,1200,5],
                        [3,6,1200,5],
                        [4,6,1500,4]])

        Impedances = np.array([[92.70,143.10,1.136,0.417,0.0], #(0)
                            [93.36,119.88,13.64,0.472,0.0], #(1)
                            [62.24,79.920,5.520,0.418,0.0], #(2)
                            [77.80,99.900,3.480,0.409,0.0], #(3)
                    [58.35,74.930,3.480,0.409,0.0], #(4)
                    [46.68,59.940,3.480,0.409,0.0]]) #(5)

        DemandL = np.array([2,3,4,5,6,6])
        DemandP = np.array([2125.0,3329.0,2050.0,1257.0,1056.0,100.0])/10.0
        #node number, type 0:normal 1:EV, params
        #node 6 ev
        DemandType = np.array([0,0,0,0,0,1])
        #Typer of source 
        solar = 1.0
        wind = 2.0
        #Type of distribution
        weibull = 1.0
        beta = 2.0
        normal = 3.0
        Converters = np.array([[2,Pnom,0.08,0.09,0.38E-3,solar,beta,900,40],
                        [3,Pnom,0.10,0.09,0.41E-3,solar,beta,11,1.2],
                        [4,Pnom,0.09,0.10,0.31E-3,solar,beta,900,40]])
        
        #Organizar la estructura
        NumN = np.max([np.max(Lines[:,0]),np.max(Lines[:,1])])+1
        NumL = np.size(Lines[:,0])
        NumD = np.size(DemandL) 
        NumC = np.size(Converters[:,0])
        Zbase = Vnom*Vnom/Pnom
        Y = 1j*np.zeros((NumN,NumN))
        Yc = 1j*np.zeros((NumN,1))
        mg = 1j*np.zeros((NumL,4))
        zlin = 1j*np.zeros((NumL,1))
        A = np.zeros((NumL,NumN))
        for k in range(NumL):
            n1 = Lines[k,0]
            n2 = Lines[k,1]
            t = Lines[k,3]
            z = (Impedances[t,0]+1j*Impedances[t,1])/1000/Zbase*Lines[k,2]
            b = 1j*Impedances[t,4]*wnom/1000*Lines[k,2]*Zbase*1E-6
            mg[k,0:4] = np.array([n1,n2,z,b])
            Y[n1,n1] = Y[n1,n1] + 1.0/z + b
            Y[n1,n2] = Y[n1,n2] - 1.0/z
            Y[n2,n1] = Y[n2,n1] - 1.0/z
            Y[n2,n2] = Y[n2,n2] + 1.0/z + b
            A[k,n1] =  1.0
            A[k,n2] = -1.0
            zlin[k] = z
            Yc[n1] = Yc[n1] + b
            Yc[n2] = Yc[n2] + b
            
        # Include the demand as constant impedances
        DemandP = DemandP/Pnom
        Yd = 1j*np.zeros((NumN,1))
        for k in range(NumD):
            n1 = np.int32(DemandL[k])
            g = DemandP[k]
            Y[n1,n1] = Y[n1,n1] + g
            Yd[n1] = Yd[n1] + g
            
        # Load flow in grid mode
        Converters[:,1] = Converters[:,1]/Pnom
        s = np.zeros((NumN,))
        for k in range(NumC):
            n1 = np.int32(Converters[k,0])
            s[n1] = Converters[k,1]
            
        # Yn = csr_matrix(Y[1:,1:]) #Using scipy
        Yn = Y[1:,1:]
        Y0 = Y[1:,0]
        V  = 1j*np.ones((NumN,))
        er = 1.0
        while er>1E-8:
            V[1:] = np.linalg.solve(Yn,np.conj(s[1:]/V[1:])-Y0*V[0])
            In = Y@V
            sc = V*np.conj(In)
            er = np.linalg.norm(s[1:]-sc[1:])
            
        self.NumN = NumN  # Numbe of nodes
        self.NumL = NumL  # Number of lines
        self.NumD = NumD  # Number of loads
        self.NumC = NumC  # Number of converters
        self.lines = mg
        self.demandP = DemandP
        self.demandL = DemandL
        self.DemandType = DemandType
        self.converters = Converters
        self.Ybus = Y
        self.V = V
        self.A = A
        self.zlin = zlin
        self.Yd = Yd
        self.Yc = Yc
        self.MC = lambda:0 # Easy way to create an object
        self.MC.nd = 100 # Number of samples by default
        self.npar = 3

    def Newton_Freq(self, convs, iterations=50):
        # initialization
        w = 1.0
        Vn = np.ones((self.NumC,1)) 
        an = np.zeros((self.NumC,1))

        xi = np.expand_dims(convs[:,2], axis=1)
        zita = np.expand_dims(convs[:,3], axis=1)
        N = np.int_(convs[:,0]) #Nodes
        S = np.setdiff1d(np.arange(self.NumN),N)
        n = np.size(N)
        Pref = np.expand_dims(convs[:,1], axis=1)
        Qref = np.zeros((n,1))
        #er = 10
        for _ in range(iterations):
            ylin = np.diag( 1./(np.real(self.zlin)+w*1j*np.imag(self.zlin))[:,0] )
            Llin = np.diag( np.imag(self.zlin)[:,0] )
            Yd2 = np.real(self.Yd) + w*1j*np.imag(self.Yd)  # admitance load model
            Yb = (self.A.T @ ylin @ self.A) + np.diag(Yd2[:,0]) + np.diag(self.Yc[:,0])*w
            Db = -(self.A.T @ ylin)@(1j*Llin)@(ylin @ self.A) + 1j*np.imag(Yd2) + np.diag(self.Yc[:,0])
            Y = Yb[np.ix_(N, N)] - Yb[np.ix_(N, S)]@(np.linalg.inv(Yb[np.ix_(S, S)])@Yb[np.ix_(S, N)])
            D = Db[np.ix_(N, N)] - Db[np.ix_(N, S)]@(np.linalg.inv(Db[np.ix_(S, S)])@Db[np.ix_(S, N)])
            Dr = np.real(D)
            Di = np.imag(D)
            G = np.real(Y)
            B = np.imag(Y)
            Vs = Vn*np.exp(1j*an)
            I = Y@Vs
            Sn = Vs*np.conj(I)
            P = np.real(Sn)
            Q = np.imag(Sn)
            dP = (Pref-P)+(1.-w)/xi
            dQ = (Qref-Q)+(1.-Vn)/zita
            dT = (np.sum(an/xi)/np.sum(1./xi))
            er = np.linalg.norm(np.array([np.linalg.norm(dP),np.linalg.norm(dQ),np.abs(dT)]))
            Dpt = np.zeros((n,n))
            Dpv = np.zeros((n,n))
            Dpw = np.zeros((n,1))
            Dqt = np.zeros((n,n))
            Dqv = np.zeros((n,n))
            Dqw = np.zeros((n,1))   
            for k in range(n):
                Dpw[k] = 0.
                Dqw[k] = 0.
                for m in range(n):
                    akm = an[k,0]-an[m,0]
                    if k == m:
                        Dpt[k,k] = -B[k,k]*Vn[k,0]**2-Q[k,0]
                        Dpv[k,k] =  G[k,k]*Vn[k,0]+P[k,0]/Vn[k,0]
                        Dqt[k,k] = -G[k,k]*Vn[k,0]**2+P[k,0]
                        Dqv[k,k] = -B[k,k]*Vn[k,0]+Q[k,0]/Vn[k,0]
                    else:                
                        Dpv[k,m] = G[k,m]*Vn[k,0]*np.cos(akm)+B[k,m]*Vn[k,0]*np.sin(akm)
                        Dqv[k,m] = -(B[k,m]*Vn[k,0]*np.cos(akm)-G[k,m]*Vn[k,0]*np.sin(akm))  # TODO: check this term
                        Dpt[k,m] =  Dqv[k,m]*Vn[m,0]                                 #TODO: check this too!
                        Dqt[k,m] = -Dpv[k,m]*Vn[m,0]

                    Dpw[k] = Dpw[k] + Dr[k,m]*Vn[k,0]*Vn[m,0]*np.cos(akm)+Di[k,m]*Vn[k]*Vn[m]*np.sin(akm)
                    Dqw[k] = Dqw[k] - Di[k,m]*Vn[k,0]*Vn[m,0]*np.cos(akm)+Dr[k,m]*Vn[k]*Vn[m]*np.sin(akm)
            
            Jac = np.block([ [Dpt, Dpv, Dpw+1./xi],
                   [Dqt, Dqv+np.diag((1./zita)[:,0]), Dqw],
                   [-1./xi.T, np.zeros((1,n+1))] ])
            dX = np.linalg.solve(Jac,np.block([[dP],[dQ],[dT]]))
            an = an + dX[:n]
            Vn = Vn + dX[n:2*n]
            w  = w + dX[-1]
            if er<1E-8:
                break
        #print('Eigs: ',np.real(np.linalg.eigvals(Jac))<0.0)
        return w,Vs,P,Q,I,np.sum(P)
    
    def Montecarlo(self,xi,zita, graficar=False, flagres=False, savedata=False, flagseed=False, seed =0):
        #flagres para incluir MAE de la frecuencia en la funcion objetivo
        # Frequency variation given by p and q
        nd = self.MC.nd  # Monte Carlo number of iterations
        w = np.zeros((nd,1)) #Newton
        dp = np.zeros((nd,1))
        dv = np.zeros((nd,2))
        imax = np.zeros((nd,1))
        dpq = np.zeros((nd,2)) # caching the diference between p_ref and p
        convs = self.converters.copy()
        convs[:,2] = xi.copy() 
        convs[:,3] = zita.copy()
        solar = 1.0
        wind = 2.0
        EV = 3.0  
        #demand = self.demand.copy()
        if flagseed:
            np.random.seed(seed)
        for k in range(nd):
            #demand[:,1]= self.demand[:,1]*np.random.rand(self.NumD)*10  # Demands are uniform distributed
            Yd = 1j*np.zeros((self.NumN,1))
            for n in range(self.NumD):
                n1 = np.int32(self.demandL[n]) #node number
                if self.demandType[n]==1:
                    g = EV_Demand(10.0)
                else:
                    g = self.demandP[n] + 0.05*self.demandP[n]*np.random.randn()  # Demand value
                Yd[n1] += g
            self.Yd = Yd.copy()
            
            for c in range(self.NumC):
                Pnom = self.converters[c,1]
                dist = self.converters[c,6]
                K1 = self.converters[c,7]
                K2 = self.converters[c,8]
                if (self.converters[c,5] == solar):
                    convs[c,1] = Solar_Sce(Pnom,dist,K1,K2)
                    
                if (self.converters[c,5] == wind):
                    convs[c,1] = Wind_Sce(Pnom,dist,K1,K2)

            w[k],v,p,q,i,dp[k] = self.Newton_Freq(convs)
            dv[k,0] = np.min(np.abs(v))
            dv[k,1] = np.max(np.abs(v))
            imax[k] = np.max(np.abs(i))
            pg = p[:,0]+Yd[np.int_(convs[:,0]),0].real
            print('Yd',Yd[np.int_(convs[:,0]),0].real)
            print('p', p[:,0])
            print('pg', pg)
            print('Pref',convs[:,1])
            print('Sum P',np.sum(p))
            
            dpq[k,0] = np.max(np.abs(pg-convs[:,1]))
            dpq[k,1] = np.max(np.abs(q))
        
        #mae = self.dynamic_sim(convs, nd=100)
        #print(mae)
        #if np.isnan(mae):
        #    mae = 1.0
        #else:
        #    mae /= 520.0

        if graficar:
            fig01 = plt.figure()
            ax01 = fig01.add_subplot(1, 1, 1) 
            plt.hist(w,30)
            ax01.grid()
            ax01.set_title('$\omega$')

            fig02, (ax021, ax022) = plt.subplots(nrows=2)
            ax021.hist(dpq[:,0],30)
            ax021.grid()
            ax021.set_title('$\Delta$ p')
            ax022.hist(dpq[:,1],30)
            ax022.grid()
            ax022.set_title('$\Delta$ q')

            fig03, (ax031, ax032) = plt.subplots(nrows=2)
            ax031.hist(dv[:,0],30)
            ax031.grid()
            ax031.set_title('$v_{min}$')
            ax032.hist(dv[:,1],30)
            ax032.grid()
            ax032.set_title('$v_{max}$')

            fig04 = plt.figure()
            ax04 = fig04.add_subplot(1, 1, 1) 
            plt.hist(imax,30)
            ax04.grid()
            ax04.set_title('$imax$')

            fig01.savefig(self.path+'omega.pdf',dpi=200)
            fig02.savefig(self.path+'dpq.pdf',dpi=200)
            fig03.savefig(self.path+'dv.pdf',dpi=200)
            fig04.savefig(self.path+'imax.pdf',dpi=200)
        
        if savedata:
            np.savez(self.savename,w=w,dp=dp,dv=dv,dpq=dpq,imax=imax,
                     params=np.concatenate((xi, zita)))

        '''
        # results on w
        self.MC.mu_w = np.mean(w)
        self.MC.sigma_w = np.std(w)
    
        a = np.where(w>1.05)
        self.MC.var_pos_w = np.size(a)/nd
    
        a = np.where(w<0.95)
        self.MC.var_neg_w = np.size(a)/nd
      
        # results on vmax
        a = np.where(dv[:,1]>1.05)
        self.MC.var_pos_v = np.size(a)/nd
    
        # results on vmin
        a = np.where(dv[:,0]<0.95)
        self.MC.var_neg_v = np.size(a)/nd
    
        # results on delta_p  % deviations larger than 0.5
        a = np.where(dpq[:,0]>0.5)
        self.MC.var_dp = np.size(a)/nd
    
        # results on delta_q  % deviations larger than 0.5
        a = np.where(dpq[:,1]>0.5)
        self.MC.var_dq = np.size(a)/nd

        self.resTol = self.MC.var_pos_w+self.MC.var_neg_w+self.MC.var_pos_v+\
                      self.MC.var_neg_v+self.MC.var_dq+self.MC.var_dp 
        '''
        optmetrics = np.zeros((8,))
        # Variance minimization 
        optmetrics[0] = np.sum(dp**2)/(nd-1) #Power losses around 0
        optmetrics[1] = np.sum((w-1.)**2)/(nd-1) #w around 1
        optmetrics[2] = np.sum((dv[:,0]-1.)**2)/(nd-1) #dv min around 1
        optmetrics[3] = np.sum((dv[:,1]-1.)**2)/(nd-1) #dv max around 1
        optmetrics[4] = np.sum(dpq[:,0]**2)/(nd-1) #dp around 0
        optmetrics[5] = np.sum(dpq[:,1]**2)/(nd-1) #dq around 0
        optmetrics[6] = np.sum(imax**2)/(nd-1) #imax around 0
        #optmetrics[7] = mae
        self.resDev =  np.sum(optmetrics)
        optmetrics[7] = np.mean(dp)
        self.optmetrics = optmetrics
        
        if np.isnan(self.resDev):
            self.resDev = 10.0
                 
    def updateYd(self, seed):
        np.random.seed(seed)
        Yd = 1j*np.zeros((self.NumN,1))
        for n in range(self.NumD):
            n1 = np.int32(self.demandL[n]) #node number
            if self.demandType[n]:
                g = EV_Demand(10.0)
            else:
                g = self.demandP[n] + 0.05*self.demandP[n]*np.random.randn()  # Demand value
            Yd[n1] = Yd[n1] + g
        self.Yd = Yd
        
    def dynamic_sim(self, convs, dt=5e-5, fp=0.7, nd = 10, pw_flag=False):
        
        xi = convs[:,2].copy()
        zita = convs[:,3].copy()
        
        xi = xi.reshape(xi.size,1)
        zita = zita.reshape(zita.size,1)

        Ydb = self.Yd*(1./fp*np.exp(1j*np.arccos(fp)))
        wbase = 2.0*np.pi*50.0
        N = np.int_(convs[:,0])
        tau = (convs[:,4].copy())[:, np.newaxis]
        S = np.int_(np.setdiff1d(np.arange(self.NumN),N))
        n = N.size
        V = np.ones((n,1))
        w0 = 1.0
        Pref = (convs[:,1].copy())[:,np.newaxis]
        Qref = np.zeros((n,1))
        v0 = np.ones((n,1))
        th = np.zeros((n,1))
        p  = Pref.copy()
        q  = Qref.copy()
        gr_wci = np.zeros((nd,1))
        t = np.zeros((nd,1))
        gr_w = np.zeros((nd,n))
        gr_p = np.zeros((nd,n))
        gr_th = np.zeros((nd,n))
        gr_v = np.zeros((nd,n))
        gr_y = np.ones((nd,1))
        wci = 1.0
        try:
            for k in range(0,nd):
                ylin = np.diag(1./(np.real(self.zlin[:,0])+wci*np.imag(self.zlin[:,0])*1j))
                Yd = np.real(Ydb) + wci*1j*np.imag(Ydb)
                Yb = self.A.T@ylin@self.A + np.diag(Yd[:,0]) + np.diag(self.Yc[:,0]*wci)
                Y = Yb[np.ix_(N,N)] - Yb[np.ix_(N,S)]@np.linalg.solve(Yb[np.ix_(S,S)],Yb[np.ix_(S,N)])
                In = Y@V

                Sn = V*np.conj(In)
                P = np.real(Sn)
                Q = np.imag(Sn)
                dp = (P-p)/(tau+1e-6)
                dq = (Q-q)/(tau+1e-6)        
                w = (Pref-p)*xi + w0
                wci = np.sum(1./xi*w)/np.sum(1./xi)
                dth = wbase*(w-wci)
                t[k] = k*dt
                th = th + dth*dt
                p  = p + dp*dt
                q  = q + dq*dt
                th_ci = np.sum(th/xi)/np.sum(1./xi)
                th = th-th_ci
                Vn = (Qref-q)*zita + v0
                V = Vn*np.exp(1j*th)
                gr_w[k,:] = w[:,0]
                gr_p[k,:] = p[:,0]
                gr_th[k,:] = th[:,0]
                gr_v[k,:] = Vn[:,0]
                gr_wci[k] = wci
                gr_y[k] = np.linalg.norm(Y)
        
        
            mae = np.mean((gr_w-1.0)**2)
            if pw_flag:
                fig01 = plt.figure()
                ax01 = fig01.add_subplot(1, 1, 1) 
                #plt.plot(t[nd-50:],gr_w[nd-50:])
                plt.plot(t, gr_w)
                plt.savefig('montecarda.pdf')
        except:
            #if the dyamic simulation fails then 
            mae = 10.0
        return mae


if __name__ == "__main__":

    mg = mgrid()
    #mg.microgrid1()

    npar = mg.npar
    np.random.seed(1)
    xi = np.random.rand(npar)*0.15
    zita = np.random.rand(npar)*0.15
       
    mg.MC.nd = np.int32(500) #Sample Size
    mg.Montecarlo(xi, zita)
    mg.converters[:,2] = xi
    mg.converters[:,3] = zita
    #print(mg.optmetrics)
    #print(mg.resDev)
    #mae = mg.dynamic_sim(mg.converters, nd = 100, pw_flag=True)
    #print(mae)
    print(mg.Newton_Freq(mg.converters))