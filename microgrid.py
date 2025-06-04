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
  nEV_val = np.array([5.0, 10.0, 20.0, 30.0]) #Number of Vehicles
  idx = np.where(nEV_val == nEV)[0][0]
  shape = [1.94733,2.2784,3.71581,5.15059] #Gamma's shape parameter
  scale = [72.7711,103.043,110.621,133.027] #Gamma's scale parameter
  return np.random.gamma(shape[idx],scale=scale[idx])*1000 #EV demand in W

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
        self.MC = lambda:0 # Easy way to create an object
        self.MC.nd = 100 # Number of samples by default
        self.savename = 'data'
        self.path = ''

    def mg1(self): #Original Cigre without EV
        self.Vnom = 400.0 # Line to Line voltage in volts
        self.wnom = 2.*np.pi*60. # Nominal angular velocity
        self.Pnom = 100000.0 # Nominal power in watts
        # N1 N2 L(m) type 
        self.Lines = np.array([[0,1,35,0],
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
        self.Impedances = np.array([[0.284,0.083,1.136,0.417,0.38], #(1) OL-Twisted cable 4x120 mm2
                               [3.690,0.094,13.64,0.472,0.05], #(2) SC - 4x6 mm2 Cu
                               [1.380,0.082,5.520,0.418,0.18], #(3) SC - 4x16 mm2 Cu
                               [0.871,0.081,3.480,0.409,0.22]])#(4) SC - 4x25 mm2 Cu   
        
        self.DemandL = np.array([10,12,13,17,18])
        self.DemandP = np.array([15000.0,55000.0,47000.0,72000.0,15000.0])
        #node 10 EV
        #node number, type 0:normal 1:EV, params
        self.DemandType = np.array([0,0,0,0,0])
        solar = 1.0
        wind = 2.0
        #EV = 3.0
        weibull = 1.0
        beta = 2.0
        #normal = 3.0
        # node power Kp Kq Tau type dist K1 K2
        self.Converters = np.array([[11,30000,0.05,0.04,0.32E-3,solar,beta,900,40],
                            [12,10000,0.08,0.09,0.38E-3,solar,beta,900,40],
                            [13,10000,0.10,0.09,0.41E-3,wind,weibull,11,1.2],
                            [17,30000,0.09,0.10,0.31E-3,solar,beta,900,40],
                            [18,10000,0.08,0.08,0.34E-3,wind,weibull,11,1.2]])
        
    def mg2(self): #Meshed MG
        '''
        A Benchmark Test System for Networked Microgrids
        '''
        self.Vnom = 11000.0 # Line to Line voltage in volts
        self.wnom = 2.*np.pi*50. # Nominal angular velocity
        self.Pnom = 10000.0 # Nominal power in watts

        self.Lines = np.array([[0,1,3000,0],
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

        self.Impedances = np.array([[0.0309,0.05962,1.136,0.417,0.0], #(0)
                            [0.0389,0.04995,13.64,0.472,0.0], #(1)
                            [0.0389,0.04995,5.520,0.418,0.0], #(2)
                            [0.0389,0.04995,3.480,0.409,0.0], #(3)
                            [0.0389,0.04995,3.480,0.409,0.0], #(4)
                            [0.0389,0.04995,3.480,0.409,0.0]]) #(5)

        self.DemandL = np.array([2,3,4,5,6])
        self.DemandP = np.array([2125.0,3329.0,2050.0,10.0,1056.0])

        #DemandL = np.array([2,3,4,5,6])
        #DemandP = np.array([2125.0, 3329.0, 2050.0, 1257.0, 1056.0, 100.0])/10.0
        #node number, type 0:normal 1:EV, params
        #node 6 ev (5 en python)
        self.DemandType = np.array([0,0,0,1,0])
        #Typer of source 
        solar = 1.0
        #wind = 2.0
        #Type of distribution
        #weibull = 1.0
        beta = 2.0
        #normal = 3.0
        #Converters = np.array([[2,Pnom,0.08,0.09,0.38E-3,solar,beta,900,40],
        #                [3,Pnom,0.10,0.09,0.41E-3,solar,beta,11,1.2],
        #                [4,Pnom,0.09,0.10,0.31E-3,solar,beta,900,40]])

        self.Converters = np.array([[2,2000,0.08,0.09,0.38E-3,solar,beta,900,40],
                            [3,2400,0.10,0.09,0.41E-3,solar,beta,900,40],
                            [4,2000,0.09,0.10,0.31E-3,solar,beta,900,40]])

    def mg3(self): #Modified Cigre with EV and new Vnom
        self.mg1()
        self.Vnom = 800.0
        self.DemandType = np.array([1,0,0,0,0])
        self.initilizeMG()

    def initilizeMG(self):
        #Structure organization
        self.NumN = np.max([np.max(self.Lines[:,0]),np.max(self.Lines[:,1])])+1 # Numbe of nodes
        self.NumL = self.Lines.shape[0] # Number of lines
        self.NumD = self.DemandL.shape[0] # Number of loads
        self.NumC = self.Converters.shape[0] # Number of converters
        Zbase = self.Vnom*self.Vnom/self.Pnom
        print(Zbase)
        Y = 1j*np.zeros((self.NumN, self.NumN))
        Yc = 1j*np.zeros((self.NumN, 1))
        #mg = 1j*np.zeros((self.NumL, 4))
        zlin = 1j*np.zeros((self.NumL, 1))
        A = np.zeros((self.NumL, self.NumN))
        for k in range(self.NumL):
            n1 = self.Lines[k,0]
            n2 = self.Lines[k,1]
            t = self.Lines[k,3]
            z = (self.Impedances[t,0]+1j*self.Impedances[t,1])/1000/Zbase*self.Lines[k,2]
            b = 1j*self.Impedances[t,4]*self.wnom/1000*self.Lines[k,2]*Zbase*1E-6
            #mg[k,0:4] = np.array([n1,n2,z,b])
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
        self.DemandP = self.DemandP/self.Pnom
        Yd = np.zeros((self.NumN, 1))+0j
        for k in range(self.NumD):
            n1 = np.int32(self.DemandL[k])
            if self.DemandType[k]:
                g = EV_Demand(10.0)/self.Pnom
            else:
                g = self.DemandP[k]
            Y[n1,n1] = Y[n1,n1] + g
            Yd[n1] = Yd[n1] + g
            
        # Load flow in grid mode
        self.Converters[:,1] = self.Converters[:,1]/self.Pnom
        '''
        s = np.zeros((self.NumN,))
        #for k in range(self.NumC):
        n1 = np.int_(self.Converters[:,0]) #node
        s[n1] = self.Converters[:,1].copy()

        # Yn = csr_matrix(Y[1:,1:]) #Using scipy
        Yn = Y[1:,1:]
        Y0 = Y[1:,0]
        V  = 1j*np.ones((self.NumN,))
        er = 1.0
        while (er>1E-8):
            V[1:] = np.linalg.solve(Yn,np.conj(s[1:]/V[1:])-Y0*V[0])
            In = Y@V
            sc = V*np.conj(In)
            er = np.linalg.norm(s[1:]-sc[1:])
        '''
        #self.lines = mg
        #self.Ybus = Y
        #self.V = V
        self.A = A
        self.zlin = zlin
        self.Yd = Yd
        self.Yc = Yc

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
            Y = Yb[np.ix_(N, N)] - Yb[np.ix_(N, S)]@np.linalg.solve(Yb[np.ix_(S, S)],Yb[np.ix_(S, N)])
            D = Db[np.ix_(N, N)] - Db[np.ix_(N, S)]@np.linalg.solve(Db[np.ix_(S, S)],Db[np.ix_(S, N)])
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
        Vl = -np.linalg.solve(Yb[np.ix_(S, S)],Yb[np.ix_(S, N)]@Vs)
        Il = Yb[np.ix_(S, N)]@Vs + Yb[np.ix_(S, S)]@Vl
        Vt = np.zeros((self.NumN,1))+0j
        Vt[N] = Vs
        Vt[S] = Vl
        It = np.zeros((self.NumN,1))+0j
        It[N] = I
        It[S] = Il
        #print(np.max(Vt),np.max(It))
        Pt = np.real(Vt*np.conj(It))
        if np.sum(Pt)<0.:
            print('Negative power loss!')
            #print('w',w)
            #print('Vt', Vt)
            #print('It', It)
            print('Pt', Pt)
            print('Conv',convs[:,1])

        return w,Vt,P,Q,I,np.sum(Pt)
    
    def Montecarlo(self,xi,zita, scales = None, makefig=False, flagres=False, savedata=False, flagseed=False, saturate = False, seed =0):
        #flagres para incluir MAE de la frecuencia en la funcion objetivo
        # Frequency variation given by p and q
        nd = self.MC.nd  # Monte Carlo number of iterations
        if scales is None:
            scales = np.ones((7,))

        w = np.zeros((nd,1)) #Newton
        dp = np.zeros((nd,1))
        dv = np.zeros((nd,2))
        Vnodes = np.zeros((nd,self.NumN))
        imax = np.zeros((nd,1))
        dpq = np.zeros((nd,2)) # caching the diference between p_ref and p
        convs = self.Converters.copy()
        convs[:,2] = xi.copy()
        convs[:,3] = zita.copy()
        solar = 1.0
        wind = 2.0
        #demand = self.demand.copy()
        if flagseed:
            np.random.seed(seed)
        for k in range(nd):
            #demand[:,1]= self.demand[:,1]*np.random.rand(self.NumD)*10  # Demands are uniform distributed
            Yd = np.zeros((self.NumN,1)) + 0j
            for n in range(self.NumD):
                n1 = np.int32(self.DemandL[n]) #node number
                if self.DemandType[n]:
                    g = EV_Demand(5.0)/(self.Pnom) #in PU for EV
                else:
                    g = self.DemandP[n] + 0.05*self.DemandP[n]*np.random.randn()  # Demand value
                Yd[n1] += g
            self.Yd = Yd.copy()
            
            for c in range(self.NumC):
                Pnom = self.Converters[c,1]
                dist = self.Converters[c,6]
                K1 = self.Converters[c,7]
                K2 = self.Converters[c,8]
                if (self.Converters[c,5] == solar):
                    convs[c,1] = Solar_Sce(Pnom,dist,K1,K2)
                    
                if (self.Converters[c,5] == wind):
                    convs[c,1] = Wind_Sce(Pnom,dist,K1,K2)

            w[k],v,p,q,i,dp[k] = self.Newton_Freq(convs)
            dv[k,0] = np.min(np.abs(v))
            dv[k,1] = np.max(np.abs(v))
            imax[k] = np.max(np.abs(i))
            #pg = p[:,0]+Yd[np.int_(convs[:,0]),0].real
            dpq[k,0] = np.max(np.abs(p[:,0]-convs[:,1]))
            dpq[k,1] = np.max(np.abs(q))
            Vnodes[k,:] = np.abs(v).reshape(-1,)

        if makefig:
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

            fig05 = plt.figure()
            ax05 = fig05.add_subplot(1, 1, 1) 
            print(Vnodes.shape)
            plt.boxplot(Vnodes)
            ax05.grid()
            ax05.set_title('$V$ at each node')

            fig06 = plt.figure()
            ax06 = fig06.add_subplot(1, 1, 1) 
            plt.hist(dp,30)
            ax06.grid()
            ax06.set_title('Losses')

            fig01.savefig(self.path+'omega.pdf',dpi=200)
            fig02.savefig(self.path+'dpq.pdf',dpi=200)
            fig03.savefig(self.path+'dv.pdf',dpi=200)
            fig04.savefig(self.path+'imax.pdf',dpi=200)
            fig05.savefig(self.path+'Vnodes.pdf',dpi=200)
            fig06.savefig(self.path+'losses.pdf',dpi=200)
        
        if savedata:
            np.savez(self.savename,w=w,dp=dp,dv=dv,dpq=dpq,imax=imax,
                     params=np.concatenate((xi, zita)))

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
        self.resDev =  np.sum(optmetrics[:7])
        print(self.resDev)
        optmetrics[7] = np.mean(dp)
        self.optmetrics = optmetrics
        
        if np.isnan(self.resDev) or (self.resDev>10.0 and saturate):
            self.resDev = 10.0
        
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
    mg.mg3()
    mg.initilizeMG()
    mg.MC.nd = np.int32(500) #Sample Size
    npar = mg.NumC
    nsets = 100
    bounds = np.array([1e-7,0.15])*np.ones((2*npar,2))
    np.random.seed(0)
    X = (bounds[:,1]-bounds[:,0])*np.random.rand(nsets,2*npar)+bounds[:,0]
    #row = 52
    #print(X[row,:])
    #xi = X[row,:npar]
    #zita = X[row,npar:]

    xi = np.random.rand(npar)*0.15
    zita = np.random.rand(npar)*0.15
    #xi = np.array([0.00040558,0.09707953,0.09005888,0.08831098,0.14441555]) 
    #zita = np.array([0.00253085,0.10447239,0.12205182,0.07647113,0.0500948])

    #mg1 worst sample for Vnom = 400
    #xi = np.array([0.00040558, 0.09707953, 0.09005888, 0.08831098, 0.14441555]) 
    #zita = np.array([0.00253085,  0.10447239, 0.12205182, 0.07647113, 0.0500948])

    #BestSol = np.load('ResultsSize500/GA_mg3_BestSol.npy')
    #BestVal = np.load('ResultsSize500/GA_mg3_BestVal.npy')
    #indMin = np.argmin(BestVal)
    #xi = BestSol[indMin,:npar]
    #zita = BestSol[indMin,npar:]
    
    #scales = np.array([2.372127,8.370188e-4,0.003252,0.004098,2.462032,6.179947,9.559357])
    scales = np.ones((7,))
    mg.Converters[:,2] = xi
    mg.Converters[:,3] = zita
    mg.Montecarlo(xi, zita, makefig=True)

    print(mg.optmetrics)
    #print(mg.resDev)
    #mae = mg.dynamic_sim(mg.Converters, nd = 100, pw_flag=True)
    #print(mae)
    #print(mg.Newton_Freq(mg.Converters))