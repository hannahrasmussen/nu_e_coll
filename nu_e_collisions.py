#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sympy as sym
import numpy as np
import numba as nb

Weinberg = .4910015
x_values, w_values = np.polynomial.laguerre.laggauss(20)
x_valuese, w_valuese = np.polynomial.legendre.leggauss(20)
me = 0.511 
inf = 6457.2 


# In[2]:

x, y, p1, E2, E3, q3, q2, GF, stw = sym.symbols('x,y,p1,E2,E3,q3,q2,GF,stw')

M_1prime = 2**5 * GF**2 * (2 * stw + 1)**2 * ( x**2 - 2 * stw / (2 * stw + 1) *me**2*x )
M_2prime = 2**7 * GF**2 * (stw)**2 * ( x**2 + (2*stw + 1)/(2*stw) * me**2*x )

M_1_1 = sym.integrate( M_1prime.subs(x, ((p1+E2)**2-me**2-y**2)/2), (y, p1+E2-E3-q3, p1+E2-E3+q3) )
M_11 = sym.lambdify((p1,E2,E3,q3),M_1_1.subs([(GF,1.166e-11),(stw,(np.sin(Weinberg))**2)]))
M11 = nb.jit(M_11,nopython=True)

M_1_2 = sym.integrate( M_1prime.subs(x, ((p1+E2)**2-me**2-y**2)/2), (y, p1-q2, p1+q2) )
M_12 = sym.lambdify((p1,E2,q2),M_1_2.subs([(GF,1.166e-11),(stw,(np.sin(Weinberg))**2)]))
M12 = nb.jit(M_12,nopython=True)

M_1_3 = sym.integrate( M_1prime.subs(x, ((p1+E2)**2-me**2-y**2)/2), (y, E3+q3-p1-E2, p1+q2) )
M_13 = sym.lambdify((p1,E2,q2,E3,q3),M_1_3.subs([(GF,1.166e-11),(stw,(np.sin(Weinberg))**2)]))
M13 = nb.jit(M_13,nopython=True)

M_1_4 = sym.integrate( M_1prime.subs(x, ((p1+E2)**2-me**2-y**2)/2), (y, q2-p1, p1+E2-E3+q3) )
M_14 = sym.lambdify((p1,E2,q2,E3,q3),M_1_4.subs([(GF,1.166e-11),(stw,(np.sin(Weinberg))**2)]))
M14 = nb.jit(M_14,nopython=True)

M_2_1 = sym.integrate( M_2prime.subs(x, (y**2 + me**2 - (p1-E3)**2)/2), (y, p1-E3+E2-q2, p1-E3+E2+q2) )
M_21 = sym.lambdify((p1,E2,E3,q2),M_2_1.subs([(GF,1.166e-11),(stw,(np.sin(Weinberg))**2)]))
M21 = nb.jit(M_21,nopython=True)

M_2_2 = sym.integrate( M_2prime.subs(x, (y**2 + me**2 - (p1-E3)**2)/2), (y, p1-q3, p1+q3) )
M_22 = sym.lambdify((p1,E3,q3),M_2_2.subs([(GF,1.166e-11),(stw,(np.sin(Weinberg))**2)]))
M22 = nb.jit(M_22,nopython=True)

M_2_3 = sym.integrate( M_2prime.subs(x, (y**2 + me**2 - (p1-E3)**2)/2), (y, E3-p1-E2+q2, p1+q3) )
M_23 = sym.lambdify((p1,E2,q2,E3,q3),M_2_3.subs([(GF,1.166e-11),(stw,(np.sin(Weinberg))**2)]))
M23 = nb.jit(M_23,nopython=True)

M_2_4 = sym.integrate( M_2prime.subs(x, (y**2 + me**2 - (p1-E3)**2)/2), (y, q3-p1, p1-E3+E2+q2) )
M_24 = sym.lambdify((p1,E2,q2,E3,q3),M_2_4.subs([(GF,1.166e-11),(stw,(np.sin(Weinberg))**2)])) 
M24 = nb.jit(M_24,nopython=True)


# In[3]:


@nb.jit(nopython=True)
def trapezoid(array,dx):
    total = np.sum(dx*(array[1:]+array[:-1])/2)
    return total

@nb.jit(nopython=True)
def fe(E,T):
    return 1/(np.e**(E/T)+1)

@nb.jit(nopython=True)
def make_q_array(E_arr):
    q2_arr = E_arr**2 - 0.511**2
    q_arr = np.sqrt(q2_arr)
    for i in range(len(q2_arr)):
        if abs(q2_arr[i]) < 1e-13:
            q_arr[i] = 0
        elif q2_arr[i]  < -1e-13:
            print("Error with q_array",q2_arr[i])
            q_arr[i] = 0
    return q_arr


# In[4]:


@nb.jit(nopython=True)
def lin_int(X,x,y): #x is an array of 6 x values, y is an array of 6 y values that correspond w/ x,
                    #X is the x value we wish to find a corresponding y value for via interpolation
    P00 = y[0]
    P11 = y[1]
    P22 = y[2]
    P33 = y[3]
    P44 = y[4]
    P55 = y[5]
    
    P01 = ((X-x[1])*P00 - (X-x[0])*P11)/(x[0]-x[1])
    P12 = ((X-x[2])*P11 - (X-x[1])*P22)/(x[1]-x[2])
    P23 = ((X-x[3])*P22 - (X-x[2])*P33)/(x[2]-x[3])
    P34 = ((X-x[4])*P33 - (X-x[3])*P44)/(x[3]-x[4])
    P45 = ((X-x[5])*P44 - (X-x[4])*P55)/(x[4]-x[5])
    
    P02 = ((X-x[2])*P01 - (X-x[0])*P12)/(x[0]-x[2])
    P13 = ((X-x[3])*P12 - (X-x[1])*P23)/(x[1]-x[3])
    P24 = ((X-x[4])*P23 - (X-x[2])*P34)/(x[2]-x[4])
    P35 = ((X-x[5])*P34 - (X-x[3])*P45)/(x[3]-x[5])
    
    P03 = ((X-x[3])*P02 - (X-x[0])*P13)/(x[0]-x[3])
    P14 = ((X-x[4])*P13 - (X-x[1])*P24)/(x[1]-x[4])
    P25 = ((X-x[5])*P24 - (X-x[2])*P35)/(x[2]-x[5])
    
    P04 = ((X-x[4])*P03 - (X-x[0])*P14)/(x[0]-x[4])
    P15 = ((X-x[5])*P14 - (X-x[1])*P25)/(x[1]-x[5])
    
    P05 = ((X-x[5])*P04 - (X-x[0])*P15)/(x[0]-x[5])
    Y = P05
    
    return Y

##  Only use this if the x-values are boxsize * i
@nb.jit(nopython=True)
def interp(p4, bx, logf):
    p1_arr = np.zeros(len(logf))
    for i in range(len(logf)):
        p1_arr[i] = bx * i  
    
    j = int(p4/bx)
    if j >= len(p1_arr):
        print("Error:  extrapolation required")
        return 0
    if j < 3:
        return lin_int(p4, p1_arr[:6], logf[:6])
    elif (j > len(p1_arr) - 4):
        return lin_int(p4, p1_arr[-6:], logf[-6:])
    else:
        return lin_int(p4, p1_arr[j-3:j+3], logf[j-3:j+3])
    
@nb.jit(nopython=True)
def interp_log(p4, bx ,f):
    return np.exp(interp(p4, bx, np.log(f)))

@nb.jit(nopython=True)
def linear_extrap(X,x,y):    
    return y[0] + (y[1] - y[0])/(x[1] - x[0]) * (X - x[0])

## Note, x and y need to be numpy arrays, should be of length 2
@nb.jit(nopython=True)
def log_linear_extrap(X,x,y):
    return np.exp(linear_extrap(X,x,np.log(y)))

@nb.jit(nopython=True)
def f_first_last(f, p4_array, boxsize):
    k = max(int(p4_array[-1]/boxsize),int(p4_array[-1]/boxsize+1e-9))
    j = max(int(p4_array[0]/boxsize),int(p4_array[0]/boxsize+1e-9))+1
    
    if j<len(f): #these conditions prevent the code from calling an index of f out of f's bounds
        f_first = interp_log(p4_array[0], boxsize, f)
    else:
        f_first = log_linear_extrap(p4_array[0],np.array([0,boxsize]),np.array([f[0],f[1]]))
    if k<len(f)-1:
        f_last = interp_log(p4_array[-1], boxsize, f)
    else:
        f_last = log_linear_extrap(p4_array[-1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        
    return f_first, f_last, j, k


# In[5]:


@nb.jit(nopython=True)
def Blim(p1,E,q,T,f,bx,sub,sup,n):
    
    if (sub==1): #it so happens that for R_1, only n matters for B limits, not superscript
        if (n==1):
            UL = E
            LL = me
        elif (n==2):
            UL = (1/2)*(2*p1 + E - q + (me**2)/(2*p1 + E - q)) #E2trans
            LL = E
        elif (n==3):
            UL = (1/2)*(2*p1 + E + q + (me**2)/(2*p1 + E + q)) #E1lim
            LL = (1/2)*(2*p1 + E - q + (me**2)/(2*p1 + E - q)) #E2trans
        elif (n==4):
            UL = (1/2)*(2*p1 + E - q + (me**2)/(2*p1 + E - q)) #E2trans
            LL = me
        elif (n==5):
            UL = E
            LL = (1/2)*(2*p1 + E - q + (me**2)/(2*p1 + E - q)) #E2trans
        elif (n==6):
            UL = (1/2)*(2*p1 + E + q + (me**2)/(2*p1 + E + q)) #E1lim
            LL = E
        elif (n==7):
            UL = E
            LL = (1/2)*(2*p1 + E - q + (me**2)/(2*p1 + E - q)) #E2lim
        else: #n=8
            UL = (1/2)*(2*p1 + E + q + (me**2)/(2*p1 + E + q)) #E1lim
            LL = E
    else: #sub=2
        if (n==1):
            UL = E
            LL = me
        elif (n==2):
            if (sup==1 or sup==2 or sup==3):
                UL = (1/2)*(E + q - 2*p1 + (me**2)/(E + q - 2*p1)) #E2trans
                LL = E
            else: #sup=4
                UL = bx*len(f)
                LL = E
        elif (n==3):
            if (sup==1 or sup==2 or sup==3):
                #UL = (1/2)*(E - q - 2*p1 + (me**2)/(E - q - 2*p1)) #E1lim
                if (E - q - 2*p1)==0:
                    UL = T*100
                else:
                    E1lim = (1/2)*(E - q - 2*p1 + (me**2)/(E - q - 2*p1))
                    UL = min(E1lim,T*100)
                LL = (1/2)*(E + q - 2*p1 + (me**2)/(E + q - 2*p1)) #E2trans
            else: #sup=4
                UL = E
                LL = me
        elif (n==4):
            if (sup==1 or sup==2):
                UL = (1/2)*(E + q - 2*p1 + (me**2)/(E + q - 2*p1)) #E2trans
                LL = me
            elif (sup==3):
                UL = E
                LL = me
            else: #sup=4
                #UL = (1/2)*(E + q - 2*p1 + (me**2)/(E + q - 2*p1)) #E2trans
                if (E + q - 2*p1)==0:
                    UL = T*100
                else:
                    E2trans = (1/2)*(E + q - 2*p1 + (me**2)/(E + q - 2*p1))
                    UL = min(E2trans,T*100)
                LL = E
        elif (n==5):
            if (sup==1 or sup==2):
                UL = E
                LL = (1/2)*(E + q - 2*p1 + (me**2)/(E + q - 2*p1)) #E2trans
            elif (sup==3):
                UL = (1/2)*(E + q - 2*p1 + (me**2)/(E + q - 2*p1)) #E2trans
                LL = E
            else: #sup=4
                UL = bx*len(f)
                if (E + q - 2*p1)==0:
                    LL = T*100
                else:
                    E2trans = (1/2)*(E + q - 2*p1 + (me**2)/(E + q - 2*p1))
                    LL = min(E2trans,T*100)
                #LL = (1/2)*(E + q - 2*p1 + (me**2)/(E + q - 2*p1)) #E2trans
        elif (n==6):
            if (sup==1 or sup==2):
                #UL = (1/2)*(E - q - 2*p1 + (me**2)/(E - q - 2*p1)) #E1lim
                if (E - q - 2*p1)==0:
                    UL = T*100
                else:
                    E1lim = (1/2)*(E - q - 2*p1 + (me**2)/(E - q - 2*p1))
                    UL = min(E1lim,T*100)
                LL = E
            elif (sup==3):
                UL = bx*len(f)
                LL = (1/2)*(E + q - 2*p1 + (me**2)/(E + q - 2*p1)) #E2trans
            else: #sup=4
                UL = (1/2)*(E + q - 2*p1 + (me**2)/(E + q - 2*p1)) #E2trans
                LL = me
        elif (n==7):
            if (sup==1):
                UL = E
                LL = (1/2)*(E + q - 2*p1 + (me**2)/(E + q - 2*p1)) #E2lim
            elif (sup==2 or sup==3):
                UL = (1/2)*(E + q - 2*p1 + (me**2)/(E + q - 2*p1)) #E2trans
                LL = me
            else: #sup=4
                UL = E
                LL = (1/2)*(E + q - 2*p1 + (me**2)/(E + q - 2*p1)) #E2trans
        elif (n==8):
            if (sup==1):
                #UL = (1/2)*(E - q - 2*p1 + (me**2)/(E - q - 2*p1)) #E1lim
                if (E - q - 2*p1)==0:
                    UL = T*100
                else:
                    E1lim = (1/2)*(E - q - 2*p1 + (me**2)/(E - q - 2*p1))
                    UL = min(E1lim,T*100)
                LL = E
            elif (sup==2 or sup==3):
                UL = E
                LL = (1/2)*(E + q - 2*p1 + (me**2)/(E + q - 2*p1)) #E2trans
            else: #sup=4 
                UL = bx*len(f)
                LL = E
        elif (n==9):
            if (sup==1 or sup==4):
                UL = E
                LL = (1/2)*(E + q - 2*p1 + (me**2)/(E + q - 2*p1)) #E2lim
            else: #sup=2,3
                UL = bx*len(f)
                LL = E
        elif (n==10):
            if (sup==1 or sup==4):
                UL = bx*len(f)
                LL = E
            else: #sup=2,3
                UL = E
                LL = (1/2)*(E + q - 2*p1 + (me**2)/(E + q - 2*p1)) #E2lim
        else: #n=11
            UL = bx*len(f)
            LL = E

    return UL, LL


@nb.jit(nopython=True)
def Alim(p1,sub,sup,n):
    
    if (sub==1):
        if (sup==1):
            if (n==1 or n==2 or n==3):
                UL = np.sqrt(p1**2 + me**2) #E3cut
                LL = me
            elif (n==4 or n==5 or n==6):
                UL = me + (2*p1**2)/(me - 2*p1) #E1cut
                LL = np.sqrt(p1**2 + me**2) #E3cut
            else: #n=7,8
                UL = inf
                LL = me + (2*p1**2)/(me - 2*p1) #E1cut
        else: #sup=2
            if (n==1 or n==2 or n==3):
                UL = np.sqrt(p1**2 + me**2) #E3cut
                LL = me
            else: #n=4,5,6
                UL = inf
                LL = np.sqrt(p1**2 + me**2) #E3cut
    else: #sub=2
        if (sup==1):
            if (n==1 or n==2 or n==3):
                UL = np.sqrt(p1**2 + me**2) #E3cut
                LL = me
            elif (n==4 or n==5 or n==6):
                UL = p1 + me*(p1+me)/(2*p1+me) #E2cut
                LL = np.sqrt(p1**2 + me**2) #E3cut
            elif (n==7 or n==8):
                UL = p1 + (me**2)/(4*p1) #E1cut
                LL = p1 + me*(p1+me)/(2*p1+me) #E2cut
            else: #n=9,10
                UL = inf
                LL = p1 + (me**2)/(4*p1) #E1cut
        elif (sup==2):
            if (n==1 or n==2 or n==3):
                UL = np.sqrt(p1**2 + me**2) #E3cut
                LL = me
            elif (n==4 or n==5 or n==6):
                UL = p1 + (me**2)/(4*p1) #E1cut
                LL = np.sqrt(p1**2 + me**2) #E3cut
            elif (n==7 or n==8 or n==9):
                UL = p1 + me*(p1+me)/(2*p1+me) #E2cut
                LL = p1 + (me**2)/(4*p1) #E1cut
            else: #n=10,11
                UL = inf
                LL = p1 + me*(p1+me)/(2*p1+me) #E2cut
        elif (sup==3):
            if (n==1 or n==2 or n==3):
                UL = p1 + (me**2)/(4*p1) #E1cut
                LL = me
            elif (n==4 or n==5 or n==6):
                UL = np.sqrt(p1**2 + me**2) #E3cut
                LL = p1 + (me**2)/(4*p1) #E1cut
            elif (n==7 or n==8 or n==9):
                UL = p1 + me*(p1+me)/(2*p1+me) #E2cut
                LL = np.sqrt(p1**2 + me**2) #E3cut
            else: #n=10,11
                UL = inf
                LL = p1 + me*(p1+me)/(2*p1+me) #E2cut
        else: #sup=4
            if (n==1 or n==2 or n==3):
                UL = p1 + (me**2)/(4*p1) #E1cut
                LL = me
            elif (n==4 or n==5 or n==6):
                UL = np.sqrt(p1**2 + me**2) #E3cut
                LL = p1 + (me**2)/(4*p1) #E1cut
            elif (n==7 or n==8):
                UL = p1 + me*(p1+me)/(2*p1+me) #E2cut
                LL = np.sqrt(p1**2 + me**2) #E3cut
            else: #n=9,10
                UL = inf
                LL = p1 + me*(p1+me)/(2*p1+me) #E2cut
    
    return UL, LL

@nb.jit(nopython=True)
def M(p1,E_arr,q_arr,E_val,q_val,sub,sup,n):
    M_arr = np.zeros(len(E_arr))
    
    if (sub==1): #it so happens that for R_1, only n matters for M, not superscript
        if (n==1 or n==4):
            for i in range (len(E_arr)):
                M_arr[i] = M11(p1,E_val,E_arr[i],q_arr[i])
        elif (n==2):
            for i in range (len(E_arr)):
                M_arr[i] = M12(p1,E_val,q_val)
        elif (n==3 or n==6 or n==8):
            for i in range (len(E_arr)):
                M_arr[i] = M13(p1,E_val,q_val,E_arr[i],q_arr[i])
        else: #n=5,7
            for i in range (len(E_arr)):
                M_arr[i] = M14(p1,E_val,q_val,E_arr[i],q_arr[i])
    else: #sub=2
        #if (sup==1):
        if (((sup == 1) and (n==1 or n==4)) or ((sup==2 or sup==3) and (n==1 or n==4 or n==7)) or ((sup==4) and (n==1 or n==3 or n==6))):
            for i in range (len(E_arr)):
                M_arr[i] = M21(p1,E_arr[i],E_val,q_arr[i])
        elif ((sup==1 and n==2) or (sup==2 and n==2) or ((sup==3) and (n==2 or n==5)) or ((sup==4) and (n==2 or n==4))):
            for i in range (len(E_arr)):
                M_arr[i] = M22(p1,E_val,q_val)
        elif (((sup==1) and (n==3 or n==6 or n==8 or n==10)) or ((sup==2 or sup==3) and (n==3 or n==6 or n==9 or n==11)) or ((sup==4) and (n==5 or n==8 or n==10))):
            for i in range (len(E_arr)):
                M_arr[i] = M23(p1,E_arr[i],q_arr[i],E_val,q_val)
        elif (((sup==1) and (n==5 or n==7 or n==9)) or ((sup==2) and (n==5 or n==8 or n==10)) or ((sup==3) and (n==8 or n==10)) or ((sup==4) and (n==7 or n==9))): #n=5,7,9
            for i in range (len(E_arr)):
                M_arr[i] = M24(p1,E_arr[i],q_arr[i],E_val,q_val)
        else:
            print(sub,sup,n,M_arr[0])
        #elif (sup==2):
            #if (n==1 or n==4 or n==7):
            #    for i in range (len(E_arr)):
            #        M_arr[i] = M21(p1,E_arr[i],E_val,q_arr[i])
            #elif (n==2):
            #    for i in range (len(E_arr)):
            #        M_arr[i] = M22(p1,E_val,q_val)
            #elif (n==3 or n==6 or n==9 or n==11):
            #    for i in range (len(E_arr)):
            #        M_arr[i] = M23(p1,E_arr[i],q_arr[i],E_val,q_val)
            #else: #n=5,8,10 
            #    for i in range (len(E_arr)):
            #        M_arr[i] = M24(p1,E_arr[i],q_arr[i],E_val,q_val)
        #elif (sup==3):
            #if (n==1 or n==4 or n==7):
            #    for i in range (len(E_arr)):
            #        M_arr[i] = M21(p1,E_arr[i],E_val,q_arr[i])
            #elif (n==2 or n==5):
            #    for i in range (len(E_arr)):
            #        M_arr[i] = M22(p1,E_val,q_val)
            #elif (n==3 or n==6 or n==9 or n==11):
            #    for i in range (len(E_arr)):
            #        M_arr[i] = M23(p1,E_arr[i],q_arr[i],E_val,q_val)
            #else: #n=8,10 
            #    for i in range (len(E_arr)):
            #        M_arr[i] = M24(p1,E_arr[i],q_arr[i],E_val,q_val)
        #else: #sup=4
            #if (n==1 or n==3 or n==6):
            #    for i in range (len(E_arr)):
            #        M_arr[i] = M21(p1,E_arr[i],E_val,q_arr[i])
            #elif (n==2 or n==4):
            #    for i in range (len(E_arr)):
            #        M_arr[i] = M22(p1,E_val,q_val)
            #elif (n==5 or n==8 or n==10):
            #    for i in range (len(E_arr)):
            #        M_arr[i] = M23(p1,E_arr[i],q_arr[i],E_val,q_val)
            #else: #n=7,9 
            #    for i in range (len(E_arr)):
            #        M_arr[i] = M24(p1,E_arr[i],q_arr[i],E_val,q_val)
    return M_arr


# In[6]:


@nb.jit(nopython=True)
def B(p1,E_val,T,f,bx,sub,sup,n): #E_val can be either E3 or E2 depending on if its R_1 or R_2
     
    q_val = (E_val**2 - .511**2)**(1/2)
    UL, LL = Blim(p1,E_val,q_val,T,f,bx,sub,sup,n)
    
    if (UL<LL):
        return 0,0

    p4_arr = np.zeros(int((UL-LL)/bx)+2)
    Fp_arr = np.zeros(len(p4_arr))
    Fm_arr = np.zeros(len(p4_arr))
    p1_box = int(np.round(p1/bx,0))

    if (sub==1):
        for i in range(int((UL-LL)/bx)):
            p4_arr[i+1] = (int((p1+E_val-UL)/bx)+i+1)*bx
        p4_arr[0] = p1 + E_val - UL 
        p4_arr[-1] = p1 + E_val - LL 
        E_arr = E_val + p1 - p4_arr
        q_arr = make_q_array(E_arr)
        M_arr = M(p1,E_arr,q_arr,E_val,q_val,sub,sup,n)
        
        for i in range (int((UL-LL)/bx)):
            if int(p4_arr[i+1]/bx)>=len(f):
                break #because everything in the arrays below are already zeros so we don't need to set them as zeros
            f_holder = log_linear_extrap(p4_arr[i+1],np.array([(len(f)-2)*bx,(len(f)-1)*bx]),np.array([f[-2],f[-1]]))
            Fp_arr[i+1] = (1-f[p1_box])*(1-fe(E_val,T))*fe(E_arr[i+1],T)*f_holder
            Fm_arr[i+1] = f[p1_box]*fe(E_val,T)*(1-fe(E_arr[i+1],T))*(1-f_holder)

        f_first, f_last, j, k = f_first_last(f, p4_arr, bx)
        Fp_arr[0] = (1-f[p1_box])*(1-fe(E_val,T))*fe(E_arr[0],T)*f_first
        Fm_arr[0] = f[p1_box]*fe(E_val,T)*(1-fe(E_arr[0],T))*(1-f_first)
        Fp_arr[-1] = (1-f[p1_box])*(1-fe(E_val,T))*fe(E_arr[-1],T)*f_last
        Fm_arr[-1] = f[p1_box]*fe(E_val,T)*(1-fe(E_arr[-1],T))*(1-f_last)
         
    else: #sub==2
        for i in range(int((UL-LL)/bx)):
            p4_arr[i+1] = (int((p1+LL-E_val)/bx)+i+1)*bx
        p4_arr[0] = p1 + LL - E_val
        p4_arr[-1] = p1 + UL - E_val
        E_arr = E_val + p4_arr - p1
        q_arr = make_q_array(E_arr)
        M_arr = M(p1,E_arr,q_arr,E_val,q_val,sub,sup,n)
    
        for i in range (int((UL-LL)/bx)):
            if int(p4_arr[i+1]/bx)>=len(f):
                break #because everything in the arrays below are already zeros so we don't need to set them as zeros
            f_holder = log_linear_extrap(p4_arr[i+1],np.array([(len(f)-2)*bx,(len(f)-1)*bx]),np.array([f[-2],f[-1]]))
            Fp_arr[i+1] = (1-f[p1_box])*(1-fe(E_arr[i+1],T))*fe(E_val,T)*f_holder
            Fm_arr[i+1] = f[p1_box]*fe(E_arr[i+1],T)*(1-fe(E_val,T))*(1-f_holder)
    
        f_first, f_last, j, k = f_first_last(f, p4_arr, bx)
        Fp_arr[0] = (1-f[p1_box])*(1-fe(E_arr[0],T))*fe(E_val,T)*f_first
        Fm_arr[0] = f[p1_box]*fe(E_arr[0],T)*(1-fe(E_val,T))*(1-f_first)
        Fp_arr[-1] = (1-f[p1_box])*(1-fe(E_arr[-1],T))*fe(E_val,T)*f_last
        Fm_arr[-1] = f[p1_box]*fe(E_arr[-1],T)*(1-fe(E_val,T))*(1-f_last)
    
    igrndp_arr = Fp_arr*M_arr
    igrndm_arr = Fm_arr*M_arr
    igrlp = trapezoid(igrndp_arr[1:-1],bx)
    igrlm = trapezoid(igrndm_arr[1:-1],bx)
    igrlp = igrlp + igrndp_arr[0]*(bx*j - p4_arr[0]) + igrndp_arr[-1]*(p4_arr[-1] - bx*k)
    igrlm = igrlm + igrndm_arr[0]*(bx*j - p4_arr[0]) + igrndm_arr[-1]*(p4_arr[-1] - bx*k)
    return igrlp, igrlm

@nb.jit(nopython=True)
def A(p1,T,f,bx,sub,sup,n): 
    igrlp = 0.0
    igrlm = 0.0
    UL, LL = Alim(p1,sub,sup,n)
    if (UL == inf):
        E_arr = x_values+LL #could be E3 or E2 depending on if its R_1 or R_2
        for i in range (len(E_arr)):
            Bp, Bm = B(p1,E_arr[i],T,f,bx,sub,sup,n)
            igrlp = igrlp + (np.e**x_values[i])*w_values[i]*Bp
            igrlm = igrlm + (np.e**x_values[i])*w_values[i]*Bm     
    else:  
        E_arr = ((UL-LL)/2)*x_valuese + (UL+LL)/2 #could be E3 or E2 depending on if its R_1 or R_2
        for i in range(len(E_arr)):
            Bp, Bm = B(p1,E_arr[i],T,f,bx,sub,sup,n)
            igrlp = igrlp + w_valuese[i]*((UL-LL)/2)*Bp
            igrlm = igrlm + w_valuese[i]*((UL-LL)/2)*Bm
    return np.array([igrlp, igrlm])


# In[7]:


@nb.jit(nopython=True)
def R11R21(p1,T,f,bx):
    coefficient = 1/(2**4*(2*np.pi)**3*p1**2)
    
    A11_1 = A(p1,T,f,bx,1,1,1)
    A11_2 = A(p1,T,f,bx,1,1,2)
    A11_3 = A(p1,T,f,bx,1,1,3)
    A11_4 = A(p1,T,f,bx,1,1,4)
    A11_5 = A(p1,T,f,bx,1,1,5)
    A11_6 = A(p1,T,f,bx,1,1,6)
    A11_7 = A(p1,T,f,bx,1,1,7)
    A11_8 = A(p1,T,f,bx,1,1,8)
    
    A21_1 = A(p1,T,f,bx,2,1,1)
    A21_2 = A(p1,T,f,bx,2,1,2)
    A21_3 = A(p1,T,f,bx,2,1,3)
    A21_4 = A(p1,T,f,bx,2,1,4)
    A21_5 = A(p1,T,f,bx,2,1,5)
    A21_6 = A(p1,T,f,bx,2,1,6)
    A21_7 = A(p1,T,f,bx,2,1,7)
    A21_8 = A(p1,T,f,bx,2,1,8)
    A21_9 = A(p1,T,f,bx,2,1,9)
    A21_10 = A(p1,T,f,bx,2,1,10)
    
    integral11 = A11_1 + A11_2 + A11_3 + A11_4 + A11_5 + A11_6 + A11_7 + A11_8 
    integral21 = A21_1 + A21_2 + A21_3 + A21_4 + A21_5 + A21_6 + A21_7 + A21_8 + A21_9 + A21_10
    integral = integral11 + integral21
    if abs((integral[0]-integral[1])/(integral[0]+integral[1]))<10**-14:
        return 0
    else:
        net = coefficient*(integral[0]-integral[1])
        return net

@nb.jit(nopython=True)
def R11R22(p1,T,f,bx):
    coefficient = 1/(2**4*(2*np.pi)**3*p1**2)
    
    A11_1 = A(p1,T,f,bx,1,1,1)
    A11_2 = A(p1,T,f,bx,1,1,2)
    A11_3 = A(p1,T,f,bx,1,1,3)
    A11_4 = A(p1,T,f,bx,1,1,4)
    A11_5 = A(p1,T,f,bx,1,1,5)
    A11_6 = A(p1,T,f,bx,1,1,6)
    A11_7 = A(p1,T,f,bx,1,1,7)
    A11_8 = A(p1,T,f,bx,1,1,8)
    
    A22_1 = A(p1,T,f,bx,2,2,1)
    A22_2 = A(p1,T,f,bx,2,2,2)
    A22_3 = A(p1,T,f,bx,2,2,3)
    A22_4 = A(p1,T,f,bx,2,2,4)
    A22_5 = A(p1,T,f,bx,2,2,5)
    A22_6 = A(p1,T,f,bx,2,2,6)
    A22_7 = A(p1,T,f,bx,2,2,7)
    A22_8 = A(p1,T,f,bx,2,2,8)
    A22_9 = A(p1,T,f,bx,2,2,9)
    A22_10 = A(p1,T,f,bx,2,2,10)
    A22_11 = A(p1,T,f,bx,2,2,11)
    
    integral11 = A11_1 + A11_2 + A11_3 + A11_4 + A11_5 + A11_6 + A11_7 + A11_8 
    integral22 = A22_1 + A22_2 + A22_3 + A22_4 + A22_5 + A22_6 + A22_7 + A22_8 + A22_9 + A22_10 + A22_11
    integral = integral11 + integral22
    if abs((integral[0]-integral[1])/(integral[0]+integral[1]))<10**-14:
        return 0
    else:
        net = coefficient*(integral[0]-integral[1])
        return net

@nb.jit(nopython=True)
def R11R23(p1,T,f,bx):
    coefficient = 1/(2**4*(2*np.pi)**3*p1**2)
    
    A11_1 = A(p1,T,f,bx,1,1,1)
    A11_2 = A(p1,T,f,bx,1,1,2)
    A11_3 = A(p1,T,f,bx,1,1,3)
    A11_4 = A(p1,T,f,bx,1,1,4)
    A11_5 = A(p1,T,f,bx,1,1,5)
    A11_6 = A(p1,T,f,bx,1,1,6)
    A11_7 = A(p1,T,f,bx,1,1,7)
    A11_8 = A(p1,T,f,bx,1,1,8)
    
    A23_1 = A(p1,T,f,bx,2,3,1)
    A23_2 = A(p1,T,f,bx,2,3,2)
    A23_3 = A(p1,T,f,bx,2,3,3)
    A23_4 = A(p1,T,f,bx,2,3,4)
    A23_5 = A(p1,T,f,bx,2,3,5)
    A23_6 = A(p1,T,f,bx,2,3,6)
    A23_7 = A(p1,T,f,bx,2,3,7)
    A23_8 = A(p1,T,f,bx,2,3,8)
    A23_9 = A(p1,T,f,bx,2,3,9)
    A23_10 = A(p1,T,f,bx,2,3,10)
    A23_11 = A(p1,T,f,bx,2,3,11)
    
    integral11 = A11_1 + A11_2 + A11_3 + A11_4 + A11_5 + A11_6 + A11_7 + A11_8 
    integral23 = A23_1 + A23_2 + A23_3 + A23_4 + A23_5 + A23_6 + A23_7 + A23_8 + A23_9 + A23_10 + A23_11
    integral = integral11 + integral23
    if abs((integral[0]-integral[1])/(integral[0]+integral[1]))<10**-14:
        return 0
    else:
        net = coefficient*(integral[0]-integral[1])
        return net

@nb.jit(nopython=True)
def R12R24(p1,T,f,bx):
    coefficient = 1/(2**4*(2*np.pi)**3*p1**2)
    
    A12_1 = A(p1,T,f,bx,1,2,1)
    A12_2 = A(p1,T,f,bx,1,2,2)
    A12_3 = A(p1,T,f,bx,1,2,3)
    A12_4 = A(p1,T,f,bx,1,2,4)
    A12_5 = A(p1,T,f,bx,1,2,5)
    A12_6 = A(p1,T,f,bx,1,2,6)
    
    A24_1 = A(p1,T,f,bx,2,4,1)
    A24_2 = A(p1,T,f,bx,2,4,2)
    A24_3 = A(p1,T,f,bx,2,4,3)
    A24_4 = A(p1,T,f,bx,2,4,4)
    A24_5 = A(p1,T,f,bx,2,4,5)
    A24_6 = A(p1,T,f,bx,2,4,6)
    A24_7 = A(p1,T,f,bx,2,4,7)
    A24_8 = A(p1,T,f,bx,2,4,8)
    A24_9 = A(p1,T,f,bx,2,4,9)
    A24_10 = A(p1,T,f,bx,2,4,10)

    integral12 = A12_1 + A12_2 + A12_3 + A12_4 + A12_5 + A12_6
    integral24 = A24_1 + A24_2 + A24_3 + A24_4 + A24_5 + A24_6 + A24_7 + A24_8 + A24_9 + A24_10
    integral = integral12 + integral24
    
    if abs((integral[0]-integral[1])/(integral[0]+integral[1]))<10**-14:
        return 0
    else:
        net = coefficient*(integral[0]-integral[1])
        return net


@nb.jit(nopython=True, parallel=True)
def driver(p_arr,T,f,bx):
    bx = p_arr[1]-p_arr[0] #why do we do this immediately if we send boxsize as an argument?
    output_arr = np.zeros(len(p_arr))
    for i in nb.prange (1,len(p_arr)):
        if p_arr[i]<.15791:
            output_arr[i] = R11R21(p_arr[i],T,f,bx)
        elif p_arr[i]<.18067:
            output_arr[i] = R11R22(p_arr[i],T,f,bx)
        elif p_arr[i]<.2555:
            output_arr[i] = R11R23(p_arr[i],T,f,bx)
        else:
            output_arr[i] = R12R24(p_arr[i],T,f,bx)
    return output_arr


# In[8]:


@nb.jit(nopython=True)
def B1_11(p1,E2,T,f,boxsize): #p1 is momentum of incoming neutrino, E2 is energy of incoming electron, T is temp, f is array of f values
    p1_box = int(np.round(p1/boxsize,0)) 
    a = .511
    B = E2
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = make_q_array(E3_array)
#    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M11_array = np.zeros(len(E3_array))
    M11_array[0] = M11(p1,E2,E3_array[0],q3_array[0])
    M11_array[-1] = M11(p1,E2,E3_array[-1],q3_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M11_array[i+1] = M11(p1,E2,E3_array[i+1],q3_array[i+1])

    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)


    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M11_array
    integrandminus_array = Fminus_array*M11_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A1_11(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino,T is the temperature at the time of the collision,and f is the array of f values from UDC
    e3cut = (p1**2 + .511**2)**(1/2)
    E2_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E2_array)):
        Bplus, Bminus = B1_11(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus]) 


# In[14]:


@nb.jit(nopython=True)
def B2_11(p1,E2,T,f,boxsize): #p1 is momentum of incoming neutrino, E2 is energy of incoming electron, T is temp, f is array of f values
    p1_box = int(np.round(p1/boxsize,0))
    q2 = (E2**2-.511**2)**(1/2)
    a = E2
    B = .5*(2*p1 + E2 - q2 + (.511**2)/(2*p1 + E2 - q2)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = make_q_array(E3_array)
#    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M12_value = M12(p1,E2,q2)
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)]         
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)

    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)

    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M12_value
    integrandminus_array = Fminus_array*M12_value
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A2_11(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E2_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E2_array)):
        Bplus, Bminus = B2_11(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[15]:


@nb.jit(nopython=True)
def B3_11(p1,E2,T,f,boxsize):
    p1_box = int(np.round(p1/boxsize,0)) 
    q2 = (E2**2-.511**2)**(1/2)
    a = .5*(2*p1 + E2 - q2 + (.511**2)/(2*p1 + E2 - q2)) #e2trans
    B = .5*(2*p1 + E2 + q2 + (.511**2)/(2*p1 + E2 + q2)) #e1lim
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = make_q_array(E3_array)
#    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M13_array = np.zeros(len(E3_array))
    M13_array[0] = M13(p1,E2,q2,E3_array[0],q3_array[0])
    M13_array[-1] = M13(p1,E2,q2,E3_array[-1],q3_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M13_array[i+1] = M13(p1,E2,q2,E3_array[i+1],q3_array[i+1])

    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)


    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M13_array
    integrandminus_array = Fminus_array*M13_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A3_11(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E2_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E2_array)):
        Bplus, Bminus = B3_11(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[16]:


@nb.jit(nopython=True)
def B4_11(p1,E2,T,f,boxsize):
    p1_box = int(np.round(p1/boxsize,0)) 
    q2 = (E2**2-.511**2)**(1/2)
    a = .511 #MeV
    B = .5*(2*p1 + E2 - q2 + (.511**2)/(2*p1 + E2 - q2)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = make_q_array(E3_array)
#    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M11_array = np.zeros(len(E3_array))
    M11_array[0] = M11(p1,E2,E3_array[0],q3_array[0])
    M11_array[-1] = M11(p1,E2,E3_array[-1],q3_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M11_array[i+1] = M11(p1,E2,E3_array[i+1],q3_array[i+1])

    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)

    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M11_array
    integrandminus_array = Fminus_array*M11_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A4_11(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = .511 + (2*p1**2)/(.511-2*p1)
    E2_array = ((e1cut-e3cut)/2)*x_valuese + (e1cut+e3cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E2_array)):
        Bplus, Bminus = B4_11(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[17]:


@nb.jit(nopython=True)
def B5_11(p1,E2,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q2 = (E2**2-.511**2)**(1/2)
    a = .5*(2*p1 + E2 - q2 + (.511**2)/(2*p1 + E2 - q2)) #e2trans
    B = E2
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = make_q_array(E3_array)
#    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M14_array = np.zeros(len(E3_array))
    M14_array[0] = M14(p1,E2,q2,E3_array[0],q3_array[0])
    M14_array[-1] = M14(p1,E2,q2,E3_array[-1],q3_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M14_array[i+1] = M14(p1,E2,q2,E3_array[i+1],q3_array[i+1])

    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)


    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M14_array
    integrandminus_array = Fminus_array*M14_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A5_11(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = .511 + (2*p1**2)/(.511-2*p1)
    E2_array = ((e1cut-e3cut)/2)*x_valuese + (e1cut+e3cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E2_array)):
        Bplus, Bminus = B5_11(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])



# In[18]:


@nb.jit(nopython=True)
def B6_11(p1,E2,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q2 = (E2**2-.511**2)**(1/2)
    a = E2
    B = .5*(2*p1 + E2 + q2 + (.511**2)/(2*p1 + E2 + q2)) #e1lim
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = make_q_array(E3_array)
#    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M13_array = np.zeros(len(E3_array))
    M13_array[0] = M13(p1,E2,q2,E3_array[0],q3_array[0])
    M13_array[-1] = M13(p1,E2,q2,E3_array[-1],q3_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M13_array[i+1] = M13(p1,E2,q2,E3_array[i+1],q3_array[i+1])

    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)

    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M13_array
    integrandminus_array = Fminus_array*M13_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A6_11(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = .511 + (2*p1**2)/(.511-2*p1)
    E2_array = ((e1cut-e3cut)/2)*x_valuese + (e1cut+e3cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E2_array)):
        Bplus, Bminus = B6_11(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[19]:


@nb.jit(nopython=True)
def B7_11(p1,E2,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q2 = (E2**2-.511**2)**(1/2)
    a = .5*(2*p1 + E2 - q2 + (.511**2)/(2*p1 + E2 - q2)) #e2lim
    B = E2
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
        
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = make_q_array(E3_array)
#    q3_array = (E3_array**2-.511**2)**(1/2)
    q3_array[-1] = 0
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M14_array = np.zeros(len(E3_array))
    M14_array[0] = M14(p1,E2,q2,E3_array[0],q3_array[0])
    M14_array[-1] = M14(p1,E2,q2,E3_array[-1],q3_array[-1])
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M14_array[i+1] = M14(p1,E2,q2,E3_array[i+1],q3_array[i+1])

    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)

    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M14_array
    integrandminus_array = Fminus_array*M14_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A7_11(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = .511 + (2*p1**2)/(.511-2*p1)
    E2_array = x_values+e1cut
    Bplus_array = np.zeros(len(E2_array))
    Bminus_array = np.zeros(len(E2_array))
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E2_array)):
        Bplus_array[i], Bminus_array[i] = B7_11(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus]) 


# In[20]:


@nb.jit(nopython=True)
def B8_11(p1,E2,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q2 = (E2**2-.511**2)**(1/2)
    a = E2
    B = .5*(2*p1 + E2 + q2 + (.511**2)/(2*p1 + E2 + q2)) #e1lim
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = make_q_array(E3_array)
#    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M13_array = np.zeros(len(E3_array))
    M13_array[0] = M13(p1,E2,q2,E3_array[0],q3_array[0])
    M13_array[-1] = M13(p1,E2,q2,E3_array[-1],q3_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M13_array[i+1] = M13(p1,E2,q2,E3_array[i+1],q3_array[i+1])

    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)

    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M13_array
    integrandminus_array = Fminus_array*M13_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A8_11(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = .511 + (2*p1**2)/(.511-2*p1)
    E2_array = x_values+e1cut
    Bplus_array = np.zeros(len(E2_array))
    Bminus_array = np.zeros(len(E2_array))
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E2_array)):
        Bplus_array[i], Bminus_array[i] = B8_11(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus])


# In[22]:


@nb.jit(nopython=True)
def B1_12(p1,E2,T,f,boxsize):
    #    p1_box = max(int(p1/boxsize),int(p1/boxsize+1e-10))
    p1_box = int(np.round(p1/boxsize,0)) 
    a = .511
    B = E2
    #    p4_array = np.zeros(int((B-a)/boxsize)+2)
    len_p4 = int((p1+E2-a)/boxsize) - int((p1+E2-B)/boxsize) + 2
    p4_array = np.zeros(len_p4)
    for i in range(len_p4-2):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = make_q_array(E3_array)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M11_array = np.zeros(len(E3_array))
    M11_array[0] = M11(p1,E2,E3_array[0],q3_array[0])
    M11_array[-1] = M11(p1,E2,E3_array[-1],q3_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
            f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M11_array[i+1] = M11(p1,E2,E3_array[i+1],q3_array[i+1])

    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)

    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M11_array
    integrandminus_array = Fminus_array*M11_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + 0.5*(integrandplus_array[0]+integrandplus_array[1])*(boxsize*j - p4_array[0]) + 0.5*(integrandplus_array[-2]+integrandplus_array[-1])*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + 0.5*(integrandminus_array[0]+integrandminus_array[1])*(boxsize*j - p4_array[0]) + 0.5*(integrandminus_array[-2]+integrandminus_array[-1])*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus


@nb.jit(nopython=True)
def B1_12_old(p1,E2,T,f,boxsize):
    a = .511
    B = E2
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = make_q_array(E3_array)
#    q3_array = make_q_array(E3_array)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M11_array = np.zeros(len(E3_array))
    M11_array[0] = M11(p1,E2,E3_array[0],q3_array[0])
    M11_array[-1] = M11(p1,E2,E3_array[-1],q3_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M11_array[i+1] = M11(p1,E2,E3_array[i+1],q3_array[i+1])
        
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)

    Fplus_array[0] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[int(p1/boxsize)])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[int(p1/boxsize)]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M11_array
    integrandminus_array = Fminus_array*M11_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + 0.5*(integrandplus_array[0]+integrandplus_array[1])*(boxsize*j - p4_array[0]) + 0.5*(integrandplus_array[-2]+integrandplus_array[-1])*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + 0.5*(integrandminus_array[0]+integrandminus_array[1])*(boxsize*j - p4_array[0]) + 0.5*(integrandminus_array[-2]+integrandminus_array[-1])*(p4_array[-1] - boxsize*k)
#    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
#    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A1_12(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E2_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E2_array)):
        Bplus, Bminus = B1_12(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus]) 


# In[23]:


@nb.jit(nopython=True)
def B2_12(p1,E2,T,f,boxsize):
    p1_box = int(np.round(p1/boxsize,0)) 
    q2 = (E2**2-.511**2)**(1/2)
    a = E2
    B = .5*(2*p1 + E2 - q2 + (.511**2)/(2*p1 + E2 - q2)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = make_q_array(E3_array)
#    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M12_value = M12(p1,E2,q2)
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)]         
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M12_value
    integrandminus_array = Fminus_array*M12_value
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A2_12(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E2_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E2_array)):
        Bplus, Bminus = B2_12(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus]) 


# In[24]:


@nb.jit(nopython=True)
def B3_12(p1,E2,T,f,boxsize):
    p1_box = int(np.round(p1/boxsize,0)) 
    q2 = (E2**2-.511**2)**(1/2)
    a = .5*(2*p1 + E2 - q2 + (.511**2)/(2*p1 + E2 - q2)) #e2trans
    B = .5*(2*p1 + E2 + q2 + (.511**2)/(2*p1 + E2 + q2)) #e1lim
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = make_q_array(E3_array)
#    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M13_array = np.zeros(len(E3_array))
    M13_array[0] = M13(p1,E2,q2,E3_array[0],q3_array[0])
    M13_array[-1] = M13(p1,E2,q2,E3_array[-1],q3_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M13_array[i+1] = M13(p1,E2,q2,E3_array[i+1],q3_array[i+1])
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M13_array
    integrandminus_array = Fminus_array*M13_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A3_12(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E2_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E2_array)):
        Bplus, Bminus = B3_12(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus]) 


# In[25]:


@nb.jit(nopython=True)
def B4_12(p1,E2,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q2 = (E2**2-.511**2)**(1/2)
    a = .511 #MeV
    B = .5*(2*p1 + E2 - q2 + (.511**2)/(2*p1 + E2 - q2)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
        
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = make_q_array(E3_array)
#    q3_array = (E3_array**2-.511**2)**(1/2)
    q3_array[-1] = 0
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M11_array = np.zeros(len(E3_array))
    M11_array[0] = M11(p1,E2,E3_array[0],q3_array[0])
    M11_array[-1] = M11(p1,E2,E3_array[-1],q3_array[-1])
    
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M11_array[i+1] = M11(p1,E2,E3_array[i+1],q3_array[i+1])
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)

    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M11_array
    integrandminus_array = Fminus_array*M11_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A4_12(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E2_array = x_values+e3cut #x_values and w_values defined above
    Bplus_array = np.zeros(len(E2_array))
    Bminus_array = np.zeros(len(E2_array))
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E2_array)):
        Bplus_array[i], Bminus_array[i] = B4_12(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus])


# In[26]:


@nb.jit(nopython=True)
def B5_12(p1,E2,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q2 = (E2**2-.511**2)**(1/2)
    a = .5*(2*p1 + E2 - q2 + (.511**2)/(2*p1 + E2 - q2)) #e2trans
    B = E2
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
        
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = make_q_array(E3_array)
#    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M14_array = np.zeros(len(E3_array))
    M14_array[0] = M14(p1,E2,q2,E3_array[0],q3_array[0])
    M14_array[-1] = M14(p1,E2,q2,E3_array[-1],q3_array[-1])
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M14_array[i+1] = M14(p1,E2,q2,E3_array[i+1],q3_array[i+1])
        
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M14_array
    integrandminus_array = Fminus_array*M14_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A5_12(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E2_array = x_values+e3cut #x_values and w_values defined above
    Bplus_array = np.zeros(len(E2_array))
    Bminus_array = np.zeros(len(E2_array))
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E2_array)):
        Bplus_array[i], Bminus_array[i] = B5_12(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus])


# In[27]:


@nb.jit(nopython=True)
def B6_12(p1,E2,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q2 = (E2**2-.511**2)**(1/2)
    a = E2
    B = .5*(2*p1 + E2 + q2 + (.511**2)/(2*p1 + E2 + q2)) #e1lim
    #    p4_array = np.zeros(int((B-a)/boxsize)+2)
    len_p4 = int(np.round((p1+E2-a)/boxsize,0)) - int(np.round((p1+E2-B)/boxsize,0)) + 2
    p4_array = np.zeros(len_p4)
    for i in range(len_p4-2):
        p4_array[i+1] = (int((p1+E2-B)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + E2 - B 
    p4_array[-1] = p1 + E2 - a
    E3_array = p1 + E2 - p4_array
    q3_array = make_q_array(E3_array)
#    q3_array = (E3_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E3_array))
    Fminus_array = np.zeros(len(E3_array))
    M13_array = np.zeros(len(E3_array))
    M13_array[0] = M13(p1,E2,q2,E3_array[0],q3_array[0])
    M13_array[-1] = M13(p1,E2,q2,E3_array[-1],q3_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[i+1],T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[i+1],T))*(1-f_holder)
        M13_array[i+1] = M13(p1,E2,q2,E3_array[i+1],q3_array[i+1])
        
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)

    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[0],T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[0],T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2,T))*fe(E3_array[-1],T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2,T)*(1-fe(E3_array[-1],T))*(1-f_last)
    integrandplus_array = Fplus_array*M13_array
    integrandminus_array = Fminus_array*M13_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + 0.5*(integrandplus_array[0]+integrandplus_array[1])*(boxsize*j - p4_array[0]) + 0.5*(integrandplus_array[-2]+integrandplus_array[-1])*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + 0.5*(integrandminus_array[0]+integrandminus_array[1])*(boxsize*j - p4_array[0]) + 0.5*(integrandminus_array[-2]+integrandminus_array[-1])*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A6_12(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E2_array = x_values+e3cut #x_values and w_values defined above
    Bplus_array = np.zeros(len(E2_array))
    Bminus_array = np.zeros(len(E2_array))
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E2_array)):
        Bplus_array[i], Bminus_array[i] = B6_12(p1,E2_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus])


# In[29]:


@nb.jit(nopython=True)
def B1_21(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    a = .511
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0])
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)]         
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1])
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)

    
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A1_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E3_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B1_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[30]:


@nb.jit(nopython=True)
def B2_21(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M22_value = M22(p1,E3,q3)
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)

    
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M22_value
    integrandminus_array = Fminus_array*M22_value
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A2_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E3_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B2_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[31]:


@nb.jit(nopython=True)
def B3_21(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    if (E3 - q3 - 2*p1)==0:
        B = T*100
    else:
        e1lim = .5*(E3 - q3 - 2*p1 + (.511**2)/(E3 - q3 - 2*p1))
        B = min(e1lim,T*100)
    if (B<a):
        return 0,0

    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3)
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3)
        
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)

    
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A3_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E3_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B3_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[32]:


@nb.jit(nopython=True)
def B4_21(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .511
    B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0])
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1])
        
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)

    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A4_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e3cut)/2)*x_valuese + (e3cut+e2cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B4_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[33]:


@nb.jit(nopython=True)
def B5_21(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3)
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3)
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
    
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A5_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e3cut)/2)*x_valuese + (e3cut+e2cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B5_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[34]:


@nb.jit(nopython=True)
def B6_21(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    if (E3 - q3 - 2*p1)==0:
        B = T*100
    else:
        e1lim = .5*(E3 - q3 - 2*p1 + (.511**2)/(E3 - q3 - 2*p1))
        B = min(e1lim,T*100)
    if (B<a):
        return 0,0

    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3)
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3)
        
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
    
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A6_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e3cut)/2)*x_valuese + (e3cut+e2cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B6_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[35]:


@nb.jit(nopython=True)
def B7_21(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2lim
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3)
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3)
            
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)

    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A7_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-e2cut)/2)*x_valuese + (e1cut+e2cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B7_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-e2cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-e2cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[36]:


@nb.jit(nopython=True)
def B8_21(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    if (E3 - q3 - 2*p1)==0:
        B = T*100
    else:
        e1lim = .5*(E3 - q3 - 2*p1 + (.511**2)/(E3 - q3 - 2*p1))
        B = min(e1lim,T*100)
    if (B<a):
        return 0,0
        
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3)
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3)
        
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A8_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    n = 100
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-e2cut)/2)*x_valuese + (e1cut+e2cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B8_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-e2cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-e2cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[37]:


@nb.jit(nopython=True)
def B9_21(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2lim
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3)
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3)
            
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A9_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = x_values+e1cut
    Bplus_array = np.zeros(len(E3_array))
    Bminus_array = np.zeros(len(E3_array))
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E3_array)):
        Bplus_array[i], Bminus_array[i] = B9_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus]) 


# In[38]:


@nb.jit(nopython=True)
def B10_21(p1,E3,T,f,boxsize):  #in this case boxsize shouldn't really apply right?
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3)
    
    for i in range (len(E2_array)-1):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3)
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A10_21(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = x_values+e1cut
    Bplus_array = np.zeros(len(E3_array))
    Bminus_array = np.zeros(len(E3_array))
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E3_array)):
        Bplus_array[i], Bminus_array[i] = B10_21(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus]) 


# In[40]:


@nb.jit(nopython=True)
def B1_22(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    a = .511
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0])
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1])
        
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A1_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E3_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B1_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[41]:


@nb.jit(nopython=True)
def B2_22(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M22_value = M22(p1,E3,q3)
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)]     
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
     
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M22_value
    integrandminus_array = Fminus_array*M22_value
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A2_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E3_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B2_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[42]:


@nb.jit(nopython=True)
def B3_22(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    if (E3 - q3 - 2*p1)==0:
        B = T*100
    else:
        e1lim = .5*(E3 - q3 - 2*p1 + (.511**2)/(E3 - q3 - 2*p1))
        B = min(e1lim,T*100)
    if (B<a):
        return 0,0
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3)
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)]     
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3)
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A3_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    E3_array = ((e3cut-.511)/2)*x_valuese + (e3cut+.511)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B3_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[43]:


@nb.jit(nopython=True)
def B4_22(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .511
    B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0])
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f[int(p4_array[i+1]/boxsize)]
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f[int(p4_array[i+1]/boxsize)])
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1])
        
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A4_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-e3cut)/2)*x_valuese + (e3cut+e1cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B4_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[44]:


@nb.jit(nopython=True)
def B5_22(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3)
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3)
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A5_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-e3cut)/2)*x_valuese + (e3cut+e1cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B5_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[45]:


@nb.jit(nopython=True)
def B6_22(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    if (E3 - q3 - 2*p1)==0:
        B = T*100
    else:
        e1lim = .5*(E3 - q3 - 2*p1 + (.511**2)/(E3 - q3 - 2*p1))
        B = min(e1lim,T*100)
    if (B<a):
        return 0,0

    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3)
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3)
        
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A6_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-e3cut)/2)*x_valuese + (e3cut+e1cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B6_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[46]:


@nb.jit(nopython=True)
def B7_22(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .511
    B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0])
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1])
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A7_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e1cut)/2)*x_valuese + (e2cut+e1cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B7_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e1cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e1cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[47]:


@nb.jit(nopython=True)
def B8_22(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3)
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3)
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A8_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e1cut)/2)*x_valuese + (e2cut+e1cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B8_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e1cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e1cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[48]:


@nb.jit(nopython=True)
def B9_22(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3)
    
    for i in range (len(E2_array)-1):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3)
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A9_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e1cut)/2)*x_valuese + (e2cut+e1cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B9_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e1cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e1cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[49]:


@nb.jit(nopython=True)
def B10_22(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0))
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2lim
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3)
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3)
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A10_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = x_values+e2cut
    Bplus_array = np.zeros(len(E3_array))
    Bminus_array = np.zeros(len(E3_array))
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E3_array)):
        Bplus_array[i], Bminus_array[i] = B10_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus]) 


# In[50]:


@nb.jit(nopython=True)
def B11_22(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3)
    
    for i in range (len(E2_array)-1):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3)
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A11_22(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = x_values+e2cut #x_values established above
    Bplus_array = np.zeros(len(E3_array))
    Bminus_array = np.zeros(len(E3_array))
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E3_array)):
        Bplus_array[i], Bminus_array[i] = B11_22(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus]) 


# In[52]:


@nb.jit(nopython=True)
def B1_23(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .511
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0])
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1])
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus


@nb.jit(nopython=True)
def A1_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-.511)/2)*x_valuese + (e1cut+.511)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B1_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[53]:


@nb.jit(nopython=True)
def B2_23(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M22_value = M22(p1,E3,q3)
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M22_value
    integrandminus_array = Fminus_array*M22_value
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A2_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-.511)/2)*x_valuese + (e1cut+.511)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B2_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[54]:


@nb.jit(nopython=True)
def B3_23(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    if (E3 - q3 - 2*p1)==0:
        B = T*100
    else:
        e1lim = .5*(E3 - q3 - 2*p1 + (.511**2)/(E3 - q3 - 2*p1))
        B = min(e1lim,T*100)
    if (B<a):
        return 0,0

    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3)
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3)
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A3_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-.511)/2)*x_valuese + (e1cut+.511)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B3_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[55]:


@nb.jit(nopython=True)
def B4_23(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .511
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0])
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1])
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A4_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e3cut-e1cut)/2)*x_valuese + (e1cut+e3cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B4_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-e1cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-e1cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[56]:


@nb.jit(nopython=True)
def B5_23(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M22_value = M22(p1,E3,q3)
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M22_value
    integrandminus_array = Fminus_array*M22_value
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A5_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e3cut-e1cut)/2)*x_valuese + (e1cut+e3cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B5_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-e1cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-e1cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[57]:


@nb.jit(nopython=True)
def B6_23(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    e2trans = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1))
    a = e2trans
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3)
    
    for i in range (len(E2_array)-1):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3)
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A6_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e3cut-e1cut)/2)*x_valuese + (e1cut+e3cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B6_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-e1cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-e1cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[58]:


@nb.jit(nopython=True)
def B7_23(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .511
    B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0])
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1])
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A7_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e3cut)/2)*x_valuese + (e2cut+e3cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B7_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[59]:


@nb.jit(nopython=True)
def B8_23(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3)
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3)
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A8_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e3cut)/2)*x_valuese + (e2cut+e3cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B8_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[60]:


@nb.jit(nopython=True)
def B9_23(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3)
    
    for i in range (len(E2_array)-1):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3)
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A9_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e3cut)/2)*x_valuese + (e2cut+e3cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B9_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[61]:


@nb.jit(nopython=True)
def B10_23(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2lim
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3)
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3)
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A10_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = x_values+e2cut
    Bplus_array = np.zeros(len(E3_array))
    Bminus_array = np.zeros(len(E3_array))
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E3_array)):
        Bplus_array[i], Bminus_array[i] = B10_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus])


# In[62]:


@nb.jit(nopython=True)
def B11_23(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3)
    
    for i in range (len(E2_array)-1):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)] 
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3)
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A11_23(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = x_values+e2cut #x_values established above
    Bplus_array = np.zeros(len(E3_array))
    Bminus_array = np.zeros(len(E3_array))
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E3_array)):
        Bplus_array[i], Bminus_array[i] = B11_23(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus])


# In[64]:


@nb.jit(nopython=True)
def B1_24(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .511
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0])
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1])
    
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A1_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-.511)/2)*x_valuese + (e1cut+.511)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B1_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[65]:


@nb.jit(nopython=True)
def B2_24(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M22_value = M22(p1,E3,q3)
    
    for i in range (len(E2_array)-1):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
 
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)

    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M22_value
    integrandminus_array = Fminus_array*M22_value
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A2_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e1cut-.511)/2)*x_valuese + (e1cut+.511)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B2_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e1cut-.511)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e1cut-.511)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[66]:


@nb.jit(nopython=True)
def B3_24(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .511
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0])
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1])
     
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A3_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e3cut-e1cut)/2)*x_valuese + (e1cut+e3cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B3_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-e1cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-e1cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[67]:


@nb.jit(nopython=True)
def B4_24(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    if (E3 + q3 - 2*p1)==0:
        B = T*100 #e2trans
    else:
        B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M22_value = M22(p1,E3,q3)
    
    for i in range (int((B-a)/boxsize)):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
 
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)

    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M22_value
    integrandminus_array = Fminus_array*M22_value
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A4_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e3cut-e1cut)/2)*x_valuese + (e1cut+e3cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B4_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-e1cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-e1cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[68]:


@nb.jit(nopython=True)
def B5_24(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    if (E3 + q3 - 2*p1)==0:
        a = T*100 #e2trans
    else:
        a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3)
    
    for i in range (len(E2_array)-1):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3)
 
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)

    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A5_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e1cut = p1 + (.511**2)/(4*p1)
    E3_array = ((e3cut-e1cut)/2)*x_valuese + (e1cut+e3cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B5_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e3cut-e1cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e3cut-e1cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[69]:


@nb.jit(nopython=True)
def B6_24(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .511
    B = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M21_array = np.zeros(len(E2_array))
    M21_array[0] = M21(p1,E2_array[0],E3,q2_array[0])
    M21_array[-1] = M21(p1,E2_array[-1],E3,q2_array[-1])
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M21_array[i+1] = M21(p1,E2_array[i+1],E3,q2_array[i+1])
     
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M21_array
    integrandminus_array = Fminus_array*M21_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A6_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e3cut)/2)*x_valuese + (e2cut+e3cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B6_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[70]:


@nb.jit(nopython=True)
def B7_24(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2trans
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3)
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3)
 
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A7_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e3cut)/2)*x_valuese + (e2cut+e3cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B7_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[71]:


@nb.jit(nopython=True)
def B8_24(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3)
    
    for i in range (len(E2_array)-1):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3)
 
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A8_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e3cut = (p1**2 + .511**2)**(1/2)
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = ((e2cut-e3cut)/2)*x_valuese + (e2cut+e3cut)/2
    integralplus = 0.0
    integralminus = 0.0
    for i in range(len(E3_array)):
        Bplus, Bminus = B8_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + w_valuese[i]*((e2cut-e3cut)/2)*Bplus
        integralminus = integralminus + w_valuese[i]*((e2cut-e3cut)/2)*Bminus
    return np.array([integralplus,integralminus])


# In[72]:


@nb.jit(nopython=True)
def B9_24(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = .5*(E3 + q3 - 2*p1 + (.511**2)/(E3 + q3 - 2*p1)) #e2lim
    B = E3
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M24_array = np.zeros(len(E2_array))
    M24_array[0] = M24(p1,E2_array[0],q2_array[0],E3,q3)
    M24_array[-1] = M24(p1,E2_array[-1],q2_array[-1],E3,q3)
    for i in range (int((B-a)/boxsize)):
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M24_array[i+1] = M24(p1,E2_array[i+1],q2_array[i+1],E3,q3)
     
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)
        
    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M24_array
    integrandminus_array = Fminus_array*M24_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A9_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = x_values+e2cut
    Bplus_array = np.zeros(len(E3_array))
    Bminus_array = np.zeros(len(E3_array))
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E3_array)):
        Bplus_array[i], Bminus_array[i] = B9_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus]) 


# In[73]:


@nb.jit(nopython=True)
def B10_24(p1,E3,T,f,boxsize): 
    p1_box = int(np.round(p1/boxsize,0)) 
    q3 = (E3**2 - .511**2)**(1/2)
    a = E3
    B = boxsize*len(f)
    if (a>B):
        return 0,0 #because we've gone too high so the integrals are effectively 0?
    p4_array = np.zeros(int((B-a)/boxsize)+2)
    for i in range(int((B-a)/boxsize)):
        p4_array[i+1] = (int((p1+a-E3)/boxsize)+i+1)*boxsize  #array of momenta of outgoing neutrinos
    p4_array[0] = p1 + a - E3 
    p4_array[-1] = p1 + B - E3
    E2_array = E3 + p4_array - p1
    q2_array = make_q_array(E2_array)
#    q2_array = (E2_array**2-.511**2)**(1/2)
    Fplus_array = np.zeros(len(E2_array))
    Fminus_array = np.zeros(len(E2_array))
    M23_array = np.zeros(len(E2_array))
    M23_array[0] = M23(p1,E2_array[0],q2_array[0],E3,q3)
    M23_array[-1] = M23(p1,E2_array[-1],q2_array[-1],E3,q3)
    
    for i in range (len(E2_array)-1):
        if int(p4_array[i+1]/boxsize)>=len(f):
            break #because everything in the arrays below are already zeros so we don't need to set them as zeros
        f_holder = log_linear_extrap(p4_array[i+1],np.array([(len(f)-2)*boxsize,(len(f)-1)*boxsize]),np.array([f[-2],f[-1]]))
        #if int(p4_array[i+1]/boxsize)<len(f): #this condition prevents the code from calling an index of f out of f's bounds
        #f_holder = f[int(p4_array[i+1]/boxsize)]
        Fplus_array[i+1] = (1-f[p1_box])*(1-fe(E2_array[i+1],T))*fe(E3,T)*f_holder
        Fminus_array[i+1] = f[p1_box]*fe(E2_array[i+1],T)*(1-fe(E3,T))*(1-f_holder)
        M23_array[i+1] = M23(p1,E2_array[i+1],q2_array[i+1],E3,q3)
 
    f_first, f_last, j, k = f_first_last(f, p4_array, boxsize)

    Fplus_array[0] = (1-f[p1_box])*(1-fe(E2_array[0],T))*fe(E3,T)*f_first
    Fminus_array[0] = f[p1_box]*fe(E2_array[0],T)*(1-fe(E3,T))*(1-f_first)
    Fplus_array[-1] = (1-f[p1_box])*(1-fe(E2_array[-1],T))*fe(E3,T)*f_last
    Fminus_array[-1] = f[p1_box]*fe(E2_array[-1],T)*(1-fe(E3,T))*(1-f_last)
    integrandplus_array = Fplus_array*M23_array
    integrandminus_array = Fminus_array*M23_array
    integralplus = trapezoid(integrandplus_array[1:-1],boxsize)
    integralminus = trapezoid(integrandminus_array[1:-1],boxsize)
    integralplus = integralplus + integrandplus_array[0]*(boxsize*j - p4_array[0]) + integrandplus_array[-1]*(p4_array[-1] - boxsize*k)
    integralminus = integralminus + integrandminus_array[0]*(boxsize*j - p4_array[0]) + integrandminus_array[-1]*(p4_array[-1] - boxsize*k)
    return integralplus, integralminus

@nb.jit(nopython=True)
def A10_24(p1,T,f,boxsize): #where p1 is the momentum of the incoming neutrino and T is the temperature at the time of the collision
    e2cut = p1 + .511*(p1 + .511)/(2*p1 + .511)
    E3_array = x_values+e2cut #x_values established above
    Bplus_array = np.zeros(len(E3_array))
    Bminus_array = np.zeros(len(E3_array))
    integralplus = 0.0
    integralminus = 0.0
    for i in range (len(E3_array)):
        Bplus_array[i], Bminus_array[i] = B10_24(p1,E3_array[i],T,f,boxsize)
        integralplus = integralplus + (np.e**x_values[i])*Bplus_array[i]*w_values[i]
        integralminus = integralminus + (np.e**x_values[i])*Bminus_array[i]*w_values[i]
    return np.array([integralplus,integralminus])


# In[74]:


@nb.jit(nopython=True)
def R11R21_old(p1,T,f,boxsize):
    coefficient = 1/(2**4*(2*np.pi)**3*p1**2)
    integral11 = (A1_11(p1,T,f,boxsize)+A2_11(p1,T,f,boxsize)+A3_11(p1,T,f,boxsize)+A4_11(p1,T,f,boxsize)
                +A5_11(p1,T,f,boxsize)+A6_11(p1,T,f,boxsize)+A7_11(p1,T,f,boxsize)+A8_11(p1,T,f,boxsize))
    integral21 = (A1_21(p1,T,f,boxsize)+A2_21(p1,T,f,boxsize)+A3_21(p1,T,f,boxsize)+A4_21(p1,T,f,boxsize)
                +A5_21(p1,T,f,boxsize)+A6_21(p1,T,f,boxsize)+A7_21(p1,T,f,boxsize)+A8_21(p1,T,f,boxsize)
                +A9_21(p1,T,f,boxsize)+A10_21(p1,T,f,boxsize))
    integral = integral11 + integral21
    if abs((integral[0]-integral[1])/(integral[0]+integral[1]))<10**-14:
        return 0
    else:
        return coefficient*(integral[0]-integral[1])

@nb.jit(nopython=True)
def R11R22_old(p1,T,f,boxsize):
    coefficient = 1/(2**4*(2*np.pi)**3*p1**2)
    integral11 = (A1_11(p1,T,f,boxsize)+A2_11(p1,T,f,boxsize)+A3_11(p1,T,f,boxsize)+A4_11(p1,T,f,boxsize)
                  +A5_11(p1,T,f,boxsize)+A6_11(p1,T,f,boxsize)+A7_11(p1,T,f,boxsize)+A8_11(p1,T,f,boxsize))
    integral22 = (A1_22(p1,T,f,boxsize)+A2_22(p1,T,f,boxsize)+A3_22(p1,T,f,boxsize)+A4_22(p1,T,f,boxsize)
              +A5_22(p1,T,f,boxsize)+A6_22(p1,T,f,boxsize)+A7_22(p1,T,f,boxsize)+A8_22(p1,T,f,boxsize)
              +A9_22(p1,T,f,boxsize)+A10_22(p1,T,f,boxsize)+A11_22(p1,T,f,boxsize))
    integral = integral11 + integral22
    if abs((integral[0]-integral[1])/(integral[0]+integral[1]))<10**-14:
        return 0
    else:
        return coefficient*(integral[0]-integral[1])

@nb.jit(nopython=True)
def R11R23_old(p1,T,f,boxsize):
    coefficient = 1/(2**4*(2*np.pi)**3*p1**2)
    integral11 = (A1_11(p1,T,f,boxsize)+A2_11(p1,T,f,boxsize)+A3_11(p1,T,f,boxsize)+A4_11(p1,T,f,boxsize)
                  +A5_11(p1,T,f,boxsize)+A6_11(p1,T,f,boxsize)+A7_11(p1,T,f,boxsize)+A8_11(p1,T,f,boxsize))
    integral23 = (A1_23(p1,T,f,boxsize)+A2_23(p1,T,f,boxsize)+A3_23(p1,T,f,boxsize)
              +A4_23(p1,T,f,boxsize)+A5_23(p1,T,f,boxsize)+A6_23(p1,T,f,boxsize)
              +A7_23(p1,T,f,boxsize)+A8_23(p1,T,f,boxsize)+A9_23(p1,T,f,boxsize)
              +A10_23(p1,T,f,boxsize)+A11_23(p1,T,f,boxsize))
    integral = integral11 + integral23
    if abs((integral[0]-integral[1])/(integral[0]+integral[1]))<10**-14:
        return 0
    else:
        return coefficient*(integral[0]-integral[1])

@nb.jit(nopython=True)
def R12R24_old(p1,T,f,boxsize):
    coefficient = 1/(2**4*(2*np.pi)**3*p1**2)
    integral12 = (A1_12(p1,T,f,boxsize)+A2_12(p1,T,f,boxsize)+A3_12(p1,T,f,boxsize)+A4_12(p1,T,f,boxsize)
                +A5_12(p1,T,f,boxsize)+A6_12(p1,T,f,boxsize))
    integral24 = (A1_24(p1,T,f,boxsize)+A2_24(p1,T,f,boxsize)+A3_24(p1,T,f,boxsize)+A4_24(p1,T,f,boxsize)
                +A5_24(p1,T,f,boxsize)+A6_24(p1,T,f,boxsize)+A7_24(p1,T,f,boxsize)+A8_24(p1,T,f,boxsize)
                +A9_24(p1,T,f,boxsize)+A10_24(p1,T,f,boxsize))
    integral = integral12 + integral24
    
    if abs((integral[0]-integral[1])/(integral[0]+integral[1]))<10**-14:
        return 0
    else:
        return coefficient*(integral[0]-integral[1])


@nb.jit(nopython=True, parallel=True)
def driver_old(p_array,T,f,boxsize):
    boxsize = p_array[1]-p_array[0]
    output_array = np.zeros(len(p_array))
    for i in nb.prange (1,len(p_array)):
        #    for i in nb.prange (1,len(p_array)):
        if p_array[i]<.15791:
            output_array[i] = R11R21_old(p_array[i],T,f,boxsize)
        elif p_array[i]<.18067:
            output_array[i] = R11R22_old(p_array[i],T,f,boxsize)
        elif p_array[i]<.2555:
            output_array[i] = R11R23_old(p_array[i],T,f,boxsize)
        else:
            output_array[i] = R12R24_old(p_array[i],T,f,boxsize)
    return output_array

