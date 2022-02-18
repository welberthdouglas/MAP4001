import pandas as pd
import numpy as np
from toolz import curry


@curry
def fixed_point_approx(initial_guess:float, 
                       h:float, 
                       t_k:float,
                       approx_func,
                       threshold:float = 0.0001,
                       max_it:int = 1e5) -> list:
    
    approx =[initial_guess]
    conv = [1e10]
    
    it = 0
    while max(conv) > threshold:
        y_k1 = approx_func(initial_guess, h, t_k, approx[-1])
        
        approx.append(y_k1)
        conv = abs(approx[-1] - approx[-2])
        
        if it == max_it:
            raise Exception(f"Fixed point method did not converged for t_(k+1) = {t_k + h} in {it} iterations. \n initial_guess ={approx[0]}")
        
        it += 1
    return approx[-1]

@curry
def edo_approx_multistep(y:list, f:list, h:float, method:str):
    
    # EXPLICIT METHODS
    if method == "hw4-method":
        y_k = y[-1] + h*(-f[-3] + (3/2)*f[-2] + (1/2)*f[-1])
    
    elif method == "AB-2":
        y_k = y[-1] + (h/2)*(3*f[-1] -f[-2])
    
    elif method == "Leap-Frog":
        y_k = y[-2] + 2*h*f[-1]
    
    else:
        raise Exception(f"Invalid value '{method}' for parameter method")
    
    return y_k.flatten()

@curry
def edo_approx(y_k:float, 
               h:float, 
               t:float, 
               y_dot,
               method:str = "runge-kutta")->float:
    
    # EXPLICIT METHODS
    if method == "euler-explicit":
        y_k1 = y_k + h*(y_dot(t,y_k))
        
        
    elif method == "euler-improved":
        euler_yk1 = y_k + h*(y_dot(t+h,y_k))
        y_k1 = y_k + (h/2)*(y_dot(t,y_k) + y_dot(t+h,euler_yk1))
        
    elif method == "runge-kutta":
        
        K1 = y_dot(t,y_k)
        K2 = y_dot(t + h/2, y_k + (h/2)*K1)
        K3 = y_dot(t + h/2, y_k + (h/2)*K2)
        K4 = y_dot(t + h, y_k + h*K3)

        y_k1 = y_k + (h/6)*(K1 + 2*K2 + 2*K3 + K4)
    
    
    # IMPLICIT METHODS
    elif method == "euler-implicit":
        
        def euler_implicit_func(initial_guess:float, 
                              h:float, 
                              t_k:float, 
                              approx:float)->float:
            
            return initial_guess + h*y_dot(t_k + h,approx)
        
        y_k1 = fixed_point_approx(y_k,h,t,euler_implicit_func)
    
    
    elif method == "trapezoidal":
        
        def trap_implicit_func(initial_guess:float,
                     h:float,
                     t_k:float,
                     approx:float)->float:
            
            return initial_guess + (h/2)*(y_dot(t_k,initial_guess) + y_dot(t_k + h,approx))
        
        y_k1 = fixed_point_approx(y_k,h,t,trap_implicit_func)
    
    else:
        raise Exception(f"Invalid value '{method}' for parameter method")
    
    return y_k1.flatten()



class EDONumSol:
    
    def __init__(self,y_dot,y_0=None,t_0=None,t_f=None,h=None):
        self.y_0 = y_0
        self.t_0 = t_0
        self.t_f = t_f
        self.h = h
        self.y_dot = y_dot
        self.num_var = len(y_0)
    
    
    def solve(self, t_0 = None, t_f = None, y_0 = None,h = None, method = "runge-kutta"):
        
        input_params = [t_0,t_f,y_0,h]
        params = [self.t_0,self.t_f,self.y_0,self.h]
        t_0,t_f,y_0,h = [s if p==None else p for s,p in zip(params,input_params)] 
            
        
        y = [y_0]
        t = [t_0]
        n = int((t_f - t_0)/ h)
        
        y_approx_func = edo_approx(method = method, y_dot = self.y_dot)
        
        for k in range(1,n+1):
            y.append(y_approx_func(y[-1],h,t[-1]))
            t.append(t[0] + k*h)
            
        return t,y
    
    def solve_multistep(self, 
                        nsteps, 
                        method, 
                        t_0 = None, 
                        t_f = None, 
                        y_0 = None,
                        h = None, 
                        t_init=None, 
                        y_init=None, 
                        initialization_method = "runge-kutta"):
        
        input_params = [t_0,t_f,y_0,h]
        params = [self.t_0,self.t_f,self.y_0,self.h]
        t_0,t_f,y_0,h = [s if p==None else p for s,p in zip(params,input_params)] 
            
        
        y = [y_0]
        t = [t_0]
        n = int((t_f - t_0)/ h)
        
        #initialization
        if (t_init == None) or (y_init==None):
            t_f = self.t_0 + (nsteps-1)*self.h
            t,y = self.solve(t_f = t_f, method = initialization_method)
        else:
            t = t_init.copy()
            y = y_init.copy()
        
        print(t)
        print(y)
        f = [self.y_dot(t_,y_).flatten() for t_,y_ in zip(t,y)]
        
        for k in range(nsteps,n+1):
            y.append(edo_approx_multistep(y, f, h, method))
            t.append(t[0] + k*h)
            
            f.pop(0)
            f.append(self.y_dot(t[-1],y[-1]))
            
        return t,y
    
    def convergence_analysis(self, 
                             t_eval, 
                             h_0, 
                             max_evals, 
                             method, 
                             y_true = None):
        
        k = [i for i in range(0,max_evals)]
        h_k = [h_0/(2**k_) for k_ in k]
        
        
        conv_table = pd.DataFrame({"k":k,"h_k":h_k})
        
        # Y results for all h_0/2**(k)
        y_approx = []
        for h_1 in h_k:
                result = self.solve(t_f = t_eval, h = h_1, method = method)[1][-1]
                y_approx.append(result)
        
        for i in range(self.num_var):
            conv_table[f"y_var{i}"] = np.array(y_approx).swapaxes(0,1)[i]
            
        
        # Error and convergence estimative  TODO REFACTOR THIS MESS
        D1 = lambda x: (abs(x[-3] - x[-2]))
        D2 = lambda x: (abs(x[-2] - x[-1]))
        
        for i in range(self.num_var):
            conv_table[f"D1_var{i}"] = conv_table[f"y_var{i}"].rolling(3).apply(D1,raw=True)
            conv_table[f"D2_var{i}"] = conv_table[f"y_var{i}"].rolling(3).apply(D2,raw=True)
            conv_table[f"p_tilde_var{i}"] = np.log2(conv_table[f"D1_var{i}"]/conv_table[f"D2_var{i}"])
            conv_table[f"C_var{i}"] = conv_table[f"D1_var{i}"]/((conv_table["h_k"]**conv_table[f"p_tilde_var{i}"])*(1-2**-conv_table[f"p_tilde_var{i}"]))
            conv_table[f"e_tilde_var{i}"] = conv_table[f"C_var{i}"]*conv_table["h_k"]**conv_table[f"p_tilde_var{i}"]
        
        
        if y_true:
            y_exact = y_true(t_eval)
            
            error = abs(y_exact - np.array(y_approx))
            p_est = lambda x: np.log2(x[-2]/ x[-1])
            
            for i in range(self.num_var):

                conv_table[f"error_var{i}"] = error.swapaxes(0,1)[i]
                conv_table[f"p_var{i}"] = conv_table[f"error_var{i}"].rolling(2).apply(p_est,raw=True)
        
        out_cols = ["k","h_k"] + [f"y_var{i}" for i in range(self.num_var)] + [f"e_tilde_var{i}" for i in range(self.num_var)] + [f"p_tilde_var{i}" for i in range(self.num_var)] 
        
        if y_true:
            out_cols += [f"error_var{i}" for i in range(self.num_var)] + [f"p_var{i}" for i in range(self.num_var)] 
            
        return conv_table[out_cols]