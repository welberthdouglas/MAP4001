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
    """
    Returns the fixed point approximation for a defined function
    """
    
    approx =[initial_guess]
    conv = [1e10]  
    
    it = 0
    while max(conv) > threshold: # checking for convergence
        y_k1 = approx_func(initial_guess, h, t_k, approx[-1])
        
        approx.append(y_k1)
        conv = abs(approx[-1] - approx[-2])
        
        if it == max_it: #stops the loop if it = max_iteration
            raise Exception(f"Fixed point method did not converged for t_(k+1) = {t_k + h} in {it} iterations. \n initial_guess ={approx[0]}")
        
        it += 1
    return approx[-1]

@curry
def edo_approx_multistep(y:list, f:list, h:float, method:str):
    """
    Returns the discretization function for multistep methods
    """
    
    # EXPLICIT METHODS
    if method == "HW4":
        y_k = y[-1] + h*(-f[-3] + (3/2)*f[-2] + (1/2)*f[-1])

    elif method == "AB-2":
        y_k = y[-1] + (h/2)*(3*f[-1] - f[-2])
        
    elif method == "AB-3":
        y_k = y[-1] + (h/12)*(5*f[-3] - 16*f[-2] + 23*f[-1])
    
    elif method == "AB-4":
        y_k = y[-1] + (h/24)*((55*f[-1]) - (59*f[-2]) + (37*f[-3]) -(9*f[-4]))
    
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
               method:str = "runge-kutta",
               k= None)->float:
    
    """
    Returns the discretization function for single-step methods
    """
    
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
    
    elif method == "noise":  #noise function with consistence zero - used to test computational modes 
        k[0] += 1
        y_k1 = y_k + h*(-1)**k[0]
    
    else:
        raise Exception(f"Invalid value '{method}' for parameter method")
    
    return y_k1.flatten()



class EDONumSol:
    
    """
    Class to describe numerical solutions to ODEs
    
    Attributes:
      y_0 - initial value
      t_0 - initial time of evaluation
      t_f - final time of evaluation
      h - integration step
      y_dot - ODE in the Cauchy format
      num_var - number of variables of the ODE (num_var > 1 => vectorial ODE)
      nsteps - number of steps (for multistep methods only)
      k - counter for the noise initialization method
    
    methods:
      solve - solves the ODE with a single step method
      solve_multistep - solves the ODE with a multistep method
      convergence_analysis - performs a convergence analysis of the specified method and ODE
    
    """
    
    def __init__(self,y_dot,y_0=None,t_0=None,t_f=None,h=None):
        self.y_0 = y_0
        self.t_0 = t_0
        self.t_f = t_f
        self.h = h
        self.y_dot = y_dot
        self.num_var = len(y_0)
        self.multistep_methods = {"HW4":3,"AB-2":2,"AB-3":3,"AB-4":4,"Leap-Frog":2}
        self.k = [0]
    
    @curry
    def solve(self, 
              t_0 = None, 
              t_f = None, 
              y_0 = None,
              h = None, 
              method = "runge-kutta"):
        
        # getting parameters values in case they are not inputed in the solve call
        input_params = [t_0,t_f,y_0,h]
        params = [self.t_0,self.t_f,self.y_0,self.h]
        t_0,t_f,y_0,h = [s if p==None else p for s,p in zip(params,input_params)] 
            
        
        y = [y_0]
        t = [t_0]
        n = int((t_f - t_0)/ h) # calculating number of steps
        
        # initializing discretization function
        y_approx_func = edo_approx(method = method, y_dot = self.y_dot, k = self.k) 
        
        # getting numerical approximations for the interval (y(t_0),y(t_f)]
        for k in range(1,n+1):
            y.append(y_approx_func(y[-1],h,t[-1]))
            t.append(t_0 + k*h)
            
        return t,y
    
    @curry
    def solve_multistep(self, 
                        method, 
                        t_0 = None, 
                        t_f = None, 
                        y_0 = None,
                        h = None,
                        y_true=None, 
                        initialization_method = "euler-explicit"):
        
        # getting parameters values in case they are not inputed in the solve call
        input_params = [t_0,t_f,y_0,h]
        params = [self.t_0,self.t_f,self.y_0,self.h]
        t_0,t_f,y_0,h = [s if p==None else p for s,p in zip(params,input_params)] 
            
        #getting the corresponding number of initialization steps
        self.nsteps = self.multistep_methods[method]
        
        y = [y_0]
        t = [t_0]
        n = int((t_f - t_0)/ h) # calculating number of steps
        
        #initialization
        if (y_true == None) or (initialization_method != "exact"): # initialization with single-step method
            self.k = [0] # reseting counter in case of noise initialization
            t_f = self.t_0 + (self.nsteps-1)*h
            t,y = self.solve(t_f = t_f,h=h, method = initialization_method)
            
        elif (y_true) and (initialization_method == "exact"): # initialization with exact solution 
            t_f = self.t_0 + (self.nsteps-1)*h
            t = [i for i in np.linspace(t_0,t_f,self.nsteps)]
            y = [np.array(y_true(t_)) for t_ in t]
        
        else:
            raise Exception(f"To use exact initialization the y_true parameter must be note None")
        
        
        #storing initialized values for y_dot
        f = [self.y_dot(t_,y_).flatten() for t_,y_ in zip(t,y)]
        
        # mutistep integration loop
        for k in range(self.nsteps,n+1):
            y.append(edo_approx_multistep(y, f, h, method))
            t.append(t_0 + k*h)
            
            f.pop(0)
            f.append(self.y_dot(t[-1],y[-1]))

        return t,y
    
    @curry
    def convergence_analysis(self, 
                             h_0, 
                             max_evals, 
                             method, 
                             initialization_method,
                             t_eval = None, 
                             y_true = None):
        
        k = [i for i in range(0,max_evals)]
        h_k = [h_0/(2**k_) for k_ in k]
        
        
        conv_table = pd.DataFrame({"k":k,"h_k":h_k})
        
        if method in self.multistep_methods:
            solve_func = self.solve_multistep(initialization_method = initialization_method, y_true= y_true)
        else:
            solve_func = self.solve
        
        y_approx = []
        t_approx = []
        for h_1 in h_k:
            t_result,result = solve_func(t_f = t_eval, h = h_1, method = method)
            y_approx.append(result)
            t_approx.append(t_result)
        
        for i in range(self.num_var):
            if i > 1:
                conv_table[f"y_var{i}"] = np.array(y_approx).swapaxes(0,1)[i]
            else:
                conv_table[f"y_var{i}"] = np.array(y_approx)        
        
        if y_true:
            if t_eval:
                # Error and convergence estimative  TODO REFACTOR THIS MESS
                D1 = lambda x: (abs(x[-3] - x[-2]))
                D2 = lambda x: (abs(x[-2] - x[-1]))
                
                for i in range(self.num_var):
                    conv_table[f"D1_var{i}"] = conv_table[f"y_var{i}"].rolling(3).apply(D1,raw=True)
                    conv_table[f"D2_var{i}"] = conv_table[f"y_var{i}"].rolling(3).apply(D2,raw=True)
                    conv_table[f"p_tilde_var{i}"] = np.log2(conv_table[f"D1_var{i}"]/conv_table[f"D2_var{i}"])
                    conv_table[f"C_var{i}"] = conv_table[f"D1_var{i}"]/((conv_table["h_k"]**conv_table[f"p_tilde_var{i}"])*(1-2**-conv_table[f"p_tilde_var{i}"]))
                    conv_table[f"e_tilde_var{i}"] = conv_table[f"C_var{i}"]*conv_table["h_k"]**conv_table[f"p_tilde_var{i}"]

                y_exact = y_true(t_eval)
                error = abs(y_exact - np.array(y_approx))
                
            else:
                y_exact = np.array([np.array([np.array(y_true(t_)) for t_ in t_k]) for t_k in t_approx])
                error = np.array([np.max(arr) for arr in abs(y_exact - np.array(y_approx))])
            
            p_est = lambda x: np.log2(x[-2]/ x[-1])
            
            for i in range(self.num_var):
                if i > 1:
                    conv_table[f"error_var{i}"] = error.swapaxes(0,1)[i]
                else:
                    conv_table[f"error_var{i}"] = error
                conv_table[f"p_var{i}"] = conv_table[f"error_var{i}"].rolling(2).apply(p_est,raw=True)
        
        out_cols = ["k","h_k"]
        
        if t_eval:
            out_cols += [f"y_var{i}" for i in range(self.num_var)]  + [f"e_tilde_var{i}" for i in range(self.num_var)] + [f"p_tilde_var{i}" for i in range(self.num_var)] 
        
        if y_true:
            out_cols += [f"error_var{i}" for i in range(self.num_var)] + [f"p_var{i}" for i in range(self.num_var)]
            
        return conv_table[out_cols]