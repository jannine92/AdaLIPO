import numpy as np
import matplotlib.pyplot as plt
import time

import cmaes as cmaes



def timerfunc(func):
    """
    A timer decorator
    """
    def function_timer(*args, **kwargs):
        """
        A nested function for timing other functions
        """
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = np.float(end - start)
        msg = "The runtime for {func} took {time} seconds to complete"
        print(msg.format(func=func.__name__,
                         time=runtime))
        
        return value
    return function_timer



# ### Optimizer


# Lipschitz function
def L(xp_t1,fx,X,k):
    L_xp_t1 = np.amax([fx[index] - k * np.linalg.norm((xp_t1 - X[index]), ord=2) 
                    for index in range(len(X))])
    return L_xp_t1



def meta_L(xp,fx,X,k):
    return np.array([L(xp_t1,fx,X,k) for xp_t1 in xp])



@timerfunc
def adaptive_lipo(n, t_max, hyperparameters, f):
    """
    Args:
        n: number of iterations
        hyperparameters: possible inputs of f (or bounds for hyperparameters?)
        f: objective function 
        p: probability of exploitation vs (1-p) probability of exploration
    Returns:
        The estimated value that maximizes f
    """

    use_cmaes = True
    
    # Initialization 
    min_bounds,max_bounds = np.array(hyperparameters).T
    x1 = np.array((np.random.uniform(low=min_bounds, high=max_bounds, size=len(hyperparameters)))) 
    
    X = [x1]
    fx = [f(x1)]
    k = [0]
    t = 0
    
    # Iterations
    overall_time = []
    overall_rs_time = []
    time_k_update = []
    while t < t_max: 
        start_time = time.time()
        
        r_val = []
        # sample from L(x)
        
        if use_cmaes:

            gen = cmaes.minimize(meta_L, hyperparameters, fx, X, k[t], number_samples=20)

            cmaes_time = []
            for i in range(n): # TODO: exit criterion 
                start_time_rs = time.time()
                params,evaluation = next(gen)
                #print("Params: ", params, " Evaluation: ", evaluation)
                #print('len r_val: ', r_val)
#                 if len(r_val) > 0:
#                     if np.absolute(r_val[-1][1] - evaluation) < 0.001:
#                         print('finished after ', i, ' iterations')
#                         break
                r_val.append((params,evaluation))
            
                end_time_rs = time.time()
                time_needed_rs = end_time_rs - start_time_rs
                cmaes_time.append(time_needed_rs)
                overall_rs_time.append(time_needed_rs)
                #print('time taken for cmaes loop', i, ' : ', time_needed_cmaes)
        #         plt.plot(np.arange(0,n),cmaes_time,marker='o', color='deepskyblue')
        #         plt.title('time for cmaes loop')
        #         plt.show()
        
        else:
            for a in range(0,n):
                start_time_rs = time.time()

                xp_t1 = np.array((np.random.uniform(low=min_bounds, high=max_bounds, size=len(hyperparameters)))) 

                # function to minimize
                L_xp_t1 = np.amax([fx[index] - k[t] * np.linalg.norm((xp_t1 - X[index]), ord=2)
                                   for index in range(len(X))])

                r_val.append((xp_t1,L_xp_t1))
                
                end_time_rs = time.time()
                time_needed_rs = end_time_rs - start_time_rs
                overall_rs_time.append(time_needed_rs)
                
        # sort after L_xp_t1 and choose the xp_t1 value with the smallest L value
        best_x = sorted(r_val,key=lambda x: x[1])[0][0]
        X.append(best_x) 
        
        
        if len(X) >= (t+2):
            #calc_f_time = time.time()
            fx.append(f(X[t+1]))
            #calc_f_time_end = time.time()
            #print("Time needed to calculate f for timestep", t, " : ", calc_f_time_end - calc_f_time)
            
            t += 1
            
            start_k_update = time.time()
            k.append(np.amax([(np.absolute(fx[i]-fx[j])/np.linalg.norm((X[i]-X[j]), ord=1)) #try: ord=2
                              for i in range(len(X)) for j in range(len(X)) if i != j 
                              and np.linalg.norm(X[i]-X[j], ord=1)!=0 
                             ])) # slope of all X's in X

            print("t:{}, k:{} with x:{} and obj:{}".format(t-1,k[t],X[t],fx[t]))
            end_time = time.time()
            time_loop = end_time - start_time
            time_k_update.append(end_time-start_k_update)
            overall_time.append(time_loop)
    if use_cmaes:            
        rs_time_reshaped = np.array(overall_rs_time).reshape((t_max,n))
        rs_time_sum = []
        for i in range(0,t_max):
            rs_time_sum.append(np.sum(rs_time_reshaped[i]))
    else:
        rs_time_sum = overall_rs_time
    
    min_obj = np.argmin(fx) # model position of minimum object
    print('Minimum: ', np.amin(fx), ' for X: ', X[min_obj], ' with k: ', k[min_obj])
    return min_obj, X[min_obj]

