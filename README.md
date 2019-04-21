# AdaLIPO with CMA-ES for Optimization

This repository contains an implementation of Adaptive LIPO (AdaLIPO) finding a global **minimum** based on the paper from Malherbe and Vayatis (2017).     
The AdaLIPO algorithm optimizes a function by approximating its Lipschitz function. It is assumed that the Lipschitz constant is unknown. Therefore the constant is updated along with the optimization process.  
   
The original AdaLIPO algorithm samples the points to evaluate from a random distribution. 
This implementation includes random sampling but also a guided random search algorithm instead: CMA-ES (Hansen and Ostermeier, 2001).
CMA-ES is an evolutionary algorithm which samples the next evaluation points according to a multivariate normal distribution. The mean vector and covariance matrix of this distribution are computed from previous evaluations.  

This repository consists of the following files:
* **AdaLIPO_test**: algorithm can be tested directly with the objective functions `sum of squares` and `ackley`and either as 1D or as nD version
* **AdaLIPO**: contains the function `adaptive_lipo`which can be directly used as optimizer 
* **cmaes**: used as sampling algorithm for AdaLIPO. It is integrated in both AdaLIPO files as default and therefore needs to be imported as well. If you want to use random search instead, set ` use_cmaes=False`


### Requirements: 
- Python 3.6
- NumPy 1.12 or higher

## Sources 
Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in
evolution strategies. *Evolutionary computation*, 9 (2), 159–195.  
Malherbe, C., & Vayatis, N. (2017). Global optimization of Lipschitz functions. arXiv:1703.02628  

