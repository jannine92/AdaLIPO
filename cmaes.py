import numpy as np
import matplotlib.pyplot as plt

def minimize(func,bounds,fx,X,k,x0=None,number_samples=100):
    min_bounds,max_bounds = np.array(bounds).T

    if x0 is None:
        x0 = ((min_bounds+max_bounds)/2.).reshape(-1)

    d = len(x0)
    sigma = np.diag([1 for _ in range(d)]) 
    mu = x0

    while True:
        samples = np.random.multivariate_normal(mu,sigma,number_samples)
        samples = np.clip(samples, min_bounds, max_bounds) 
        # plt.plot(samples[:,0],samples[:,1],color='black',marker="o",linewidth=0)
        evalu = np.array(func(samples,fx,X,k))
        samples = samples[evalu.argsort()][:d*2+1]

        # plt.plot(samples[:,0],samples[:,1],color='purple',marker="o",linewidth=0)
        # plt.xlim(-1,1)
        # plt.ylim(-1,1)
        # plt.show()
        sigma = a_cov(np.array(samples),mu)
        mu = np.mean(samples,axis=0)

        yield samples[0],evalu[0]

def a_cov(data,mu):
    m = data - mu
    cov = m.T.dot(m) / data.shape[0]
    return cov

if __name__ == "__main__":
    def y(X):
        return [x[0]**2+np.sin(10*x[1]) for x in X]

    gen = minimize(y,np.array([[-1,1],[-1,1]]))
    while True:
        print(next(gen))
