import theano  #to generate and deal with lots of parallel path (ideal for looking at statistics!)
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import matplotlib.pyplot as plt
import time



class StochasticEquationSolver():
    
    def __init__(self, seed = 31234):
        self.name = None
        self.srng = RandomStreams(seed=seed)
        self.rv_n = None #self.srng.normal(c.shape, std=0.05)
        self.num_samples = None # has to be add 
        self.dt0 = None #0.1
        self.total_time = None #10
        self.total_steps = None #int(total_time/dt0)
        self.c_initial_value = None
        
    def init_run(self, num_samples, dt0, total_time, c_initial_value):
        self.num_samples = num_samples
        self.dt0 = dt0
        self.total_time = total_time
        self.total_steps = int(total_time/dt0)
        self.c_initial_value = c_initial_value
        self.c0 = theano.shared(self.c_initial_value*np.ones(self.num_samples, dtype='float32'))
            #define symbolic variables
        self.dt = T.fscalar("dt")
        self.c = T.fvector("c")
        self.rv_n = self.srng.normal(self.c.shape, std=0.05) #not a global variable
        
    def g(self, c):
        """ Corresponds to the function G in the Ito's equation dy = f(y,t)dt + G(y)dW """
        raise NotImplementedError("to be defined in subclass")
        
    def evolve(self,c):
        raise NotImplementedError("to be defined in subclass")
                
    def euler(self,c, n, k, l, dt):
        return T.cast(c + dt*evolve(c, n, k, l) + T.sqrt(dt)*c*rv_n, 'float32')

    def rk4(self, c, dt):
        '''
        Adapted from
        http://people.sc.fsu.edu/~jburkardt/c_src/stochastic_rk/stochastic_rk.html
        
        Ok for solving the Ito's equation in which F and G does not depends explicitely of the time: 
        dy(x,t)/dt = F(y(x,t)) + G(y(x,t))dW(x,t)
        '''
        a21 =   2.71644396264860
        a31 = - 6.95653259006152
        a32 =   0.78313689457981
        a41 =   0.0
        a42 =   0.48257353309214
        a43 =   0.26171080165848
        a51 =   0.47012396888046
        a52 =   0.36597075368373
        a53 =   0.08906615686702
        a54 =   0.07483912056879
 
        q1 =   2.12709852335625
        q2 =   2.73245878238737
        q3 =  11.22760917474960
        q4 =  13.36199560336697

        x1 = c
        k1 = dt * self.evolve(x1) + T.sqrt(dt) * self.g(c) * self.rv_n

        x2 = x1 + a21 * k1
        k2 = dt * self.evolve(x2) + T.sqrt(dt) * self.g(c) * self.rv_n

        x3 = x1 + a31 * k1 + a32 * k2
        k3 = dt * self.evolve(x3) + T.sqrt(dt) * self.g(c) * self.rv_n

        x4 = x1 + a41 * k1 + a42 * k2
        k4 = dt * self.evolve(x4) + T.sqrt(dt) * self.g(c) * self.rv_n

        return T.cast(x1 + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4, 'float32')
    
    
    def run(self):
       
        c = self.c
        dt = self.dt

        #create loop
        #first symbolic loop with everything
        (cout, updates) = theano.scan(fn=self.rk4,
                                 outputs_info=[c], #output shape
                                 non_sequences=[dt], #fixed parameters
                                 n_steps=self.total_steps)

        #compile it
        sim = theano.function(inputs=[dt], 
                               outputs=cout, 
                               givens={c:self.c0}, 
                               updates=updates, 
                               allow_input_downcast=True)

        print "running sim..."
        start = time.clock()
        cout = sim(self.dt0)
        diff = (time.clock() - start)
        print "done in", diff, "s at ", diff/self.num_samples, "s per path"
        
        self.cout = cout
        
    def plot_series(self, N=50):
        
        downsample_factor_t = int(0.1/self.dt0) #always show 10 points per time unit
        downsample_factor_p = int(self.num_samples/N)
        x = np.linspace(0, self.total_time, int(self.total_steps/downsample_factor_t))
        plt.subplot()
        plt.title(self.name)
        plt.xlabel('time')
        plt.ylabel('y')
        plt.plot(x, self.cout[::downsample_factor_t, ::downsample_factor_p])

        
    def plot_histo(self):
        
        plt.subplot()
        plt.title(self.name)
        bins = np.linspace(0, 1.2, 50)
        plt.hist(self.cout[int(1/self.dt0)], bins, alpha = 0.5, 
                    normed=True, histtype='bar',  
                    label=['Time one'])
        plt.hist(self.cout[int(2/self.dt0)], bins, alpha = 0.5, 
                    normed=True, histtype='bar',  
                    label=['Time two'])
        plt.hist(self.cout[-1], bins, alpha = 0.5, 
                    normed=True, histtype='bar',  
                    label=['Final time'])
        plt.legend()
       
    def mean_ensemble(self):
        """ index is the time index up to where the mean is computed."""
        downsample_factor_t = int(0.1/self.dt0)
        x = np.linspace(0, self.total_time, int(self.total_steps/downsample_factor_t))
        # x = np.linspace(0, index, index)/self.total_steps*downsample_factor_t*self.total_time #vector time
        mean = (np.sum(self.cout[::downsample_factor_t, :], axis=1)/self.num_samples)
        print "Average over realisations and paths: ", np.sum(mean)/len(mean)
        plt.subplot()
        plt.title('Mean value (average over the realisations)')
        plt.xlabel('time')
        plt.plot(x, mean)
        return np.sum(mean)/len(mean)
        
    def mean_alongpaths(self):
        x = np.arange(self.num_samples)+1
        mean = (np.sum(self.cout[: :], axis=0)/self.total_steps)
        print "Average over paths and realisations: ", np.sum(mean)/len(mean)
        plt.subplot()
        plt.title('Mean value (average over the time)')
        plt.xlabel('realisations')
        plt.plot(x[::200], mean[::200])
        return np.sum(mean)/len(mean)
    
    def var_ensemble(self):
        """ index is the time index up to where the mean is computed."""
        downsample_factor_t = int(0.1/self.dt0)
        x = np.linspace(0, self.total_time, int(self.total_steps/downsample_factor_t))
        # x = np.linspace(0, index, index)/self.total_steps*downsample_factor_t*self.total_time #vector time
        mean = (np.sum((self.cout[::downsample_factor_t, :])**2, axis=1)/self.num_samples)
        plt.subplot()
        plt.title('Variance value (average over the realisations)')
        plt.xlabel('time')
        plt.title("Variance")
        plt.plot(x, mean)
            
