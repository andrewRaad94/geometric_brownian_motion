
from math import exp,sqrt
import numpy as np
import pdb
import matplotlib.pyplot as plt
import pandas as pd


class simulator:
    def __init__(self) -> None:
        pass 

    def run_simulations(self, n : int ):
        '''
        Determine how many brownian motion simulations you would like to generate
        
        Returns
        --------
        'dict' 
            A dictionary in which the key, value pairs are the index of each simulation and a list of the generated outcomes
        
        '''
        results = {}

        for i in range(1, n + 1,1):
            results[i] = self.calculate()
        return results

    def plot_simulations(self, results):
        '''
        Plot all simulations once they are ready
        '''
        results = pd.DataFrame(results)
        plt.figure(figsize=(15, 10))
        plt.plot(results)
        self.add_graph_information()
        plt.show()
        return
    
    def first_passage_barrier(self, simulations, barrier):
        count = 0
        for i in simulations.keys():
            if max(simulations[i]) > barrier:
                count += 1
        return count/self.simulations
    
    def __call__(self):
        '''
        Call method to generate and plot all simulations after all variables are created
        '''
        simulations = self.run_simulations(self.simulations)
        self.plot_simulations(simulations)
        return




class geometric_brownian_motion_simulation(simulator):
    '''
    This class once instantiated will generate and plot price paths that were simulated with geometric brownian motion.
        Parameters
    ----------
    initial price : `float`
        The initial price of a stock at time 0.
    drift_term : `float`
        The tendency of the asset to move in a particular direction (i.e the overall trend).
    sigma : `float`
        The 'volatility' of the asset.
    '''

    def __init__(self, initial_price, drift_term, sigma, increments, time_interval, n_simulations):
        super().__init__()
        self.initial_price = initial_price
        self.drift_term = drift_term 
        self.sigma = sigma 
        self.increments = increments 
        self.time_interval = time_interval
        self.simulations = n_simulations

    def calculate(self):
        '''
        Generate the path of the stock using gbm
        '''
        increments_per_period = int(self.increments/ self.time_interval)
        dt = 1/increments_per_period #for each period, there will be dt time steps
        price_vector = np.zeros(self.increments+1)
        price_vector[0] = self.initial_price
         
        path = np.exp((self.drift_term - 0.5 * self.sigma**2 ) * dt + self.sigma * np.random.normal(0, np.sqrt(dt), size = self.increments ))
        price_vector[1:] = path
        price = price_vector.cumprod()
        return price 

    def add_graph_information(self):
        plt.xlabel("Increments")
        plt.ylabel("Stock Value")
        plt.title(f"Realizations of {self.simulations} Geometric Brownian Motion simulations using $\mu = {self.drift_term}$ and $\sigma = {self.sigma}$")


class standard_brownian_motion_simulation(simulator):
    '''
    This class once instantiated will generate and plot price paths that were simulated with standard brownian motion.
        Parameters
    ----------
    increments : `int`
        The number of increments that will be used for the whole time period.
    time_interval : `int`
        The number of time periods to be used.
    n_simulations : `int`
        The number of simulations to generate and plot.
    '''
    def __init__(self, increments, time_interval, n_simulations):
        self.increments = increments 
        self.time_interval = time_interval
        self.simulations = n_simulations

    def calculate(self):
        increments_per_period = int(self.increments/ self.time_interval)
        dt = 1/increments_per_period  #for each period, there will be dt time steps

        # By definition, standard brownian motion starts at point 0
        y = 0

        #create a list to store the paths
        y_path = []
            # Generate a random step in the x and y direction and update position
        dy = np.random.normal(0, np.sqrt(dt), size = self.increments)
        y_path = dy.cumsum()

        return y_path

    def add_graph_information(self):
        plt.xlabel("Increments")
        plt.ylabel("Value")
        plt.title(f"Realizations of {self.simulations} Standard Brownian Motion simulations")

if __name__ == "__main__":
    gbm = standard_brownian_motion_simulation(252,1,500)
    gbm()