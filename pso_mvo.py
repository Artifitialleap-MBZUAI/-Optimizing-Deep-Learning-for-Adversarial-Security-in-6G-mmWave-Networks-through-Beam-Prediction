# -*- coding: utf-8 -*-

"""PSO module

Copyright (c) 2017 Future Processing sp. z.o.o.

@author: Pablo Ribalta Lorenzo
@email: pribalta@future-processing.com)
@date: 10.04.2017

This module encapsulates all the functionality related to Particle
Swarm Optimization, including the algorithm itself and the Particles
"""
import os
import copy
import numpy as np
import pandas as pd
import random
import time
import math
import sklearn
from numpy import asarray
from sklearn.preprocessing import normalize
from solution import solution


class Particle(object):
    """Particle class for PSO

    This class encapsulates the behavior of each particle in PSO and provides
    an efficient way to do bookkeeping about the state of the swarm in any given
    iteration.

    Args:
        lower_bound (np.array): Vector of lower boundaries for particle dimensions.
        upper_bound (np.array): Vector of upper boundaries for particle dimensions.
        dimensions (int): Number of dimensions of the search space.
        objective function (function): Black-box function to evaluate.

    """
    def __init__(self,
                 lower_bound,
                 upper_bound,
                 dimensions,
                 objective_function):
        self.reset(dimensions, lower_bound, upper_bound, objective_function)

    def reset(self,
              dimensions,
              lower_bound,
              upper_bound,
              objective_function):
        """Particle reset

        Allows for reset of a particle without reallocation.

		Args:
			lower_bound (np.array): Vector of lower boundaries for particle dimensions.
			upper_bound (np.array): Vector of upper boundaries for particle dimensions.
			dimensions (int): Number of dimensions of the search space.

        """
        position = []
        for i in range(dimensions):
            if lower_bound[i] < upper_bound[i]:
                position.extend(np.random.randint(lower_bound[i], upper_bound[i] + 1, 1, dtype=int))
            elif lower_bound[i] == upper_bound[i]:
                position.extend(np.array([lower_bound[i]], dtype=int))
            else:
                assert False

        self.position = [position]

        self.velocity = [np.multiply(np.random.rand(dimensions),
                                     (upper_bound - lower_bound)).astype(int)]

        self.best_position = self.position[:]

        self.function_value = [objective_function(self.best_position[-1])]
        self.best_function_value = self.function_value[:]

    def update_velocity(self, omega, phip, phig, best_swarm_position):
        """Particle velocity update

		Args:
			omega (float): Velocity equation constant.
			phip (float): Velocity equation constant.
			phig (float): Velocity equation constant.
			best_swarm_position (np.array): Best particle position.

        """
        random_coefficient_p = np.random.uniform(size=np.asarray(self.position[-1]).shape)
        random_coefficient_g = np.random.uniform(size=np.asarray(self.position[-1]).shape)

        self.velocity.append(omega
                             * np.asarray(self.velocity[-1])
                             + phip
                             * random_coefficient_p
                             * (np.asarray(self.best_position[-1])
                                - np.asarray(self.position[-1]))
                             + phig
                             * random_coefficient_g
                             * (np.asarray(best_swarm_position)
                                - np.asarray(self.position[-1])))

        self.velocity[-1] = self.velocity[-1].astype(int)

    def update_position(self, lower_bound, upper_bound, objective_function):
        """Particle position update

		Args:
			lower_bound (np.array): Vector of lower boundaries for particle dimensions.
			upper_bound (np.array): Vector of upper boundaries for particle dimensions.
			objective function (function): Black-box function to evaluate.

        """
        new_position = self.position[-1] + self.velocity[-1]

        if np.array_equal(self.position[-1], new_position):
            self.function_value.append(self.function_value[-1])
        else:
            mark1 = new_position < lower_bound
            mark2 = new_position > upper_bound

            new_position[mark1] = lower_bound[mark1]
            new_position[mark2] = upper_bound[mark2]

            self.function_value.append(objective_function(self.position[-1]))

        self.position.append(new_position.tolist())

        if self.function_value[-1] > self.best_function_value[-1]:
            self.best_position.append(self.position[-1][:])
            self.best_function_value.append(self.function_value[-1])



class PsoMVO(object):
    """PSO wrapper

    This class contains the particles and provides an abstraction to hold all the context
    of the PSO algorithm

    Args:
        swarmsize (int): Number of particles in the swarm
        maxiter (int): Maximum number of generations the swarm will run

    """
    def __init__(self, swarmsize=100, maxiter=100):
        self.max_generations = maxiter
        self.swarmsize = swarmsize

        self.omega = 0.5
        self.phip = 0.5
        self.phig = 0.5

        self.minstep = 1e-4
        self.minfunc = 1e-4

        self.best_position = [None]
        self.best_function_value = [1]

        self.particles = []

        self.retired_particles = []

    def run(self, function, lower_bound, upper_bound, kwargs=None):
        """Perform a particle swarm optimization (PSO)

		Args:
			objective_function (function): The function to be minimized.
			lower_bound (np.array): Vector of lower boundaries for particle dimensions.
			upper_bound (np.array): Vector of upper boundaries for particle dimensions.

		Returns:
			best_position (np.array): Best known position
			accuracy (float): Objective value at best_position
			:param kwargs:

        """
        if kwargs is None:
            kwargs = {}

        objective_function = lambda x: function(x, **kwargs)
        assert hasattr(function, '__call__'), 'Invalid function handle'

        assert len(lower_bound) == len(upper_bound), 'Invalid bounds length'

        lower_bound = np.array(lower_bound)
        upper_bound = np.array(upper_bound)

        assert np.all(upper_bound > lower_bound), 'Invalid boundary values'


        dimensions = len(lower_bound)

        self.particles = self.initialize_particles(lower_bound,
                                                   upper_bound,
                                                   dimensions,
                                                   objective_function)

        # Start evolution
        generation = 1
        
        while generation <= self.max_generations:
            #print(generation)
  
            for particle in self.particles:               
                
                particle.update_velocity(self.omega, self.phip, self.phig, self.best_position[-1])
                particle.update_position(lower_bound, upper_bound, objective_function)

                if particle.best_function_value[-1] == 0:
                    self.retired_particles.append(copy.deepcopy(particle))
                    particle.reset(dimensions, lower_bound, upper_bound, objective_function)
                elif particle.best_function_value[-1] < self.best_function_value[-1]:
                    stepsize = np.sqrt(np.sum((np.asarray(self.best_position[-1])
                                               - np.asarray(particle.position[-1])) ** 2))

                    if np.abs(np.asarray(self.best_function_value[-1])
                              - np.asarray(particle.best_function_value[-1])) \
                            <= self.minfunc:
                        return particle.best_position[-1], particle.best_function_value[-1]
                    elif stepsize <= self.minstep:
                        return particle.best_position[-1], particle.best_function_value[-1]
                    else:
                        self.best_function_value.append(particle.best_function_value[-1])
                        self.best_position.append(particle.best_position[-1][:])

            #print("best:",self.best_function_value[-1])
            output2 = pd.DataFrame({"best":[self.best_function_value[-1]]})
            output2.to_csv(os.path.join("output2", "best.csv"), mode='a', index=False,header=False) 
            
            fit=[]
            for i in range(self.swarmsize):          
                fit.append(self.particles[i].function_value[1])
            fit=np.array(fit)
            
            output_data = fit
            output2 = pd.DataFrame({"pso_function_value"+str(i+1): [output_data[i]] for i in range(len(output_data))})
            output2.to_csv(os.path.join("output2", "pso_function_value.csv"), mode='a', index=False, header=False)

            pso_pop=[]
            for i in range(self.swarmsize):          
                pso_pop.append(self.particles[i].position[-1])
            pso_pop=np.array(pso_pop)
            
            output_data = pso_pop
            output2 = pd.DataFrame({"pso_pop"+str(i+1): [output_data[i]] for i in range(len(output_data))})
            output2.to_csv(os.path.join("output2", "pso_pop.csv"), mode='a', index=False, header=False)
              
            mvo_improve=self.mvo(objective_function,lower_bound,upper_bound,dimensions, self.swarmsize,self.particles,generation,self.best_function_value) 
            #print(mvo_improve)
   
            j=0
            for i in range(self.swarmsize): 
                if i> round(self.swarmsize/2):
                  #print("i=",i)
                  #print("j=",j)
                  
                  self.particles[i].position[-1]= mvo_improve[j] 
                  
                      
                  j=j+1
            generation += 1
           

       
     
        
        return self.best_position[-1], self.best_function_value[-1]

    def initialize_particles(self,
                             lower_bound,
                             upper_bound,
                             dimensions,
                             objective_function):
        """Initializes the particles for the swarm

		Args:
			objective_function (function): The function to be minimized.
			lower_bound (np.array): Vector of lower boundaries for particle dimensions.
			upper_bound (np.array): Vector of upper boundaries for particle dimensions.
			dimensions (int): Number of dimensions of the search space.

		Returns:
			particles (list): Collection or particles in the swarm

        """
        particles = []
        for _ in range(self.swarmsize):
            particles.append(Particle(lower_bound,
                                      upper_bound,
                                      dimensions,
                                      objective_function))
            if particles[-1].best_function_value[-1] < self.best_function_value[-1]:
                self.best_function_value.append(particles[-1].best_function_value[-1])
                self.best_position.append(particles[-1].best_position[-1])


        self.best_position = [self.best_position[-1]]
        self.best_function_value = [self.best_function_value[-1]]

        return particles
    
    def mvo(self,objective_function, lower_bound, upper_bound, dimensions, N,x,generation,best_function_value):
        #print (lower_bound)
        #print (upper_bound)
        #print (dimensions)
        #print (N)
        #x=np.array(x)
        #print(x)
        #print("value01=",x[1].function_value[-1])
        N=round(N/2)
        #print("N=",N)
        output2 = pd.DataFrame({"generation":[generation]})
        output2.to_csv(os.path.join("output2", "Inflation_rates.csv"), mode='a', index=False,header=False) 
           
        
        lower_bound=list(lower_bound)
        upper_bound=list(upper_bound)
        
        Universes =[]
        
        for i in range(N):
          
          Universes.append(x[i].position[-1])
        Universes=np.array(Universes)
        #Universes=np.array([[167,3,3,86],[200,3,3,108],[41,2,1,127],[154,3,1,115]])
        #print(Universes)
        #Universes=Universes[-1]
        
        "parameters"
        # dim=30
        # lower_bound=-100
        # upper_bound=100
        WEP_Max = 1
        WEP_Min = 0.2
        Max_time=10
        # N=50
        if not isinstance(lower_bound, list):
            lower_bound = [lower_bound] * dimensions
        if not isinstance(upper_bound, list):
            upper_bound = [upper_bound] * dimensions
    
        #Universes = np.copy(x)
        # i in range(dimensions):
        #    Universes[:, i] = np.random.uniform(0, 1, N) * (upper_bound[i] - lower_bound[i]) + lower_bound[i]
    
        Sorted_universes = np.copy(Universes)
    
        convergence = np.zeros(Max_time)
    
        Best_universe = [0] * dimensions
        Best_universe_Inflation_rate = float("inf")
    
        s = solution()
    
        Time = 1
        ############################################
        print("MVO is optimizing")
    
        timerStart = time.time()
        s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        while Time < Max_time + 1:
            

            
            "Eq. (3.3) in the paper"
            WEP = WEP_Min + Time * ((WEP_Max - WEP_Min) / Max_time)
    
            TDR = 1 - (math.pow(Time, 1 / 6) / math.pow(Max_time, 1 / 6))
    
            Inflation_rates = [0] * len(Universes)
    
            for i in range(0, N):
                for j in range(dimensions):
 
                    Universes[i, j] = np.clip(Universes[i, j], lower_bound[j], upper_bound[j])
    
                Inflation_rates[i] = objective_function(Universes[i, :])
    
                if Inflation_rates[i] < Best_universe_Inflation_rate:
    
                    Best_universe_Inflation_rate = Inflation_rates[i]
                    Best_universe = np.array(Universes[i, :])
    
            sorted_Inflation_rates = np.sort(Inflation_rates)
            sorted_indexes = np.argsort(Inflation_rates)
    
            for newindex in range(0, N):
                Sorted_universes[newindex, :] = np.array(
                    Universes[sorted_indexes[newindex], :]
                )
    
            normalized_sorted_Inflation_rates = np.copy(self.normr(-sorted_Inflation_rates))
    
            Universes[0, :] = np.array(Sorted_universes[0, :])
    
            for i in range(1, N):
                Back_hole_index = i
                for j in range(0, dimensions):
                    r1 = random.random()
    
                    if r1 < normalized_sorted_Inflation_rates[i]:
                        White_hole_index = self.RouletteWheelSelection(-sorted_Inflation_rates)
    
                        if White_hole_index == -1:
                            White_hole_index = 0
                        White_hole_index = 0
                        Universes[Back_hole_index, j] = Sorted_universes[
                            White_hole_index, j
                        ]
    
                    r2 = random.random()
    
                    if r2 < WEP:
                        r3 = random.random()
                        if r3 < 0.5:
                            Universes[i, j] = Best_universe[j] + TDR * (
                                (upper_bound[j] - lower_bound[j]) * random.random() + lower_bound[j]
                            )  # random.uniform(0,1)+lower_bound);
                        if r3 > 0.5:
                            Universes[i, j] = Best_universe[j] - TDR * (
                                (upper_bound[j] - lower_bound[j]) * random.random() + lower_bound[j]
                            )  # random.uniform(0,1)+lower_bound);
    
            convergence[Time - 1] = Best_universe_Inflation_rate
            if Time % 1 == 0:
                print(
                    [
                        "MVO At iteration "
                        + str(Time)
                        + " the best fitness is "
                        + str(Best_universe_Inflation_rate)
                        + " PSO At iteration "
                        + str(generation)
                    ]
                )
                
                
                
  
                #inflation_rates = [[1.2, 2.3, 3.4, 4.5]]
                output_data = Inflation_rates
                output2 = pd.DataFrame({"Inflation_rate_"+str(i+1): [output_data[i]] for i in range(len(output_data))})
                output2.to_csv(os.path.join("output2", "Inflation_rates.csv"), mode='a', index=False, header=False)
                    
                
                output_data = Universes
                output2 = pd.DataFrame({"mvo_pop"+str(i+1): [output_data[i]] for i in range(len(output_data))}) 
                output2.to_csv(os.path.join("output2", "mvo_pop.csv"), mode='a', index=False,header=False) 
     
                output2 = pd.DataFrame({"mvo":[str(Best_universe_Inflation_rate)]})
                output2.to_csv(os.path.join("output2", "mvo.csv"), mode='a', index=False,header=False) 
              
    
            Time = Time + 1
        # timerEnd = time.time()
        # s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        # s.executionTime = timerEnd - timerStart
        # s.convergence = convergence
        # s.optimizer = "MVO"
        # s.objfname = objective_function.__name__
        for i in range(0, N):
           if Inflation_rates[i] > x[1].function_value[-1]:
               print("solution does not improve",i)
               print(Inflation_rates[i])
               print(x[1].function_value[-1])
               Universes[i,:]=np.array(x[i].position[-1])
               print(Universes[i,:]) 
           else:
               print("solution improve",i)              
           if Inflation_rates[i] < self.best_function_value[-1]:
               self.best_function_value[-1]=Inflation_rates[i] 
               self.best_position[-1]=Universes[i,:]
                
        return Universes
    
    def normr(self,Mat):
        """normalize the columns of the matrix
        B= normr(A) normalizes the row
        the dtype of A is float"""
        Mat = Mat.reshape(1, -1)
        # Enforce dtype float
        if Mat.dtype != "float":
            Mat = asarray(Mat, dtype=float)
    
        # if statement to enforce dtype float
        B = normalize(Mat, norm="l2", axis=1)
        B = np.reshape(B, -1)
        return B


    def randk(self,t):
        if (t % 2) == 0:
            s = 0.25
        else:
            s = 0.75
        return s
    
    
    def RouletteWheelSelection(self,weights):
        accumulation = np.cumsum(weights)
        p = random.random() * accumulation[-1]
        chosen_index = -1
        for index in range(0, len(accumulation)):
            if accumulation[index] > p:
                chosen_index = index
                break
    
        choice = chosen_index
    
        return choice