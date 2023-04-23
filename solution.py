# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:28:33 2023

@author: Ammar.Abasi
"""

class solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual = []
        self.convergence = []
        self.optimizer = ""
        self.objfname = ""
        self.startTime = 0
        self.endTime = 0
        self.executionTime = 0
        self.lower_bound = 0
        self.upper_bound = 0
        self.dimensions = 0
        self.popnum = 0
        self.maxiers = 0