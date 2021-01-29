# -*- coding: utf-8 -*-
"""
This program uses simple Monte Carlo to approximate pi
See slides for pseudocode and graphics

Created on Tue Jan 21 22:53:14 2020

@author: mroll
"""

import random

# Initialize some stuff. You need this if you run it multiple times
inside_circle = 0
total_points = 0

# Create a for loop that runs 1000 times, making counter go from 0 to 999
for counter in range (0,1000): 
    X = random.random()
    Y = random.random()
    # Test to see if our (X,Y) is in the circle or not (under the line)
    if(X**2+Y**2<1): # note that what you do when if is true is indented
        inside_circle = inside_circle+1
    total_points = total_points+1 # Always update total points, no indent
    
print("pi approximation is", 4*inside_circle/total_points)