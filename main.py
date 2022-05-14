import vtk
from math import sin, cos, pi, sqrt
import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt

class Field():
    
    def __init__(self, U0, N, L, l, R, x, y):
    
        self.U0 = U0            # Wave amplitude
        self.N = N              # The number of angles
        self.L = L              # Distance from aperture to screen
        self.k = 2 * pi/l       # Wave number
        self.R = R              # Radius of the circumscribed circle
        
        self.len = 2*R/sin(pi/N)
        self.A = np.zeros((N+1, 2), dtype=np.complex) # Coordinates of vertexes
        self.S = np.zeros((x+1, y+1), dtype=np.complex)   # Coordinates of screen mesh
        self.E = np.zeros((x+1, y+1), dtype=np.float)   # Distribution of the field 
                                                    # on the screen
        self.I = np.zeros((x, y), dtype=np.float)   # Distribution of the 
                                                    # intensity on the screen
                                                    
        self.x = x              # Amount of horizontal steps
        self.y = y              # Amount of vertical steps
        self.a = 2 * R          # Side of the screen
        
    def make_mesh(self):
        
        dx = self.a/self.x
        dy = self.a/self.y
        for i in range(self.y+1):
            y =  self.a/2 - dy*i
            for j in range(self.x+1):
                x = -self.a/2 + dx*j
                self.S[i][j] = x + 1j * y
    
    def make_aperture(self):
        phi = 2*pi/self.N
        for n in range(self.N):
            self.A[n][0] = self.R * cos(n * phi)
            self.A[n][1] = self.R * sin(n * phi)
        self.A[self.N] = self.A[0]
        self.A = np.apply_along_axis(lambda args: [complex(*args)], 1, self.A)
    
    def print_aperture(self):
        X = self.S.real
        Y = self.S.imag
        x = self.A.real
        y = self.A.imag
        
        fig, axs = plt.subplots()
        axs.set_aspect('equal', 'box')
        plt.plot(x, y)
        plt.scatter(X, Y)
        plt.grid()
        plt.show()      
   
f = Field(1, 6, 1, 0.0000006, 1, 50, 50)
f.make_aperture()
f.make_mesh()             
f.print_aperture()
