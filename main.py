import vtk
from math import sin, cos, pi, sqrt, exp
import numpy as np
import quadpy
from scipy.integrate import quad
from matplotlib import pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def complex_projection(v1, v2):
    a = np.array([v1.real, v1.imag]).reshape(1, 2)[0]
    b = np.array([v2.real, v2.imag]).reshape(1, 2)[0]
    a_on_b = (np.dot(a, b)/np.dot(b, b)) * b
    return np.absolute(a_on_b[0] + 1j * a_on_b[1])


class Field():
    
    def __init__(self, U0, N, L, l, R, x, y):
    
        self.U0 = U0            # Wave amplitude
        self.N = N              # The number of angles
        self.L = L              # Distance from aperture to screen
        self.k = 2 * pi/l       # Wave number
        self.R = R              # Radius of the circumscribed circle
        
        self.len = 2*R/sin(pi/N)
        self.A = np.zeros((N+1, 2), dtype=np.complex128)     # Coordinates of vertexes
        self.S = np.zeros((x+1, y+1), dtype=np.complex128)   # Coordinates of screen mesh
        self.E = np.zeros((x+1, y+1), dtype=np.float64)      # Distribution of the field 
                                                             # on the screen
        self.I = np.zeros((x+1, y+1), dtype=np.float64)      # Distribution of the 
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

    
    def make_integral_sum(self, x, y):
        p = self.S[x][y]
        summ = 0
        
        for j in range(self.N):
            qj  = self.A[j]
            qj_ = self.A[j+1]
            Lj  = qj_ - qj
            tj  = Lj/np.absolute(Lj)
            t0  = complex_projection((qj - p), tj)
            t1  = (np.absolute(Lj) + t0)[0]
            print(np.absolute(qj - p)**2 - t0**2)
            a = sqrt(abs(np.absolute(qj - p)**2 - t0**2))
            b = self.k/(2*self.L)
            if (a!=0):
                I, err = (quadpy.quad(lambda x: np.exp(1j * b * x**2) / 
                         (x**2 + abs(a)**2), t0, t1, epsabs=100, epsrel=100, limit=495))
                I_ = a * np.exp(1j * b * a**2) * I
                summ += I_
            else:
                summ += 0
        return(summ)
    
    
    def calculate_E(self):
        pol = []
        for i in range(self.N):
            pol.append((self.A[i].real[0], self.A[i].imag[0]))
        print(pol)
        polygon = Polygon(pol)
        for i in range(self.y+1):
            for j in range(self.x+1):
                I = self.make_integral_sum(i, j)
                point = Point(self.S[i][j].real, self.S[i][j].imag)
                if (polygon.contains(point)):
                    e = 1
                else:
                    e = 0
                self.E[i][j] += self.U0 * (e - 1/(2*pi) * I)
    
    def calculate_I(self): 
    
        for i in range(self.y+1):
            for j in range(self.x+1):
                self.I[i][j] = self.E[i][j] * np.conj(self.E[i][j]) / self.U0**2 
                
    def snapshot(self):
    
        structuredGrid = vtk.vtkStructuredGrid()
        points = vtk.vtkPoints()
        E = vtk.vtkDoubleArray()
        E.SetName("E")
        
        for i in range(self.y+1):
            for j in range(self.x+1):
                points.InsertNextPoint(self.S[i][j].real, self.S[i][j].imag, 0)
                E.InsertNextValue(self.I[i][j])
        
        structuredGrid.SetDimensions(self.x+1, self.y+1, 1)
        structuredGrid.SetPoints(points)

        structuredGrid.GetPointData().AddArray(E)
        
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetInputDataObject(structuredGrid)
        writer.SetFileName("diff.vts")
        writer.Write()
   
f = Field(1, 6, 0.5, 0.0000007, 0.001, 200, 200)
f.make_aperture()
f.make_mesh()             
#f.print_aperture()
f.calculate_E()
f.calculate_I()
f.snapshot()
