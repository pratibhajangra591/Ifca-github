import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
from scipy.integrate import quad
import seaborn as sns
from scipy.interpolate import interp1d
import matplotlib.colors as colors
from matplotlib import rcParams
from scipy.interpolate import InterpolatedUnivariateSpline
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif"
})


π = np.pi
Ω_cdm = 0.85
G = 4.4959e-15            #in units of M☉^-1 pc^3 yr^-2
c = 0.3068                #in units of pc yr^-1
ρ_eq = 3.1808e3           #in units of M☉ pc^-3 with ρ_eq=2.1548e-16 kg m^-3
pc = 3.0857e16            # in meters
yr = 3.154e7              # in units of seconds
t_eq = 1.5923e12/yr      # in units of seconds
t_m = 13.78e9             #in units of yrs corresponding to t_0=13.78Gyr
t_0 = 13.78e9             #in units of yrs corresponding to t_0=13.78Gyr

σ_eq = 0.005
ρ_m = 4e19                #ρ_m = 4e19 M☉ Gpc^-3



file = np.load('tmofj0_ref_multipeak.npz')
m = file['arr_0']
b = file['arr_1']
a_i_ref =  file['arr_2']
ji_ref_list = file['arr_3']
tm_ref_list = file['arr_4']


y = ji_ref_list
x = tm_ref_list

def k(m_1, m_2):
    return ((3/85) * (c**5))/((G**3)* m_1 * m_2 * (m_1 + m_2))
 

 

    
def C_ref(m_1, m_2, a_i):
    beta = 0.75
    gamma  =  0.65
    delta = -0.89
    a_i_ref = 1.67e2 # in units of pc 
    m1_ref = 1      # in units of solar mass.
    m2_ref = 1e-3   # in units of solar mass.
    return ((a_i/a_i_ref)**(beta)) * ((m_1 /m1_ref)**(gamma)) * ((m_2 /m2_ref)**(delta))







def interpolation(x, y):
    return InterpolatedUnivariateSpline(x, y, ext = 2, k= 1) # j_i = g(t_m)



def interpolation_prime(x, y):
    return  interpolation(x, y).derivative()


#t_static = t_m/C = (j_i**m)*(10**b)

def ji_interpolation(x, y, t_static):
    return interpolation(x, y)(t_static)
            
       
def ji_prime_interpolation(x, y, t_static):
    return interpolation_prime(x, y)(t_static) 

            
    
    
#t_static = t_m/C = (j_i**m)*(10**b)

def ji_fitting_function(m, b, t_static):
    return   (10**(-b/m)) * (t_static**(1/m))     



def ji_prime_fitting_function(m, b, t_static): 
    return  (1/m) * (10**(-b/m)) * (t_static**((1-m)/m))


j_grid = np.geomspace(1e-25, 1.0, 100)
lhs_grid = np.zeros(len(j_grid))

for i, j in enumerate(j_grid):
    if j < 1e-3:
        def f(j):
            return (j**m) * (10**b)
    else:
        def f(j):
            x = ji_ref_list
            y = tm_ref_list
            return interpolation(x, y)(j)

    lhs_grid[i] = (j**7)/f(j)

j_c_interp = interp1d(lhs_grid, j_grid)
#j_c = j_c_interp((C_ref(m_1, m_2, a_i))/(k(m_1, m_2) * (a_i**4)))
