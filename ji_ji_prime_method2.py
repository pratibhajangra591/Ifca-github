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




def k(m_1, m_2):
    return ((3/85) * (c**5))/((G**3)* m_1 * m_2 * (m_1 + m_2))
 

def f(j_i):
        if np.all(j_i)< 1e-3:
            return (j_i**m) * (10**b) 
        else:
            spl =  InterpolatedUnivariateSpline(ji_ref_list, tm_ref_list, ext = 2, k= 1)
            return (spl(j_i)) 
    

    
def C_ref(m_1, m_2, a_i):
    beta = 0.75
    gamma  =  0.65
    delta = -0.89
    a_i_ref = 1.67e2 # in units of pc 
    m1_ref = 1      # in units of solar mass.
    m2_ref = 1e-3   # in units of solar mass.
    return ((a_i/a_i_ref)**(beta)) * ((m_1 /m1_ref)**(gamma)) * ((m_2 /m2_ref)**(delta))




j_grid = np.geomspace(1e-6, 1.0, 100)
lhs_grid = (j_grid**7)/f(j_grid)
j_c_interp = interp1d(lhs_grid, j_grid,fill_value="extrapolate")


#j_c = j_c_interp((C_ref(m_1, m_2, a_i))/(k(m_1, m_2) * (a_i**4)))



def ji_interpolation(x, y, variable):
    g =  InterpolatedUnivariateSpline(x, y, ext = 2, k= 1) # j_i = g(t_m)
    return g(variable)
            
       
def ji_prime_into_C_interpolation(x, y, variable):
    g =  InterpolatedUnivariateSpline(x, y, ext = 2, k= 1) # j_i = g(t_m)
    g_prime = g.derivative()
    return g_prime(variable) 

            
    
    

def ji_into_C_fitting_function(m, b, t_m):   # t_m/C = (j_i**m) * (10**b)
    return   (10**(-b/m)) * (t_m**(1/m))



def ji_into_C_prime_fitting_function(m, b, t_m):   # t_m/C = (j_i**m) * (10**b)
    return  (1/m) * (10**(-b/m)) * (t_m**(-(m-1)/m))
