#=================================================================================================
#--------------------------------      Clifford Attractor     ------------------------------------
#=================================================================================================

#-----------------------     new_X = sin(a * Y) + c * cos(a * X)     -----------------------------
#-----------------------     new_Y = sin(b * X) + d * cos(b * Y)     -----------------------------

#=================================================================================================

import numpy as np
import pandas as pd
import panel as pn
import datashader as ds
from numba import jit
from datashader import transfer_functions as tf

#=================================================================================================

@jit(nopython=True)
def Clifford_trajectory(x0, y0, n, a, b, c, d):
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    
    for i in np.arange(n-1):

        x[i+1] = np.sin(a*y[i]) + c*np.cos(a*x[i])
        y[i+1] = np.sin(b*x[i]) + d*np.cos(b*y[i])
    
    return x, y

#=================================================================================================

def Clifford_plot(x0=0, y0=0, n=100000, a=1.6, b=-0.7, c=-1.8, d=-1.9, cmap=["pink", "purple"]):
    
    cvs = ds.Canvas(plot_width=700, plot_height=700)
    x, y = Clifford_trajectory(x0, y0, n, a, b, c, d)
    agg = cvs.points(pd.DataFrame({'x':x, 'y':y}), 'x', 'y')
    
    return tf.shade(agg, cmap)

#=================================================================================================

pn.extension()
pn.interact(Clifford_plot, n=(1,1000000))

#=================================================================================================

# The value of this attractor can be changed freely. Try it in the jupyter notebook.

