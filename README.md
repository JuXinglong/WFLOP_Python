# Wind farm layout optimization problem (WFLOP) Python toolbox
## Reference
[1] Xinglong Ju, and Victoria C. P. Chen. "A MARS Python version using truncated linear function." 2019. [![DOI](https://zenodo.org/badge/226974692.svg)](https://zenodo.org/badge/latestdoi/226974692)<br/>
[2] Victoria C.P. Chen, David Ruppert, and Christine A. Shoemaker. "Applying experimental design and regression splines to high-dimensional continuous-state stochastic dynamic programming." *Operations Research*  47.1 (1999): 38-53. [![DOI](/IMAGES/10.1287opre.47.1.38.svg)](https://doi.org/10.1287/opre.47.1.38)<br/>
[3] Xinglong Ju, and Feng Liu. "Wind farm layout optimization using self-informed genetic algorithm with information guided exploitation." *Applied Energy* 248 (2019): 429-445. [![DOI](/IMAGES/j.apenergy.2019.04.084.svg)](https://doi.org/10.1016/j.apenergy.2019.04.084)<br/>
[4] Xinglong Ju, Victoria C. P. Chen, Jay M. Rosenberger, and Feng Liu. "Knot Optimization for Multivariate Adaptive Regression Splines." In *IISE Annual Conference*. Proceedings, Institute of Industrial and Systems Engineers (IISE), 2019.

If you have any questions, please email Feng Liu (*fliu0@mgh.harvard.edu*), or Xinglong Ju (*xinglong.ju@mavs.uta.edu*).

<p align="center"> 
    <img width="400" src="/IMAGES/wfi.png" alt="Wind farm land cells illustration"/><br/>
    Wind farm land cells illustration.
</p>

## WFLOP Python toolbox guide
### Import necessary libraries
```python
import numpy as np
import pandas as pd
import MARS  # MARS (Multivariate Adaptive Regression Splines) regression class
import WindFarmGeneticToolbox  # wind farm layout optimization using genetic algorithms classes
from datetime import datetime
import os
```

### Wind farm settings and algorithm settings
```python
# parameters for the genetic algorithm
elite_rate = 0.2
cross_rate = 0.6
random_rate = 0.5
mutate_rate = 0.1

# wind farm size, cells
rows = 21
cols = 21
cell_width = 77.0 * 2 # unit : m

#
N = 60  # number of wind turbines
pop_size = 100  # population size, number of inidividuals in a population
iteration = 1  # number of genetic algorithm iterations
```

### Create a WindFarmGenetic object
```python
# all data will be save in data folder
data_folder = "data"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# create an object of WindFarmGenetic
wfg = WindFarmGeneticToolbox.WindFarmGenetic(rows=rows, cols=cols, N=N, pop_size=pop_size,
                                             iteration=iteration, cell_width=cell_width, elite_rate=elite_rate,
                                             cross_rate=cross_rate, random_rate=random_rate, mutate_rate=mutate_rate)
# set wind distribution
# wind distribution is discrete (number of wind speeds) by (number of wind directions)
# wfg.init_4_direction_1_speed_12()
wfg.init_1_direction_1_N_speed_12()
```

### Generate initial populations
```python
init_pops_data_folder = "data/init_pops"
if not os.path.exists(init_pops_data_folder):
    os.makedirs(init_pops_data_folder)
# n_init_pops : number of initial populations
n_init_pops = 60
for i in range(n_init_pops):
    wfg.gen_init_pop()
    wfg.save_init_pop("{}/init_{}.dat".format(init_pops_data_folder,i))
```

### Create results folder
```python
# results folder
# adaptive_best_layouts_N60_9_20190422213718.dat : best layout for AGA of run index 9
# result_CGA_20190422213715.dat : run time and best eta for CGA method
results_data_folder = "data/results"
if not os.path.exists(results_data_folder):
    os.makedirs(results_data_folder)

n_run_times = 1  # number of run times
# result_arr stores the best conversion efficiency of each run
result_arr = np.zeros((n_run_times, 2), dtype=np.float32)
```

### Run conventional genetic algorithm (CGA)
```python
# CGA method
CGA_results_data_folder = "{}/CGA".format(results_data_folder)
if not os.path.exists(CGA_results_data_folder):
    os.makedirs(CGA_results_data_folder)
for i in range(0, n_run_times):  # run times
    print("run times {} ...".format(i))
    wfg.load_init_pop("{}/init_{}.dat".format(init_pops_data_folder, i))
    run_time, eta = wfg.conventional_genetic_alg(ind_time=i, result_folder=CGA_results_data_folder)
    result_arr[i, 0] = run_time
    result_arr[i, 1] = eta
time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = "{}/result_CGA_{}.dat".format(CGA_results_data_folder, time_stamp)
np.savetxt(filename, result_arr, fmt='%f', delimiter="  ")
```

### Run adaptive genetic algorithm (AGA)
```python
# AGA method
AGA_results_data_folder = "{}/AGA".format(results_data_folder)
if not os.path.exists(AGA_results_data_folder):
    os.makedirs(AGA_results_data_folder)
for i in range(0, n_run_times):  # run times
    print("run times {} ...".format(i))
    wfg.load_init_pop("{}/init_{}.dat".format(init_pops_data_folder, i))
    run_time, eta = wfg.adaptive_genetic_alg(ind_time=i, result_folder=AGA_results_data_folder)
    result_arr[i, 0] = run_time
    result_arr[i, 1] = eta
time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = "{}/result_AGA_{}.dat".format(AGA_results_data_folder, time_stamp)
np.savetxt(filename, result_arr, fmt='%f', delimiter="  ")
```

### Run self-informed genetic algorithm (SIGA)
#### Generate wind distribution surface
```python
wds_data_folder = "data/wds"
if not os.path.exists(wds_data_folder):
    os.makedirs(wds_data_folder)
# mc : monte-carlo
n_mc_samples = 10000

# each layout is binary list and the length of the list is (rows*cols)
# 1 indicates there is a wind turbine in that cell
# 0 indicates there is no wind turbine in the cell
# in "mc_layout.dat", there are 'n_mc_samples' line and each line is a layout.

# generate 'n_mc_samples' layouts and save it in 'mc_layout.data' file
WindFarmGeneticToolbox.LayoutGridMCGenerator.gen_mc_grid(rows=rows, cols=cols, n=n_mc_samples, N=N,
                                                         lofname="{}/{}".format(wds_data_folder, "mc_layout.dat"))
# read layouts from 'mc_layout.dat' file
layouts = np.genfromtxt("{}/{}".format(wds_data_folder,"mc_layout.dat"), delimiter="  ", dtype=np.int32)

# generate dataset to build wind farm distribution surface
wfg.mc_gen_xy(rows=rows, cols=cols, layouts=layouts, n=n_mc_samples, N=N, xfname="{}/{}".format(wds_data_folder, "x.dat"),
              yfname="{}/{}".format(wds_data_folder, "y.dat"))

# parameters for MARS regression method
n_variables = 2
n_points = rows * cols
n_candidate_knots = [rows, cols]
n_max_basis_functions = 100
n_max_interactions = 4
difference = 1.0e-3

x_original = pd.read_csv("{}/{}".format(wds_data_folder,"x.dat"), header=None, nrows=n_points, delim_whitespace=True)
x_original = x_original.values

y_original = pd.read_csv("{}/{}".format(wds_data_folder,"y.dat"), header=None, nrows=n_points, delim_whitespace=True)
y_original = y_original.values

mars = MARS.MARS(n_variables=n_variables, n_points=n_points, x=x_original, y=y_original,
                 n_candidate_knots=n_candidate_knots, n_max_basis_functions=n_max_basis_functions,
                 n_max_interactions=n_max_interactions, difference=difference)
mars.MARS_regress()
# save wind distribution model to 'wds.mars'
mars.save_mars_model_to_file()
with open("{}/{}".format(wds_data_folder,"wds.mars"), "wb") as mars_file:
    pickle.dump(mars, mars_file)
```
#### SIGA method
```python
SIGA_results_data_folder = "{}/SIGA".format(results_data_folder)
if not os.path.exists(SIGA_results_data_folder):
    os.makedirs(SIGA_results_data_folder)
# wds_mars_file : wind distribution surface MARS model file
wds_mars_file = "{}/{}".format(wds_data_folder, "wds.mars")
for i in range(0, n_run_times):  # run times
    print("run times {} ...".format(i))
    wfg.load_init_pop("{}/init_{}.dat".format(init_pops_data_folder, i))
    run_time, eta = wfg.self_informed_genetic_alg(ind_time=i, result_folder=SIGA_results_data_folder,
                                                   wds_file=wds_mars_file)
    result_arr[i, 0] = run_time
    result_arr[i, 1] = eta
time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = "{}/result_self_informed_{}.dat".format(SIGA_results_data_folder, time_stamp)
np.savetxt(filename, result_arr, fmt='%f', delimiter="  ")
```
