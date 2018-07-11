-----------------------------------
TOOLS TO ANALYZE CHANNEL EVOLUTION
-----------------------------------

Program: Python 3.6 
Editor: PyCharm
Packages: Numpy 1.14.3, Matplotlib, 2.1.2
Modules: Glob

INTRO
--------------------
There are 5 scripts to analyze the results from the Guerbe torrent simulation. 84 different scenarios have been tested, 
of which each one returned 101 digital elevation models (DEMs) and hourly sediment yield values. This totals in 8400 DEMs 
and 73,584,000 sediment rates, which have been analyzed using the following scripts:

+ Tool 1: load in DEMs
+ Tool 2: spatially distinct change in channel elevation
+ Tool 3: mean change in channel elevation (over time)
+ Tool 4: sediment yield with potential channel fill (over time)
+ Tool 5: bonus

REQUIRED DATA
--------------------
The model saved the outputs in the following formats: 
+ The model exported DEMs as elev.dat(x).txt files, which could be read directily in ArcGIS; each year a DEM was saved.
+ The sediment yield was saved in one file per scenario in the format (x).dat; the file was hourly updated with a new row 
containing information about the sediment yield.

TOOL DESCRIPTION
--------------------
+ Tool 1: Load in DEMs
ACCESS FILES FROM FOLDERS & SUBFOLDERS: This script imports all files that are stored in different folders and subfolders. 
The name of these folders and files follow a pattern and can be read in automatically after creating a path to each file. 
The folders and subfolders represent different scenarios from a simulation and the files are digital elevation models (DEM). 
The folders differ by the numbers 0-100 with the increment of 10 and three location scenarios. the name of the subfolders 
is written in the list "floods”.

+ Tool 2: Spatially distinct change in channel elevation
PROFILE & ELEVATION DIFFERENCE ALONG THE CHANNEL: This script firstly creates the longitudinal profile of the channel at 
the beginning and at the end of the simulation. It also cuts the profile into three parts so the differences are better 
visible. The profiles are generated for two maintenance scenarios. more could be added: change in 'doPlot_prof' the index 
in the variable `finalDEM[i]` to the requested scenario (0=0% maint, 1=10% maint etc.). Secondly, the spatially distributed 
elevation differences are calculated. This for 4 different maintenance scn. Same here, the number of these can be changed 
in the `doPlot_diff` function ('DEMdiff[i]'). Additionally, the number and the relative share of cells which are below a 
certain erosion/deposition threshold can be calculated (e.g. x% of all values lie below n)

+ Tool 3: Mean change in channel elevation (over time)
MEAN ELEVATION DIFFERENCE: This script calculates the mean elevation difference of the total channel after a certain number 
of years. You can choose to calculate the total difference after a 100 years of simulation ('diff_yrs = [100]') or define the 
years of difference you want to look at (e.g. always calculate the difference after 20 years, calculate the difference after 
the flood events, or only after the big flood events). Depending if you chose the first option (difference after 100 years) or
the second option (continous difference during the 100 years), different figures will be plotted. The figures present either the 
mean channel change after the whole simulation time for the different scenarios or the evolution of the channel change within 
the 100 years of simulation. If 'diff_yrs = [100]', all arrays exported to ArcGIS files, which include geometric information. 
Additionally, a function for a unique cholor scheme is developed.

+ Tool 4: Sediment yield with potential channel fill (over time)
SED YIELD + DOWNSTREAM FILL: this script deals with the sediment yield generated at the output of the model. it calculates the 
total sum of sediment yield, cumulative sediment yield (evolution of sediment yield over simulation time), potential downstream 
channel fill in percentage and meters, and the cross section specific analysis of channel aggradation. 

+ Tool 5: Bonus
OUTPUT COMPARISON + SYNTHETIC INPUT: This script includes two additional tools. The first one is comparing two different 
simulation runs, to check whether the model can reproduce twice similar answers. The second part is producing bar plots for the 
different synthtetic water and sediment input.
