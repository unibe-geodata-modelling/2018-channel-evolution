# ACCESS FILES FROM FOLDERS & SUBFOLDERS: This script imports all files that are stored in different folders and
# subfolders. The name of these folders and files follow a pattern and can be read in automatically after creating a
# path to each file. The folders and subfolders represent different scenarios from a simulation and the files are
# digital elevation models (DEM). The folders differ by the numbers 0-100 with the increment of 10 and three location
# scenarios. the name of the subfolders is written in the list "floods”.

# -IMPORT LIBRARIES & VARIABLES HERE-
import glob
import numpy as np

# -DEFINE FUNCTIONS HERE-
def doPaths(floods, maintenances, locations, path, path2):
    '''create list of paths to specific folders and subfolders by changing certain parts of a path'''
    scenario_list = []
    for maintenance in maintenances:                        # loop over maintenance scenarios (scn)
        for flood in floods:                                # loop over flood scenarios
            paths = path.format(maintenance, flood)         # include two arguments to the path
            scenario_list.append(paths)
    for location in locations:
        for flood in floods:                                # loop over location scenarios
            paths2 = path2.format(location, flood)
            scenario_list.append(paths2)
    print('\n''path list created''\n')
    return scenario_list
def doElev(length, scenariolist):
    '''get all the files within the specific paths. sort them by string length'''
    elev_list = []
    sorted_list = []
    for x in length:                                                                  # loop over the length of the list
        elev_list.append(glob.glob(scenariolist[x]))    # import all DEMs within one folder. all DEMs end with .dat(number)
        sort = sorted(elev_list[x], key=len)
        sorted_list.append(sort)
    elev = np.array(sorted_list)                                     # change list into array with 101 cols, 66 rows
    print('\n''elev list created''\n''\n')
    return elev
def doDEM(scenario, year, array):
    '''read in all files from created path array. store them in a 4D array'''
    DEM = []
    for row in range(scenario):
        inter_list = []       # use intermediate list to store all files from nested loop and later append them to "DEM"
        for col in range(year):
            read_files = np.genfromtxt(array[row][col], skip_header=6,                         # skip ArcGIS information
                                       skip_footer=52, usecols=range(76, 203), delimiter=' ')  # load only cells with c.d.
            inter_list.append(read_files)
        DEM.append(inter_list)
        if (row % 6 == 0):
            print('DEM ' + str(row) + ' is created. Only ' + str(scenario-row) + ' to go!')
    DEM = np.array(DEM)                                                                     # convert list into 4D array
    print('\n''YES, all done!''\n')
    return DEM

# -DEFINE GLOBAL VARIABLES HERE-
# define folders and subfolders names. names represent the different flood, maintenance and location scenarios (snc).
maintenances = range(0, 110, 10)
floods = ['2apart', '2close', '2med', 'first', 'middle', 'last']
locations = ['high', 'mid', 'low']
# define paths of where folders are located
path = 'U:simulations/dem_reach/longSim{}percMaintain/{}/elev.dat*.txt'    # maintenance path which needs to be adjusted
path2 = 'U:simulations/dem_reach/longSimLocation/{}/{}/elev.dat*.txt'         # location path which needs to be adjusted

# -CALL FUNCTIONS HERE-
# create path list to each subfolder
path_list = doPaths(floods, maintenances, locations, path, path2)

# get all the DEM files within the subfolder, sort them and store them in a 4D array (scenarios, years, x-elev, y-elev)
#  scenarios: maintenance&location(14)*flood(6)
# years: 100 years of simulation, 1 DEM per year -> 101 DEMs in total
# x-&y-elev: elevation at x-coord, at y-coord
elev = doElev(range(len(path_list)), path_list)

# read in files from the paths created in the array "elev". iterate through ech row and col. nested for-loop procedure:
# take 1st row of the "elev_list" and iterate through all cols, then go on to the 2nd row and iterate rough each col
# again etc. 1st line of output represents the scenario at elev_list[0,0], 2nd line the scenario at elev_list[0,1] etc.
DEM = doDEM(elev.shape[0], elev.shape[1], elev)

# delete variables that are not needed anymore
del(maintenances, floods, locations, path, path2, path_list)
