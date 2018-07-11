# OUTPUT COMPARISON + SYNTHETIC INPUT: this script includes two additional tools. the first one is comparing two
# different simulation runs, to check whether the model can reproduce twice similar answers. the second part is
# producing the synthetic water input data for the simulation.

# -IMPORT LIBRARIES & VARIABLES HERE-
import numpy as np
import glob
import matplotlib.pyplot as plt

# -DEFINE FUNCTIONS HERE-
def doRead():
    '''read in output results model runs'''
    dat_data = []
    for x in range(files.shape[0]):
        for y in range(files.shape[1]):
            read_files = np.genfromtxt(files[x][y], delimiter=' ', usecols=(1, 4), skip_header=3) # only read in Qw & Qs
            dat_data.append(read_files)
        print('read in '+str(x)+' of ' +str(files.shape[0]))
    dat_data = np.array(dat_data)
    print('read\'em all in''\n')

    return dat_data
def doSum():
    '''calculate sum of sediment yield for all scenarios'''
    sum = []
    for x in range(dat_data.shape[0]):
        sum_Qs = np.sum(dat_data[x, :, 1])
        sum.append(sum_Qs)
    sum = np.array(sum).reshape(6, 2)       # reshape it two array with 6 rows and 2 cols
    return sum
def doDiff():
    '''calculate the difference of sed yield between the tow simulation rusn'''
    diff_perc = []
    for x in range(6):
        diff = (sum[x, 0]-sum[x, 1])/sum[x, 0]*100
        diff_perc.append(diff)
    diff_perc = np.array(diff_perc).reshape(6, 1)
    return diff_perc
def doHydro(time,n, z):
    '''fill the created hydrograph with any number (n) and repeat the number for a certain number of times (z)'''
    hydro = []
    for x in range(len(flood)):
        discharge = np.repeat(n, z)
        np.put(discharge, time[x][:], flood[x][:])
        hydro.append(discharge)
    hydro = np.array(hydro)
    return hydro
def doPlot(xlabel, ylabel, ytick1, ytick2, ax_size, l_size, title, save):
    '''plot the hydro- or sedigraph'''
    floods = ['(a) 2 apart', '(b) 2 close', '(c) 2 med', '(d) 1 first', '(e) 1 middle', '(f) 1 last']
    palette2 = np.array(['#e3ce8d', '#db786c', '#8e729d', '#7ba6d0', '#7ba47b', '#8d8d8d'])
    fig = plt.figure(figsize=(19, 12))
    fig.suptitle(title, fontsize=24, fontweight=1, color='black').set_position([.5, 0.96])
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    fig.text(0.5, 0.055, xlabel, ha='center', fontsize=ax_size)
    fig.text(0.07, 0.5, ylabel, va='center', rotation='vertical', fontsize=ax_size)

    for x in range(len(floods)):
        ax = fig.add_subplot(3, 2, (x+1), sharey=ax)
        plt.title(floods[x], fontsize=l_size, loc='left')
        plt.subplots_adjust(wspace=0.15, hspace=0.25)
        ax.plot(years, hydro[x, :], color=palette2[x], linestyle='-', linewidth=1, label='_nolegend_')
        plt.xticks(range(0, 876100, (87600*2)), range(0, 110, 20), fontsize=l_size)
        plt.yticks(ytick1, ytick2, fontsize=l_size)
    plt.savefig(save, dpi=450, bbox_inches='tight')

# -READ IN FILES HERE-
files1 = np.array(glob.glob('U:simulations/2nd try/**/*.dat'))
files2 = np.array(glob.glob('U:simulations/3rd try/**/*.dat'))
files = np.append(np.vstack(files1), np.vstack(files2), axis=1)

# -CALL FUNCTIONS HERE-
# ---------------------- 1 output comparison ----------------------
# this script is comparing 6 scenarios from two different simulation runs
# read in water and sediment output
dat_data = doRead()

# calculate sum of sediment yield (2nd column) over 100 years (all rows) for all scenarios
sum = doSum()

# calculate difference of sed yield between the two runs
diff_perc = doDiff()

# combine two cols of sum plus the diff_perc col in one array, round everything to one decimal
comparison = np.round(np.append(sum, diff_perc, axis=1), 1)

# export file
np.savetxt('U:simulations/analysis/python/run_comparison.txt', comparison[:], delimiter=' ', comments='')

# ---------------------- 2 synthetic input ----------------------
# create synthetic hydrograph and sedigraph
# flood magnitude
apart = np.array([100, 30, 30, 30, 50, 30, 30, 30, 50, 30, 30, 30, 50, 100])
close = np.array([100, 30, 30, 100, 50, 30, 30, 30, 50, 30, 30, 30, 50, 30])
med = np.array([100, 30, 30, 30, 50, 30, 30, 100, 50, 30, 30, 30, 50, 30])
first = np.array([100, 30, 30, 30, 50, 30, 30, 30, 50, 30, 30, 30, 50, 30])
middle = np.array([30, 30, 30, 30, 50, 30, 30, 100, 50, 30, 30, 30, 50, 30])
last = np.array([30, 30, 30, 30, 50, 30, 30, 30, 50, 30, 30, 30, 50, 100])
flood = np.ndarray.tolist(np.concatenate([[apart], [close], [med], [first], [middle], [last]]))

# flood times for each flood scenario
at = np.array([8746, 64985, 129935, 194885, 259835, 324785, 389735, 454661, 519611, 584561,
               649511, 714461, 779411, 844361])
ct = np.array([8746, 64985, 129935, 194909, 259859, 324809, 389759, 454685, 519635, 584585,
               649535, 714485, 779435, 844385])
met = np.array([8746, 64985, 129935, 194885, 259835, 324785, 389735, 454661, 519635, 584585,
                649535, 714485, 779435, 844385])
ft = np.array([8746, 64985, 129935, 194885, 259835, 324785, 389735, 454661, 519611, 584561,
               649511, 714461, 779411, 844361])
mit = np.array([8745, 64961, 129911, 194861, 259811, 324761, 389711, 454637, 519587, 584537,
                649487, 714437, 779387, 844337])
l = np.array([8745, 64961, 129911, 194861, 259811, 324761, 389711, 454661, 519611, 584561,
              649511, 714461, 779411, 844361])
time = np.ndarray.tolist(np.concatenate([[at], [ct], [met], [ft], [mit], [l]]))

# fill hydrographs between the flood events with zeros
hydro = doHydro(time, 0, 876000)

# plot hydrographs
years = np.arange(0, 876000, 1)             # x value for plotting
years2 = np.arange(0, 100, 1)
# define plot properties
xlabel = "Time [years]"
ylabel = "Water input [$\mathregular{m^3 s^{-1}}$]"
ytick1 = range(3, 120, 20)
ytick2 = range(0, 120, 20)
ax_size = 16
l_size = 14
title = "Synthetic hydrographs"
save = 'U:simulations/analysis/python/bonus/hydrograph.png'
doPlot(xlabel, ylabel, ytick1, ytick2, ax_size, l_size, title, save)

# plot sedigraphs
# define plot properties
ylabel = "Sediment input [$\mathregular{m^3}$]"
ytick1 = range(3, 120, 20)
ytick2 = range(0, 2400, 400)
title = "Synthetic sedigraphs"
save = 'U:simulations/analysis/python/bonus/sedigraph.png'
doPlot(xlabel, ylabel, ytick1, ytick2, ax_size, l_size, title, save)
