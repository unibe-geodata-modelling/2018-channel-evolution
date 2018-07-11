# PROFILE & ELEVATION DIFFERENCE ALONG THE CHANNEL: This script firstly creates the longitudinal profile of the channel
# at the beginning and at the end of the simulation. It also cuts the profile into three parts so the differences are
# better visible. The profiles are generated for two maintenance scenarios. more could be added: change in 'doPlot_prof'
# the index in the variable `finalDEM[i]` to the requested scenario (0=0% maint, 1=10% maint etc.). Secondly, the
# spatially distributed elevation differences are calculated. This for 4 different maintenance scn. Same here, the
# number of these can be changed in the `doPlot_diff` function ('DEMdiff[i]'). Additionally, the number and the relative
# share of cells which are below a certain erosion/deposition threshold can be calculated (e.g. x% of all values lie below n)

# -IMPORT LIBRARIES & VARIABLES HERE-
import numpy as np
import matplotlib.pyplot as plt
from tool1_LoadDEMs import DEM

# -DEFINE FUNCTIONS HERE-
def doDEMdiff(scenarios):
    '''creates the difference for each cell between predefined years for each scenario'''
    DEMdiff = []
    for scenario in range(scenarios):
        DEMdiff_list = DEM[scenario, 100, :] - DEM[scenario, 0, :]
        DEMdiff.append(DEMdiff_list)
    DEMdiff = np.array(DEMdiff)

    DEMdiffzero = []  # tranfer zero values into nan
    for scenario in range(DEM.shape[0]):
        DEMdiff0 = np.where(DEMdiff[scenario, :, :] == 0, np.nan, DEMdiff[scenario, :, :])
        DEMdiffzero.append(DEMdiff0)
    DEMdiffzero = np.array(DEMdiffzero)
    print('difference calculations finished''\n')
    return DEMdiffzero
def doProfile_prof (in1, in2, scenarios, DEM):
    '''mask all generated arrays with the thalweg array, so only the values that belong to the thalweg are analyzed'''
    # load in thalweg file, created in ArcGIS with flow accumulation, which has the same extent as the "cut" DEM
    profile = np.genfromtxt(in1, skip_header=6, delimiter=' ')
    # load in initial DEM in the same extent as the other DEMs
    start = np.genfromtxt(in2, skip_header=6, skip_footer=52, usecols=range(76, 203), delimiter=' ')
    # index array to switch order of rows from last to first
    index = np.arange(profile.shape[0]-1, -1, -1)

    # create profile for the initial DEM (the same for all scenarios)
    thal_start = np.where(profile == True, start, np.nan)        # use thalweg as mask to only get DEM values from thalweg
    thal_start = thal_start[index, :]                            # switch order of rows with index array
    thal_start = np.array((thal_start[~np.isnan(thal_start)]))   # only get values that are not nan (~ opposite of is.nan)

    # create profile for the final DEM (loop over all 84 scenarios)
    thal = []
    for x in range(scenarios):
        thal_0 = np.where(profile == True, DEM[x, 100, :, :], np.nan)
        thal.append(thal_0)
    thal = np.array(thal)

    thal_i = []
    for x in range(scenarios):
        i = thal[x, index, :]
        thal_i.append(i)
    thal_i = np.array(thal_i)

    thal_end = []
    for x in range(scenarios):
        thal_e = np.array((thal_i[x, :, :][~np.isnan(thal_i[x, :, :])]))
        thal_end.append(thal_e)
    thal_end = np.array(thal_end)
    print('profile built along thalweg''\n')
    return thal_start, thal_end
def doProfile_diff (paths, scenarios, DEMdiff):
    '''mask all generated arrays with the thalweg array, so only the values that belong to the thalweg are analyzed'''
    # load in thalweg file, created in ArcGIS with flow accumulation, which has the same extent as the DEM
    thalweg = np.genfromtxt(paths, skip_header=6, delimiter=' ')

    index = np.arange(thalweg.shape[0]-1, -1, -1)

    # create profile for the final DEM (loop over all 84 scenarios)
    thal = []
    for x in range(scenarios):
        thal_0 = np.where(thalweg == True, DEMdiff[x, :, :], np.nan)
        thal.append(thal_0)
    thal = np.array(thal)

    thal_i = []
    for x in range(scenarios):
        i = thal[x, index, :]
        thal_i.append(i)
    thal_i = np.array(thal_i)

    thal = []
    for x in range(scenarios):
        thal_e = np.array((thal_i[x, :, :][~np.isnan(thal_i[x, :, :])]))
        thal.append(thal_e)
    thal = np.array(thal)

    print('profile built along thalweg''\n')
    return thal
def doNewArray(input):
    '''create arrays to original elev array: specific maintenance scn ("perc"), location scn ("loc"), flood scn ("flood")'''
    perc_loc = np.repeat(np.arange(0, 7, 0.5), 6).reshape(input.shape[0], 1)
    new_array = np.append(np.vstack(input), perc_loc, axis=1)  # combine the new created perc_loc
    floods = np.array(14 * ['2apart', '2close', '2med', 'a_first', 'b_middle', 'c_last']).reshape(input.shape[0], 1)
    new_array = np.append(new_array, floods, axis=1)  # append third column to STD array
    return new_array
def doMean(prof):
    '''create new array which is sorted in the right way for analysis'''
    new_array = doNewArray(prof)

    # sort array by maintenance scn
    a = new_array[:, range(0, prof.shape[1])]
    b = new_array[:, -1]    # flood scn
    c = new_array[:, -2]    # maint scn

    sorts = []
    for x in range(prof.shape[1]):
        ind = np.lexsort((a[:, x], b, c))  # create array with the specified order
        sort = np.array([(a[:, x][i], b[i], c[i]) for i in ind])  # apply the "sorting array" to the original array
        sorts.append(sort)

    sorts = np.array(sorts)
    comb = np.concatenate(sorts[:, :, 0]).reshape(prof.shape[1], prof.shape[0])

    split = np.array(np.split(comb[:, :], 14, axis=1))                  # split array into different flood scenarios (6)

    # calculate mean of flood scn for different maintenance scn
    mean = []
    for x in range(split.shape[0]):  # loop through all scenarios
        interlist = []
        for y in range(split.shape[1]):
            mean_m = np.mean((split[x, y, :]).astype('float'))  # build mean of DEMdiff for each cell during 100yrs
            interlist.append(mean_m)
        mean.append(interlist)
    mean = np.array(mean)

    # calculate min + max of flood scn for different maintenance scn
    min = []
    for x in range(split.shape[0]):  # loop through all scenarios
        interlist = []
        for y in range(split.shape[1]):
            mean_m = np.min((split[x, y, :]).astype('float'))  # build mean of DEMdiff for each cell during 100yrs
            interlist.append(mean_m)
        min.append(interlist)
    min = np.array(min)

    max = []
    for x in range(split.shape[0]):  # loop through all scenarios
        interlist = []
        for y in range(split.shape[1]):
            mean_m = np.max((split[x, y, :]).astype('float'))  # build mean of DEMdiff for each cell during 100yrs
            interlist.append(mean_m)
        max.append(interlist)
    max = np.array(max)

    print('flood mean, min and max calculated.''\n')
    return mean, min, max
def doPlot_prof(initialDEM, finalDEM, xlabel, ylabel, ax_size, l_size, title, save1, save2):
    '''plot the total longitudinal profile of two maintenance scn (0+100% maintenance)'''
    legend = np.array(['final DEM', 'initial DEM'])
    # plot 0% maintenance effort
    plt.figure(figsize=(19, 12))
    # plt.suptitle(title, fontsize=24, fontweight=1, color='black').set_position([.5, 0.96])
    plt.subplots_adjust(hspace=0.25)
    plt.subplots_adjust(left=0.06, bottom=0.07, right=0.9, top=0.9, hspace=0.25)

    plt.subplot(211)
    plt.xticks(np.arange(0.5,  460, 7.1), ('0', '100', '200', '300', '400', '500', '600', '700', '800', '900',
                                           '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800',
                                           '1900', '2000', '2100'), fontsize=l_size)
    plt.plot(initialDEM, linewidth=1, color='tomato', linestyle='--', label=legend[1])
    plt.plot(finalDEM[0], linewidth=1, color='maroon', label=legend[0])
    plt.legend(loc='right', fontsize=l_size, frameon=False)
    plt.title('0% maintenance effort', fontsize=ax_size, loc='left')
    plt.xlim(0, 150)
    plt.axvline(x=50.2, color='black', linewidth=0.7)
    plt.axvline(x=99.9, color='black', linewidth=0.7)
    plt.axvspan(0, 50.2, color='maroon', alpha=0.14, lw=0)
    plt.axvspan(50.2, 100, color='maroon', alpha=0.07, lw=0)
    plt.axvspan(100, 150, color='maroon', alpha=0.015, lw=0)
    plt.text(47, 1200, '(a)', fontsize=l_size)
    plt.text(96.5, 1200, '(b)', fontsize=l_size)
    plt.text(146.7, 1200, '(c)', fontsize=l_size)
    plt.xlabel(xlabel, labelpad=9, fontsize=ax_size)
    plt.ylabel(ylabel, labelpad=8, fontsize=ax_size)
    plt.yticks(fontsize=l_size)

    # zoom in on 3 channel sections
    plt.subplot(234)
    plt.xticks(np.arange(1, 60, 6.9), ('0', '100', '200', '300', '400', '500', '600', '700'), fontsize=l_size)
    plt.plot(initialDEM, linewidth=1, color='tomato', linestyle='--', label=legend[1])
    plt.plot(finalDEM[0], linewidth=1, color='maroon', label=legend[0])
    plt.xlim(0, 50)
    plt.ylim(995, 1215)
    plt.legend(loc='lower left', fontsize=l_size, frameon=False)
    plt.text(46, 1200, '(a)', fontsize=l_size)
    plt.ylabel(ylabel, labelpad=8, fontsize=ax_size)
    plt.yticks(np.arange(1000, 1250, 50), np.arange(1000, 1250, 50), fontsize=l_size)
    plt.axvspan(0, 60, color='maroon', alpha=0.14, lw=0)

    plt.subplot(235)
    plt.xticks(np.arange(51, 110, 6.9), ('700', '800', '900', '1000', '1100', '1200', '1300', '1400'), fontsize=l_size)
    plt.plot(initialDEM, linewidth=1, color='tomato', linestyle='--', label=legend[1])
    plt.plot(finalDEM[0], linewidth=1, color='maroon', label=legend[0])
    plt.xlim(50, 100)
    plt.ylim(845, 1065)
    plt.legend(loc='lower left', fontsize=l_size, frameon=False)
    plt.text(96, 1050, '(b)', fontsize=l_size)
    plt.xlabel(xlabel, labelpad=9, fontsize=ax_size)
    plt.yticks(np.arange(850, 1100, 50), range(850, 1100, 50), fontsize=l_size)
    plt.axvspan(50, 120, color='maroon', alpha=0.07, lw=0)

    plt.subplot(236)
    plt.xticks(np.arange(101, 160, 6.9), ('1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100'), fontsize=l_size)
    plt.plot(initialDEM, linewidth=1, color='tomato', linestyle='--', label=legend[1])
    plt.plot(finalDEM[0], linewidth=1, color='maroon', label=legend[0])
    plt.xlim(100, 150)
    plt.ylim(745, 965)
    plt.legend(loc='lower left',  fontsize=l_size, frameon=False)
    plt.text(146.5, 950,'(c)', fontsize=l_size)
    plt.yticks(np.arange(750, 1000, 50), range(750, 1000, 50), fontsize=l_size)
    plt.axvspan(100, 170, color='maroon', alpha=0.015, lw=0)

    plt.savefig(save1, dpi=300, bbox_inches='tight')

    # plot 100% maintenance effort
    plt.figure(figsize=(19, 12))
    # plt.suptitle(title, fontsize=24, fontweight=1, color='black').set_position([.5, 0.96])
    plt.subplots_adjust(left=0.06, bottom=0.07, right=0.9, top=0.9, hspace=0.25)
    plt.subplot(211)
    plt.xticks(np.arange(0.5, 460, 7.1), ('0', '100', '200', '300', '400', '500', '600', '700', '800', '900',
                                          '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800',
                                          '1900', '2000', '2100'), fontsize=l_size)
    plt.plot(initialDEM,linewidth=1, color='royalblue', linestyle='--', label=legend[1])
    plt.plot(finalDEM[10], linewidth=1, color='navy', label=legend[0])
    plt.legend(loc='right', fontsize=l_size, frameon=False)
    plt.title('100% maintenance effort', fontsize=ax_size, loc='left')
    plt.axvline(x=50.2, color='black', linewidth=0.6)
    plt.axvline(x=99.9, color='black', linewidth=0.6)
    plt.axvspan(0, 50.2, color='navy', alpha=0.13, lw=0)
    plt.axvspan(50.2, 100, color='navy', alpha=0.07, lw=0)
    plt.axvspan(100, 150, color='navy', alpha=0.015, lw=0)
    plt.text(47, 1200,'(d)', fontsize=l_size)
    plt.text(96.5, 1200,'(e)', fontsize=l_size)
    plt.text(147, 1200,'(f)', fontsize=l_size)
    plt.xlabel(xlabel, labelpad=9, fontsize=ax_size)
    plt.ylabel(ylabel, labelpad=8, fontsize=ax_size)
    plt.xlim(0, 150)
    plt.yticks(fontsize=l_size)

    # zoom in on 3 channel sections
    plt.subplot(234)
    plt.xticks(np.arange(1, 60, 6.9), ('0', '100', '200', '300', '400', '500', '600', '700'), fontsize=l_size)
    plt.plot(initialDEM, linewidth=1, color='royalblue', linestyle='--', label=legend[1])
    plt.plot(finalDEM[10], linewidth=1, color='navy', label=legend[0])
    plt.xlim(0, 50)
    plt.ylim(995, 1215)
    plt.legend(loc='lower left', fontsize=l_size, frameon=False)
    plt.text(46, 1200, '(d)', fontsize=l_size)
    plt.ylabel(ylabel, labelpad=8, fontsize=ax_size)
    plt.yticks(np.arange(1000, 1250, 50), np.arange(1000, 1250, 50), fontsize=l_size)
    plt.axvspan(0, 60, color='navy', alpha=0.13, lw=0)

    plt.subplot(235)
    plt.xticks(np.arange(51, 110, 6.9), ('700', '800', '900', '1000', '1100', '1200', '1300', '1400'), fontsize=l_size)
    plt.plot(initialDEM, linewidth=1, color='royalblue', linestyle='--', label=legend[1])
    plt.plot(finalDEM[10], linewidth=1, color='navy', label=legend[0])
    plt.xlim(50, 100)
    plt.ylim(845, 1065)
    plt.legend(loc='lower left', fontsize=l_size, frameon=False)
    plt.text(96, 1050, '(e)', fontsize=l_size)
    plt.xlabel(xlabel, labelpad=9, fontsize=ax_size)
    plt.yticks(np.arange(850, 1100, 50), range(850, 1100, 50), fontsize=l_size)
    plt.axvspan(50, 120, color='navy', alpha=0.07, lw=0)

    plt.subplot(236)
    plt.xticks(np.arange(101, 160, 6.9), ('1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100'), fontsize=l_size)
    plt.plot(initialDEM, linewidth=1, color='royalblue', linestyle='--', label=legend[1])
    plt.plot(finalDEM[10], linewidth=1, color='navy', label=legend[0])
    plt.xlim(100, 150)
    plt.ylim(745, 965)
    plt.legend(loc='lower left', fontsize=l_size, frameon=False)
    plt.text(146.5, 950, '(f)', fontsize=l_size)
    plt.axvspan(100, 170, color='navy', alpha=0.015, lw=0)
    plt.yticks(np.arange(750, 1000, 50), range(750, 1000, 50), fontsize=l_size)
    plt.savefig(save2, dpi=300, bbox_inches='tight')
    print('longitudinal profile plotted''\n')
def doPlot_diff(mean, min, max, xlabel, ylabel, l_size, ax_size, title1, title2, title3, save1, save2):
    '''plot the elevation difference along the longitudinal profile of the channel for four different maintenance scn'''
    # maint scn
    fig = plt.figure(figsize=(19, 12))
    # fig.suptitle(title1, fontsize=24, fontweight=1, color='black').set_position([.5, 0.96])
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    fig.text(0.5, 0.051, xlabel, ha='center', fontsize=ax_size)
    fig.text(0.006, 0.5, ylabel, va='center', rotation='vertical', fontsize=ax_size)
    # fig.text(0.5, 0.91, title2, ha='center', fontsize=ax_size, style='italic')

    maint = [0, 3, 7, 10]
    maintenance = ['0% maintenance effort', '30% maintenance effort', '70% maintenance effort', '100% maintenance effort']
    sigma = [r'$\bar x=5.6$ m', r'$\bar x=3.1$ m', r'$\bar x=1.7$ m', r'$\bar x=0.9$ m']
    label = ['(a)', '(b)', '(c)', '(d)']

    for x in maint:
        y = maint.index(x)
        ax = fig.add_subplot(4, 1, y+1, sharey=ax)
        # fig.tight_layout()
        plt.subplots_adjust(left=0.046, bottom=0.096, right=0.99, top=0.96, hspace=0.5)
        plt.plot(mean[x], linewidth=1, color='black', linestyle='-', label='Mean')
        plt.plot(min[x], linewidth=0.25, color='grey', linestyle='-')
        plt.plot(max[x], linewidth=0.25, color='grey', linestyle='-')
        plt.fill_between(np.arange(min.shape[1]), max[x, range(min.shape[1])], min[x, range(min.shape[1])],
                         color='grey', alpha=0.18, label='Range')
        plt.xticks(np.arange(0.5, 460, 7.1), np.arange(0, 2200, 100), fontsize=l_size)
        plt.yticks(np.arange(-20, 30, 10), np.arange(-20, 30, 10), fontsize=l_size)
        plt.title(maintenance[y], fontsize=ax_size, loc='left')
        plt.xlim(-.5, 151)
        plt.ylim(-22, 22)
        plt.axvspan(0, 50.2, color='grey', alpha=0.0, lw=0)
        plt.axhline(y=0, color='black', linewidth=0.6)
        plt.text(1, -18, sigma[y], fontsize=l_size)
        plt.axhspan(-.5, 22, color='green', alpha=0.04, lw=0)
        plt.axhspan(-.5, -22, color='tomato', alpha=0.04, lw=0)
        plt.text(141, -18, 'Erosion', alpha=0.85, color='tomato', fontsize=l_size)
        plt.text(138.3, 15.5, 'Deposition', alpha=0.85, color='green', fontsize=l_size)
    plt.legend(ncol=2, fontsize=l_size, framealpha=0, bbox_to_anchor=(0.6, 0.012), loc=3)
    plt.savefig(save1, dpi=300, bbox_inches='tight')

    # loc scn
    fig = plt.figure(figsize=(19, 12))
    # fig.suptitle(title, fontsize=24, fontweight=1, color='black').set_position([.5, 0.96])
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    fig.text(0.5, 0.26, xlabel, ha='center', fontsize=ax_size)
    fig.text(0.006, 0.62, ylabel, va='center', rotation='vertical', fontsize=ax_size)
    # fig.text(0.5, 0.91, title3, ha='center', fontsize=ax_size, style='italic')

    loc = [11, 12, 13]
    location = ['\'High\' maintenance location ', '\'Mid\' maintenance location ', '\'Low\' maintenance location ']
    sigma = [r'$\bar x=3.7$ m', r'$\bar x=4$ m', r'$\bar x=4.7$ m']
    for x in loc:
        y = loc.index(x)
        ax = fig.add_subplot(4, 1, y+1, sharey=ax)
        # fig.tight_layout()
        plt.plot(mean[x], linewidth=1, color='black', linestyle='-', label='Mean')
        plt.plot(min[x], linewidth=0.25, color='grey', linestyle='-')
        plt.plot(max[x], linewidth=0.25, color='grey', linestyle='-')
        plt.subplots_adjust(left=0.046, bottom=0.07, right=0.99, top=0.96, hspace=0.5)
        plt.fill_between(np.arange(min.shape[1]), max[x, range(min.shape[1])], min[x, range(min.shape[1])],
                         color='grey', alpha=0.18, label='Range')
        plt.xticks(np.arange(0.5, 460, 7.1), np.arange(0, 2200, 100), fontsize=l_size)
        plt.yticks(np.arange(-20, 30, 10), np.arange(-20, 30, 10), fontsize=l_size)
        plt.title(location[y], fontsize=ax_size, loc='left')
        plt.xlim(-.5, 151)
        plt.ylim(-22, 22)
        plt.axvspan(0, 50.2, color='grey', alpha=0.0, lw=0)
        plt.axhline(y=0, color='black', linewidth=0.6)
        plt.text(1, -18, sigma[y], fontsize=l_size)
        plt.axhspan(-.5, 22, color='green', alpha=0.04, lw=0)
        plt.axhspan(-.5, -22, color='tomato', alpha=0.04, lw=0)
        plt.text(141.1, -18, 'Erosion', alpha=0.85, color='tomato', fontsize=l_size)
        plt.text(138.4, 15.6, 'Deposition', alpha=0.85, color='green', fontsize=l_size)
        # plt.text(0.5, 16, label[y], fontsize=l_size)
    plt.legend(ncol=2, fontsize=l_size, framealpha=0, bbox_to_anchor=(0.6, 0.012), loc=3)
    plt.savefig(save2, dpi=300, bbox_inches='tight')
    print('mean and range of elevation change plotted''\n')
def doHigherThan(n):
    highervalues=[]
    relvalues=[]
    for x in range(diff_mean.shape[0]):
        hv = len(diff_mean[x][np.where(diff_mean[x] < n)])
        rv = hv/151*100
        highervalues.append(hv)
        relvalues.append(rv)
    relvalues = np.round(relvalues, 1)
    print('\n''How many values are smaller than ' + str(n) + ' for each maintenance scenario (100%, 90%, ...)?''\n'
          + str(highervalues))
    print('\n''What is the relative share?''\n' + str(relvalues))

# -READ IN FILES HERE-
# define where profile and initial DEM are located to load in and where outputs should be saved at
in_path = 'U:simulations/analysis/python/profile/profile_old.txt'
start_DEM = 'U:simulations/analysis/python/profile/elevSlide2.txt'
out_path1 = 'U:simulations/analysis/python/profile/DEM/profile_DEM{x}.txt'
out_path2 = 'U:simulations/analysis/python/profile/profile.txt'

# -CALL FUNCTIONS HERE-
## ---------------------- 1 longitudinal profile ----------------------
# mask the two different DEMs (initial and final DEM) with the profile line (created in ArcGIS)
startDEM, endDEM = doProfile_prof(in_path, start_DEM, DEM.shape[0], DEM)

# sort the different maintenance scenarios. calculate mean of all flood scenarios for the different maintenance scenarios
profile_mean, mi, ma = doMean(endDEM)
del(mi, ma)
# plot the longitudinal profile
# define plot properties
xlabel = "Length [m]"
ylabel = "Elevation [m a.s.l.]"
ax_size = 18
l_size = 15
title = "Longitudinal profile of channel after 100 years of simulation"
save0 = 'U:simulations/analysis/python/profile/longitudinalProfile0perc.png'
save100 = 'U:simulations/analysis/python/profile/longitudinalProfile100perc.png'

doPlot_prof(startDEM, profile_mean, xlabel, ylabel, ax_size, l_size, title, save0, save100)

## ---------------------- 2 DEMdiff profile ----------------------
# calculate elevation difference of DEM between year 100 and 0
DEMdiff = doDEMdiff(DEM.shape[0])  # shape of first dimension = 66, shape of second dimension = 100-1

# mask the DEMdiff file with the profile line (created in ArcGIS)
DEMdiff_thal = doProfile_diff(in_path, DEMdiff.shape[0], DEMdiff)

# sort the different maintenance scenarios. calculate mean of all flood scenarios for the different maintenance scenarios
diff_mean, diff_min, diff_max = doMean(DEMdiff_thal)

# calculate number of values smaller than x for different maintenance scenarios to get the p-quantile of the dataset
n = 5.5   # threshold value, how many values are smaller than this number?
doHigherThan(n)

# plot the elevation difference
# define plot properties
ylabel = "Change in elevation [m]"
title1 = "Change in channel elevation after 100 years of simulation"
title2 = "Maintenance effort"
title3 = "Maintenance location"
save1 = 'U:simulations/analysis/python/profile/ElevDiff_maint.png'
save2 = 'U:simulations/analysis/python/profile/ElevDiff_loc.png'
doPlot_diff(diff_mean, diff_min, diff_max, xlabel, ylabel, l_size, ax_size, title1, title2, title3, save1, save2)
