# MEAN ELEVATION DIFFERENCE: This script calculates the mean elevation difference of the total channel after a certain
# number of years. You can choose to calculate the total difference after a 100 years of simulation ('diff_yrs = [100]')
# or define the years of difference you want to look at (e.g. always calculate the difference after 20 years, calculate
# the difference after the flood events, or only after the big flood events). Depending if you chose the first option
# (difference after 100 years) or the second option (continuous difference during the 100 years), different figures will
# be plotted. The figures present either the mean channel change after the whole simulation time for the different
# scenarios or the evolution of the channel change within the 100 years of simulation. If 'diff_yrs = [100]', all
# arrays exported to ArcGIS files, which include geometric information. Additionally, a function for a unique cholor scheme 
# is developed.

# -IMPORT LIBRARIES & VARIABLES HERE-
import numpy as np
import matplotlib.pyplot as plt
from tool1_LoadDEMs import DEM

# -DEFINE FUNCTIONS HERE-
def doDEMdiff(scenarios, diffyear):
    '''reates the difference for each cell between predefined years for each scenario'''
    DEMdiff = []
    for scenario in range(scenarios):
        DEMdiff_list = DEM[scenario, diffyear, :, :] - DEM[scenario, 0, :, :]
        DEMdiff.append(DEMdiff_list)
    DEMdiff = np.array(DEMdiff)
    DEMdiffzero = DEMdiff

    DEMdiffnan = []  # tranfer zero values into nan
    for scenario in range(DEM.shape[0]):
        interlist = []
        for diff in range(len(diffyear)):
            DEMdiff0 = np.where(DEMdiff[scenario, diff, :, :] == 0, np.nan, DEMdiff[scenario, diff, :, :])
            interlist.append(DEMdiff0)
        DEMdiffnan.append(interlist)
    DEMdiffnan = abs(np.array(DEMdiffnan))

    print('\n''difference calculations finished''\n')
    return DEMdiffzero, DEMdiffnan

def doProfile (paths, scenarios, diffyear, DEMdiff):
    '''mask all generated arrays with the thalweg array, so only the values that belong to the thalweg are analyzed'''
    # load in thalweg file, created in ArcGIS with flow accumulation, which has the same extent as the DEM
    thalweg = np.genfromtxt(paths, skip_header=6, delimiter=' ')

    thal = []
    for x in range(scenarios):
        interlist = []
        for y in range(len(diffyear)):
            thal_0 = np.where(thalweg == True, DEMdiff0[x, y,:, :], 0)    # mask DEMdiff array to get global mean and std
            interlist.append(thal_0)
        thal.append(interlist)
    thalzero = np.array(thal)

    thal = []
    for x in range(scenarios):
        interlist = []
        for y in range(len(diffyear)):
            thal_0 = np.where(thalweg == True, DEMdiff[x, y, :, :], np.nan)    # mask DEMdiff array to get global mean and std
            interlist.append(thal_0)
        thal.append(interlist)
    thalnan = np.array(thal)
    print('thalweg read in and zeros changed to NaN''\n')
    return thalzero, thalnan

def doStatistics(scenarios, diffyear, elev_diff):
    """calculate mean, std of difference of selected years"""
    mean_DEMdiff = []
    for scenario in range(scenarios):  # loop through all scenarios
        interlist = []
        for diff in range(len(diffyear)):
            mean_DEMdiff_m = np.nanmean(
                elev_diff[scenario, diff, :, :])  # build mean of DEMdiff for each cell during 100yrs
            interlist.append(mean_DEMdiff_m)
        mean_DEMdiff.append(interlist)
    mean_DEMdiff = np.array(mean_DEMdiff)

    std_DEMdiff = []
    for scenario in range(scenarios):
        interlist = []
        for diff in range(len(diffyear)):
            std_DEMdiff_s = np.nanstd(
                elev_diff[scenario, diff, :, :])  # build std of DEMdiff for each cell during 100yrs
            interlist.append(std_DEMdiff_s)
        std_DEMdiff.append(interlist)
    std_DEMdiff = np.array(std_DEMdiff)

    def rmse(diff):
        return np.sqrt(np.nanmean((abs(diff))**2))
    rmse_DEMdiff = []
    for scenario in range(scenarios):
        interlist = []
        for diff in range(len(diffyear)):
            rmse_DEMdiff_s = rmse(
                elev_diff[scenario, diff, :, :])  # build std of DEMdiff for each cell during 100yrs
            interlist.append(rmse_DEMdiff_s)
        rmse_DEMdiff.append(interlist)
    rmse_DEMdiff = np.array(rmse_DEMdiff)
    print('statistic calculations finished''\n')
    return mean_DEMdiff, std_DEMdiff, rmse_DEMdiff

def doNewArray(input):
    '''create arrays to original elev array: specific maintenance scn ("perc"), location scn ("loc"), flood scn ("flood")'''
    perc_loc = np.repeat(np.arange(0, 7, 0.5), 6).reshape(input.shape[0], 1)
    new_array = np.append(np.vstack(input), perc_loc, axis=1)  # combine the new created perc_loc
    floods = np.array(14 * ['2apart', '2close', '2med', 'a_first', 'b_middle', 'c_last']).reshape(input.shape[0], 1)
    new_array = np.append(new_array, floods, axis=1)  # append third column to STD array
    return new_array

def doArray(elev_diff):
    '''create new array which is sorted in the right way for analyzing summary statistics of difference'''
    new_array = doNewArray(elev_diff)
    # sort array by following order: 1st by flood scenarios (3rd col), 2nd by maintenance/location scenarios (2nd col)
    a = []
    for x in range(elev_diff.shape[1]):
        b = new_array[:, x]
        a.append(b)
    a = np.array(a)
    b = new_array[:, -2]
    c = new_array[:, -1]

    sorts = []
    for x in range(elev_diff.shape[1]):
        ind = np.lexsort((a[x, :], b, c))                           # create array with the specified order
        sort = np.array([(a[x, :][i], b[i], c[i]) for i in ind])    # apply the "sorting array" to the original array
        sorts.append(sort)
    sorts = np.array(sorts)

    if elev_diff.shape[1]>1:                                        # loop through diff_yrs
        splits = []
        for x in range(elev_diff.shape[1]):
            split = np.array(np.split(sorts[x, :, :], 6))           # split array into different flood scn (6)
            splits.append(split)
        splits = np.array(splits)
    else:                                                           # no need to loop through diff_yrs
        splits = np.array(np.split(sorts[0, :, :], 6))
        splits = np.array(splits)
    # location & maintenance scenarios need to be split in order to plot them separately
    split1 = []
    split2 = []

    if elev_diff.shape[1]>1:                                        # loop through maint scn AND diff_yrs
        for x in range(splits.shape[0]):
            interlist1 = []
            interlist2 = []
            for y in range(splits.shape[1]):
                first = splits[x, y, :11, :]
                last = splits[x, y, -3:, :]
                interlist1.append(first)
                interlist2.append(last)
            split1.append(interlist1)
            split2.append(interlist2)
        split1 = np.array(split1)
        split2 = np.array(split2)
    else:                                                           # only loop through maint scn
        for x in range(splits.shape[0]):
            first = splits[x, :11, :]
            last = splits[x, -3:, :]
            split1.append(first)
            split2.append(last)
        split1 = np.array(split1)
        split2 = np.array(split2)
    print('array created''\n')
    return split1, split2

def doMinMax(elev_diff):
    '''calculates the min and the max value for each flood scenario. depending on how many diffyears are analyzed
    (if 1 or more), the calculation method is adapted'''
    new_array = doNewArray(elev_diff)

    if elev_diff.shape[1]>1:
        # sort array by following order: 1st by flood scenarios (3rd col), 2nd by maintenance/location scenarios (2nd col)
        sort_list=[]
        for x in range(elev_diff.shape[1]):
            a = new_array[:, x]
            b = new_array[:, -1]
            c = new_array[:, -2]
            ind = np.lexsort((b, c))  # create array with the specified order
            sort = np.array([(a[i], b[i], c[i]) for i in ind])  # apply the "sorting array" to the original array
            sort_list.append(sort[:, 0])
        sort_list.append((sort[:, -1]))
        sort_list.append(sort[:, -2])
        sort_list = np.array(sort_list)

        splits = np.array(np.hsplit(sort_list[:, :], 14))
        splits = np.array(splits)

        min_f = []
        for x in range(splits.shape[0]):  # loop through all scenarios
            interlist=[]
            for y in range(splits.shape[1]-2):
                min = np.min((splits[x, y, :]).astype('float'))   # min of all flood scn
                interlist.append(min)
            min_f.append(interlist)
        min_flood = np.array(min_f)

        max_f = []
        for x in range(splits.shape[0]):  # loop through all scenarios
            interlist=[]
            for y in range(splits.shape[1]-2):
                min = np.max((splits[x, y, :]).astype('float'))   # min of all flood scn
                interlist.append(min)
            max_f.append(interlist)
        max_flood = np.array(max_f)

        perc_loc2 = np.vstack(np.append((0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5), np.array([5.5, 6, 6.5])))

        min = np.append(min_flood[:], perc_loc2, axis=1)
        min1 = np.array([(min[x, :]) for x in range(11)])
        min2 = np.array([(min[x, :]) for x in range(11, 14)])

        max = np.append(max_flood[:], perc_loc2, axis=1)
        max1 = np.array([(max[x, :]) for x in range(11)])
        max2 = np.array([(max[x, :]) for x in range(11, 14)])
    else:
        # sort array by following order: 1st by flood scenarios (3rd col), 2nd by maintenance/location scenarios (2nd col)
        a = new_array[:, 0]
        b = new_array[:, -1]
        c = new_array[:, -2]
        ind = np.lexsort((a, b, c))  # create array with the specified order
        sort = np.array([(a[i], b[i], c[i]) for i in ind])  # apply the "sorting array" to the original array

        splits = np.array(np.split(sort[:, :], 14))
        splits = np.array(splits)

        min_f = []
        for x in range(splits.shape[0]):  # loop through all scenarios
            min = np.min((splits[x, :, 0]).astype('float'))   # min of all flood scn
            min_f.append(min)
        min_flood = np.array(min_f)

        max_f = []
        for x in range(splits.shape[0]):  # loop through all scenarios
            max = np.max((splits[x, :, 0]).astype('float'))   # max of all flood scn
            max_f.append(max)
        max_flood = np.array(max_f)

        perc_loc2 = np.append((0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5), np.array([5.5, 6, 6.5]))

        min = np.append(min_flood[:], perc_loc2).reshape(2,min_flood.shape[0])
        min1 = np.array([(min[:,x]) for x in range(11)])
        min2 = np.array([(min[:,x]) for x in range(11, 14)])

        max = np.append(max_flood[:], perc_loc2).reshape(2,max_flood.shape[0])
        max1 = np.array([(max[:,x]) for x in range(11)])
        max2 = np.array([(max[:,x]) for x in range(11, 14)])
    print('min and max calculated''\n')
    return min1, min2, max1, max2

def doFloodmean(elev_diff):
    '''calculates the flood mean of all flood scenarios. depending on how many diffyears are analyzed (if 1 or more),
    the calculation method is adapted'''
    new_array = doNewArray(elev_diff)

    if elev_diff.shape[1]>1:
        # sort array by following order: 1st by flood scn (3rd col), 2nd by maintenance/location scn (2nd col)
        sort_list=[]
        for x in range(elev_diff.shape[1]):
            a = new_array[:, x]
            b = new_array[:, -1]
            c = new_array[:, -2]
            ind = np.lexsort((b, c))  # create array with the specified order
            sort = np.array([(a[i], b[i], c[i]) for i in ind])  # apply the "sorting array" to the original array
            sort_list.append(sort[:, 0])
        sort_list.append((sort[:, -1]))
        sort_list.append(sort[:, -2])
        sort_list = np.array(sort_list)

        splits = np.array(np.hsplit(sort_list[:, :], 14))
        splits = np.array(splits)

        mean_flood = []
        for x in range(splits.shape[0]):  # loop through all scenarios
            interlist=[]
            for y in range(splits.shape[1]-2):
                mean_m = np.mean((splits[x, y, :]).astype('float'))  # build mean of DEMdiff for each cell during 100yrs
                interlist.append(mean_m)
            mean_flood.append(interlist)
        mean_flood = np.array(mean_flood)

        perc_loc2 = np.vstack(np.append((0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5), np.array([5.5, 6, 6.5])))
        mean_flood2 = np.append(mean_flood, perc_loc2, axis=1)

        split1 = np.array([(mean_flood2[x, :]) for x in range(11)])
        split2 = np.array([(mean_flood2[x, :]) for x in range(11, 14)])
    else:
        # sort array by following order: 1st by maintenance scenarios (3rd col), 2nd by flood scenarios (2nd col)
        a = new_array[:, 0]
        b = new_array[:, -1]
        c = new_array[:, -2]
        ind = np.lexsort((a, b, c))  # create array with the specified order
        sort = np.array([(a[i], b[i], c[i]) for i in ind])  # apply the "sorting array" to the original array

        splits = np.array(np.split(sort[:, :], 14))
        splits = np.array(splits)

        # calculate mean of flood scn for different maintenance scn
        mean_flood = []
        for x in range(splits.shape[0]):  # loop through all scenarios
            mean_m = np.mean((splits[x, :, 0]).astype('float'))  # build mean of DEMdiff for each cell during 100yrs
            mean_flood.append(mean_m)
        mean_flood = np.array(mean_flood)

        perc_loc2 = np.append((0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5), np.array([5.5, 6, 6.5]))
        mean_flood2 = np.append((mean_flood), perc_loc2).reshape(2,mean_flood.shape[0])

        split1 = np.array([(mean_flood2[:, x]) for x in range(11)])
        split2 = np.array([(mean_flood2[:, x]) for x in range(11, 14)])

    print('mean_flood calculated''\n')
    return split1, split2

def doColors(diffyear):
    '''create color palette to plot the different diff_yrs scenarios. depending on the number of diff_years,
    the palette will be extended'''
    if diffyear <= 8:
        # palette = 'gold','orangered','darkorchid','dodgerblue','limegreen','dimgrey'
        palette = np.array([['#ffe766', '#ffb499', '#e0c1ef', '#bbddff', '#adebad', '#c3c3c3'],
                            ['#ffdf32', '#ff8f66', '#cc98e5', '#8ec7ff', '#84e184', '#a5a5a5'],
                            ['#ffae19', '#ff6a32', '#b76fdb', '#61b1ff', '#5ad75a', '#878787'],
                            ['#e59400', '#ff4500', '#a346d1', '#349bff', '#32cd32', '#696969'],
                            ['#b27300', '#e53e00', '#9932cc', '#1e90ff', '#2db82d', '#5e5e5e'],
                            ['#895900', '#b23000', '#7a28a3', '#1873cc', '#238f23', '#494949'],
                            ['#6b4500', '#7f2200', '#5b1e7a', '#125699', '#196619', '#343434'],
                            ['#4c3100', '#4c1400', '#3d1451', '#0c3966', '#0f3d0f', '#1f1f1f']])
    elif 9 < diffyear <= 15:
        palette = np.array([['#ffe766', '#ffb499', '#e0c1ef', '#bbddff', '#adebad', '#c3c3c3'],
                            ['#ffe34c', '#ffa27f', '#d6adea', '#a5d2ff', '#98e698', '#b4b4b4'],
                            ['#ffdf32', '#ff8f66', '#cc98e5', '#8ec7ff', '#84e184', '#a5a5a5'],
                            ['#ffd700', '#ff7c4c', '#c184e0', '#78bcff', '#6fdc6f', '#969696'],
                            ['#ffae19', '#ff6a32', '#b76fdb', '#61b1ff', '#5ad75a', '#878787'],
                            ['#ffa500', '#ff5719', '#ad5ad6', '#4aa6ff', '#46d246', '#787878'],
                            ['#e59400', '#ff4500', '#a346d1', '#349bff', '#32cd32', '#696969'],
                            ['#b27300', '#e53e00', '#9932cc', '#1e90ff', '#2db82d', '#5e5e5e'],
                            ['#996300', '#cc3700', '#892db7', '#1b81e5', '#28a428', '#545454'],
                            ['#895900', '#b23000', '#7a28a3', '#1873cc', '#238f23', '#494949'],
                            ['#7a4f00', '#992900', '#6b238e', '#1564b2', '#1e7b1e', '#3f3f3f'],
                            ['#6b4500', '#7f2200', '#5b1e7a', '#125699', '#196619', '#343434'],
                            ['#5b3b00', '#661b00', '#4c1966', '#0f487f', '#145214', '#2a2a2a'],
                            ['#4c3100', '#4c1400', '#3d1451', '#0c3966', '#0f3d0f', '#1f1f1f'],
                            ['#3d2700', '#330d00', '#2d0f3d', '#092b4c', '#0a290a', '#0a0a0a']])
    else:
        print('the number of lines exceeds the number of colors. the color palette will be replicated as many times as '
              'necessary.')
        if diffyear % 15 == 0:
            multi = int(diffyear / 15)
        else:
            multi = np.round(diffyear / 15, 0).astype(int) + 1
        palette = np.array(multi * [['#ffe766', '#ffb499', '#e0c1ef', '#bbddff', '#adebad', '#c3c3c3'],
                                    ['#ffe34c', '#ffa27f', '#d6adea', '#a5d2ff', '#98e698', '#b4b4b4'],
                                    ['#ffdf32', '#ff8f66', '#cc98e5', '#8ec7ff', '#84e184', '#a5a5a5'],
                                    ['#ffd700', '#ff7c4c', '#c184e0', '#78bcff', '#6fdc6f', '#969696'],
                                    ['#ffae19', '#ff6a32', '#b76fdb', '#61b1ff', '#5ad75a', '#878787'],
                                    ['#ffa500', '#ff5719', '#ad5ad6', '#4aa6ff', '#46d246', '#787878'],
                                    ['#e59400', '#ff4500', '#a346d1', '#349bff', '#32cd32', '#696969'],
                                    ['#b27300', '#e53e00', '#9932cc', '#1e90ff', '#2db82d', '#5e5e5e'],
                                    ['#996300', '#cc3700', '#892db7', '#1b81e5', '#28a428', '#545454'],
                                    ['#895900', '#b23000', '#7a28a3', '#1873cc', '#238f23', '#494949'],
                                    ['#7a4f00', '#992900', '#6b238e', '#1564b2', '#1e7b1e', '#3f3f3f'],
                                    ['#6b4500', '#7f2200', '#5b1e7a', '#125699', '#196619', '#343434'],
                                    ['#5b3b00', '#661b00', '#4c1966', '#0f487f', '#145214', '#2a2a2a'],
                                    ['#4c3100', '#4c1400', '#3d1451', '#0c3966', '#0f3d0f', '#1f1f1f'],
                                    ['#3d2700', '#330d00', '#2d0f3d', '#092b4c', '#0a290a', '#0a0a0a']])
    return palette

def doPlot(diff1, diff2, flood1, flood2, min1, min2, max1, max2, diffyear, diff):
    '''create two different plots, first one for all flood scn seperately, second one for the mean of all flood scn and
    its range. the plots includes the location scenarios'''
    floods = ['(a) 2 apart', '(b) 2 close', '(c) 2 med', '(d) 1 first', '(e) 1 middle', '(f) 1 last']
    palette3 = np.array([diffyear*['#f9f0a1']])
    palette2 = np.array(['#e3ce8d', '#db786c', '#8e729d', '#7ba6d0', '#7ba47b', '#8d8d8d'])
    palette = doColors(diffyear)
    if diffyear>1:
        # all flood scn
        fig = plt.figure(figsize=(19, 12))
        # fig.suptitle(title, fontsize=24, fontweight=1, color='black').set_position([.5, 0.94])
        ax = plt.axes([0, 0, 1, 1], frameon=False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        fig.text(0.5, 0.06, xlabel_subplot, ha='center', fontsize=ax_size)
        fig.text(0.09, 0.5, ylabel, va='center', rotation='vertical', fontsize=ax_size)
        for x in range(len(floods)):
            ax = fig.add_subplot(3, 2, (x + 1), sharey=ax)
            plt.title(floods[x], fontsize=ax_size, color='black', loc='left')
            plt.subplots_adjust(wspace=0.1, hspace=0.32)
            for y in range(diffyear):
                ax.plot(diff1[y, x, :, 1], diff1[y, x, :, 0].astype(float), color=palette[y, x],
                        marker='.', markersize=2, linestyle='-', linewidth=.5, label=diff[y])
                ax.plot(diff2[y, x, :, 1], diff2[y, x, :, 0].astype(float), color=palette[y, x],
                        marker='.', markersize=2, linestyle='-', linewidth=.5, label='_nolegend_')
                legend = plt.legend(loc='upper center', ncol=np.round(diffyear/4, 0).astype(int),
                                    fontsize=10, title=legend_y)
                plt.setp(legend.get_title(), fontsize=(l_size-4))
                plt.xticks([r + 0.005 for r in range(0, 14)],
                           [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 'High', 'Mid', 'Low'],
                           fontsize=l_size-2)
                plt.yticks(fontsize=l_size-2)
                plt.axvline(x=10.5, color='black', linestyle='--', linewidth=0.5, alpha=0.6)
        plt.savefig(save1, dpi=300, bbox_inches='tight')

    else:
        # all flood scn
        floods = ['2 apart', '2 close', '2 med', '1 first', '1 middle', '1 last']
        fig = plt.figure(figsize=(19, 12))
        # fig.suptitle(title, fontsize=24, fontweight=1, color='black').set_position([.5, 0.94])
        ax = fig.add_subplot(1, 1, 1)
        for x in range(len(floods)):
            ax.plot(diff1[x, :, 1], diff1[x, :, 0].astype(float), color=palette2[x],
                    marker='.', linestyle='--', linewidth=1, label=floods[x])
            ax.plot(diff2[x, :, 1], diff2[x, :, 0].astype(float), color=palette2[x],
                    marker='.', linestyle='--', linewidth=1, label='_nolegend_')
        plt.plot(flood1[:, 1], flood1[:, 0].astype(float), color='black',
                marker='.', linestyle='-', linewidth=1, label=label_mean)
        plt.plot(flood2[:, 1], flood2[:, 0].astype(float), color='black',
                marker='.', linestyle='-', linewidth=1, label='_nolegend_')
        legend = plt.legend(ncol=2, fontsize=l_size, title=legend_h, bbox_to_anchor=(0.042, 0.05), loc=3)
        plt.setp(legend.get_title(), fontsize=l_size)
        plt.xticks([r + 0.005 for r in range(0, 14)], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 'High', 'Mid', 'Low'],
                   fontsize=l_size)
        plt.yticks(fontsize=l_size)
        plt.ylim(lim)
        plt.xlabel(xlabel, fontsize=ax_size, labelpad=8)
        plt.ylabel(ylabel, fontsize=ax_size, labelpad=15)
        ax.yaxis.grid(linestyle='--', alpha=0.3)
        plt.axvline(x=10.5, color='black', linestyle='--', linewidth=0.4)
        plt.savefig(save1, dpi=300, bbox_inches='tight')

        # mean + range flood scn
        fig = plt.figure(figsize=(19, 12))
        # fig.suptitle(title, fontsize=24, fontweight=1, color='black').set_position([.5, 0.94])
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(min1[:, 1], min1[:, 0].astype(float), color='#8e729d',
                linestyle='-', linewidth=0.7)
        ax.plot(min2[:, 1], min2[:, 0].astype(float), color='#8e729d',
                linestyle='-', linewidth=0.7, label='_nolegend_')
        ax.plot(max1[:, 1], max1[:, 0].astype(float), color='#8e729d',
                linestyle='-', linewidth=0.7, label='_nolegend_')
        ax.plot(max2[:, 1], max2[:, 0].astype(float), color='#8e729d',
                linestyle='-', linewidth=0.7, label='_nolegend_')
        ax.plot(flood1[:, 1], flood1[:, 0].astype(float), color='black',
                linestyle='-', marker='.', linewidth=1, label=label_mean)
        ax.plot(flood2[:, 1], flood2[:, 0].astype(float), color='black',
                linestyle='-', marker='.', linewidth=1, label='_nolegend_')
        plt.fill_between(min1[:, 1], max1[:, 0], min1[:, 0], color='#8e729d', alpha=0.4, label=label_range)
        plt.fill_between(min2[:, 1], max2[:, 0], min2[:, 0], color='#8e729d', alpha=0.4)
        legend = plt.legend(ncol=1, fontsize=l_size, title=legend_h, bbox_to_anchor=(0.042, 0.05), loc=3)
        plt.setp(legend.get_title(), fontsize=l_size)
        plt.xticks([r + 0.005 for r in np.arange(0, 7, 0.5)], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                                                               'High', 'Mid', 'Low'], fontsize=l_size)
        plt.ylim(lim)
        plt.yticks(fontsize=l_size)
        plt.ylabel(ylabel, fontsize=ax_size, labelpad=15)
        ax.yaxis.grid(linestyle='--', alpha=0.3)
        plt.axvline(x=5.25, color='black', linestyle='--', linewidth=0.4)
        plt.xlabel(xlabel, fontsize=ax_size, labelpad=8)
        plt.savefig(save2, dpi=300, bbox_inches='tight')
        print('plots with maintenance and location scenario''\n')
    return

def doPlot_maint(diff1, flood1, min1, max1, diffyear, diff):
    '''create two different plots, first one for all flood scn seperately, second one for the mean of all flood scn and
    its range. the plots only show the maintenance effort scenarios'''
    palette2 = np.array(['#e3ce8d', '#db786c', '#8e729d', '#7ba6d0', '#7ba47b', '#8d8d8d'])
    palette = doColors(diffyear)
    floods = ['(a) 2 apart', '(b) 2 close', '(c) 2 med', '(d) 1 first', '(e) 1 middle', '(f) 1 last']

    if diffyear>1:
        # all flood scn
        # subplot
        fig = plt.figure(figsize=(19, 12))
        # fig.suptitle(title, fontsize=24, fontweight=1, color='black').set_position([.5, 0.94])
        palette = doColors(diffyear)
        ax = plt.axes([0, 0, 1, 1], frameon=False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        fig.text(0.5, 0.06, xlabel_subplot, ha='center', fontsize=ax_size)
        fig.text(0.09, 0.5, ylabel, va='center', rotation='vertical', fontsize=ax_size)
        for x in range(len(floods)):
            ax = fig.add_subplot(3, 2, (x + 1), sharey=ax)
            ax.text(0.8, .9, floods[x], fontsize=l_size, color='black', transform=ax.transAxes)
            ax.text(0.02, .92, 'Year', fontsize=l_size-4, color=palette[diffyear-1, x], transform=ax.transAxes)
            plt.subplots_adjust(wspace=0.1, hspace=0.2)
            for y in range(diffyear):
                ax.plot(diff1[y, x, :, 1], diff1[y, x, :, 0].astype(float), color=palette[y, x],
                        marker='.', markersize=2, linestyle='-', linewidth=0.5)
                if y % 2 == 0:
                    ax.text(-.4, diff1[y, x, 0, 0].astype(float), diff_yrs[y],
                            alpha=1, color=palette[y, x], fontsize=l_size-4)
                else:
                    ax.text(-0.8, diff1[y, x, 0, 0].astype(float), diff_yrs[y],
                            alpha=1, color=palette[y, x], fontsize=l_size-4)
                plt.xticks([r + 0.005 for r in range(0, 11)], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=l_size)
                plt.yticks(fontsize=l_size)
                plt.xlim(-1, 10.2)
                plt.ylim(-0.2, 6.75)
        plt.savefig(save1, dpi=400, bbox_inches='tight')

        # individual plots
        for x in range(len(floods)):
            floods = ['(a) 2 apart', '(b) 2 close', '(c) 2 med', '(d) 1 first', '(e) 1 middle', '(f) 1 last']
            fig = plt.figure(figsize=(19, 12))
            palette = doColors(diffyear)

            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.85, 0.95, floods[x], fontsize=ax_size, color='black', transform=ax.transAxes)

            for y in range(diffyear):
                ax.plot(diff1[y, x, :, 1], diff1[y, x, :, 0].astype(float), color=palette[y, x],
                        marker='.', markersize=4, linestyle='-', linewidth=0.9)
                ax.text(-.25, diff1[y, x, 0, 0].astype(float), diff_yrs[y], alpha=1, color=palette[y, x], fontsize=l_size)
            ax.yaxis.grid(linestyle='--', alpha=0.3)
            plt.ylabel(ylabel, fontsize=ax_size, labelpad=15)
            plt.xlabel(xlabel_subplot, fontsize=ax_size, labelpad=(l_size-2))
            plt.xticks([r + 0.005 for r in range(0, 11)], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=l_size)
            plt.yticks(fontsize=l_size)
            plt.ylim(-0.1, 6.1)
            plt.xlim(-.5, 10.2)
            floods = ['2 apart', '2 close', '2 med', '1 first', '1 middle', '1 last']
            plt.savefig('U:simulations/analysis/python/channel change/MultiIndplot'+str(typ)+'_'+floods[x]+'_maint.png',
                        dpi=300, bbox_inches='tight')

    else:
        # all flood scn
        floods = ['2 apart', '2 close', '2 med', '1 first', '1 middle', '1 last']
        fig = plt.figure(figsize=(19, 12))
        # fig.suptitle(title, fontsize=24, fontweight=1, color='black').set_position([.5, 0.94])
        ax = fig.add_subplot(1, 1, 1)
        for x in range(len(floods)):
            ax.plot(diff1[x, :, 1], diff1[x, :, 0].astype(float), color=palette2[x],
                    marker='.', linestyle='--', linewidth=1, label=floods[x])
        plt.plot(flood1[:,1], flood1[:,0].astype(float), color='black',
                 marker='.', linestyle='-', linewidth=1, label=label_mean)
        legend = plt.legend(ncol=2, fontsize=l_size, title=legend_h, bbox_to_anchor=(0.042, 0.05), loc=3)
        plt.setp(legend.get_title(), fontsize=l_size)
        ax.yaxis.grid(linestyle='--', alpha=0.3)
        plt.xticks([r + 0.005 for r in range(0,11)], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=l_size)
        plt.yticks(fontsize=l_size)
        plt.ylim(lim)
        plt.xlabel(xlabel, fontsize=ax_size, labelpad=8)
        plt.ylabel(ylabel, fontsize=ax_size, labelpad=15)
        ax.yaxis.grid(linestyle='--', alpha=0.3)
        plt.savefig(save1, dpi=300, bbox_inches='tight')

        # mean + range flood scn
        fig = plt.figure(figsize=(19, 12))
        # fig.suptitle(title, fontsize=24, fontweight=1,
        #              color='black').set_position([.5, 0.94])
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(min1[:, 1], min1[:, 0].astype(float), color='#8e729d',
                linestyle='-', linewidth=0.7)
        ax.plot(max1[:, 1], max1[:, 0].astype(float), color='#8e729d',
                linestyle='-', linewidth=0.7, label='_nolegend_')
        ax.plot(flood1[:, 1], flood1[:, 0].astype(float), color='black',
                linestyle='-', marker='.', linewidth=1, label=label_mean)
        plt.fill_between(min1[:, 1], max1[:, 0], min1[:, 0], color='#8e729d', alpha=0.4, label=label_range)
        legend = plt.legend(ncol=1, fontsize=l_size, title=legend_h, bbox_to_anchor=(0.042, 0.05), loc=3)
        plt.setp(legend.get_title(), fontsize=l_size)
        ax.yaxis.grid(linestyle='--', alpha=0.3)
        plt.xticks([r + 0.005 for r in np.arange(0, 5.5, 0.5)], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=l_size)
        plt.yticks(fontsize=l_size)
        plt.ylim(lim)
        plt.ylabel(ylabel, fontsize=ax_size, labelpad=15)
        plt.xlabel(xlabel, fontsize=ax_size, labelpad=8)
        ax.yaxis.grid(linestyle='--', alpha=0.3)
        plt.savefig(save2, dpi=300, bbox_inches='tight')
        print('plots with maintenance scenario''\n')
    return

def doPlot_loc(diff1, diff2, flood1, flood2, min2, max2, diffyear, diff):
    '''create two different plots, first one for all flood scn seperately, second one for the mean of all flood scn and
    its range. the plots only show the maintenance location scenarios'''
    palette3 = np.array([diffyear*['#f9f0a1']])
    palette2 = np.array(['#e3ce8d', '#db786c', '#8e729d', '#7ba6d0', '#7ba47b', '#8d8d8d'])
    palette = doColors(diffyear)

    if diffyear>1:
        # all flood scn
        # subplot
        floods = ['(a) 2 apart', '(b) 2 close', '(c) 2 med', '(d) 1 first', '(e) 1 middle', '(f) 1 last']
        fig = plt.figure(figsize=(19, 12))
        # fig.suptitle(title, fontsize=24, fontweight=1, color='black').set_position([.5, 0.95])
        ax = plt.axes([0, 0, 1, 1], frameon=False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        fig.text(0.5, 0.06, xlabel_subplot, ha='center', fontsize=ax_size)
        fig.text(0.09, 0.5, ylabel, va='center', rotation='vertical', fontsize=ax_size)
        for x in range(len(floods)):
            ax = fig.add_subplot(3, 2, (x + 1), sharey=ax)
            ax.text(0.78, .9, floods[x], fontsize=l_size, color='black', transform=ax.transAxes)
            ax.text(0.02, .65, 'Year', fontsize=l_size-4, color=palette[diffyear-1, x], transform=ax.transAxes)
            plt.subplots_adjust(wspace=0.1, hspace=0.2)
            for y in range(diffyear):
                ax.plot(2.38, diff1[y, x, 3, 0].astype(float), color='grey', marker='x', markersize=5.5,
                        label=label_30 if y == 0 else '')
                ax.plot(diff2[y, x, :, 1], diff2[y, x, :, 0].astype(float), color=palette[y, x],
                        marker='.', markersize=3.5, linestyle='-', linewidth=0.9)
                # ax.text(2.19, 3.8, '30% mainte-''\n''nance effort' if y == 0 else '', fontsize=l_size-4, color='grey')
                if y % 2 == 0:
                    ax.text(-.1, diff2[y, x, 0, 0].astype(float), diff_yrs[y],
                            alpha=1, color=palette[y, x], fontsize=l_size-4)
                else:
                    ax.text(-.2, diff2[y, x, 0, 0].astype(float), diff_yrs[y],
                            alpha=1, color=palette[y, x], fontsize=l_size-4)
            plt.xticks(np.arange(0, 3, 1), ['High', 'Mid', 'Low'], fontsize=l_size)
            plt.yticks(fontsize=l_size)
            plt.xlim(-0.25, 2.1)
            plt.ylim(-0.2, 6.75)
        plt.savefig(save1, dpi=400, bbox_inches='tight')

        # individual plots
        for x in range(len(floods)):
            floods = ['(a) 2 apart', '(b) 2 close', '(c) 2 med', '(d) 1 first', '(e) 1 middle', '(f) 1 last']
            fig = plt.figure(figsize=(19, 12))
            palette = doColors(diffyear)
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.02, 0.95, floods[x], fontsize=ax_size, color='black', transform=ax.transAxes)
            for y in range(diffyear):
                ax.plot(2.4, diff1[y, x, 3, 0].astype(float), color='grey', marker='x', markersize=8,)
                ax.plot(diff2[y, x, :, 1], diff2[y, x, :, 0].astype(float), color=palette[y, x],
                        marker='.', markersize=4, linestyle='-', linewidth=0.9)
                if y % 2 == 0:
                    ax.text(2.03, diff2[y, x, 2, 0].astype(float), diff_yrs[y],
                            alpha=1, color=palette[y, x], fontsize=(l_size-2))
                else:
                    ax.text(2.08, diff2[y, x, 2, 0].astype(float), diff_yrs[y],
                            alpha=1, color=palette[y, x], fontsize=(l_size-2))
            ax.text(2.3, 3.8, '30% mainte-''\n''nance effort', fontsize=l_size, color='grey')
            # ax.text(1.7, 6, 'after ... years', fontsize=l_size, color=palette[y-2, x])

            ax.yaxis.grid(linestyle='--', alpha=0.3)
            plt.ylabel(ylabel, fontsize=ax_size, labelpad=15)
            plt.xlabel(xlabel_subplot, fontsize=ax_size, labelpad=l_size)
            plt.xticks(np.arange(0, 3, 1), ['High', 'Mid', 'Low'], fontsize=l_size)
            plt.yticks(fontsize=l_size)
            plt.xlim(-.1, 2.6)
            plt.ylim(-0.05, 6.2)
            floods = ['2 apart', '2 close', '2 med', '1 first', '1 middle', '1 last']
            plt.savefig('U:simulations/analysis/python/channel change/MultiIndplot'+str(typ)+'_'+floods[x]+'_loc.png',
                        dpi=300, bbox_inches='tight')

    else:
        # all flood scn
        floods = ['2 apart', '2 close', '2 med', '1 first', '1 middle', '1 last']
        fig = plt.figure(figsize=(19, 12))
        # fig.suptitle(title, fontsize=24, fontweight=1, color='black').set_position([.5, 0.94])
        ax = fig.add_subplot(1, 1, 1)

        for x in range(len(floods)):
            ax.plot(diff2[x, :, 1], diff2[x, :, 0].astype(float), color=palette2[x],
                    marker='.', linestyle='--', linewidth=1, label=floods[x])
        plt.plot(flood2[:, 1], flood2[:, 0].astype(float), color='black',
                marker='.', linestyle='-', linewidth=1, label=label_mean)
        plt.plot(2, flood1[3, 0].astype(float), color='grey', marker='x', markersize=9, linestyle='-', linewidth=1)
        plt.plot(1, flood1[3, 0].astype(float), color='grey', marker='x', markersize=9, linestyle='-', linewidth=1)
        plt.plot(0, flood1[3, 0].astype(float), color='grey',
                 marker='x', markersize=9, linestyle='-.', linewidth=1, label=label_30)
        legend = plt.legend(ncol=2, fontsize=l_size, title=legend_h, bbox_to_anchor=(0.042, 0.05), loc=3)
        plt.setp(legend.get_title(), fontsize=l_size)
        ax.yaxis.grid(linestyle='--', alpha=0.3)
        plt.xticks(np.arange(0, 3, 1), ['High', 'Mid', 'Low'], fontsize=l_size)
        plt.yticks(fontsize=l_size)
        plt.xlabel(xlabel, fontsize=ax_size, labelpad=8)
        plt.ylabel(ylabel, fontsize=ax_size, labelpad=15)
        ax.yaxis.grid(linestyle='--', alpha=0.3)
        plt.ylim(lim)
        plt.axhline(flood1[3, 0].astype(float), xmin=0.045, xmax=0.955, color='grey', linestyle='-.', linewidth=1, alpha=1)
        plt.savefig(save1, dpi=400, bbox_inches='tight')

        # mean + range flood scn
        fig = plt.figure(figsize=(19, 12))
        # fig.suptitle(title, fontsize=24, fontweight=1, color='black').set_position([.5, 0.94])
        ax = fig.add_subplot(1, 1, 1)

        ax.fill_between(min2[:, 1], max2[:, 0], min2[:, 0], color='#8e729d', alpha=0.4, label=label_range)
        ax.plot(min2[:, 1], min2[:, 0].astype(float), color='#8e729d', linestyle='-', linewidth=0.7)
        ax.plot(max2[:, 1], max2[:, 0].astype(float), color='#8e729d', linestyle='-', linewidth=0.7)
        ax.plot(flood2[:, 1], flood2[:, 0].astype(float), color='black', marker='.', linewidth=1, label=label_mean)
        ax.plot(6.5, flood1[3, 0].astype(float), color='grey', linestyle='-.', marker='x', markersize=9, label=label_30)
        ax.plot(6, flood1[3, 0].astype(float), color='grey', linestyle='-', marker='x', markersize=9)
        ax.plot(5.5, flood1[3, 0].astype(float), color='grey', linestyle='-', marker='x', markersize=9)
        handles, labels = plt.gca().get_legend_handles_labels()     # change order of labels in legend
        order = [0, 2, 1]
        legend = plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol=1, fontsize=l_size,
                            title=legend_h, bbox_to_anchor=(0.042, 0.05), loc=3)
        plt.setp(legend.get_title(), fontsize=l_size)
        ax.yaxis.grid(linestyle='--', alpha=0.3)
        plt.xticks(np.arange(5.5, 7, 0.5), ['High', 'Mid', 'Low'], fontsize=l_size)
        plt.yticks(fontsize=l_size)
        plt.ylabel(ylabel, fontsize=ax_size, labelpad=15)
        ax.yaxis.grid(linestyle='--', alpha=0.3)
        plt.xlabel(xlabel, fontsize=ax_size, labelpad=8)
        plt.ylim(lim)
        plt.axhline(flood1[3, 0].astype(float), xmin=0.045, xmax=0.955, color='grey', linestyle='-.', linewidth=1, alpha=1)
        plt.savefig(save2, dpi=300, bbox_inches='tight')
        print('plots with location scenario''\n')
    return

def doExport(scenarios):
    '''export all arrays to ArcGIS files which include geometric information or to normal files'''
    ArcGIS = 'ncols 127' '\n' 'nrows 115' '\n' 'xllcorner 602510.99199495' '\n'\
             'yllcorner 175232.74791014' '\n' 'cellsize 15' '\n' 'NODATA_value 0.000000000000000'

    diff_name = 'U:simulations/analysis/python/channel change/DEMs/DEMdiff{x}.asc'
    scenario_list = 'U:simulations/analysis/python/list.txt'

    for scenario in range(scenarios):
        np.savetxt(diff_name.format(x=scenario), DEMdiff_thal0[scenario, 0, :, :], delimiter=' ', comments='', header=ArcGIS)
    # np.savetxt(scenario_list, elev[:, :], delimiter=' ', comments='', fmt="%s")
    print('\n''saved them ALL''\n')


# -DEFINE GLOBAL VARIABLES HERE-
# years from which the difference should be calculated. (uncomment which one you want to calculate!)
#_______________________________________________________________________________________________________________________

# diff_yrs = [100]                                                         # difference years reflect first & last year
# typ = ''
# after_during = "after"
diff_yrs = list(range(1, 100, 15))                                        # a: difference years reflect random years
n = 6
typ = 'Random'+str(n)
after_during = "during"
# diff_yrs = [1, 4, 21, 24, 50, 53, 95, 98]                                # b: difference years reflect big events
# typ = 'BigEvents'
# diff_yrs = [1, 8, 15, 23, 30, 38, 45, 52, 60, 67, 75, 82, 89, 97]        # c: difference years reflect all events
# typ = 'AllEvents'
# after_during = "during"

#_______________________________________________________________________________________________________________________

# flood scn
floods = ['2 apart', '2 close', '2 med', '1 first', '1 middle', '1 last']

# mask for where to calculate the difference; thalweg = narrow/channel = wide. (uncomment which one you want to calculate!)
profile = 'U:simulations/analysis/python/channel change/thalweg.txt'
# profile = 'U:simulations/analysis/python/channel change/channel.txt'

# -CALL FUNCTIONS HERE-
# calculate elevation difference of DEM between predefined years
DEMdiff0, DEMdiff = doDEMdiff(DEM.shape[0], diff_yrs)

# mask the DEMdiff file with the profile line (created in ArcGIS)
DEMdiff_thal0, DEMdiff_thal = doProfile(profile, DEMdiff.shape[0], diff_yrs, DEMdiff)

# calculate the mean, std and rmse of the total channel DEM
mean, std, rmse = doStatistics(DEMdiff.shape[0], diff_yrs, DEMdiff)

# thalweg only
mean_thal, std_thal, rmse_thal = doStatistics(DEMdiff_thal.shape[0], diff_yrs, DEMdiff_thal)

# calculate the mean, std and rmse of the thalweg only channel DEM
mean1_t, mean2_t = doArray(mean_thal)

# calculate the min and max of the elevation difference (thalweg only). if you want the whole channel, change input to
# total DEM (DEMdiff). the min max can only be calculated if DEMdiff is calculated between year 100 and 0 and not
# multiple with multiple diff_yrs.
min1, min2, max1, max2 = doMinMax(mean_thal)

# calculate the mean elevation diff for all flood scn for the different maintenance scn. also here only for diff_yrs=100.
floodm_t1, floodm_t2 = doFloodmean(mean_thal)

# plot maintenance and location scn
# define plot properties
xlabel = "                                        Maintenance effort [%]" \
         "                                                                 Location"
xlabel_subplot = "Maintenance effort [%] and maintenance location"
ylabel = "Mean change in elevation [m]"
legend_h = "Hydrograph"
legend_y = "Year"
label_mean = "Mean of hydrographs"
label_range = "Range of hydrographs"
label_30 = "30 % maintenance"
lim = -0.35, 6.35
ax_size = 18
l_size = 16
title = "Change in elevation"+after_during+"100 years of simulation"

if mean_thal.shape[1] == 1:
    diff_yrs = 0                                                  # set diff_yrs to zero, because its not defined
    save1 = 'U:simulations/analysis/python/channel change/SingleAll'+str(typ)+'_maint+loc.png'
    save2 = 'U:simulations/analysis/python/channel change/SingleRange'+str(typ)+'_maint+loc.png'
    save3 = 'U:simulations/analysis/python/channel change/SingleMean'+str(typ)+'_maint+loc.png'
    doPlot(mean1_t, mean2_t, floodm_t1, floodm_t2, min1, min2, max1, max2, mean_thal.shape[1], diff_yrs)
else:
    save1 = 'U:simulations/analysis/python/channel change/MultiAll'+str(typ)+'_maint+loc.png'
    save2 = 'U:simulations/analysis/python/channel change/MultiRange'+str(typ)+'_maint+loc.png'
    save3 = 'U:simulations/analysis/python/channel change/MultiMean'+str(typ)+'_maint+loc.png'
    doPlot(mean1_t, mean2_t, floodm_t1, floodm_t2, min1, min2, max1, max2, mean_thal.shape[1], diff_yrs)

# plot maintenance scn only
xlabel = "Maintenance effort [%]"
xlabel_subplot = "Maintenance effort [%]"

if mean_thal.shape[1] == 1:
    save1 = 'U:simulations/analysis/python/channel change/SingleAll_maint.png'
    save2 = 'U:simulations/analysis/python/channel change/SingleRange_maint.png'
    doPlot_maint(mean1_t, floodm_t1, min1, max1, mean_thal.shape[1], diff_yrs)
else:
    save1 = 'U:simulations/analysis/python/channel change/MultiSubplot'+str(typ)+'_maint.png'
    save2 = ''
    doPlot_maint(mean1_t, floodm_t1, min1, max1, mean_thal.shape[1], diff_yrs)

# plot location scn only
xlabel = "Maintenance location"
xlabel_subplot = "Maintenance location"

if mean_thal.shape[1] == 1:
    save1 = 'U:simulations/analysis/python/channel change/SingleAll'+str(typ)+'_loc.png'
    save2 = 'U:simulations/analysis/python/channel change/SingleRange'+str(typ)+'_loc.png'
    doPlot_loc(mean1_t, mean2_t, floodm_t1, floodm_t2, min2, max2, mean_thal.shape[1], diff_yrs)
else:
    save1 = 'U:simulations/analysis/python/channel change/MultiSubplot'+str(typ)+'_loc.png'
    save2 = ''
    doPlot_loc(mean1_t, mean2_t, floodm_t1, floodm_t2, min2, max2, mean_thal.shape[1], diff_yrs)

# export DEMdiff in ArcGIS readable format
if DEMdiff_thal.shape[1] == 1:
    doExport(DEMdiff.shape[0])
else:
    print('ERROR: exports only DEMs if diff_yrs = [100]. otherwise too many DEMs too store. if you want them to be '
          'stored anyways, you have to add a second loop which loops over the diff_yrs''\n')
