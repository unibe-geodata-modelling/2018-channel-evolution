# SED YIELD + DOWNSTREAM FILL: this script deals with the sediment yield generated at the output of the model. it
# calculates the total sum of sediment yield, cumulative sediment yield (evolution of sediment yield over simulation
# time), potential downstream channel fill in percentage and meters, and the cross section specific analysis of channel
# aggradation.

# -IMPORT LIBRARIES & VARIABLES HERE-
import numpy as np
import glob
import matplotlib.pyplot as plt

# -DEFINE FUNCTIONS HERE-
def doRead(files):
    '''read in the .dat file which includes the hourly sediment yield values'''
    print('\n''IMPORTAMT MESSAGE:''\n''\n''sorry to tell you that this is gonna take A LOOOOOONG time (5-10min). maybe'
          ' you wanna go wash some dishes or start cooking dinner, while this is loading in the files.''\n')
    dat_data = []
    for x in range(files.shape[0]):
        read_files = np.genfromtxt(files[x], delimiter=' ', usecols=(1, 4), skip_header=3)            # only read in Qw & Qs
        dat_data.append(read_files)
        if (x % 6 == 0):
            print('file ' + str(x) + ' is created. Only ' + str(files.shape[0]-x) + ' to go!')
    dat_data = np.array(dat_data)
    print('\n''well, that took a while, thank\'s for being so patient. hope dinner or dishes or even both are done by now''\n')
    return dat_data

def doSumSed(file):
    '''calculate sum of sediment yield for all scenarios'''
    sumsed = []
    for x in range(file.shape[0]):
        sum_Qs = np.sum(file[x, :, 1])
        sumsed.append(sum_Qs)
    sumsed = np.array(sumsed)
    return sumsed

def doCumsum(file):
    '''calculate cumulative sum of sediment yield for all scenarios'''
    cumsum = []
    for x in range(0, file.shape[0]):
        csum_Qs = np.cumsum(file[x, :, 1])
        cumsum.append(csum_Qs)
        if x % 14 == 0:
            print('cumsum ' + str(x) +' done. only ' + str(file.shape[0]-x) + ' to go!')
    cumsum = np.array(cumsum)

    # get cumsum of each year (otherwise array is too big to calculate)
    cumsum = cumsum[:, np.arange(0, cumsum.shape[1], (24*365))]
    print('ALL done! cumsum is calculated.''\n')
    return cumsum

def doNewArray(input):
    '''create arrays to original elev array: specific maintenance scn ("perc"), location scn ("loc"), flood scn ("flood")'''
    perc_loc = np.repeat(np.arange(0, 7, 0.5), 6).reshape(input.shape[0], 1)
    new_array = np.append(np.vstack(input), perc_loc, axis=1)  # combine the new created perc_loc
    floods = np.array(14 * ['2apart', '2close', '2med', 'a_first', 'b_middle', 'c_last']).reshape(input.shape[0], 1)
    new_array = np.append(new_array, floods, axis=1)  # append third column to STD array
    return new_array

def doArray(sed_a, sed_m):
    '''create new array which is sorted in the right way for analyzing summary statistics of difference'''
    new_array = doNewArray(sed_a)

    # sort array by following order: 1st by flood scenarios (3rd col), 2nd by maintenance/location scenarios (2nd col)
    a1 = new_array[:, 0]
    a2 = new_array[:, 1]
    b = new_array[:, -2]
    c = new_array[:, -1]

    ind = np.lexsort((a1, a2, b, c))  # create array with the specified order
    sort = np.array([(a1[i], a2[i], b[i], c[i]) for i in ind])  # apply the "sorting array" to the original array

    splits = np.array(np.split(sort[:, :], 6))
    splits = np.array(splits)

    # location & maintenance scenarios need to be split in order to plot them
    split1 = []
    split2 = []
    for x in range(splits.shape[0]):
        first = splits[x, :11, :]
        last = splits[x, -3:, :]
        split1.append(first)
        split2.append(last)
    split1 = np.array(split1)
    split2 = np.array(split2)

    perc_loc2 = np.append((5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, 0), np.array([5.5, 6, 6.5]))
    m_sed2 = np.append(np.append(sed_m[:,0], sed_m[:, 1]), perc_loc2).reshape(3, sed_m.shape[0])

    msplit1 = np.array([(m_sed2[:, x]) for x in range(11)])
    msplit2 = np.array([(m_sed2[:, x]) for x in range(11, 14)])

    print('array created''\n')
    return split1, split2, msplit1, msplit2

def doMinMax(sed):
    '''calculates the min and the max value for each flood scenario. depending on if only 1 or 2 sediment yield values
    are analyzed, the calculation method is adapted'''
    new_array = doNewArray(sed)

    if len(sed.shape) == 2:
        a1 = new_array[:, 0]
        a2 = new_array[:, 1]
        b = new_array[:, -1]
        c = new_array[:, -2]

        # sort array by following order: 1st by flood scenarios (3rd col), 2nd by maintenance/location scenarios (2nd col)
        ind = np.lexsort((a1, a2, b, c))  # create array with the specified order
        sort = np.array([(a1[i], a2[i], b[i], c[i]) for i in ind])  # apply the "sorting array" to the original array

        splits = np.array(np.split(sort[:, :], 14))
        splits = np.array(splits)

        min_p_f = []
        min_m_f = []
        for x in range(splits.shape[0]):  # loop through all scenarios
            min_p = np.min((splits[x, :, 0]).astype('float'))   # mean of fill [%]
            min_m = np.min((splits[x, :, 1]).astype('float'))   # mean of fill [m]
            min_p_f.append(min_p)
            min_m_f.append(min_m)
        min_flood = np.append(min_p_f, min_m_f).reshape(2,splits.shape[0])

        max_p_f = []
        max_m_f = []
        for x in range(splits.shape[0]):  # loop through all scenarios
            max_p = np.max((splits[x, :, 0]).astype('float'))
            max_m = np.max((splits[x, :, 1]).astype('float'))
            max_p_f.append(max_p)
            max_m_f.append(max_m)
        max_flood = np.append(max_p_f, max_m_f).reshape(2,splits.shape[0])

        perc_loc2 = np.append((0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5), np.array([5.5, 6, 6.5]))

        min = np.append(min_flood[:], perc_loc2).reshape(3,min_flood.shape[1])
        min1 = np.array([(min[:, x]) for x in range(11)])
        min2 = np.array([(min[:, x]) for x in range(11, 14)])

        max = np.append(max_flood[:], perc_loc2).reshape(3,max_flood.shape[1])
        max1 = np.array([(max[:, x]) for x in range(11)])
        max2 = np.array([(max[:, x]) for x in range(11, 14)])

    else:
        a = new_array[:, 0]
        b = new_array[:, -1]
        c = new_array[:, -2]

        ind = np.lexsort((a, b, c))  # create array with the specified order
        sort = np.array([(a[i], b[i], c[i]) for i in ind])  # apply the "sorting array" to the original array

        splits = np.array(np.split(sort[:, :], 14))
        splits = np.array(splits)

        min_p_f = []
        for x in range(splits.shape[0]):  # loop through all scenarios
            min_p = np.min((splits[x, :, 0]).astype('float'))   # mean of fill [%]
            min_p_f.append(min_p)
        min_flood = np.array(min_p_f)

        max_p_f = []
        for x in range(splits.shape[0]):  # loop through all scenarios
            max_p = np.max((splits[x, :, 0]).astype('float'))
            max_p_f.append(max_p)
        max_flood = np.array(max_p_f)

        perc_loc2 = np.append((0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5), np.array([5.5, 6, 6.5]))

        min = np.append(min_flood[:], perc_loc2).reshape(2, min_flood.shape[0])
        min1 = np.array([(min[:, x]) for x in range(11)])
        min2 = np.array([(min[:, x]) for x in range(11, 14)])

        max = np.append(max_flood[:], perc_loc2).reshape(2, max_flood.shape[0])
        max1 = np.array([(max[:, x]) for x in range(11)])
        max2 = np.array([(max[:, x]) for x in range(11, 14)])
    print('min and max calculated''\n')
    return min1, min2, max1, max2

def doFloodmean(elev_diff):
    '''calculates the flood mean of all flood scenarios'''
    new_array = doNewArray(elev_diff)

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
    mean_flood2 = np.append(mean_flood, perc_loc2).reshape(2, mean_flood.shape[0])

    split1 = np.array([(mean_flood2[:, x]) for x in range(11)])
    split2 = np.array([(mean_flood2[:, x]) for x in range(11, 14)])

    print('mean_flood calculated''\n')
    return split1, split2

def doSplit(cumsum):
    '''split the new created array into the maitnenance effort and location scenarios'''
    perc_loc = np.repeat(np.array([0, 5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.5, 6, 6.5]), 6).reshape(cumsum.shape[0], 1)
    new_array = np.append(np.vstack(cumsum), perc_loc, axis=1)  # combine the new created perc_loc
    floods = np.array(int(cumsum.shape[0]/6) * ['2apart', '2close', '2med', 'a_first',
                                                'b_middle', 'c_last']).reshape(cumsum.shape[0], 1)
    new_array = np.append(new_array, floods, axis=1)  # append third column to STD array

    # sort array by maintenance scn
    a = new_array[:, range(0, cumsum.shape[1])]
    b = new_array[:, -1]    # flood scn
    c = new_array[:, -2]    # maint scn

    sorts = []
    for x in range(a.shape[1]):
        ind = np.lexsort((a[:, x], b, c))  # create array with the specified order
        sort = np.array([(a[:, x][i], b[i], c[i]) for i in ind])  # apply the "sorting array" to the original array
        sorts.append(sort)
    sorts = np.array(sorts)
    comb = np.concatenate(sorts[:, :, 0]).reshape(a.shape[1], a.shape[0])
    split = np.array(np.split(comb[:, :], (cumsum.shape[0]/6), axis=1)).astype('float32')  # split array into different
                                                                                           # flood scenarios (6)

    # substract offset (174400m^3) from arrays
    offset = np.transpose((6*[(np.arange(0, 174400, 1744))]))
    split_off = []
    for x in range(split.shape[0]):
        interlist = []
        for y in range(split.shape[2]):
            off = split[x, :, y]-offset[:, y]
            interlist.append(off)
        split_off.append(interlist)
    split_off = np.array(split_off)

    # calculate mean of flood scn for different maintenance scn
    offset = np.arange(0, 174400, 1744)
    mean = []
    for x in range(split.shape[0]):  # loop through all scenarios
        interlist = []
        for y in range(split.shape[1]):
            mean_m = np.mean((split[x, y, :]).astype('float'))  # build mean of DEMdiff for each cell during 100yrs
            interlist.append(mean_m)
        mean.append(interlist)
    mean = np.array(mean)
    mean_off = mean-offset

    # calculate min + max of flood scn for different maintenance scn
    min = []
    for x in range(split.shape[0]):  # loop through all scenarios
        interlist = []
        for y in range(split.shape[1]):
            mean_m = np.min((split[x, y, :]).astype('float'))  # build mean of DEMdiff for each cell during 100yrs
            interlist.append(mean_m)
        min.append(interlist)
    min = np.array(min)
    min_off = min-offset

    max = []
    for x in range(split.shape[0]):  # loop through all scenarios
        interlist = []
        for y in range(split.shape[1]):
            mean_m = np.max((split[x, y, :]).astype('float'))  # build mean of DEMdiff for each cell during 100yrs
            interlist.append(mean_m)
        max.append(interlist)
    max = np.array(max)
    max_off = max-offset
    print('profile split into maintenance scenarios. mean of different flood scenarios calculated.''\n')
    return split_off

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

def doPlot(scenario, diff1, diff2, flood1, flood2, min1, min2, max1, max2, color, xlabel, ylabel1, ylabel2,
           title, save1, save2):
    '''create two different plots, first one for all flood scn seperately, second one for the mean of all flood scn and
    its range. the plots includes the location scenarios'''
    print('let\'s plot now''\n')
    # 1st plot: all flood scn
    floods = ['2 apart', '2 close', '2 med', '1 first', '1 middle', '1 last']
    palette = np.array(['#e3ce8d', '#db786c', '#8e729d', '#7ba6d0', '#7ba47b', '#8d8d8d'])

    fig = plt.figure(figsize=(19, 12))
    # fig.suptitle(title, fontsize=24, fontweight=1, color='black').set_position([.5, 0.94])
    ax1 = fig.add_subplot(1, 1, 1)
    for x in range(scenario):
        ax1.plot(diff1[x, :, 2], diff1[x, :, 0].astype(float), color=palette[x],
                 marker='.', linestyle='--', linewidth=1, label=floods[x])
        ax1.plot(diff2[x, :, 2], diff2[x, :, 0].astype(float), color=palette[x],
                 marker='.', linestyle='--', linewidth=1)
    ax1.plot(flood1[:, -1], flood1[:, 0].astype(float), color='black',
             marker='o', linestyle='-', linewidth=1.25, label=lab_mean)
    ax1.plot(flood2[:, -1], flood2[:, 0].astype(float), color='black',
             marker='o', linestyle='-', linewidth=1.25)
    legend = plt.legend(ncol=2, fontsize=l_size, title=legend_h, bbox_to_anchor=(0.042, 0.05), loc=3)
    plt.setp(legend.get_title(), fontsize=l_size)
    plt.ylabel(ylabel1, fontsize=ax_size, labelpad=15)
    plt.xlabel(xlabel, fontsize=ax_size, labelpad=8)
    ax1.yaxis.grid(linestyle='--', alpha=0.3)
    plt.xticks(fontsize=l_size)
    plt.yticks(fontsize=l_size)
    plt.ylim(lim_range)

    if max1.shape[1] > 2:
        ax2 = ax1.twinx()
        ax2.plot(flood1[:, 2], flood1[:, 1].astype(float), color='black', linestyle='-', marker='.', linewidth=0.01, alpha=0)
        plt.yticks(np.arange(0, 4.5, 0.5), np.arange(0, 4.5, 0.5), fontsize=l_size)
        plt.ylabel(ylabel2, fontsize=ax_size, labelpad=15)
        plt.ylim(-0.075, 3.83)
        plt.yticks(fontsize=l_size)
    plt.xticks([r + 0.005 for r in range(0, 14)], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 'High', 'Mid', 'Low'],
               fontsize=l_size)
    plt.axvline(x=10.5, color='black', linestyle='--', linewidth=0.4)
    # plt.axvspan(0, 2, color='grey', alpha=0.05, lw=0)
    # plt.axvspan(8, 10, color='grey', alpha=0.05, lw=0)
    plt.savefig(save1, dpi=300, bbox_inches='tight')

    # 2nd plot: range of flood scn
    fig = plt.figure(figsize=(19, 12))
    # fig.suptitle(title, fontsize=24, fontweight=1, color='black').set_position([.5, 0.94])
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(min1[:, -1], min1[:, 0].astype(float), color=color, linestyle='-', linewidth=0.7)
    ax1.plot(min2[:, -1], min2[:, 0].astype(float), color=color, linestyle='-', linewidth=0.7)
    ax1.plot(max1[:, -1], max1[:, 0].astype(float), color=color, linestyle='-', linewidth=0.7)
    ax1.plot(max2[:, -1], max2[:, 0].astype(float), color=color, linestyle='-', linewidth=0.7)
    ax1.plot(flood1[:, -1], flood1[:, 0].astype(float), color='black',
             linestyle='-', marker='.', linewidth=1, label=lab_mean)
    ax1.plot(flood2[:, -1], flood2[:, 0].astype(float), color='black',
             linestyle='-', marker='.', linewidth=1)
    plt.fill_between(min1[:, -1], max1[:, 0], min1[:, 0], color=color, alpha=0.4)
    plt.fill_between(min2[:, -1], max2[:, 0], min2[:, 0], color=color, alpha=0.4, label=lab_range)
    legend = plt.legend(ncol=1, fontsize=l_size, title=legend_h, bbox_to_anchor=(0.042, 0.05), loc=3)
    plt.setp(legend.get_title(), fontsize=l_size)
    ax1.yaxis.grid(linestyle='--', alpha=0.3)
    plt.ylabel(ylabel1, fontsize=ax_size, labelpad=15)
    plt.xlabel(xlabel, fontsize=ax_size, labelpad=8)
    plt.xticks(fontsize=l_size)
    plt.yticks(fontsize=l_size)
    plt.ylim(lim_range)

    if max1.shape[1] > 2:
        ax2 = ax1.twinx()
        ax2.plot(flood1[:, -1], flood1[:, 1].astype(float), color='black',
                 linestyle='-', linewidth=1, label=lab_mean, alpha=0)
        plt.yticks(np.arange(0, 4.5, 0.5), np.arange(0, 4.5, 0.5), fontsize=l_size)
        plt.ylabel(ylabel2, fontsize=ax_size, labelpad=15)
        plt.ylim(-0.075, 3.83)
        plt.yticks(fontsize=l_size)
    plt.xticks([r + 0.005 for r in np.arange(0, 7, 0.5)], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                                                           'High', 'Mid', 'Low'], fontsize=l_size)
    plt.axvline(x=5.25, color='black', linestyle='--', linewidth=0.4)
    # plt.axvspan(0, 1, color='grey', alpha=0.05, lw=0)
    # plt.axvspan(4, 5, color='grey', alpha=0.05, lw=0)
    plt.savefig(save2, dpi=300, bbox_inches='tight')

def doPlot_maint(scenario, diff1, flood1, min1, max1, color, xlabel, ylabel1, ylabel2, title, save1, save2):
    '''create two different plots, first one for all flood scn seperately, second one for the mean of all flood scn and
    its range. the plots only show the maintenance effort scenarios'''
    # 1st plot: all flood scn
    floods = ['2 apart', '2 close', '2 med', '1 first', '1 middle', '1 last']
    fig = plt.figure(figsize=(19, 12))
    # fig.suptitle(title, fontsize=24, fontweight=1, color='black').set_position([.5, 0.94])
    palette = np.array(['#e3ce8d', '#db786c', '#8e729d', '#7ba6d0', '#7ba47b', '#8d8d8d'])

    ax1 = fig.add_subplot(1, 1, 1)
    for x in range(scenario):
        ax1.plot(diff1[x, :, 2], diff1[x, :, 0].astype(float), color=palette[x],
                 marker='.', linestyle='--', linewidth=1, label=floods[x])
    ax1.plot(flood1[:, -1], flood1[:, 0].astype(float), color='black',
             marker='o', linestyle='-', linewidth=1.25, label=lab_mean)

    legend = plt.legend(ncol=2, fontsize=l_size, title=legend_h, bbox_to_anchor=(0.042, 0.03), loc=3)
    plt.setp(legend.get_title(), fontsize=l_size)
    ax1.yaxis.grid(linestyle='--', alpha=0.3)
    plt.ylabel(ylabel1, fontsize=ax_size, labelpad=l_size)
    plt.xlabel(xlabel, fontsize=ax_size, labelpad=l_size)
    plt.yticks(fontsize=l_size)
    plt.xticks(fontsize=l_size)
    plt.ylim(lim_range)

    if max1.shape[1] > 2:
        ax2 = ax1.twinx()
        ax2.plot(max1[:, 2], flood1[:, 1].astype(float), color='maroon', alpha=0, linestyle='-.', marker='x')
        plt.ylabel(ylabel2, fontsize=ax_size, labelpad=15)
        plt.yticks(np.arange(0, 4.5, 0.5), np.arange(0, 4.5, 0.5), fontsize=l_size)
        plt.ylim(-0.075, 3.83)
        plt.yticks(fontsize=l_size)
    plt.xticks([r + 0.005 for r in range(0, 11)], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=l_size)

    plt.savefig(save1, dpi=475, bbox_inches='tight')

    # 2nd plot: range of flood scn
    fig = plt.figure(figsize=(19, 12))
    # fig.suptitle(title, fontsize=24, fontweight=1, color='black').set_position([.5, 0.94])
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(min1[:, -1], min1[:, 0].astype(float), color=color,
             linestyle='-', linewidth=0.7)
    ax1.plot(max1[:, -1], max1[:, 0].astype(float), color=color,
             linestyle='-', linewidth=0.7)
    ax1.plot(flood1[:, -1], flood1[:, 0].astype(float), color='black',
             linestyle='-', marker='.', linewidth=1, label=lab_mean)
    plt.fill_between(min1[:, -1], max1[:, 0], min1[:, 0], color=color, alpha=0.4, label=lab_range)
    ax1.yaxis.grid(linestyle='--', alpha=0.3)
    plt.ylabel(ylabel1, fontsize=ax_size, labelpad=l_size)
    plt.xlabel(xlabel, fontsize=ax_size, labelpad=l_size)
    plt.xticks(fontsize=l_size)
    plt.yticks(fontsize=l_size)
    plt.ylim(lim_range)
    legend = plt.legend(ncol=1, fontsize=l_size, title=legend_h, bbox_to_anchor=(0.042, 0.05), loc=3)
    plt.setp(legend.get_title(), fontsize=l_size)
    # get second y-axis
    if max1.shape[1] > 2:
        ax2 = ax1.twinx()
        ax2.plot(flood1[:, -1], flood1[:, 1].astype(float), color='maroon', linestyle='-.', linewidth=0, alpha=0)
        plt.ylabel(ylabel2, fontsize=ax_size, labelpad=15)
        plt.yticks(np.arange(0, 4.5, 0.5), np.arange(0, 4.5, 0.5), fontsize=l_size)
        plt.ylim(-0.075, 3.83)
        # plt.legend(ncol=1, fontsize=l_size, bbox_to_anchor=(0.042, 0.03), loc=3)
    plt.xticks([r + 0.005 for r in np.arange(0, 5.5, 0.5)], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=l_size)
    plt.savefig(save2, dpi=400, bbox_inches='tight')

def doPlot_loc(scenario, diff1, diff2, flood1, flood2, min1, min2, max1, max2, color, xlabel, ylabel1, ylabel2,
               title, save1, save2):
    '''create two different plots, first one for all flood scn seperately, second one for the mean of all flood scn and
    its range. the plots only includes the location scenarios'''
    # 1st plot: all flood scn
    floods = ['2 apart', '2 close', '2 med', '1 first', '1 middle', '1 last']
    fig = plt.figure(figsize=(19, 12))
    # fig.suptitle(title, fontsize=24, fontweight=1, color='black').set_position([.5, 0.94])
    palette = np.array(['#e3ce8d', '#db786c', '#8e729d', '#7ba6d0', '#7ba47b', '#8d8d8d'])

    ax1 = fig.add_subplot(1, 1, 1)
    for x in range(scenario):
        ax1.plot(diff2[x, :, 2], diff2[x, :, 0].astype(float), color=palette[x],
                 marker='.', linestyle='--', linewidth=1, label=floods[x])
    ax1.plot(flood2[:, -1], flood2[:, 0].astype(float), color='black',
             marker='o', linestyle='-', linewidth=1.3, label=lab_mean)
    ax1.plot(2, flood1[z, 0].astype(float), color='grey', marker='x', linestyle='-.', markersize=9, label=label_30)
    ax1.plot(1, flood1[z, 0].astype(float), color='grey', marker='x', linestyle='-', markersize=9)
    ax1.plot(0, flood1[z, 0].astype(float), color='grey', marker='x', linestyle='-', markersize=9)
    legend = plt.legend(ncol=2, fontsize=l_size, title=legend_h, bbox_to_anchor=(0.042, 0.03), loc=3)
    plt.setp(legend.get_title(), fontsize=l_size)
    ax1.yaxis.grid(linestyle='--', alpha=0.3)
    plt.ylabel(ylabel1, fontsize=ax_size, labelpad=l_size)
    plt.xlabel(xlabel, fontsize=ax_size, labelpad=l_size)
    plt.yticks(fontsize=l_size)
    plt.xticks(fontsize=l_size)
    plt.ylim(lim_range)
    plt.axhline(flood1[z, 0].astype(float), xmin=0.06, xmax=0.844, color='grey', linestyle='-.', linewidth=1.6, alpha=1)

    if max1.shape[1] > 2:
        ax2 = ax1.twinx()
        ax2.plot(flood1[:, -1], flood1[:, 1].astype(float), color='maroon', linestyle='-.', linewidth=0, alpha=0)
        plt.ylabel(ylabel2, fontsize=ax_size, labelpad=15)
        plt.yticks(np.arange(0, 4.5, 0.5), np.arange(0, 4.5, 0.5), fontsize=l_size)
        plt.ylim(-0.075, 3.83)
        plt.yticks(fontsize=l_size)

    plt.xticks(np.arange(0, 3, 1), ['High', 'Mid', 'Low'], fontsize=l_size)
    # plt.yticks(fontsize=l_size)
    plt.xlim(-.15, 2.4)
    plt.savefig(save1, dpi=300, bbox_inches='tight')

    # 2nd plot: range of flood scn
    fig = plt.figure(figsize=(19, 12))
    # fig.suptitle(title, fontsize=24, fontweight=1, color='black').set_position([.5, 0.94])
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.fill_between(min2[:, -1], max2[:, 0], min2[:, 0],color=color, alpha=0.4, label=lab_range)
    ax1.plot(min2[:, -1], min2[:, 0].astype(float), color=color, linestyle='-', linewidth=0.7)
    ax1.plot(max2[:, -1], max2[:, 0].astype(float), color=color, linestyle='-', linewidth=0.7)
    ax1.plot(flood2[:, -1], flood2[:, 0].astype(float), color='black',
             linestyle='-', marker='.', linewidth=1, label=lab_mean)
    ax1.plot(6.5, flood1[z, 0].astype(float), color='grey', linestyle='-.', marker='x', markersize=9, label=label_30)
    ax1.plot(6, flood1[z, 0].astype(float), color='grey', linestyle='-', marker='x', markersize=9)
    ax1.plot(5.5, flood1[z, 0].astype(float), color='grey', linestyle='-', marker='x', markersize=9)
    handles, labels = plt.gca().get_legend_handles_labels()     # change order of labels in legend
    order = [0, 2, 1]

    ax1.yaxis.grid(linestyle='--', alpha=0.3)
    plt.ylabel(ylabel1, fontsize=ax_size, labelpad=l_size)
    plt.xlabel(xlabel, fontsize=ax_size, labelpad=l_size)
    plt.xticks(np.arange(5.5, 7, 0.5), ['High', 'Mid', 'Low'], fontsize=l_size)
    plt.yticks(fontsize=l_size)
    plt.ylim(lim_range)
    plt.axhline(flood1[z, 0].astype(float), xmin=0.077, xmax=0.844, color='grey', linestyle='-.', linewidth=1.2, alpha=1)
    legend = plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol=1, fontsize=l_size,
                        title=legend_h, bbox_to_anchor=(0.042, 0.05), loc=3)
    plt.setp(legend.get_title(), fontsize=l_size)

    # get second y-axis
    if max1.shape[1] > 2:
        ax2 = ax1.twinx()
        ax2.plot(flood1[:, -1], flood1[:, 1].astype(float), color='maroon', linestyle='-.', marker='x',
                 linewidth=0, label=lab_mean2, alpha=0)
        plt.ylabel(ylabel2, fontsize=ax_size, labelpad=15)
        plt.yticks(np.arange(0, 4.5, 0.5), np.arange(0, 4.5, 0.5), fontsize=l_size)
        plt.ylim(-0.075, 3.83)
        # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol=1, fontsize=l_size,
        #                     title=legend_h, bbox_to_anchor=(0.042, 0.03), loc=3)
    plt.xlim(5.4, 6.7)
    plt.savefig(save2, dpi=300, bbox_inches='tight')

def doPlot_cross(crosssection, xlabel, ylabel, l_size, ax_size, title, save):
    '''sediment aggregation in four different downstream cross sections for analyze the remaining potential water depth'''
    palette = np.array(['#e3ce8d', '#ffa584', '#db786c', '#e796d8', '#8e729d', '#7ba6d0', '#198c8c',
                        '#7ba47b', '#bc856c', '#8d8d8d', '#383838'])
    fig = plt.figure(figsize=(19, 12))
    # fig.suptitle(title, fontsize=24, fontweight=1, color='black').set_position([.5, 0.94])
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(width, height1, color='black')
    plt.plot(width, height2, color='black')
    plt.plot(width, height3, color='black')
    plt.plot(width, height4, color='black')

    plt.text(44.5, height1[-1]+0.1, crosssection[0], color='black', fontsize=l_size)
    plt.text(44.5, height2[-1]+0.1, crosssection[1], color='black', fontsize=l_size)
    plt.text(44.5, height3[-1]+0.1, crosssection[2], color='black', fontsize=l_size)
    plt.text(44.5, height4[-1]+0.1, crosssection[3], color='black', fontsize=l_size)
    plt.text(0, height1[0]+0.1, crosssection[0], color='black', fontsize=l_size)
    plt.text(0, height2[0]+0.1, crosssection[1], color='black', fontsize=l_size)
    plt.text(0, height3[0]+0.1, crosssection[2], color='black', fontsize=l_size)
    plt.text(0, height4[0]+0.1, crosssection[3], color='black', fontsize=l_size)

    plt.axhline(fill_max1[0,1], xmin=0.2, xmax=0.8,color='#5d4535', linewidth=0.7, alpha=0.6)
    plt.axhline(fill_max1[5,1], xmin=0.22, xmax=0.78, color='#5d4535', linewidth=0.7, alpha=0.6)
    plt.axhline(fill_max1[10,1], xmin=0.24, xmax=0.76,color='#5d4535', linewidth=0.7, alpha=0.6)
    # plt.axhline(height1[0]-0.05, xmin=0.18, xmax=0.814,color='grey', linewidth=0.7, alpha=0.6)
    # plt.axhline(height2[0]-0.05, xmin=0.194, xmax=0.795,color='grey', linewidth=0.7, alpha=0.6)
    # plt.axhline(height3[0]-0.05, xmin=0.195, xmax=0.805,color='grey', linewidth=0.7, alpha=0.6)
    # plt.axhline(height4[0]-0.05, xmin=0.195, xmax=0.805,color='grey', linewidth=0.7, alpha=0.6)

    plt.text(16.55, fill_max1[0,1]-0.61, '2.7 m fill (0% maintenance effort)', alpha=1, color='#5d4535',
             fontsize=16, fontweight='bold')
    plt.text(16.5, fill_max1[5,1]-0.455, '1.6 m fill (50% maintenance effort)', alpha=1, color='#5d4535',
             fontsize=16, fontweight='bold')
    plt.text(16.455, fill_max1[10,1]-0.4555, '0.8 m fill (100% maintenance effort)', alpha=1, color='#5d4535',
             fontsize=16, fontweight='bold')

    plt.xlabel(xlabel, fontsize=ax_size, labelpad=10)
    plt.ylabel(ylabel, fontsize=ax_size, labelpad=10)
    # ax.yaxis.grid(linestyle='--', alpha=0.3)
    plt.yticks(fontsize=l_size)
    plt.xticks(fontsize=l_size)

    plt.savefig(save, dpi=300, bbox_inches='tight')
    print('cross sections plotted''\n')

def doPlot_cum_maint(scenario, cums, ylabel, title1, title2, save):
    '''create plot with 6 subplots for flood scenarios, each plot presenting the cumulative sediment yield over time'''
    # subplot for all flood scn
    maint = ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
    floods = ['(a) 2 apart', '(b) 2 close', '(c) 2 med', '(d) 1 first', '(e) 1 middle', '(f) 1 last']
    palette = np.array(['#e3ce8d', '#ffa584', '#db786c', '#e796d8', '#8e729d', '#7ba6d0', '#198c8c',
                         '#7ba47b', '#bc856c', '#8d8d8d', '#383838'])
    palette2 = np.array(['#e3ce8d', '#db786c', '#8e729d', '#7ba6d0', '#7ba47b', '#8d8d8d'])
    palette3 = doColors(scenario[0]-3)

    fig = plt.figure(figsize=(19, 12))
    # fig.suptitle(title1, fontsize=24, fontweight=1, color='black').set_position([.5, 0.965])
    # fig.text(0.5, 0.91, title2, ha='center', fontsize=ax_size, style='italic')
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    fig.text(0.52, 0.05, xlabel, ha='center', fontsize=ax_size)
    fig.text(0.006, 0.5, ylabel, va='center', rotation='vertical', fontsize=ax_size)

    for x in range(scenario[1]):
        ax = fig.add_subplot(3, 2, (x+1), sharey=ax)
        ax.text(.02, .9, floods[x], fontsize=l_size, color='black', transform=ax.transAxes)
        ax.text(.96, .75, 'Maintenance effort', fontsize=s_size, color=palette3[scenario[0]-6, x],
                transform=ax.transAxes, rotation=270)
        plt.subplots_adjust(left=0.075, bottom=0.1, right=0.99, top=0.99, wspace=0.17, hspace=0.02)
        for y in range(scenario[0]-3):
            ax.plot(cums[y, x, :], color=palette3[y, x], linestyle='-', linewidth=1)
            if y % 2 == 0:
                ax.text(100, cums[y, x, 99], maint[y], alpha=1, color=palette3[y, x], fontsize=s_size)
            else:
                ax.text(100, cums[y, x, 99], maint[y], alpha=1, color=palette3[y, x], fontsize=s_size)
            plt.xticks(np.arange(0, 110, 10), np.arange(0, 110, 10), fontsize=l_size)
            plt.yticks(np.arange(0, 600000, 75000), np.arange(0, 600000, 75000), fontsize=l_size)
            plt.xlim(-2, 113)
            plt.ylim(lim_range)
    plt.savefig(save, dpi=400, bbox_inches='tight', pad_inches=0)

    # individual plot for each flood scn, different colors for maintenance (flood colors)
    for x in range(scenario[1]):
        floods = ['(a) 2 apart', '(b) 2 close', '(c) 2 med', '(d) 1 first', '(e) 1 middle', '(f) 1 last']
        fig = plt.figure(figsize=(19, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.02, 0.95, floods[x], fontsize=ax_size, color='black', transform=ax.transAxes)
        for y in range(scenario[0]-3):
            ax.plot(cums[y, x, :], color=palette3[y, x], linestyle='-', linewidth=1)
            if y==scenario[0]-4:
                z=4
                ax.text(100, cums[scenario[0]-z, x, 99], maint[scenario[0]-z], alpha=1, color=palette3[10, x], fontsize=l_size)
                z=z+1
                ax.text(100, cums[scenario[0]-z, x, 99], maint[scenario[0]-z], alpha=1, color=palette3[9, x], fontsize=l_size)
                z=z+1
                ax.text(100, cums[scenario[0]-z, x, 99], maint[scenario[0]-z], alpha=1, color=palette3[8, x], fontsize=l_size)
                z=z+1
                ax.text(100, cums[scenario[0]-z, x, 99], maint[scenario[0]-z], alpha=1, color=palette3[7, x], fontsize=l_size)
                z=z+1
                ax.text(100, cums[scenario[0]-z, x, 99], maint[scenario[0]-z], alpha=1, color=palette3[6, x], fontsize=l_size)
                z=z+1
                ax.text(100, cums[scenario[0]-z, x, 99], maint[scenario[0]-z], alpha=1, color=palette3[5, x], fontsize=l_size)
                z=z+1
                ax.text(100, cums[scenario[0]-z, x, 99], maint[scenario[0]-z], alpha=1, color=palette3[4, x], fontsize=l_size)
                z=z+1
                ax.text(100, cums[scenario[0]-z, x, 99], maint[scenario[0]-z], alpha=1, color=palette3[3, x], fontsize=l_size)
                z=z+1
                ax.text(100, cums[scenario[0]-z, x, 99], maint[scenario[0]-z], alpha=1, color=palette3[2, x], fontsize=l_size)
                z=z+1
                ax.text(100, cums[scenario[0]-z, x, 99], maint[scenario[0]-z], alpha=1, color=palette3[1, x], fontsize=l_size)
                z=z+1
                ax.text(100, cums[scenario[0]-z, x, 99], maint[scenario[0]-z], alpha=1, color=palette3[0, x], fontsize=l_size)
        ax.yaxis.grid(linestyle='--', alpha=0.3)
        plt.ylabel(ylabel, fontsize=ax_size, labelpad=l_size)
        plt.xlabel(xlabel, fontsize=ax_size, labelpad=l_size)
        plt.yticks(fontsize=l_size)
        plt.xticks(np.arange(0, 110, 10), np.arange(0, 110, 10), fontsize=l_size)
        plt.ylim(lim_range)
        plt.xlim(-3, 106)
        floods = ['2 apart', '2 close', '2 med', '1 first', '1 middle', '1 last']
        plt.savefig('U:simulations/analysis/python/sed yield/CumSumIndplot'+floods[x]+'_diffcol2_maint.png', dpi=300,
                    bbox_inches='tight')

def doPlot_cum_loc(scenario, cums, ylabel, title1, title2, save):
    '''create plot with 6 subplots for flood scenarios, each plot presenting the cumulative sediment yield over time'''
    # subplot for all flood scn
    maint = ['High', 'Low', 'Mid']
    floods = ['(a) 2 apart', '(b) 2 close', '(c) 2 med', '(d) 1 first', '(e) 1 middle', '(f) 1 last']
    palette = np.array(['#e3ce8d', '#ffa584', '#db786c', '#e796d8', '#8e729d', '#7ba6d0', '#198c8c',
                         '#7ba47b', '#bc856c', '#8d8d8d', '#383838'])
    palette2 = np.array(['#e3ce8d', '#db786c', '#8e729d', '#7ba6d0', '#7ba47b', '#8d8d8d'])
    palette3 = doColors(scenario[0]-11)

    fig = plt.figure(figsize=(19, 12))
    # fig.suptitle(title1, fontsize=24, fontweight=1, color='black').set_position([.5, 0.965])
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    fig.text(0.52, 0.05, xlabel, ha='center', fontsize=ax_size)
    fig.text(0.006, 0.5, ylabel, va='center', rotation='vertical', fontsize=ax_size)
    # fig.text(0.5, 0.91, title2, ha='center', fontsize=ax_size, style='italic')

    for x in range(scenario[1]):
        ax = fig.add_subplot(3, 2, (x+1), sharey=ax)
        ax.text(0.02, .9, floods[x], fontsize=l_size, color='black', transform=ax.transAxes)
        ax.text(.96, .9, 'Maintenance location', fontsize=s_size, color=palette3[scenario[0]-11, x],
                transform=ax.transAxes, rotation=270)
        plt.subplots_adjust(left=0.075, bottom=0.1, right=0.99, top=0.99, wspace=0.17, hspace=0.02)
        for y in range(scenario[0]):
            ax.plot(cums[y, x, :], color=palette3[y*2, x], linestyle='-', linewidth=1)
            ax.text(100, cums[y, x, 99], maint[y], alpha=1, color=palette3[y*2, x], fontsize=s_size)
            plt.xlim(-2, 113)
            plt.ylim(lim_range)
            plt.xticks(np.arange(0, 110, 10), np.arange(0, 110, 10), fontsize=l_size)
            plt.yticks(np.arange(0, 600000, 75000), np.arange(0, 600000, 75000), fontsize=l_size)
    plt.savefig(save, dpi=400, bbox_inches='tight')

# -DEFINE GLOBAL VARIABLES HERE-
# font sizes for plot
ax_size = 18        # axis label
l_size = 16         # tick size, legend size
s_size = 12         # small in plot labels

# -READ IN FILES HERE-
# create list with paths from where to read in the files
loc = np.array(glob.glob('U:simulations/dem_reach/****/***/**/*.dat'))
main = np.array(glob.glob('U:simulations/dem_reach/***/**/*.dat'))
files = np.append(main, loc)

fill_mean = ('U:simulations/analysis/python/sed yield/mean_sedyield.csv')
fill_all = ('U:simulations/analysis/python/sed yield/all_sedyield.csv')
sum_sed = ('U:simulations/analysis/python/sed yield/sum.csv')

fill_mean = np.genfromtxt(fill_mean, delimiter=',', skip_header=1)  # mean of flood scenarios for downstream channel fill
fill_all = np.genfromtxt(fill_all, delimiter=',', skip_header=1)   # seperate flood scenarios for downstream channel fill
sum_sed = np.genfromtxt(sum_sed, delimiter=',', skip_header=1)    # sum of total sed yield (minus offset from spinoff part)

# -CALL FUNCTIONS HERE-
# ---------------------- 1 calculate sum of total sediment ----------------------
# read in water and sediment outputs
dat_data = doRead(files)

# calculate sum of sediment yield (2nd column) over 100 years (all rows) for all scenarios
sumsed = doSumSed(dat_data)

# WORK IN EXCEL: export 'sumsed' and subtract the sediment yield offset (model spinoff part (calculated in ArcGIS) from
# it. also calculate the potential fill in percentage and in meters. this is all done in excel. in a next step read in
# this newly calculated sum of the total sediment yield with its potential filling value ('sum_sed').

## ---------------------- 2.1 channel fill ----------------------
# create array for sorting and splitting of the data
fill_all1, fill_all2, fill_mean1, fill_mean2 = doArray(fill_all[:, (-2,-1)], fill_mean[:, (-2,-1)])

# calculate the min and the max of the flood scenarios
fill_min1, fill_min2, fill_max1, fill_max2 = doMinMax(fill_all[:,(-2,-1)])

# plot maint and loc scenarios combined
lim_range = -2, 102
titel_fill = "Downstream channel fill after 100 years of simulation"
xlabel = "                                        Maintenance effort [%]" \
         "                                                                 Location"
ylabel_fill1 = "Channel fill [%]"
ylabel_fill2 = "Channel fill [m]"
lab_mean = "Mean of hydrographs  for channel fill [%]"
lab_mean2 = "Mean of hydrographs for channel fill [m]"
lab_range = "Range of hydrographs for channel fill [%]"
lab_range2 = "Range of hydrographs for channel fill [m]"
label_30 = "30 % maintenance"
legend_h = "Hydrograph"
save_fill1 = 'U:simulations/analysis/python/sed yield/FillAll_maint+loc.png'
save_fill2 = 'U:simulations/analysis/python/sed yield/FillRange_maint+loc.png'

doPlot(fill_all1.shape[0], fill_all1, fill_all2, fill_mean1, fill_mean2, fill_min1, fill_min2,
       fill_max1, fill_max2, '#7ba6d0', xlabel, ylabel_fill1, ylabel_fill2,
       titel_fill, save_fill1, save_fill2)

# plot maint and loc scenarios separately
# maint scn
xlabel = "Maintenance effort [%]"
save_fill1 = 'U:simulations/analysis/python/sed yield/FillAll_maint.png'
save_fill2 = 'U:simulations/analysis/python/sed yield/FillRange_maint.png'
doPlot_maint(fill_all1.shape[0], fill_all1, fill_mean1, fill_min1, fill_max1, '#7ba6d0', xlabel, ylabel_fill1,
             ylabel_fill2, titel_fill, save_fill1, save_fill2)
# loc scn
z = -4          # corresponds to the 30 % maintenance value, which is unfortunately not always at the same position...
xlabel = "Maintenance location"
save_fill3 = 'U:simulations/analysis/python/sed yield/FillAll_loc.png'
save_fill4 = 'U:simulations/analysis/python/sed yield/FillRange_loc.png'
doPlot_loc(fill_all1.shape[0], fill_all1, fill_all2, fill_mean1, fill_mean2, fill_min1, fill_min2, fill_max1, fill_max2,
           '#7ba6d0', xlabel, ylabel_fill1, ylabel_fill2, titel_fill, save_fill3, save_fill4)

## ---------------------- 2.2 cross section ----------------------
# generate cross sections with spatial dimension values derived from geo.map.admin
width = np.arange(50)
height1 = np.array([4.25,4.25,4.25,4.25,4.25,4.25,4.25,4.25,4,3,2,1,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0.5,1,2,3,4,4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5])
height2 = np.array([5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5,5,4,3,2,1,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,1,2,3,
                    4,5,6,6,6,6,6,6,6,6,6])
height3 = np.array([5,5,5,5,5,5,5,5,5,4,3,2,1,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,1,2,3,4,5,5.5,5.5,5.5,
                    5.5,5.5,5.5,5.5,5.5])
height4 = np.array([3,3,3,3,3,3,3,3,3,2,1,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,1,2,3,4,4,4,4,4,4,
                    4,4])

# plot cross section
# define plot properties
title = 'Cross sections in downstream channel'
crosssection = ['Cross section 1', 'Cross section 2', 'Cross section 3', 'Cross section 4']
xlabel = 'Width [m]'
ylabel = 'Height [m]'
save = 'U:simulations/analysis/python/sed yield/Crossection.png'
doPlot_cross(crosssection, xlabel, ylabel, l_size, ax_size, title, save)

## ---------------------- 3 sed yield ----------------------
# create array for sorting and splitting of the data
sum_sed1, sum_sed2, fill_mean1, fill_mean2 = doArray(sum_sed[:, 1], fill_mean[:, (-2, -1)])

# calculate the min and the max of the flood scenarios
sum_min1, sum_min2, sum_max1, sum_max2 = doMinMax(sum_sed[:, 1])

# calculate the mean of the flood scenarios
sum_mean1, sum_mean2 = doFloodmean(sum_sed[:, 1])

# plot maint and loc scenarios combined
# define plot properties
lim_range = -10000, 510000
titel_sedsum = "Total sediment yield after 100 years of simulation"
xlabel = "                                        Maintenance effort [%]" \
         "                                                                 Location"
ylabel_sedsum = "Sum of sediment yield [m\u00b3]"
save_sedsum1 = 'U:simulations/analysis/python/sed yield/SedSum_maint+loc.png'
save_sedsum2 = 'U:simulations/analysis/python/sed yield/SedSumRange_maint+loc.png'

doPlot(sum_sed1.shape[0], sum_sed1, sum_sed2, sum_mean1, sum_mean2, sum_min1, sum_min2, sum_max1, sum_max2, '#5C9C88',
       xlabel, ylabel_sedsum, ylabel_fill2, titel_sedsum,save_sedsum1, save_sedsum2)

# plot maint and loc scenarios separately
# maint scn
xlabel = "Maintenance effort [%]"
save_sedsum1 = 'U:simulations/analysis/python/sed yield/SedSum_maint.png'
save_sedsum2 = 'U:simulations/analysis/python/sed yield/SedSumRange_maint.png'
doPlot_maint(sum_sed1.shape[0], sum_sed1, sum_mean1, sum_min1, sum_max1, '#5C9C88', xlabel, ylabel_sedsum, ylabel_fill2,
             titel_sedsum, save_sedsum1, save_sedsum2)

# loc scn
z = 3
xlabel = "Maintenance location"
save_sedsum3 = 'U:simulations/analysis/python/sed yield/SedSum_loc.png'
save_sedsum4 = 'U:simulations/analysis/python/sed yield/SedSumRange_loc.png'
doPlot_loc(sum_sed1.shape[0], sum_sed1, sum_sed2, sum_mean1, sum_mean2, sum_min1, sum_min2, sum_max1, sum_max2,
           '#5C9C88', xlabel, ylabel_sedsum, ylabel_fill2, titel_sedsum, save_sedsum3, save_sedsum4)

## ---------------------- 4 cum sum ----------------------
# calculate cumsum of four different maintenance scenarios (0%, 30%, 70%, 100%)
cumsum = doCumsum(dat_data)

# calculate the min and the max of the flood scenarios
cums_off = doSplit(cumsum)

# plot cumsum
# define plot properties
title_cum = "Cumulative sediment yield during 100 years of simulation "
title_maint = "Maintenance effort"
xlabel = "Time [years]"
ylabel_cum = "Cumulative sediment yield [m\u00b3]"
lim_range = -15000, 486000
save_cum = 'U:simulations/analysis/python/sed yield/CumSumSubplot_maint.png'
doPlot_cum_maint(cums_off.shape, cums_off, ylabel_cum, title_cum, title_maint, save_cum)


cums_off = cums_off[range(11, 14)]

title_loc = "Maintenance location"
save_cum = 'U:simulations/analysis/python/sed yield/CumSumSubplot_loc.png'
doPlot_cum_loc(cums_off.shape, cums_off, ylabel_cum, title_cum, title_loc, save_cum)
