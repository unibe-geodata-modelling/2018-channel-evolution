import numpy as np
import glob, os
import numpy as np
import matplotlib.pyplot as plt

# packages:
# python 2.7.0
# numpy 1.15.0
# matplotlib 2.2.2

# settings
load_max_files = 66
year = 8760
max_rows = year*100  # 8760 -> 1y; 17520 -> 2y;  35040 -> 4y
load_cols = range(4, 14)
skiprows = 2
use_cols = range(0, 9)
max_rows = max_rows / skiprows

# declare custom objects

# creating a template to store the grainsize data sets. The template has a parameter for name of the set and its data.
class GrainsizeSet(object):
    name = ""
    data = []


# create template to store information about grainsize name, scenario and the max value
class Grainsize(object):
    name = ""
    scenario = ""
    max_grainsize = 0

labels = []
class Label(object):
    name = ""
    color = ""

# declare global variables
selected_files = []
index = 0
# declare a list to store all gainsize related max values
grainsizes = []

# chart style
defaultFontSize = 16
headerFontSize = 20
numberFontSize = 11
legendFontSize = 12


# define labels and colors for each column
label = Label()
label.name = "0.0001m"
label.color = "fuchsia"
labels.append(label)
label = Label()
label.name = "0.005m"
label.color = "purple"
labels.append(label)
label = Label()
label.name = "0.02m"
label.color = "blue"
labels.append(label)
label = Label()
label.name = "0.04m"
label.color = "cyan"
labels.append(label)
label = Label()
label.name = "0.08m"
label.color = "green"
labels.append(label)
label = Label()
label.name = "0.12m"
label.color = "gold"
labels.append(label)
label = Label()
label.name = "0.2m"
label.color = "orange"
labels.append(label)
label = Label()
label.name = "0.4m"
label.color = "red"
labels.append(label)
label = Label()
label.name = "1.5m"
label.color = "black"
labels.append(label)

# Look for dat files
for root, dirs, files in os.walk("C:\LocalDrive\Seminar_DMA2\input"):

    # Get data for each file
    for file in files:
        if file.endswith(".dat") and (index < load_max_files):

            # increment index in order to compare with load max files setttings
            index += 1

            # get filename
            filename = os.path.join(root, file)
            print("Loading: " + filename)

            # load data
            data = np.genfromtxt(filename, delimiter=' ', max_rows=max_rows*skiprows, usecols=load_cols)

            # filter data set
            data = data[0::skiprows]

            # create new grainsize set
            grainsize_set = GrainsizeSet()
            grainsize_set.name = os.path.basename(file)

            print ("Processing: " + grainsize_set.name)

            # load one column
            column = np.array(data)
            first_row_data = data[:,0]
            # calculate percentage for all grainsize columns compared to the total column
            calculated = column / first_row_data[:,None] * 100

            # declare figure for plot charts
            fig, ax = plt.subplots()
            fig.set_size_inches(30.5, 15.5)

            # chart specific settings
            title = 'Absolute grain-size distribution during scenario ' + grainsize_set.name[:-4]

            # create new plot with absolute data (not calculated data=relative data)
            x = np.arange(max_rows)

            plt.ylabel('Absolute amount of certain grain-size', fontsize=defaultFontSize)
            plt.yticks(fontsize=numberFontSize)
            plt.xlabel('years', fontsize=defaultFontSize)
            plt.title(title, fontsize=headerFontSize)
            plt.xticks(np.arange(0, max_rows, step=year/skiprows), np.arange(0, max_rows / (year/skiprows)), fontsize= numberFontSize, rotation ='vertical')
            #plt.xticks(fontsize = labelFontsize, position = 'vertical')
            # add data
            for count in use_cols:
                # skip the first column with total values
                y = data[:, count+1]
                ax.plot(x, y, color=labels[count].color, label=labels[count].name)

            # add legend
            ax.legend(loc=1, fontsize = legendFontSize, title= "Grain-size fractions")

            # save as image
            plt.savefig(os.path.join("C:\LocalDrive\Seminar_DMA2\output", "absolute-" + grainsize_set.name + ".png"))

            # declare figure for plot charts
            fig, ax = plt.subplots()
            fig.set_size_inches(30.5, 15.5)

            # chart specific settings
            title = 'Relative grain-size distribution during scenario ' + grainsize_set.name[:-4]

            # create new plot with relative data (calculated data)
            x = np.arange(max_rows)

            plt.ylabel('% of certain grain-size', fontsize=defaultFontSize)
            plt.yticks(np.arange(0, 101, 10), fontsize = numberFontSize)
            plt.xlabel('years', fontsize=defaultFontSize)
            plt.title(title, fontsize=headerFontSize)
            plt.xticks(np.arange(0, max_rows, step=year / skiprows), np.arange(0, max_rows / (year / skiprows)), fontsize= numberFontSize, rotation ='vertical')


            # add data
            for count in use_cols:
                # skip the first column with total values
                y = calculated[:, count+1]
                ax.plot(x, y, color=labels[count].color, label=labels[count].name)

            # add legend
            ax.legend(loc=1, fontsize= legendFontSize, title= "Grain-size fractions")

            # save as image
            plt.savefig(os.path.join("C:\LocalDrive\Seminar_DMA2\output", "relative-" + grainsize_set.name + ".png"))


            plt.close('all')

            # get max value for each grainsize column
            for count in use_cols:
                grainsize = Grainsize()
                grainsize.name = labels[count-1].name
                grainsize.scenario = grainsize_set.name
                grainsize.max_grainsize = np.max(data[:, count])
                grainsizes.append(grainsize)


print("finished")
