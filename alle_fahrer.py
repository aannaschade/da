import functools
import locale
from turtle import color
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import pingouin  # ergänzt
import researchpy  # ergänzt
import scipy
import seaborn as sns
from scipy import optimize
from scipy.stats import ttest_rel
from sklearn.cluster import DBSCAN
from sympy import *
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker
import locale

plt.rcParams['axes.formatter.use_locale'] = True
mpl.rcParams['axes.labelpad'] = 10
mpl.rcParams['axes.xmargin'] = 0.01
plt.rcParams['image.cmap'] = 'tab20b'
locale.setlocale(
    category=locale.LC_ALL,
    locale="German"  # Note: do not use "de_DE" as it doesn't work
)
from adjustText import adjust_text


plt.rcParams['hatch.linewidth'] = 0.8

import matplotlib.colors as cm
all_colors = [cm.to_hex(plt.cm.tab20b(i)) for i in range(20)]
dark_colors = all_colors[::4]
s_colors = all_colors[1::4]
td_colors = all_colors[2::4]
tt_colors = all_colors[3::4]
new_colormap = td_colors + tt_colors + dark_colors +s_colors
colorx = []

def swapPositions(list, pos1, pos2):
     
    # popping both the elements from list
    first_ele = list.pop(pos1)  
    second_ele = list.pop(pos2-1)
    
    # inserting in each others positions
    list.insert(pos1, second_ele) 
    list.insert(pos2, first_ele) 
    return list

pos1, pos2  = 1, 2
swapPositions(new_colormap, pos1-1, pos2-1)
my_cmap = ListedColormap(new_colormap, name='my_colormap_name')

# apply German locale settings
locale.setlocale(locale.LC_ALL, 'de_DE')

################################################################ 

df32 = pd.read_csv('./dist/alle_fahrer.csv', encoding='utf-8', decimal=',', sep=';', engine='python')

plt.figure(figsize=(6.6, 3))
ypos = np.arange(len(df32['ec_driver']))
plt.ylabel('Spezifischer\nEnergieverbrauch [Wh/tkm]')
plt.xlabel('Fahrer')
barlist = plt.bar(ypos, df32['ec_driver'],  width=0.5, color=new_colormap[0])
barlist[0].set_color(new_colormap[1])
plt.xlabel('Fahrer')
plt.xticks([], [])

plt.legend(['Benchmark'], loc='lower left', ncol=2, bbox_to_anchor=(0,1.02,1,0.2))
plt.show() 

################################################################
print(len(df32['ec_driver']))
print(max(df32['ec_driver']))
print(min(df32['ec_driver']))

a = df32['ec_driver']
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(6.6,3),
                               sharex=True,
                               gridspec_kw={
                                   'height_ratios': [1, 0.1]
                               })
fig.subplots_adjust(hspace=0.05)
ypos = np.arange(len(df32['ec_driver']))

ax1.bar(ypos, a, color=new_colormap[0], width=0.7, label='Benchmark')
ax2.bar(ypos, a, color=new_colormap[0], width=0.7, label='Benchmark')

ax1.set_ylim(17, 28)  # outliers only
ax2.set_ylim(0, 2)  # most of the data
#limit range of y-axis to the data only

# remove x-axis line's between the two sub-plots
ax1.spines['bottom'].set_visible(False)  # 1st subplot bottom x-axis
ax2.spines['top'].set_visible(False)  # 2nd subplot top x-axis

# 1st x-axis: move ticks from bottom to top
ax1.xaxis.tick_top()
# 2nd x-axis: ticks on the bottom
ax2.xaxis.tick_bottom()

# 1st subplot y-axis: remove first tick
ax1.set_yticks(ax1.get_yticks()[1:])
# 2nd subplot y-axis: remove the last
ax2.set_yticks(ax2.get_yticks()[:-1])

# now draw the cut
d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(
    marker=[(-1, -d), (1, d)],
    markersize=12,  # "length" of cut-line
    linestyle='none',
    mec='k',  # ?
    mew=1,  # line thickness
    clip_on=False
)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs, rasterized=True)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs, rasterized=True)

ax1.set_xticks([], [])
ax1.get_children()[0].set_color(new_colormap[1])
ax2.get_children()[0].set_color(new_colormap[1])


ax1.legend(loc='lower left', ncol=2, bbox_to_anchor=(0,1.02,1,0.2), prop={'size': 10})
ax1.set_ylabel('Gewichteter spezifischer\nEnergieverbrauch [Wh/tkm]')
ax2.set_xlabel('Fahrer')
#plt.legend(['Benchmark'], loc='lower left', ncol=2, bbox_to_anchor=(0,1.02,1,0.2))
plt.savefig('übersicht_alle_fahrer.pgf', format='pgf', bbox_inches='tight')
plt.show()


