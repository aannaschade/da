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
mpl.rcParams['axes.xmargin'] = 0.03
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

################################################################ EP Fahrzeuge

df_one = pd.read_csv('./dist/ein_fahrer.csv', encoding='utf-8', decimal=',', sep=';', engine='python')
print(min(df_one['e_temp_ind']))
print(max(df_one['e_temp_ind']))

N = len(df_one)+1
ind = np.arange(1, N)
data = df_one['e_temp_ind']
benchmark= 19.100172815218315
plt.figure(figsize=(6.6, 3))
plt.bar(ind, data ,color = new_colormap[0])
plt.axhline(y=benchmark, color=new_colormap[1], linestyle='dotted',label='Gewichteter spezifischer Energieverbrauch des Fahrers')
plt.ylabel('Spezifischer\nEnergieverbrauch [Wh/tkm]')
plt.xticks([], [])
plt.xlabel('Einzelfahrten')
plt.legend(loc='lower left', bbox_to_anchor=(0,1.02,1,0.2))
plt.savefig('übersicht_ein_fahrer.pgf', format='pgf', bbox_inches='tight')
plt.show()