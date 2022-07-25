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
new_colormap = td_colors + tt_colors + dark_colors +s_colors + td_colors
colorx = []

print(new_colormap)
print(len(new_colormap))

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
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
y = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
plt.plot(x,y,color='#6b6ecf')
plt.show()

bd = pd.read_csv('./dist/benchmarks.csv', encoding='utf-8', decimal=',', sep=';', engine='python')

bd = bd.sort_values('br')
groups = bd.groupby('li')
fig, ax = plt.subplots(figsize=(6.6,3))
x = bd['br']
y = bd['benchmarks']
lister = bd['li'].unique()
print(lister)
print(len(lister))
zipped = list(zip(new_colormap, lister))
print(zipped)

for name, group in groups:
       print(name)
       res = [x for (x, y) in zipped if y == name]
       res = res[0]
       ax.plot(group.br, group.benchmarks, marker='o', linestyle='', alpha= 0.5, ms=10, label=name, color = res)


bd = bd.groupby('br')['benchmarks'].mean().reset_index()
plt.plot(bd.br, bd.benchmarks, marker='x', linestyle='', color='black', ms=10, label='Mittelwert')
ax.legend(loc='lower left', ncol=6, bbox_to_anchor=(0,1.02,1,0.2), mode='expand', prop={'size': 8})
ax.margins(0.07)
plt.xticks(rotation=90)
plt.ylim((0,60))
plt.ylabel('Spezifischer\nEnergieverbrauch [Wh/tkm]', loc='bottom')
plt.savefig('benchmarks_überblick.pgf', format='pgf', bbox_inches='tight')
plt.show()