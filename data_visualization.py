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
#mpl.rcParams['axes.xmargin'] = 0.1
plt.rcParams['image.cmap'] = 'tab20b'

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

################################################################ EP Fahrzeuge

dfObj = pd.read_csv('./dist/data_vis.csv', encoding='utf-8', decimal=',', sep=';', engine='python')
lines = dfObj['nurlinie'].to_list()

dfn = dfObj.groupby(['nurbaureihe'])['econsumption', 'total_epotential'].sum().reset_index()
dfn = dfn.assign(relative=lambda x: x.total_epotential / x.econsumption)
dfn['total_epotential'] = dfn['total_epotential'].div(5 * 1000000).round(1)

print(dfn)
summa = dfn['total_epotential'].sum()
print(summa)

width = 0.35
labels = dfn['nurbaureihe']
x = np.arange(len(labels))
""" fig, ax1 = plt.subplots(figsize=(5.9,3))
ax2 = ax1.twinx()
rects1 = ax1.bar(x - width / 2, dfn['total_epotential'], width, label='absolut', color=new_colormap[0])
rects2 = ax2.bar(x + width / 2, dfn['relative'], width, label='relativ', color=new_colormap[1])
ax1.set_ylabel('Energieeinsparpotential [GWh]')
ax2.set_ylabel('Energieeinsparpotential [%]')
fig.legend(loc='upper right', bbox_to_anchor=(0.5, 1.0), ncol=2)

ab = dfn['relative']
ab = [round(num,3) for num in ab]
A_as_ticklabel = [f'{i*100:.1f}%' for i in ab]
print(A_as_ticklabel)
ax2.bar_label(rects2, labels=A_as_ticklabel, padding=3, rotation=90,  fontsize=8)

ax1.ticklabel_format(useOffset=False, style='plain', axis='y')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation =90)
ax1.bar_label(rects1, padding=3,rotation=90, fontsize=8)
ax1.axis(ymin=0, ymax=5)

ax2.axis(ymin=0, ymax=0.5)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.ylim(0,0.5)
plt.savefig('ep_baureihe.pgf', format='pgf', bbox_inches='tight', dpi=300)
plt.show() """


width = 0.35
part = []
ep_year = []
ep_year_pro = []
lines = ['Linie 1']
for i in lines:
    df_re50 = dfObj[dfObj['nurlinie'] == i].groupby('nurbaureihe')['econsumption', 'total_epotential'].sum().reset_index()
    df_re50['econsumption'] = df_re50['econsumption'].div(5 * 1000000).round(2)
    df_re50['total_epotential'] = df_re50['total_epotential'].div(5 * 1000000).round(2)
    df_re50 = df_re50.assign(relative=lambda x: x.total_epotential / x.econsumption)
    sum1 = df_re50['econsumption'].sum()
    sum2 = df_re50['total_epotential'].sum()
    divisi = sum2 / sum1
    df_re50.loc[len(df_re50.index)] = ['gesamt', sum1, sum2, divisi]
    ep_year.append(df_re50['total_epotential'].iloc[-1])
    ep_year_pro.append(df_re50['relative'].iloc[-1])


    # # create grouped barplot of df_re50
    # labels = df_re50['nurbaureihe']
    # x = np.arange(len(labels))
    # fig, ax1 = plt.subplots(figsize=(5.9,3))
    # ax2 = ax1.twinx()
    # rects1 = ax1.bar(x - width/2, df_re50['total_epotential'], width, label='absolut', color=new_colormap[0])
    # rects2 = ax2.bar(x + width/2, df_re50['relative'], width, label='relativ', color=new_colormap[1])
    # fig.legend(loc='upper right', bbox_to_anchor=(0.5, 1.0), ncol=2)
    #   #ax2.legend(loc='lower left', ncol=2, bbox_to_anchor=(0,1.02,1,0.2)) 
    # ax1.set_ylabel('Energieeinsparpotential [GWh]')
    # ax1.set_ylim([0,1.5])
    # ax2.set_ylabel('Energieeinsparpotential [%]')

    # ax1.bar_label(rects1, padding=3)
    # ax1.ticklabel_format(useOffset=False, style='plain', axis='y')
    # ax1.set_xticks(x)
    # ax1.set_xticklabels(labels)
    # ab = df_re50['relative']
    # ab = [round(num,3) for num in ab]
    # A_as_ticklabel = [f'{i*100:.1f}%' for i in ab]
    # print(A_as_ticklabel)
    # ax2.bar_label(rects2, labels=A_as_ticklabel, padding=3)
    # #ax2.bar_label(rects2, padding=3) # fmt='%1f.%%'
    # ax2.axis(ymin=0, ymax=0.4)
    # ax2.set_yticks(np.arange(0, 0.401, 0.1))
    # ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
    # ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    # plt.axhline(y=divisi, linestyle='dotted', color=new_colormap[2])
    # plt.savefig('ep_linie.pgf', format='pgf', bbox_inches='tight', dpi=300)
    # plt.show()


zippList = list(zip(lines, ep_year_pro, ep_year))



## Jährliche Entwicklung der Energieeffizienz der Fahrweise (Kapitel 8.1)
df_years = dfObj.groupby(['all_years', 'nurbaureihe'])['econsumption', 'total_epotential'].sum().reset_index()
df_years = df_years.assign(relative=lambda x: x.total_epotential / x.econsumption)
df_years.drop(df_years.columns[[2, 3]], axis=1, inplace=True)
df_years['relative'] = 1 - df_years['relative']
p_table = pd.pivot_table(df_years, values='relative', index=['all_years'], columns=['nurbaureihe'])
print(p_table)
p_df = p_table.reset_index()
#print(p_df)
p_df1 = p_df.drop(['all_years'], axis=1)
y_values = p_df1.iloc[-1].tolist()  # funktioniert
p_df = p_df1.pct_change(axis='rows', periods=4)

## Szenarienbildung und -berechnung (Kapitel 8.x)
scenarios = p_df1.pct_change(axis='rows').mean(axis=0).reset_index()
scenarios['starting_point'] = y_values
scenarios = scenarios.rename({0: 'average_improvement'}, axis=1)
scenarios['s1_2026'] = scenarios['starting_point'] + scenarios['average_improvement'] * 5  
scenarios.loc[scenarios['s1_2026'] < scenarios['starting_point'], 's1_2026'] = scenarios['starting_point']
scenarios['s1_2031'] = scenarios['s1_2026']  # fun
scenarios['s2_2026'] = scenarios['s1_2026'] + ((1 - scenarios['s1_2026']) * 0.1)  
scenarios['s2_2031'] = scenarios['s2_2026']
scenarios['s3_2026'] = scenarios['s2_2026']


def f(ref):
    if 0.90 < ref < 0.95:
        result = 0.02
    elif 0.80 < ref < 0.90:
        result = 0.05
    elif ref < 0.80:
        result = 0.10
    else:
        result = 1
    return result


scenarios['s3_2031'] = scenarios['starting_point'].apply(f)
#print(scenarios)
scenarios['s3_2031'] = scenarios['s3_2031'] + scenarios['s3_2026'] + 0.01
scenarios = scenarios
scenarios = scenarios.round(4)*100
scenarios = scenarios.drop(['average_improvement'], axis=1)
# pivot table from scenarios
# replace dots with comma in scenarios
print(scenarios)

# Trendszenarien Diagramm (Kapitel 8)
min_value = scenarios['starting_point'].idxmin()
min_values = scenarios.iloc[min_value].tolist()
min_values.pop(0)
values_s1 = [min_values[0], min_values[1], min_values[2]]
values_s2 = [min_values[0], min_values[3], min_values[4]]
values_s3 = [min_values[0], min_values[5], min_values[6]]
time = [2021, 2026, 2031]
#plt.plot(p_df['all_years'], p_df['Fzg 1'] * 100, alpha=0.5)
#plt.plot(time, values_s1, label='Worst-Case-Szenario', alpha=0.5)
#plt.plot(time, values_s2, label='Trendszenario', alpha=0.5)
#plt.plot(time, values_s3, label='Best-Case-Szenario', alpha=0.5)
#plt.show()

print(p_df)
labs = p_df.iloc[-1].tolist()
labs = [i * 100 for i in labs]
labs = [round(num, 1) for num in labs]
labs = [str(i) if i < 0 else '+' + str(i) for i in labs]
labs = [str(i) + '%' for i in labs]
print(labs)
x_values = [2021] * len(y_values)

################################################################################ Trendszarien
""" 
fig, (ax1, ax2) = plt.subplots(2, 1,
                               figsize=(6.6,3),
                               sharex=True,
                               gridspec_kw={
                                   'height_ratios': [1, 0.1]
                               })
fig.subplots_adjust(hspace=0.05)

time = [2021, 2026, 2031]
#ax1.set_xlim(left=2020, right=2031)
#ax1.set_xlim(left=2020, right=2031)

ax1.plot(p_df['all_years'], p_df['Fzg 13'], alpha=0.7, color='grey')
print(values_s1)
print(values_s2)
print(values_s3)
ax1.plot(time, values_s1, label='Worst-Case-Szenario', linestyle='-', color=new_colormap[0])
ax1.plot(time,values_s2, label='Trendszenario', linestyle=':', color=new_colormap[1])
ax1.plot(time,values_s3, label='Best-Case-Szenario', alpha=0.6,linestyle='--', color=new_colormap[2])

ax1.set_ylim(0.5, 1)  # outliers only
ax2.set_ylim(0, 0.1)  # most of the data
#limit range of y-axis to the data only

# remove x-axis line's between the two sub-plots
ax1.spines['bottom'].set_visible(False)  # 1st subplot bottom x-axis
ax2.spines['top'].set_visible(False)  # 2nd subplot top x-axis

# 1st x-axis: move ticks from bottom to top
ax1.xaxis.tick_top()
#ax1.tick_params(labeltop=False, length=0)  # no labels
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

texts = []
for i, txt in enumerate(labs):
             texts.append(ax1.text(x_values[i], y_values[i], labs))
             ax1.annotate(txt, (x_values[i], y_values[i]), size=8)
             ax2.annotate(txt, (x_values[i], y_values[i]), size=8)


ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1))
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1))

ax1.legend(loc='lower left', ncol=2, bbox_to_anchor=(0,1.02,1,0.2), mode='expand', prop={'size': 10})
plt.xlabel('Jahr')
ax1.set_ylabel('Energieeffizienz', loc='bottom')
a = [2017,2018,2019,2020,2021,2026,2031]
ax1.set_xticks(a)
ax1.set_xticklabels(ax1.get_xticks())
plt.xticks(rotation=90)
plt.savefig('trendszenario.pgf', bbox_inches='tight', dpi=300)
plt.show()


 """

################################################################################ 
#plt.figure(figsize=(6.6, 6))

#ax = p_table.plot(xticks=p_table.index, ylabel='Energieeffizienz', kind='line', xlabel='Jahr',
                  #style=['-', ':', '-.', '--', '-', ':', '-.', '--', '-',':', '-.', '--'], color=new_colormap)

#ax.legend(loc='lower left', ncol=5, bbox_to_anchor=(0, 1.02, 1, 0.2), mode='expand', prop={'size': 10})
#ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
#for i, txt in enumerate(labs):
#    plt.annotate(txt, (x_values[i]+0.5, y_values[i]), size=8)
#plt.show()
# plt.savefig('bewertung_wirksamkeit_maßnahmen.pgf', format='pgf', bbox_inches='tight')



########################################################## Entwicklung Energieeffizienz im UZ

def set_xmargin(ax, left=0.0, right=0.3):
    ax.set_xmargin(0)
    ax.autoscale_view()
    lim = ax.get_xlim()
    delta = np.diff(lim)
    left = lim[0] - delta*left
    right = lim[1] + delta*right
    ax.set_xlim(left,right)

lens = len(p_table.columns)
lins = ['-', ':', '-.', '--', '-', ':', '-.', '--', '-',':', '-.', '--',':']

fig, (ax1, ax2) = plt.subplots(2, 1,
                               figsize=(6.6,4.5),
                               sharex=True,
                               gridspec_kw={
                                   'height_ratios': [1, 0.1]
                               })
fig.subplots_adjust(hspace=0.05)

for i in range(lens):
  ax1.plot(p_table.iloc[:,i], color = new_colormap[i], linestyle=lins[i], label=p_table.columns[i])
  ax2.plot(p_table.iloc[:,i], color = new_colormap[i])

ax1.set_ylim(0.6, 0.93)  # outliers only
ax2.set_ylim(0, 0.05)  # most of the data
#limit range of y-axis to the data only

# remove x-axis line's between the two sub-plots
ax1.spines['bottom'].set_visible(False)  # 1st subplot bottom x-axis
ax2.spines['top'].set_visible(False)  # 2nd subplot top x-axis

# 1st x-axis: move ticks from bottom to top
ax1.xaxis.tick_top()
#ax1.tick_params(labeltop=False, length=0)  # no labels
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

#for i, txt in enumerate(labs):
#    ax1.annotate(txt, (x_values[i], y_values[i]), size=8)
#    ax2.annotate(txt, (x_values[i], y_values[i]), size=8)
print(x_values)
print(y_values)
set_xmargin(ax1, left=0.0, right=0.09)

ax1.annotate('+3.4%',(2021, 0.729903811322), size=7)
ax1.annotate('+2.5%',(2021, 0.8388589375707524), size=7)
ax1.annotate('-7.5%',(2021, 0.8077575880894481), size=7)
ax1.annotate('+1.2%',(2021, 0.8889612514385898), size=7)
ax1.annotate('+5.0%',(2021, 0.76577205862976118), size=7)
ax1.annotate('-0.3%',(2021, 0.87930682540817124), size=7)

ax1.annotate('-0.8%',(2021, 0.83036176740042194), size=7)
ax1.annotate('-5.1%',(2021, 0.8679703766361286), size=7)

ax1.annotate('-1.3%',(2021, 0.8495220249396399), size=7)
ax1.annotate('+7.5%',(2021, 0.66894955824189077), size=7)

ax1.annotate('+1.5%',(2021, 0.7894003142167835), size=7)
ax1.annotate('+0.7%',(2021, 0.8174218986202259), size=7)
ax1.annotate('+4.3%',(2021, 0.8581516622887626051), size=7)

#ax2.annotate('test',(2021, 0.8174218986202259), size=7)

ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1))
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1))

ax1.legend(loc='lower left', ncol=6, bbox_to_anchor=(0,1.02,1,0.2), mode='expand', prop={'size': 10})
plt.xlabel('Jahr')
plt.ylabel('Energieeffizienz')
plt.xticks(p_table.index)
#ax2.set_ylabel('Energieeffizienz', loc='center')
plt.ylabel('Energieeffizienz', loc='bottom')
plt.savefig('bewertung_wirksamkeit_maßnahmen.pgf', bbox_inches='tight', dpi=300)
plt.show()
