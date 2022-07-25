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

locale.setlocale(locale.LC_NUMERIC, "de_DE")
plt.rcParams['axes.formatter.use_locale'] = True
mpl.rcParams['axes.labelpad'] = 10
mpl.rcParams['axes.xmargin'] = 0
plt.rcParams['image.cmap'] = 'tab20b'
locale.setlocale(
    category=locale.LC_ALL,
    locale="German"  # Note: do not use "de_DE" as it doesn't work
)

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


#################### Datenimport
df1 = pd.read_csv('./dist/data_clean.csv', encoding='utf-8', decimal=',', sep=';', engine='python')
all_lines = df1['line'].unique().tolist()

### fehlende Werte ergänzen

## nicht hinterlegte Fzg-Massen ergänzen
# Kombination Fahrzeugtypen
df11 = df1[['traintype', 'trainset', 'mass']].drop_duplicates(keep='first')
df11 = df11[df11['mass'] > 0]
df11 = df11.groupby(['traintype', 'trainset']).mean().reset_index()
df3 = pd.merge(df1, df11, on=['traintype', 'trainset'], how='left')
df3 = df3.assign(spec_econ_mass=lambda x: (x.econsumption * 1000) / (x.distance * x.mass_y))


#### Ausreißer über gesamten Datensatz erkennen und entfernen oder korrigieren
#df3['train'] = df3['traintype'].map(str) + '+' + df3['trainset'].map(str)
stds = 3
df3['outliers'] = df3[['line', 'train', 'spec_econ_mass']].groupby(['line', 'train']).transform(
    lambda group: (group - group.mean()).abs().div(group.std())) > stds
df3 = df3[~df3.outliers]
print(len(df1))


# pro Abfahrtsort am Tag Durchschnittstemperatur
df3['temperature'] = df3['temperature'].fillna(5) 

# Dataframe für eine Linie und einen Zugtyp
df4 = df3[['line', 'train', 'driver', 'mass_y', 'distance', 'spec_econ_mass', 'temperature', 'econsumption']]

# Clusteralgorithmus DBSCAN bei Linie 11 anwenden
traintys = []
data_s1msdb = []
df_s1mdsb = df4[df4['line'] == 'Linie 11']
traintys = df_s1mdsb['train'].unique().tolist()
for n in traintys: 
        dfdbscan = df_s1mdsb[df_s1mdsb['train'] == n]
        if len(dfdbscan) > 3500:
          dbscan_cluster = DBSCAN(eps=2.4, min_samples=150)
          X = dfdbscan[['temperature','spec_econ_mass']].to_numpy()
          dbscan_cluster.fit(X)
          #plt.figure(figsize=(3,3))
          #plt.ylim(0,75)
          #plt.scatter(dfdbscan['temperature'], dfdbscan['spec_econ_mass'], rasterized=True, s = 0.3, alpha=0.2, color=new_colormap[0])
          # xticks with distance 10
          #plt.xticks(np.arange(-15, 40, 15))
          #plt.xlabel("Außentemperatur [°C]")
          #plt.ylabel("Spezifischer\nEnergieverbrauch [Wh/tkm]")
          #plt.savefig('dbscan_davor.pgf', format='pgf', bbox_inches='tight', dpi=300)
          #plt.show()
          labels=dbscan_cluster.labels_
          N_clus=len(set(labels))-(1 if -1 in labels else 0)
          n_noise = list(dbscan_cluster.labels_).count(-1)
          # Trennung der Cluster
          dfdbscan['cluster'] = dbscan_cluster.labels_
          dfdbscan.loc[(dfdbscan.cluster == 0), 'line'] = 'Linie 11_0'
          dfdbscan.loc[(dfdbscan.cluster == 1), 'line'] = 'Linie 11_1'
          i = [-1,0,1]
          #plt.figure(figsize=(3,3))

          for j in i:
            db = dfdbscan[(dfdbscan.cluster == j)]
           # plt.scatter(db['temperature'], db['spec_econ_mass'], rasterized=True, s = 0.3, alpha=0.2, color=new_colormap[j+1], label='Cluster '+str(j+1))
          #leg = plt.legend(['Rauschen','Cluster 2','Cluster 1'], loc='lower left', ncol=2, bbox_to_anchor=(0, 1.02, 1, 0.2), mode='expand', markerscale=15)
          #colors = [new_colormap[0], new_colormap[1], new_colormap[2]]
          #for i, j in enumerate(leg.legendHandles):
          #  j.set_color(colors[i])  	
          #plt.ylim(0,75)
          #plt.xticks(np.arange(-15, 40, 15))
          #plt.xlabel("Außentemperatur [°C]")
          #plt.savefig('dbscan_danach.pgf', format='pgf', bbox_inches='tight', dpi=300)
          #plt.show()

          stds = 2.5
          dfdbscan['outliers_'] = dfdbscan[['line','train','spec_econ_mass']].groupby(['line', 'train']).transform(lambda group: (group - group.mean()).abs().div(group.std())) > stds
          dfdbscan = dfdbscan[~dfdbscan.outliers_]
          data_s1msdb.append(dfdbscan)

data_s1msdb = pd.concat(data_s1msdb)

df_nex = df4.merge(data_s1msdb.drop_duplicates(),
                   on=['train', 'driver', 'temperature', 'spec_econ_mass', 'mass_y', 'distance'], how='left',
                   indicator=True)
df_nex['line_x'] = df_nex.apply(
    lambda x: x['line_y'] if x['line_y'] == 'Linie 4_0' or x['line_y'] == 'Linie 4_1' else x['line_x'], axis=1)
df_nex = df_nex.rename(columns={'line_x': 'line'})

all_lines.extend(['Linie 11_0', 'Linie 11_1'])

lines = ['Linie 1']  #all_lines
df_nw = df_nex[df_nex['line'].isin(lines)]  # Linienauswahl aus Liste heraus
dfs = dict(tuple(df_nw.groupby('train')))
listes = pd.unique(df_nw['train']).tolist()

# Funktion definieren
def polyfit(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    # calculate r-squared
    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results = ssreg / sstot
    return round(results, 4)


dataframelist = []

for i in lines:
    df222 = df_nw[df_nw['line'] == i]
    listes = pd.unique(df222['train']).tolist()

    for trains in listes:
        df223 = df222[df222['train'] == trains]

        if len(df223['train']) > 3500:
            df_t = df223
            print(i)
            print(trains)

            # Streudiagramm spez. Energieverbrauch in Abhängigkeit der Außentemperatur
            x = df_t['temperature']
            y = df_t['spec_econ_mass']
            #plt.figure(figsize=(6.6,3))
            #plt.scatter(x, y, s = 0.3, alpha = 0.2, rasterized=True, color = new_colormap[0])
            #plt.xlabel("Außentemperatur [°C]")
            #plt.ylabel("Spezifischer\nEnergieverbrauch [Wh/tkm]")
            #plt.ylim(0, 40)
            #plt.savefig('temperaturverlauf_entwurf.pgf', format='pgf', bbox_inches='tight', dpi=300)
            #plt.show()

            # quadratische Regressionsfunktion
            model = np.poly1d(np.polyfit(x, y, 2))
            polyline = np.linspace(-20, 40, 50)
            print(polyfit(x, y, 2))
            #plt.plot(polyline, model(polyline), label=model, color=new_colormap[1])
            minimum = scipy.optimize.minimize(model, x0=0)
            minimum.y = model(minimum.x)
            #plt.vlines(x=minimum.x, ymin=0, ymax=minimum.y, colors=new_colormap[1], ls=':')
            #plt.xticks(np.arange(-15, 40, 15))
            #plt.savefig('regression.pgf', format='pgf', bbox_inches='tight', dpi=300)
            #plt.show()

            ### Trennung der Anteile umsetzen
            df_t = df_t.assign(e_temp_ind=lambda x: x.spec_econ_mass - (model(x.temperature) - model(minimum.x)), \
                               e_temp_dep=lambda x: model(x.temperature) - model(minimum.x))
            df_t = df_t.assign(e_temp_ind_abs=lambda x: x.e_temp_ind * (x.distance * x.mass_y) / 1000, \
                               e_temp_dep_abs=lambda x: x.e_temp_dep * (x.distance * x.mass_y) / 1000)

            # temperaturunabhängiger Anteil Fahrgastkomfort
            comfort = df_t['e_temp_dep_abs'].sum()
            drive = df_t['e_temp_ind_abs'].sum()
            d = comfort / (drive + comfort)
            mass = df_t['mass_y'].unique()
            reference = df_t['e_temp_ind'].mean()
            reference = reference * 0.1 * mass / 1000

            df_t = df_t.assign(e_temp_dep=lambda x: x.e_temp_dep + reference, \
                               e_temp_ind=lambda x: x.e_temp_ind - reference)
            
            #plt.figure(figsize=(6.6,3))
            a = df_t['temperature']
            weights = np.ones_like(a) / len(a)
            binBoundaries = np.linspace(0,-5,15,20)
            plt.figure(figsize=(6.6,3))
            plt.hist(a, weights=weights, bins = 30, color=new_colormap[0], rwidth=0.7, rasterized=True)
            plt.xlabel("Außentemperatur [°C]")	
            plt.ylabel("Relative Häufigkeitsdichte")
            plt.xlim((-20, 40))
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            plt.savefig('histogramm.pgf', format='pgf', bbox_inches='tight', dpi=300)
            plt.show()
    
            #plt.figure(figsize=(6.6,3))
            #plt.scatter(df_t['temperature'], df_t['e_temp_ind'], s=0.3, alpha=0.2, color = new_colormap[0], rasterized=True)
            #plt.xlabel("Außentemperatur [°C]")
            #plt.ylabel("Spezifischer\nEnergieverbrauch [Wh/tkm]")
            #plt.ylim((0, 40))
            #plt.savefig('bereinigt.pgf', format='pgf', bbox_inches='tight', dpi=300)
            #plt.show()

            dataframelist.append(df_t)
            # Tabelle Anteile Fahrgastkomfort
            start = -20
            step = 10
            num = 6
            a = np.arange(0, num) * step + start
            dict = {'a': a}
            df = pd.DataFrame(dict)
            df = df.assign(e_temp_dep=lambda x: (model(x.a) - model(minimum.x)) * mass / 1000)
            df['e_temp_dep'] = df['e_temp_dep'].round(3)
            df = df.transpose()
            #print(df.to_latex(index=False))

df_final = pd.concat(dataframelist, axis=0, ignore_index=True)

################# Fahrweise
df3 = df3[['train', 'temperature', 'mass_y', 'distance', 'driver', 'spec_econ_mass', 'econsumption', 'year', 'weekday',
           'ref_time', 'abweichung_fahrplan', 'latestart', 'lateend', 'delay_reduction']]
dfstart = pd.merge(df_nex, df3, on=['train', 'driver', 'spec_econ_mass', 'temperature', 'distance', 'mass_y'],
                   how='left', indicator='value')  # drop bei merge die die nur rechts sind
df33 = pd.merge(df_final, dfstart,
                on=['line', 'train', 'driver', 'spec_econ_mass', 'temperature', 'distance', 'mass_y'], how='left')
# print column names from df33

# je baureihe in line
lister = df33.groupby(['line', 'train']).size().reset_index()
lister['combined'] = lister[['line', 'train']].values.tolist()
listxxx = lister['combined'].tolist()

years = df3['year'].unique().tolist()

all_years = []
dataframepool = []
linie = []
total_epotential = []
nurlinie = []
econsumption = []
nurbaureihe = []
list_rest = []
benchmarks = []
br = []
li = []
for i in listxxx:
    print(i)  # ab hier
    df44 = df33[(df33['line'] == i[0]) & (df33['train'] == i[1])]
    list_rest.append(df44)
    df2 = df44.groupby(['driver']).agg({'distance': 'sum', 'e_temp_ind': 'size'}).reset_index()

    df3a = pd.merge(df44, df2, on=['driver'], how='left')
    df3a = df3a[df3a['e_temp_ind_y'] >= 10]
    if len(df3a.index) == 0:
        continue
    df3a = df3a.assign(ec_driver=lambda x: x.e_temp_ind_x * (x.distance_x / x.distance_y))
    df32 = df3a.groupby('driver')['ec_driver'].sum().reset_index().sort_values('ec_driver')
    df32.to_csv('./dist/alle_fahrer.csv', index=False, sep=';', encoding='utf-8')

    benchmark = df32['ec_driver'].min()
    print(benchmark)
    benchmarks.append(benchmark)
    br.append(i[1])
    li.append(i[0])

    b_drivers = df32['driver'].iloc[0:5].to_list()
    b_driver = df32['driver'].iloc[0]


    # Diagramm bester Fahrer
    df_one = df44[df44['driver'] == b_driver]
    df_one.to_csv('./dist/ein_fahrer.csv', index=False, sep=';', encoding='utf-8')
    xxx = df_one['temperature'].unique().size.mean().head()
    df3act = df44.assign(ec_pot=lambda x: (x.e_temp_ind - benchmark) * (
                (x.distance * x.mass_y) / 1000))  # achtung hier ggf. auch negative Werte, ev_pot in kWh
    # print df3act und dann bei total ep ggf. alle Fahrten mit Verspäteter Abfahrt und Verspätungsabbau rauslöschen, consumption bleibt die selbe
    for x in years:
        dfr = df3act[(df3act['year'] == x)]
        total_econsumption = dfr['econsumption'].sum()
        econsumption.append(total_econsumption)
        # Berechnung Einsparpotential unter den Bedingungen von Pünktlichkeit
        total_ep = dfr.loc[
            (dfr['ec_pot'] > 0) & (dfr['latestart'] <= 2) & (dfr['delay_reduction'] <= 3), 'ec_pot'].sum()
        total_epotential.append(total_ep)
        linie.append(i)
        nurlinie.append(i[0])
        nurbaureihe.append(i[1])
        all_years.append(x)

    # Bewertung der EE gemessen am besten - 100 % Energieeffizienz des Fahrers
    # df32 = df32.assign(ranking_ee = lambda x: reference_min / x.ec_driver)
    # df32 = df32[df32['ec_driver'].round(4)]
    # print(df32) # gibt Fahrer, ec_driver und ranking_ee aus - Ergebnisse von ranking_ee mit linie und baureihe je fahrer speichern
    # df32 = df32.drop(['ec_driver'], axis=1)
    # str1 = ','.join(i)
    # df325 = df32.rename(columns={'ranking_ee': str1})
    # dataframepool.append(df325) # liste von Dataframes mit EE-Ranking-Werten der Fahrer

    ########## Auslastung (kapitel 5.2)
    #
    df_load = df44[
        ['line', 'train', 'weekday', 'driver', 'ref_time', 'abweichung_fahrplan', 'latestart', 'lateend', 'e_temp_ind']]
    days_of_week = ['MO', 'DI', 'MI', 'DO', 'FR']


    def load(df_load):
        if (df_load['weekday'] in days_of_week) and (5 <= df_load['ref_time'] <= 9) or (
                14 <= df_load['ref_time'] <= 18):  # anpassen an in ausarbeitung genannte zeiten
            return 'peak'
        else:
            return 'off-peak'


    df_load['load'] = df_load.apply(load, axis=1)

    #
    df333 = df_load.groupby(['driver', 'load'])['e_temp_ind'].agg(['mean', 'size']).reset_index()
    df333 = df333[df333['size'] >= 5]
    num = df333['driver'].value_counts()
    df333 = df333[df333['driver'].isin(num[num == 2].index)]
    df333 = df333.rename(columns={'mean': 'e_temp_ind'})
    list_offpeak = df333.loc[df333['load'] == 'off-peak', 'e_temp_ind'].to_list()
    list_peak = df333.loc[df333['load'] == 'peak', 'e_temp_ind'].to_list()

    Z = list(zip(list_offpeak, list_peak))


    def takeSecond(elem):
        return elem[1]


    Z.sort(key=takeSecond)
    # plt.figure(figsize=(6.6,3))
    xs = [x[0] for x in Z]
    ys = [x[1] for x in Z]
    # plt.plot(ys)
    # plt.legend(['NVZ', 'HVZ'])
    # plt.xlabel('Fahrernummer')
    # plt.ylabel('mittlerer spezifischer Energieverbrauch [Wh/tkm]')
    # plt.show()


    # Durchführung T-Test für abhängige Stichproben
    stat, p = ttest_rel(list_offpeak, list_peak)
    list_offpeak = pd.Series(list_offpeak)
    list_peak = pd.Series(list_peak)
    des, res = researchpy.ttest(list_offpeak, list_peak, 'off-peak', 'peak', paired=True)
    # print(des.to_latex())
    # print(res)
    # Berechnung CLES
    if len(list_offpeak) > 1:
        effect_size = pingouin.compute_effsize(list_offpeak, list_peak, eftype='CLES')
        # print(effect_size)

    ########## Pünktlichkeit (Kapitel 5.3)

    ## Histogramm verspätete Abfahrt in Minuten - Anzahl der Fahrten
    #plt.figure(figsize=(2.8,3))
    a = df44['latestart']
    weights = np.ones_like(a) / len(a)
    #plt.hist(a, weights=weights, bins= 500, color=new_colormap[0])
    #plt.yticks(np.arange(0, 0.9, 0.2))
    #plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    #plt.ylabel('Relative Häufigkeitkeitsdichte')
    #plt.xlabel('Abweichungen zwischen\nAbfahrts- und Ankunftszeit [min]')
    #plt.xlim((-1, 13))
    #plt.savefig('verspätete-abfahrt.pgf', format='pgf', bbox_inches='tight', dpi=300)
    #plt.show()

    ## Histogramm Verspätungsabbau
    #plt.figure(figsize=(2.8,3))
    a = df44['delay_reduction']
    weights = np.ones_like(a) / len(a)
    binBoundaries = np.linspace(0,-5,15,20)
    #plt.hist(a, weights=weights, bins= 500, color=new_colormap[0]) # bins=binBoundaries)
    #plt.yticks(np.arange(0, 0.4, 0.1))
    #plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    #plt.ylabel('Relative Häufigkeitkeitsdichte')
    #plt.xlabel('Verspätungsabbau [min]\n')
    #plt.xlim((-2, 13))
    #plt.savefig('verspätungsabbau.pgf', format='pgf', bbox_inches='tight', dpi=300)
    #plt.show()



# df_merged = reduce(lambda left,right: pd.merge(left,right,on=['driver'], how='outer'), dataframepool) #.fillna('void')

# df_merged['ee_ranking_total'] = df_merged.iloc[:, 1:].mean(axis=1)
# df_merged_described = df_merged.describe(percentiles=None)
# df_merged1 = df_merged.iloc[:, 1:7].astype(float) / df_merged.max(axis=0) # erste spalte raus und letzte auch
# fig, ax0 = plt.subplots()
# ax0.pcolor(df_merged1.iloc[:, 1:7], cmap = 'viridis')
# plt.show()
# print(df_merged)

# Diagramm benchmarks
bd = list(zip(br, li, benchmarks))
bd = pd.DataFrame(bd, columns=['br', 'li', 'benchmarks'])
bd.to_csv('./dist/benchmarks.csv', index=False, sep=';', encoding='utf-8')

bd = bd.sort_values('br')
groups = bd.groupby('li')
# fig, ax = plt.subplots()
# x = bd['br']
# y = bd['benchmarks']
# jittered_y = y + 0.1 * np.random.rand(len(y)) -0.05
# jittered_x = x + 0.1 * np.random.rand(len(x)) -0.05
# for name, group in groups:
#    ax.plot(group.br, group.benchmarks, marker='o', linestyle='', ms=12, label=name) ######### sortieren nach name der baureihe
# ax.legend(loc='lower left', ncol=6, bbox_to_anchor=(0,1.02,1,0.2), mode= 'expand', prop={'size': 8})
# ax.margins(0.07)
# plt.xticks(rotation=90)
# plt.ylim((0,60))
bd = bd.groupby('br')['benchmarks'].mean().reset_index()
# plt.plot(bd.br, bd.benchmarks, marker='x', linestyle='', ms=12, label='mean')
# plt.ylabel('Spezifischer Energieverbrauch [Wh/tkm]')
# plt.xlabel('Fahrzeugkombinationen')
# plt.show()

# Diagramm Einsparpotential für Linie und Baureihe
zippedList = list(zip(nurlinie, nurbaureihe, econsumption, total_epotential, all_years))
dfObj = pd.DataFrame(zippedList, columns=['nurlinie', 'nurbaureihe', 'econsumption', 'total_epotential', 'all_years'])
dfObj = dfObj.replace(['Linie 11_0', 'Linie 11_1'], 'Linie 11') 
dfObj.to_csv('./dist/data_vis.csv', index=False, sep=';', encoding='utf-8')

######## EP für alle Fahrzeuge gesamt

dfn = dfObj.groupby(['nurbaureihe'])['econsumption', 'total_epotential'].sum().reset_index()
dfn = dfn.assign(relative=lambda x: x.total_epotential / x.econsumption)
dfn['total_epotential'] = dfn['total_epotential'].round(2)
width = 0.35
labels = dfn['nurbaureihe']
x = np.arange(len(labels))
#fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()
#rects1 = ax1.bar(x - width / 2, dfn['total_epotential'] / (1000000 * 5), width, label='absolut', color=new_colormap[0])
#rects2 = ax2.bar(x + width / 2, dfn['relative'], width, label='relativ', color=new_colormap[1])
#ax1.set_ylabel('Energieeinsparpotential [GWh]')
#ax2.set_ylabel('Energieeinsparpotential [%]')
#fig.legend(loc='lower left', ncol=2, bbox_to_anchor=(0, 1.02, 1, 0.2))  # ggf. expand rausnehmen
#ax1.bar_label(rects1, padding=3)
#ax1.ticklabel_format(useOffset=False, style='plain', axis='y')
#ax1.set_xticks(x)
#ax1.set_xticklabels(labels)
#print(rects2) ###################################################### hier
##dfn['relative'] = dfn['relative'].round(4)
#dfn['relative'] = pd.Series(["{0:.2f}%".format(val * 100) for val in dfn['relative']], index = dfn.index)
#ax2.bar_label(rects2, padding=3)
#ax2.axis(ymin=0, ymax=0.4)
#ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1))
#plt.ylim(0,0.8)
#plt.show()

######## EP für alle Baureihen einer Linie

# in lines sind alle Linien schon einmal drin # Tabellenkopf
part = []
ep_year = []
ep_year_pro = []
for i in lines:
    df_re50 = dfObj[dfObj['nurlinie'] == i].groupby('nurbaureihe')[
        'econsumption', 'total_epotential'].sum().reset_index()
    df_re50['econsumption'] = df_re50['econsumption'].div(5 * 1000000).round(2)
    df_re50['total_epotential'] = df_re50['total_epotential'].div(5 * 1000000).round(2)
    df_re50 = df_re50.assign(relative=lambda x: x.total_epotential / x.econsumption)
    sum1 = df_re50['econsumption'].sum()
    sum2 = df_re50['total_epotential'].sum()
    divisi = sum2 / sum1
    df_re50.loc[len(df_re50.index)] = ['gesamt', sum1, sum2, divisi]
    ep_year.append(df_re50['total_epotential'].iloc[-1])
    ep_year_pro.append(df_re50['relative'].iloc[-1])
    # create grouped barplot of df_re50
    labels = df_re50['nurbaureihe']
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    rects1 = ax1.bar(x - width/2, df_re50['total_epotential'], width, label='absolut')
    rects2 = ax2.bar(x + width/2, df_re50['relative'], width, label='relativ')
    ax1.set_ylabel('Energieeinsparpotential [GWh]')
    ax2.set_ylabel('Energieeinsparpotential [%]')
    fig.legend(loc='lower left', ncol=2, bbox_to_anchor=(0,1.02,1,0.2)) #ggf. expand rausnehmen
    ax1.bar_label(rects1, padding=3)
    ax1.ticklabel_format(useOffset=False, style='plain', axis='y')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax2.bar_label(rects2, padding=3) # fmt='%1f.%%'
    ax2.axis(ymin=0, ymax=0.4)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1))

    plt.axhline(y=divisi, linestyle='dotted') # Farbe der relativen EP
    plt.show()
    line_tractionenergy = df33[df33['line'] == i]
    line_tractionenergy = line_tractionenergy['econsumption'].sum()
    total_tractionenergy = df_nw[df_nw['line'] == i]
    total_tractionenergy = total_tractionenergy['econsumption_x'].sum()
    part.append(line_tractionenergy / total_tractionenergy)

zippList = list(zip(lines, ep_year_pro, ep_year, part))
df_8 = pd.DataFrame(zippList, columns=['lines', 'ep_year_pro', 'ep_year', 'part'])
df_8 = df_8[df_8['ep_year_pro'].notna()]
df_8 = df_8.sort_values(by=['ep_year_pro'], ascending=False)
df_8['ep_year_pro'] = df_8['ep_year_pro'] * 100
df_8['ep_year_pro'] = df_8['ep_year_pro'].round(2)
df_8['part'] = df_8['part'] * 100
df_8['part'] = df_8['part'].round(2)
summe = df_8['ep_year'].sum()
print(df_8.to_latex(index=False))
print(summe)

## Jährliche Entwicklung der Energieeffizienz der Fahrweise (Kapitel 8.1)
df_years = dfObj.groupby(['all_years', 'nurbaureihe'])['econsumption', 'total_epotential'].sum().reset_index()
df_years = df_years.assign(relative=lambda x: x.total_epotential / x.econsumption)
df_years.drop(df_years.columns[[2, 3]], axis=1, inplace=True)
df_years['relative'] = 1 - df_years['relative']
p_table = pd.pivot_table(df_years, values='relative', index=['all_years'], columns=['nurbaureihe'])
p_df = p_table.reset_index()
#print(p_df)
p_df1 = p_df.drop(['all_years'], axis=1)
y_values = p_df1.iloc[-1].tolist()  # funktioniert

## Szenarienbildung und -berechnung (Kapitel 8.x)
scenarios = p_df1.pct_change(axis='rows').mean(axis=0).reset_index()
scenarios['starting_point'] = y_values
scenarios = scenarios.rename({0: 'average_improvement'}, axis=1)
scenarios['s1_2026'] = scenarios['starting_point'] + scenarios['average_improvement'] * 5  # fun
scenarios.loc[scenarios['s1_2026'] < scenarios['starting_point'], 's1_2026'] = scenarios['starting_point']
scenarios['s1_2031'] = scenarios['s1_2026']  # fun
scenarios['s2_2026'] = scenarios['s1_2026'] + ((1 - scenarios['s1_2026']) * 0.1)  # fun
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
scenarios = scenarios * 100
scenarios = scenarios.round(2)
scenarios = scenarios.drop(['average_improvement'], axis=1)
# pivot table from scenarios
#print(scenarios)

# Trendszenarien Diagramm (Kapitel 8)
min_value = scenarios['starting_point'].idxmin()
min_values = scenarios.iloc[min_value].tolist()
min_values.pop(0)
values_s1 = [min_values[0], min_values[1], min_values[2]]
values_s2 = [min_values[0], min_values[3], min_values[4]]
values_s3 = [min_values[0], min_values[5], min_values[6]]
time = [2021, 2026, 2031]
plt.plot(p_df['all_years'], p_df['146/1+5'] * 100, alpha=0.5)
plt.plot(time, values_s1, label='Worst-Case-Szenario', alpha=0.5)
plt.plot(time, values_s2, label='Trendszenario', alpha=0.5)
plt.plot(time, values_s3, label='Best-Case-Szenario', alpha=0.5)
plt.ylim((60, 100))
plt.legend(loc='lower left', ncol=3, bbox_to_anchor=(0, 1.02, 1, 0.2))
plt.xlabel('Jahr')
plt.ylabel('Energieeffizienz [%]')
plt.show()

# add plus to positive values in p_df
p_df = p_df1.pct_change(axis='rows', periods=4)
print(p_df)

labs = p_df.iloc[-1].tolist()
labs = [i * 100 for i in labs]
labs = [round(num, 1) for num in labs]
labs = [str(i) if i < 0 else '+' + str(i) for i in labs]
labs = [str(i) + '%' for i in labs]
print(labs)
x_values = [2021] * len(y_values)
plt.figure(figsize=(6.6, 6))

ax = p_table.plot(xticks=p_table.index, ylabel='Energieeffizienz', kind='line', xlabel='Jahr',
                  style=['-', ':', '-.', '--', '-', ':', '-.', '--', '-',':', '-.', '--'])

ax.legend(loc='lower left', ncol=5, bbox_to_anchor=(0, 1.02, 1, 0.2), mode='expand', prop={'size': 8})
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
for i, txt in enumerate(labs):
    plt.annotate(txt, (x_values[i], y_values[i]), size=8)
plt.show()
# plt.savefig('bewertung_wirksamkeit_maßnahmen.pgf', format='pgf', bbox_inches='tight')

## Abbildung Trendszenarien für BR mit niedrigster Energieeffizienz bei der Fahrweise
# np.arange(2017, 2031, 15)