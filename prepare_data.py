import os
import pandas as pd
import locale

locale.setlocale(locale.LC_NUMERIC, "de_DE")


def prepare_data():
    """
    This function prepares the data for the model.
    :return:
    """

    # Read all csv files in /sources directory
    files = [f for f in os.listdir('./sources') if f.endswith('.csv')]

    # Create a list of dataframes
    dataframes = []

    # For each file, read the data and append it to the list of dataframes, first line is column names and separated
    # by ;
    for file in files:
        dataframes.append(pd.read_csv('./sources/' + file, encoding='utf-8', decimal=',', sep=';', engine='python'))

    # Concatenate all dataframes into one dataframe
    data = pd.concat(dataframes)

    # Rename columns to a data readable naming convention
    data = data.rename({
        'Fz-Nummern Zug': 'vehiclenumber',
        'Ma Ranking Name': 'driver',
        'BR Zug Verbund alle Fz': 'traintype',
        'EV Zug [kWh]': 'econsumption',
        'Zeitstempel Ab Ist': 'departuretime',
        'Zeitstempel An Ist': 'arrivaltime',
        'Marketinglinie': 'line',
        'T mit [°C]': 'temperature',
        'Tag Ab Plan': 'day',
        'Monat Ab Plan': 'month',
        'Jahr Ab Plan': 'year',
        'km Zug [km]': 'distance',
        'Last Zug [t] AM': 'mass',
        'WT Ab Plan': 'weekday',
        'Zeitstempel Ab Plan': 'p_departuretime',
        'Zeitstempel An Plan': 'p_arrivaltime',
        'Fahrweg der Marketinglinie': 'route',
        'Anz Rzw': 'trainset',
        'Bahnhof AbBf': 'departure',
        'Bahnhof AnBf': 'destination',
        'Verspätungsabbau LA Zug [min]': 'delay_reduction'
    }, axis=1)


    # Adjust date formats
    data['departuretime'] = pd.to_datetime(data['departuretime'], format='%d.%m.%Y %H:%M:%S')
    data['arrivaltime'] = pd.to_datetime(data['arrivaltime'], format='%d.%m.%Y %H:%M:%S')
    data['p_arrivaltime'] = pd.to_datetime(data['p_arrivaltime'], format='%d.%m.%Y %H:%M:%S')
    data['p_departuretime'] = pd.to_datetime(data['p_departuretime'], format='%d.%m.%Y %H:%M:%S')

    # Add new variables
    data['latestart'] = ((data.departuretime - data.p_departuretime) / pd.Timedelta(hours=1)) * 60
    data['lateend'] = (data.arrivaltime - data.p_arrivaltime) / pd.Timedelta(hours=1) * 60
    data['duration'] = (data.arrivaltime - data.departuretime) / pd.Timedelta(hours=1)
    data['p_duration'] = (data.p_arrivaltime - data.p_departuretime) / pd.Timedelta(hours=1)
    data['datehour'] = data['departuretime'].dt.hour
    data['dateminute'] = data['departuretime'].dt.minute
    data = data.assign(def_time=lambda x: x.datehour + x.dateminute / 60,
                       ref_time=lambda x: x.def_time + 0.5 * x.duration)

    # Save data to a separate file
    data.to_csv('./dist/data_prepare.csv', sep=';', index=False, encoding='utf-8')

    df1 = data.assign(spec_econ=lambda x: x.econsumption / x.distance,
                      spec_erecovery=lambda x: x.erecovery / x.distance,
                      spec_edemand=lambda x: x.edemand / x.distance,
                      speed=lambda x: x.distance / x.duration,
                      abweichung_fahrplan=lambda x: x.duration - x.p_duration)

    df1 = df1[df1['line'].str.contains('BB') == False]
    df1 = df1[df1['line'].str.contains('BY') == False]
    df1 = df1[df1['line'].str.contains('Ba') == False]
    df1 = df1[
        df1['line'].str.contains('DINO_Ba|Dummy_o|ST RB76|SdrLr|RB53 DN|RB 76|Nostalg|ST LR 18|Lr|ST LT 1|ST LT 3|ST S47MD') == False]
    df1 = df1[df1['line'].str.contains('\.') == False]

    # Merge same lines with different namings
    df1['line'] = df1['line'].replace(['SN RE 50', 'RE50SX'], 'RE 50')
    df1['line'] = df1['line'].replace(['S 1 SO', 'SN S1 DD', 'SN S1VDD'], 'S1 Dresden')
    df1['line'] = df1['line'].replace(['S 2 SO', 'SN S2 DD'], 'S2 Dresden')
    df1['line'] = df1['line'].replace(['S 3 SO', 'SN S3 DD'], 'S3 Dresden')
    df1['line'] = df1['line'].replace(['S1   EN'], 'S1 ENORM')
    df1['line'] = df1['line'].replace(['S2MDSB2', 'ST S2MD'], 'S2 MDSB')
    df1['line'] = df1['line'].replace(['S5xMDSB', 'SN S5xMD'], 'S5x MDSB')
    df1['line'] = df1['line'].replace(['SN S5MD'], 'S5 MDSB')
    df1['line'] = df1['line'].replace(['S8MDSB2', 'ST S8MD'], 'S8 MDSB')
    df1['line'] = df1['line'].replace(['S9MDSB2','ST S9MD'], 'S9 MDSB')
    df1['line'] = df1['line'].replace(['SN S1MD'], 'S1 MDSB')
    df1['line'] = df1['line'].replace(['SN S3MD'], 'S3 MDSB')
    df1['line'] = df1['line'].replace(['SN S4MD'], 'S4 MDSB')
    df1['line'] = df1['line'].replace(['ST S 1'], 'S1 ME')
    df1['line'] = df1['line'].replace(['SN S6MD'], 'S6 MDSB')
    df1['line'] = df1['line'].replace(['ST RE13', 'RE13MDS'], 'RE 13')
    df1['line'] = df1['line'].replace(['RE14MDS','ST RE14','RE14ES'], 'RE 14')
    df1['line'] = df1['line'].replace(['RB42MDS'], 'RE 42')
    df1['line'] = df1['line'].replace(['RE30 EN','ST RE 30'], 'RE 30')
    df1['line'] = df1['line'].replace(['RE20 EN','ST RE 20'], 'RE 20')
    df1['line'] = df1['line'].replace(['RB81MDS','RB80MDS','RB80/81'], 'RB 80')
    df1['line'] = df1['line'].replace(['RE3MDSB','ST RE3'], 'RE 3')
    df1['line'] = df1['line'].replace(['ST RB 32','RB32 EN'], 'RB 32')
    df1['line'] = df1['line'].replace(['RB40 EN','ST RB 40'], 'RB 40')
    df1['line'] = df1['line'].replace(['RB75MDS'], 'RB 75')
    df1['line'] = df1['line'].replace(['ST RE 18','RE 18 H'], 'RE 18')
    df1['line'] = df1['line'].replace(['RB51MDS','ST RB51'], 'RB 51')
    df1['line'] = df1['line'].replace(['RB51MDS'], 'RB 51')



    df1['train'] = df1['traintype'].map(str) + '+' + df1['trainset'].map(str)
    value = ['146/1+0', '143/1+0', '112/1+0']
    df1 = df1[~df1['train'].isin(value)]

    # Filter dataframe to unique lines
    unique_lines = df1['line'].unique()
    unique_traintypes = df1['train'].unique()

    # Create a map to anonymize the lines
    # Replace all renamed lines in the dataframe and save the mapping in a file
    line_map = {}
    traintype_map = {}

    for i, line in enumerate(unique_lines):
        line_map[line] = 'Linie ' + str(i + 1)

        # Save the line_map to file in directory ./dist
        with open('./dist/line_map.txt', 'w') as f:
            for key, value in line_map.items():
                f.write(key + '\t' + value + '\n')

        df1['line'] = df1['line'].replace(line, line_map[line])

    for i, train in enumerate(unique_traintypes):
        traintype_map[train] = 'Fzg ' + str(i + 1)

        # Save the traintype_map to file in directory ./dist
        with open('./dist/traintype_map.txt', 'w') as f:
            for key, value in traintype_map.items():
                f.write(key + '\t' + value + '\n')

        df1['train'] = df1['train'].replace(train, traintype_map[train])

    """ Clean up dataframe with some indicators"""

    # Remove staging journeys (Bereitstellungsfahrten)
    df1 = df1[df1['distance'] > 5]

    # Remove unnecessary data
    # EV <= 0

    df1 = df1[df1.econsumption > 0]

    # v <= 200 km/h (in RV)
    df1 = df1[df1.speed < 200]

    # Route
    df1 = df1[df1.distance < 200]

    # Export df1 to a csv
    df1.to_csv('./dist/data_clean.csv', index=False, sep=';', encoding='utf-8')

    pass


if __name__ == '__main__':
    prepare_data()
