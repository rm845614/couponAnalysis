# Coupon analysis (dataCoupon)

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from sklearn.linear_model import LinearRegression
import os

matplotlib.use('Agg')
target = 'analyzing the coupon data to avoid over sampling the pipelines'

# ----------------------------------------------------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------------------------------------------------
outlier_threshold = 2
excel_name = 'dataCoupon2'


# ----------------------------------------------------------------------------------------------------------------------
# Implemented Functions
# ----------------------------------------------------------------------------------------------------------------------
def clean_data(df):
    df = df.rename(columns={'Field observations (color, appearance of scale or corrosion product, '
                            'erosion/mechanical damage, others etc.)': 'Field_Observations',
                            'Unnamed: 2': 'Coupon Area', 'Initial weight   (g)': 'Initial weight (g)'})
    df.columns = df.columns.str.rstrip()
    df.columns = df.columns.str.replace(' ', '_')
    df.replace(" ", np.nan, inplace=True)
    df.dropna(subset=['Coupon_Type', 'Location_Description', 'Date_In', 'Date_Out',
                      'Field_Observations', 'General_Corrosion_Rate_(mpy)'], inplace=True)
    df.drop_duplicates(inplace=True)
    df = df[df['General_Corrosion_Rate_(mpy)'] > 0.0]
    df.dropna(axis='columns', thresh=int(0.9 * len(df)), inplace=True)
    df = df.reset_index(drop=True)
    df['Location_Description'] = df['Location_Description'].str.replace('"', ' inches')
    df['Location_Description'] = df['Location_Description'].str.replace(',  ', ', ')
    df['Location_Description'] = df['Location_Description'].str.replace('/', '_')
    df['Field_Observations'] = df['Field_Observations'].str.replace('UNUSAL', 'UNUSUAL')
    df['Field_Observations'] = df['Field_Observations'].str.replace('Bent coupon', 'BENT')
    df['Date_In'] = pd.to_datetime(df['Date_In'])
    df['Date_Out'] = pd.to_datetime(df['Date_Out'])
    return df


def read_data(file_name):
    root = '{}/dataSummary'.format(file_name)
    if not os.path.exists(root):
        os.makedirs(root)
    # ---------------------------------
    sheet_names = pd.ExcelFile('{}.xlsx'.format(file_name)).sheet_names
    df = pd.DataFrame()
    for sheet_name in sheet_names:
        if sheet_name == sheet_names[0]:
            df = pd.read_excel('{}.xlsx'.format(file_name), sheet_name=sheet_name, header=3)
        else:
            df2 = pd.read_excel('{}.xlsx'.format(file_name), sheet_name=sheet_name, header=3)
            df = pd.concat([df, df2], ignore_index=True, sort=False)
    df = clean_data(df)
    excel_output(df, root=root, file_name='{}Cleaned'.format(file_name))
    return df


def columns_stats(df, file_name):
    root = '{}/dataSummary'.format(file_name)
    if not os.path.exists(root):
        os.makedirs(root)
    # ---------------------------------
    statistics = pd.DataFrame()
    for column in df.columns:
        if (column != 'Coupon_Serial_Number') and (column != 'Date_In') and (column != 'Date_Out') and \
                (column != 'Initial_weight_(g)') and (column != 'Final_Weight_(g)') and \
                (column != 'Weight_Difference_(g)') and (column != 'General_Corrosion_Rate_(mpy)'):
            if column == df.columns[0]:
                statistics = pd.DataFrame(df[column].value_counts()).reset_index(drop=False)
                statistics.rename(columns={'index': column, column: 'samples_num'}, inplace=True)
            else:
                temp = pd.DataFrame(df[column].value_counts()).reset_index(drop=False)
                temp.rename(columns={'index': column, column: 'samples_num'}, inplace=True)
                statistics = pd.concat([statistics, temp], axis=1)
    excel_output(statistics, root=root, file_name='columnsStats')
    return statistics


def view_data(df, num, file_name):
    locations = df['Location_Description'].unique()
    for loc in locations:
        df1 = df.loc[df['Location_Description'] == loc].reset_index(drop=True)
        if len(df1) == 1:
            df2 = df1
        else:
            df2 = df1.groupby(['Date_In', 'Date_Out']).mean().reset_index(drop=False)
            _std = df2['General_Corrosion_Rate_(mpy)'].std()
            _mean = df2['General_Corrosion_Rate_(mpy)'].mean()
            threshold = _mean + num * _std
            df2 = df2[df2['General_Corrosion_Rate_(mpy)'] <= threshold].reset_index(drop=True)
        df2['X_error'] = 0.5 * (df2['Date_Out'] - df2['Date_In'])
        df2['Date'] = 0.5 * (pd.to_numeric(df2['Date_In']) + pd.to_numeric(df2['Date_Out']))
        _X = df2.loc[:, 'Date'].to_numpy().reshape(-1, 1)
        _y = df2.loc[:, 'General_Corrosion_Rate_(mpy)'].to_numpy()
        regression(df2, _X, _y, loc, file_name)


def regression(df, _x, _y, loc, file_name):
    reg = LinearRegression().fit(_x, _y)
    df['Predicted_Corrosion_Rate_(mpy)'] = pd.Series(reg.predict(_x))
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'X_error', 'General_Corrosion_Rate_(mpy)', 'Predicted_Corrosion_Rate_(mpy)']]
    p_value = 1.0
    if len(_x) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(_x[:, -1], _y)
    # ---------------------------------
    d_cr = df.loc[len(df) - 1, 'Predicted_Corrosion_Rate_(mpy)'] - df.loc[0, 'Predicted_Corrosion_Rate_(mpy)']
    d_time = float(str(df.loc[len(df) - 1, 'Date'] - df.loc[0, 'Date']).split(' ')[0])
    slope, equation = 0, 'slope is not significant'
    if d_time != 0:
        slope = (d_cr / d_time) * 360
    if p_value < 0.2:
        equation = 'y = {:.3f} year'.format(slope)
    elif p_value == 1.0:
        equation = 'not enough data points (only 1 is found)'
    _max = df['General_Corrosion_Rate_(mpy)'].apply(np.floor).max()
    if _max > 10:
        y_axis_max = _max + 5
    elif _max > 1:
        y_axis_max = _max + 2
    else:
        y_axis_max = _max + 1.2
    # ---------------------------------
    analysis_plot(_x=df['Date'], _y=df['General_Corrosion_Rate_(mpy)'],
                  _y_pred=df['Predicted_Corrosion_Rate_(mpy)'], _x_err=df['X_error'], _y_max=y_axis_max,
                  p_value=p_value, equation=equation, label='Experimental data', title=loc, file_name=file_name)


def analysis_plot(_x, _y, _y_pred, _x_err, _y_max, p_value, equation, label, title, file_name):
    root = '{}/locationsAnalysis'.format(file_name)
    if not os.path.exists(root):
        os.makedirs(root)
    register_matplotlib_converters()
    # ---------------------------------
    fig, ax = plt.subplots(1, figsize=(12, 9))
    plt.errorbar(_x, _y, xerr=_x_err, ecolor='grey', fmt=' ', capsize=7, marker='o', color='black', label=label)
    plt.plot(_x, _y_pred, '-', c='lightcoral', linewidth=7.0, label='Linear fitting')
    plt.grid(linewidth=0.5)
    plt.ylabel('General Corrosion Rate (mpy)', fontsize=25)
    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0, _y_max)
    p_value = 'P-value = {:.3f}'.format(p_value)
    plt.text(0.05, 0.96, p_value,
             ha='left', va='top', transform=ax.transAxes, fontdict={'color': 'red', 'size': 20})
    plt.text(0.05, 0.9, equation,
             ha='left', va='top', transform=ax.transAxes, fontdict={'color': 'black', 'size': 16})
    plt.legend(loc='upper right', fontsize=20)
    if len(title) > 35:
        title = title[0:35] + '...'
    plt.title(label='Location: {}'.format(title), fontsize=27)
    plt.tight_layout()
    plt.savefig('{}/{}.png'.format(root, title))
    plt.close()


def excel_output(_object, root, file_name):
    if root != '':
        _object.to_excel('{}/{}.xls'.format(root, file_name))
    else:
        _object.to_excel('{}.xls'.format(file_name))


# --------------------------------------------------------------------------------------------------------------------
# BEGIN
# --------------------------------------------------------------------------------------------------------------------

# reading data + analysis
coupon = read_data(excel_name)
columns_stats(coupon, excel_name)
view_data(coupon, outlier_threshold, excel_name)

# --------------------------------------------------------------------------------------------------------------------
# THE END
# --------------------------------------------------------------------------------------------------------------------
print('=> DONE!')
