import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# matplotlib inline

NHL_Goalies = pd.read_excel('NHLGoalies2016_2017.xls', na_values='', sheetname=0)
GAA_ = pd.read_excel('NHLGoalies2016_2017.xls', na_values='', sheetname='5vs5')
# QUESTION 1A
# print(NHL_Goalies)
# print(NHL_Goalies.shape) # 95 rows, 111 columns (95, 111)
# print(NHL_Goalies.get('GP', default=None))    # to look at the column
NHL_Ones = pd.DataFrame(NHL_Goalies[NHL_Goalies['GP'] == 1])  # answer to Q1; locates rows with GP values = 1
# print(NHL_Ones.shape)   # 15 rows, 111 columns (15, 111)
# print(NHL_Ones)     # to see what it looks like
# print(NHL_Ones.loc[:, 'GP'])     # to see that all GP vales are 1
# QUESTION 1B
# salary = NHL_Goalies.get('Salary', default=None)
# print(salary)
# salary1 = NHL_Ones.loc[:, 'Salary']
# print(salary1)
# print(min(NHL_Goalies['Salary']))   # to find the minimum (575000.0); minimum is coming from NHL_Goalies
Adjusted_NHL_Ones = pd.DataFrame(NHL_Ones)      # creates new df with information from NHL_Ones
Adjusted_NHL_Ones.loc[:, 'Adjusted Salary'] = Adjusted_NHL_Ones['Salary']
Adjusted_NHL_Ones['Adjusted Salary'].fillna((min(NHL_Goalies['Salary'])), inplace=True)     # fills/changes NaN with min
print(Adjusted_NHL_Ones[['Salary', 'Adjusted Salary']])     # multiple columns as described in Lec 7; printed 2 columns
# print(Adjusted_NHL_Ones.shape) # 15 rows, 112 columns; to see what the shape is and that it still had 112 columns
# QUESTION 2
#workhorse = pd.concat([NHL_Goalies, GAA_], axis=1)
#print(workhorse, '\n********')
#_, i = np.unique(workhorse.columns, return_index=True)
#print(workhorse.iloc[:, i])
#print(workhorse)
print('\n******************')
workhorse = pd.DataFrame(NHL_Goalies[(NHL_Goalies['GP'] > 25) & (NHL_Goalies['GAA'] < 3)])
# print(workhorse.shape)  # (44, 111)
# print(workhorse[['GP', 'GAA']])   # this was done to double check that criteria was met
print('\n******************')
# .merge(GAA_, left_on='First Name', right_on='GAA', how='outer')
#print(workhorse)
# #print(NHL_Goalies)
# GAA_less_than_three = pd.DataFrame(GAA_[GAA_['GAA'] < 3.0])
# list1 = GAA_less_than_three.index.values.tolist()
# print(GAA_less_than_three.shape)    # 80 rows
# print(list1)
# GAA_more_than_three = pd.DataFrame(GAA_[GAA_['GAA'] >= 3.0])
# list2 = GAA_more_than_three.index.values.tolist()
# print(GAA_more_than_three.shape)    # 80 rows
# print(list2)
# print(var)
# workhorse =  pd.DataFrame(Adjusted_NHL_Ones
# print(NHL_Goalies.keys(), '\n*********************')
# print(GAA_.keys(),
print('\n*********************')
#  QUESTION 3A


def sum_missing_col_val(data_f):  # defining function for sum of missing col
    df_input = data_f
    missing_col_sums = df_input.isnull().sum(axis=0)    # creates Pandas series of col names as index & missing sums
    print(type(missing_col_sums), '\n', missing_col_sums.index.values)   # to look at type and series index
    series_to_df = missing_col_sums.to_frame()   # converts series to df
    series_to_df.columns = ['Missing']  # assigns 'Missing' to the column
    # print(type(series_to_df))   # to check type is df, <class 'pandas.core.frame.DataFrame'>
    # print(series_to_df.index.values)    # to look at val of df index
    return series_to_df


#    df_output = pd.DataFrame(:, index_output)
#     miss_val_col = df_.isnull().sum(axis=0)
#     _sum = pd.Series(df_['Value'].values, index=df_.column.values))
# print(NHL_Goalies.columns.values)
# print(is_null)
# df = pd.DataFrame(NHL_Goalies)
# df.loc[:, 'Sum of Missing Values'] = is_null
# print(df)
# print(type(NHL_Goalies))    # <class 'pandas.core.frame.DataFrame'>
df_practice = pd.DataFrame({"PIID":[38542,33629,32789],
                   "fy":["2014","2015", '5'],
                   "zone":["AZW - Acquisition Zone West", "NAZ - Northern Acquisition Zone", "SAZ - Southern Acquisition Zone"]})


def sum_missing_val_row(data_frame):
        input_df = data_frame
        missing_val_row = input_df.isnull().sum(axis=1)  # creating a pandas Series
        new_df = pd.DataFrame(input_df)
        new_df.loc[:, 'missing_values'] = missing_val_row
        return new_df


print(sum_missing_val_row(df_practice))
# print(sum_missing_col_val(NHL_Ones))
# print(sum_missing_col_val(NHL_Goalies))
#sum_missing_val_row(NHL_Goalies)

# QUESTION 4
plt.figure(1)   # plotting figure 1
width = 16  # setting width
height = 8  # setting height
plt.figure(figsize=(width, height))     # applying width and height to fig
ax = plt.subplot(121)   # creating first subplot, 1 row, 2 columns, 1st subplot
NHL_Goalies['GAA'].plot.hist(histtype='bar', color='m', edgecolor='k', linewidth=1.2)   # creating histogram and appearance
plt.xlabel('GAA values of all NHL Goalies', fontsize=14)    # adding x and y labels
plt.ylabel('Frequency of GAA values', fontsize=14)
ax.set_title('Histogram of GAA values of NHL Goalies', fontsize=16)
ax = plt.subplot(122)   # second subplot, 1 row, 2 columns, 2nd subplot
workhorse['GAA'].plot.hist(histtype='bar', color='c', edgecolor='k', linewidth=1.2)
# creating histogram of second subplot
ax.set_title('Histogram of GAA values of Goalies with GAA<3 and GP>25', fontsize=16)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
# creating dimensions of subplots within fig
plt.xlabel('GAA values of NHL Goalies w/ GAA<3 and GP>25', fontsize=14)     # x and y labels of second subplot
plt.ylabel('Frequency of GAA values', fontsize=14)

# QUESTION 5

injuries_not_known = NHL_Goalies[NHL_Goalies['Injuries'].isnull()]   # creating subset of injuries not known
print(injuries_not_known, '\n**type: ', type(injuries_not_known))   # check subset and that it is a df
merged_left = pd.merge(left=GAA_, right=injuries_not_known, how='outer', )
print(merged_left, '\n******************!!')
mergedDF = GAA_.join(injuries_not_known, lsuffix='_GAA', rsuffix='_injuries')    # merged DF on both indices to preserve order
print(mergedDF, '\n**type: ', type(mergedDF))
print(mergedDF[['First Name_GAA', 'First Name_injuries']])

# Write DataFrame to CSV
mergedDF.to_csv('TOI_2017.csv')

# for kicks read our output back into python and make sure all looks good
output_mergedDF = pd.read_csv('TOI_2017.csv', keep_default_na=False, na_values=[""])


