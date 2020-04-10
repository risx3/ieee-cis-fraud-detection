
## Overview

Imagine standing at the check-out counter at the grocery store with a long line behind you and the cashier not-so-quietly announces that your card has been declined. In this moment, you probably aren’t thinking about the data science that determined your fate. <br>
Embarrassed, and certain you have the funds to cover everything needed for an epic nacho party for 50 of your closest friends, you try your card again. Same result. As you step aside and allow the cashier to tend to the next customer, you receive a text message from your bank. “Press 1 if you really tried to spend $500 on cheddar cheese.”<br>
While perhaps cumbersome (and often embarrassing) in the moment, this fraud prevention system is actually saving consumers millions of dollars per year. Researchers from the IEEE Computational Intelligence Society (IEEE-CIS) want to improve this figure, while also improving the customer experience. With higher accuracy fraud detection, you can get on with your chips without the hassle.<br>
IEEE-CIS works across a variety of AI and machine learning areas, including deep neural networks, fuzzy systems, evolutionary computation, and swarm intelligence. Today they’re partnering with the world’s leading payment service company, Vesta Corporation, seeking the best solutions for fraud prevention industry, and now you are invited to join the challenge.<br>
In this competition, you’ll benchmark machine learning models on a challenging large-scale dataset. The data comes from Vesta's real-world e-commerce transactions and contains a wide range of features from device type to product features. You also have the opportunity to create new features to improve your results.<br>
If successful, you’ll improve the efficacy of fraudulent transaction alerts for millions of people around the world, helping hundreds of thousands of businesses reduce their fraud loss and increase their revenue. And of course, you will save party people just like you the hassle of false positives.<br>

## Data

In this, you are predicting the probability that an online transaction is fraudulent, as denoted by the binary target 'isFraud'.<br>
The data is broken into two files identity and transaction, which are joined by TransactionID. Not all transactions have corresponding identity information.

**Categorical Features - Transaction**

1. ProductCD
2. card1 - card6
3. addr1, addr2
4. P_emaildomain
5. R_emaildomain
6. M1 - M9

**Categorical Features - Identity**

1. DeviceType
2. DeviceInfo
3. id_12 - id_38

The TransactionDT feature is a timedelta from a given reference datetime (not an actual timestamp).<br>
You can find the dataset [here](https://www.kaggle.com/c/ieee-fraud-detection/data).

## Files

1. train_{transaction, identity}.csv - the training set
2. test_{transaction, identity}.csv - the test set (you must predict the isFraud value for these observations)

## So lets begin with complete EDA...


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc, os, sys
import re

from sklearn import metrics, preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

sns.set_style('darkgrid')

pd.options.display.float_format = '{:,.3f}'.format

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
```

    /kaggle/input/ieee-fraud-detection/sample_submission.csv
    /kaggle/input/ieee-fraud-detection/test_identity.csv
    /kaggle/input/ieee-fraud-detection/train_transaction.csv
    /kaggle/input/ieee-fraud-detection/test_transaction.csv
    /kaggle/input/ieee-fraud-detection/train_identity.csv
    


```python
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
```

# Load Data

### Reducing memory usage


```python
%%time
train_id = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
train_trn = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
test_id = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')
test_trn = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
```

    CPU times: user 40.3 s, sys: 4.38 s, total: 44.7 s
    Wall time: 44.8 s
    


```python
train_id = reduce_mem_usage(train_id)
train_trn = reduce_mem_usage(train_trn)
test_id = reduce_mem_usage(test_id)
test_trn = reduce_mem_usage(test_trn)
```

    Memory usage of dataframe is 45.12 MB --> 25.86 MB (Decreased by 42.7%)
    Memory usage of dataframe is 1775.15 MB --> 542.35 MB (Decreased by 69.4%)
    Memory usage of dataframe is 44.39 MB --> 25.44 MB (Decreased by 42.7%)
    Memory usage of dataframe is 1519.24 MB --> 472.59 MB (Decreased by 68.9%)
    


```python
print(train_id.shape, test_id.shape)
print(train_trn.shape, test_trn.shape)
```

    (144233, 41) (141907, 41)
    (590540, 394) (506691, 393)
    

# Data Analysis

### isFraud count


```python
fc = train_trn['isFraud'].value_counts(normalize=True).to_frame()
fc.plot.bar()
# Also print a table
fc.T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>isFraud</td>
      <td>0.965</td>
      <td>0.035</td>
    </tr>
  </tbody>
</table>
</div>




![png](notebook_files/notebook_16_1.png)


### Fraud Transaction rate per day and per week


```python
fig,ax = plt.subplots(2, 1, figsize=(16,8))

train_trn['_seq_day'] = train_trn['TransactionDT'] // (24*60*60)
train_trn['_seq_week'] = train_trn['_seq_day'] // 7
train_trn.groupby('_seq_day')['isFraud'].mean().to_frame().plot.line(ax=ax[0])
train_trn.groupby('_seq_week')['isFraud'].mean().to_frame().plot.line(ax=ax[1])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f01227e90f0>




![png](notebook_files/notebook_18_1.png)


### Fraud transaction rate by weekday, hour, month-day, and year-month


```python
import datetime

START_DATE = '2017-11-30'
startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
train_trn['Date'] = train_trn['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
train_trn['_ymd'] = train_trn['Date'].dt.year.astype(str) + '-' + train_trn['Date'].dt.month.astype(str) + '-' + train_trn['Date'].dt.day.astype(str)
train_trn['_year_month'] = train_trn['Date'].dt.year.astype(str) + '-' + train_trn['Date'].dt.month.astype(str)
train_trn['_weekday'] = train_trn['Date'].dt.dayofweek
train_trn['_hour'] = train_trn['Date'].dt.hour
train_trn['_day'] = train_trn['Date'].dt.day

fig,ax = plt.subplots(4, 1, figsize=(16,12))

train_trn.groupby('_weekday')['isFraud'].mean().to_frame().plot.bar(ax=ax[0])
train_trn.groupby('_hour')['isFraud'].mean().to_frame().plot.bar(ax=ax[1])
train_trn.groupby('_day')['isFraud'].mean().to_frame().plot.bar(ax=ax[2])
train_trn.groupby('_year_month')['isFraud'].mean().to_frame().plot.bar(ax=ax[3])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f0116032f60>




![png](notebook_files/notebook_20_1.png)


### Fraud transaction rate by day


```python
df = train_trn.groupby(['_ymd'])['isFraud'].agg(['count','mean','sum'])
df.sort_values(by='mean',ascending=False)[:10].T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>_ymd</th>
      <th>2018-1-28</th>
      <th>2018-3-27</th>
      <th>2018-2-11</th>
      <th>2018-2-3</th>
      <th>2018-1-25</th>
      <th>2018-3-25</th>
      <th>2018-4-8</th>
      <th>2018-3-24</th>
      <th>2018-2-2</th>
      <th>2018-1-31</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>2,402.000</td>
      <td>3,042.000</td>
      <td>2,304.000</td>
      <td>3,209.000</td>
      <td>2,789.000</td>
      <td>2,461.000</td>
      <td>2,348.000</td>
      <td>2,758.000</td>
      <td>4,317.000</td>
      <td>3,057.000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>0.070</td>
      <td>0.064</td>
      <td>0.059</td>
      <td>0.057</td>
      <td>0.057</td>
      <td>0.056</td>
      <td>0.055</td>
      <td>0.054</td>
      <td>0.054</td>
      <td>0.054</td>
    </tr>
    <tr>
      <td>sum</td>
      <td>168.000</td>
      <td>194.000</td>
      <td>135.000</td>
      <td>183.000</td>
      <td>158.000</td>
      <td>139.000</td>
      <td>130.000</td>
      <td>150.000</td>
      <td>234.000</td>
      <td>165.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sort_values(by='count',ascending=False)[:10].T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>_ymd</th>
      <th>2017-12-22</th>
      <th>2018-3-2</th>
      <th>2017-12-24</th>
      <th>2017-12-23</th>
      <th>2017-12-20</th>
      <th>2017-12-25</th>
      <th>2017-12-21</th>
      <th>2017-12-18</th>
      <th>2017-12-19</th>
      <th>2017-12-1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>6,852.000</td>
      <td>6,252.000</td>
      <td>6,065.000</td>
      <td>5,872.000</td>
      <td>5,749.000</td>
      <td>5,742.000</td>
      <td>5,677.000</td>
      <td>5,585.000</td>
      <td>5,526.000</td>
      <td>5,122.000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>0.021</td>
      <td>0.024</td>
      <td>0.013</td>
      <td>0.018</td>
      <td>0.023</td>
      <td>0.011</td>
      <td>0.020</td>
      <td>0.029</td>
      <td>0.021</td>
      <td>0.022</td>
    </tr>
    <tr>
      <td>sum</td>
      <td>147.000</td>
      <td>147.000</td>
      <td>76.000</td>
      <td>106.000</td>
      <td>135.000</td>
      <td>63.000</td>
      <td>116.000</td>
      <td>161.000</td>
      <td>118.000</td>
      <td>112.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# transaction-count X fraud-rate
plt.scatter(df['count'], df['mean'], s=10)
```




    <matplotlib.collections.PathCollection at 0x7f01226edba8>




![png](notebook_files/notebook_24_1.png)



```python
# transaction-count X fraud-count
plt.scatter(df['count'], df['sum'], s=10)
```




    <matplotlib.collections.PathCollection at 0x7f012246b8d0>




![png](notebook_files/notebook_25_1.png)


### Fraud transaction rate by weekday-hour


```python
train_trn['_weekday_hour'] = train_trn['_weekday'].astype(str) + '_' + train_trn['_hour'].astype(str)
train_trn.groupby('_weekday_hour')['isFraud'].mean().to_frame().plot.line(figsize=(16,3))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f0122412588>




![png](notebook_files/notebook_27_1.png)


### Fraud rate by weekday


```python
df = train_trn.groupby('_weekday')['isFraud'].mean().to_frame()
df.sort_values(by='isFraud', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>isFraud</th>
    </tr>
    <tr>
      <th>_weekday</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3</td>
      <td>0.037</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.037</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.036</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.036</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.035</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.033</td>
    </tr>
    <tr>
      <td>0</td>
      <td>0.031</td>
    </tr>
  </tbody>
</table>
</div>



### Fraud rate by hour


```python
df = train_trn.groupby('_hour')['isFraud'].mean().to_frame()
df.sort_values(by='isFraud', ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>isFraud</th>
    </tr>
    <tr>
      <th>_hour</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>7</td>
      <td>0.106</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.093</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.090</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.078</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.070</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.053</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.052</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.039</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.038</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.037</td>
    </tr>
  </tbody>
</table>
</div>



### Fraud rate by weekday-hour


```python
df = train_trn.groupby('_weekday_hour')['isFraud'].mean().to_frame()
df.sort_values(by='isFraud', ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>isFraud</th>
    </tr>
    <tr>
      <th>_weekday_hour</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>5_7</td>
      <td>0.143</td>
    </tr>
    <tr>
      <td>6_7</td>
      <td>0.142</td>
    </tr>
    <tr>
      <td>3_8</td>
      <td>0.136</td>
    </tr>
    <tr>
      <td>4_8</td>
      <td>0.135</td>
    </tr>
    <tr>
      <td>4_9</td>
      <td>0.117</td>
    </tr>
    <tr>
      <td>6_8</td>
      <td>0.110</td>
    </tr>
    <tr>
      <td>5_10</td>
      <td>0.109</td>
    </tr>
    <tr>
      <td>4_7</td>
      <td>0.106</td>
    </tr>
    <tr>
      <td>0_7</td>
      <td>0.103</td>
    </tr>
    <tr>
      <td>6_11</td>
      <td>0.102</td>
    </tr>
  </tbody>
</table>
</div>



### Fraud rate by amount bin


```python
train_trn['_amount_qcut10'] = pd.qcut(train_trn['TransactionAmt'],10)
df = train_trn.groupby('_amount_qcut10')['isFraud'].mean().to_frame()
df.sort_values(by='isFraud', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>isFraud</th>
    </tr>
    <tr>
      <th>_amount_qcut10</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>(0.25, 25.953]</td>
      <td>0.056</td>
    </tr>
    <tr>
      <td>(275.25, 31936.0]</td>
      <td>0.051</td>
    </tr>
    <tr>
      <td>(117.0, 160.0]</td>
      <td>0.043</td>
    </tr>
    <tr>
      <td>(160.0, 275.25]</td>
      <td>0.038</td>
    </tr>
    <tr>
      <td>(68.75, 100.0]</td>
      <td>0.036</td>
    </tr>
    <tr>
      <td>(35.938, 49.0]</td>
      <td>0.032</td>
    </tr>
    <tr>
      <td>(25.953, 35.938]</td>
      <td>0.032</td>
    </tr>
    <tr>
      <td>(57.938, 68.75]</td>
      <td>0.029</td>
    </tr>
    <tr>
      <td>(100.0, 117.0]</td>
      <td>0.020</td>
    </tr>
    <tr>
      <td>(49.0, 57.938]</td>
      <td>0.019</td>
    </tr>
  </tbody>
</table>
</div>



### TransactionId


```python
# Not all transactions have corresponding identity information.
# len([c for c in train_trn['TransactionID'] if c not in train_id['TransactionID'].values]) 
# 446307

# Not all fraud transactions have corresponding identity information.
fraud_id = train_trn[train_trn['isFraud'] == 1]['TransactionID']
fraud_id_in_trn = [i for i in fraud_id if i in train_id['TransactionID'].values]
print(f'fraud data count:{len(fraud_id)}, and in trn:{len(fraud_id_in_trn)}')
```

    fraud data count:20663, and in trn:11318
    

## Identity Data

Variables in this table are identity information – network connection information (IP, ISP, Proxy, etc) and digital signature (UA/browser/os/version, etc) associated with transactions. They're collected by Vesta’s fraud protection system and digital security partners. (The field names are masked and pairwise dictionary will not be provided for privacy protection and contract agreement)

**Categorical Features:**

1. DeviceType
2. DeviceInfo
3. id12 - id38


```python
train_id_trn = pd.merge(train_id, train_trn[['isFraud','TransactionAmt','TransactionID']])
train_id_f0 = train_id_trn[train_id_trn['isFraud'] == 0]
train_id_f1 = train_id_trn[train_id_trn['isFraud'] == 1]
print(train_id_f0.shape, train_id_f1.shape)

def plotHistByFraud(col, bins=20, figsize=(8,3)):
    with np.errstate(invalid='ignore'):
        plt.figure(figsize=figsize)
        plt.hist([train_id_f0[col], train_id_f1[col]], bins=bins, density=True, color=['royalblue', 'orange'])
        
def plotCategoryRateBar(col, topN=np.nan, figsize=(8,3)):
    a, b = train_id_f0, train_id_f1
    if topN == topN: # isNotNan
        vals = b[col].value_counts(normalize=True).to_frame().iloc[:topN,0]
        subA = a.loc[a[col].isin(vals.index.values), col]
        df = pd.DataFrame({'normal':subA.value_counts(normalize=True), 'fraud':vals})
    else:
        df = pd.DataFrame({'normal':a[col].value_counts(normalize=True), 'fraud':b[col].value_counts(normalize=True)})
    df.sort_values('fraud', ascending=False).plot.bar(figsize=figsize)
```

    (132915, 43) (11318, 43)
    

### id_01 - id_11


```python
plotHistByFraud('id_01')
plotHistByFraud('id_02')
plotHistByFraud('id_03')
plotHistByFraud('id_04')
plotHistByFraud('id_05')
plotHistByFraud('id_06')
plotHistByFraud('id_07')
plotHistByFraud('id_08')
plotHistByFraud('id_09')
plotHistByFraud('id_10')
plotHistByFraud('id_11')
```


![png](notebook_files/notebook_44_0.png)



![png](notebook_files/notebook_44_1.png)



![png](notebook_files/notebook_44_2.png)



![png](notebook_files/notebook_44_3.png)



![png](notebook_files/notebook_44_4.png)



![png](notebook_files/notebook_44_5.png)



![png](notebook_files/notebook_44_6.png)



![png](notebook_files/notebook_44_7.png)



![png](notebook_files/notebook_44_8.png)



![png](notebook_files/notebook_44_9.png)



![png](notebook_files/notebook_44_10.png)



```python
numid_cols = [f'id_{str(i).zfill(2)}' for i in range(1,12)]
train_id_trn[numid_cols].isna().sum().to_frame().T / len(train_id)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_01</th>
      <th>id_02</th>
      <th>id_03</th>
      <th>id_04</th>
      <th>id_05</th>
      <th>id_06</th>
      <th>id_07</th>
      <th>id_08</th>
      <th>id_09</th>
      <th>id_10</th>
      <th>id_11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.000</td>
      <td>0.023</td>
      <td>0.540</td>
      <td>0.540</td>
      <td>0.051</td>
      <td>0.051</td>
      <td>0.964</td>
      <td>0.964</td>
      <td>0.481</td>
      <td>0.481</td>
      <td>0.023</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10, 5))
sns.heatmap(train_id_trn[['isFraud','TransactionAmt']+numid_cols].corr(), annot=True, fmt='.2f')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f0122a14780>




![png](notebook_files/notebook_46_1.png)



```python
train_id_f1[['isFraud'] + numid_cols].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>isFraud</th>
      <th>id_01</th>
      <th>id_02</th>
      <th>id_03</th>
      <th>id_04</th>
      <th>id_05</th>
      <th>id_06</th>
      <th>id_07</th>
      <th>id_08</th>
      <th>id_09</th>
      <th>id_10</th>
      <th>id_11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>52</td>
      <td>1</td>
      <td>0.000</td>
      <td>169,947.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3.000</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>100.000</td>
    </tr>
    <tr>
      <td>53</td>
      <td>1</td>
      <td>0.000</td>
      <td>222,455.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>100.000</td>
    </tr>
    <tr>
      <td>54</td>
      <td>1</td>
      <td>0.000</td>
      <td>271,870.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3.000</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>100.000</td>
    </tr>
    <tr>
      <td>69</td>
      <td>1</td>
      <td>-20.000</td>
      <td>258,138.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>-1.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>100.000</td>
    </tr>
    <tr>
      <td>86</td>
      <td>1</td>
      <td>-5.000</td>
      <td>141,271.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>9.000</td>
      <td>-81.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>97.562</td>
    </tr>
    <tr>
      <td>98</td>
      <td>1</td>
      <td>-20.000</td>
      <td>550,210.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>-1.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>100.000</td>
    </tr>
    <tr>
      <td>185</td>
      <td>1</td>
      <td>-25.000</td>
      <td>59,967.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>1.000</td>
      <td>-12.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>100.000</td>
    </tr>
    <tr>
      <td>201</td>
      <td>1</td>
      <td>-5.000</td>
      <td>30,602.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>-12.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>95.062</td>
    </tr>
    <tr>
      <td>235</td>
      <td>1</td>
      <td>-5.000</td>
      <td>4,235.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>100.000</td>
    </tr>
    <tr>
      <td>236</td>
      <td>1</td>
      <td>0.000</td>
      <td>36,004.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>100.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_id_f0[['isFraud'] + numid_cols].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>isFraud</th>
      <th>id_01</th>
      <th>id_02</th>
      <th>id_03</th>
      <th>id_04</th>
      <th>id_05</th>
      <th>id_06</th>
      <th>id_07</th>
      <th>id_08</th>
      <th>id_09</th>
      <th>id_10</th>
      <th>id_11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>0.000</td>
      <td>70,787.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>100.000</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0</td>
      <td>-5.000</td>
      <td>98,945.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>-5.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>100.000</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0</td>
      <td>-5.000</td>
      <td>191,631.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>100.000</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0</td>
      <td>-5.000</td>
      <td>221,832.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>-6.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>100.000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>0.000</td>
      <td>7,460.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>100.000</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0</td>
      <td>-5.000</td>
      <td>61,141.000</td>
      <td>3.000</td>
      <td>0.000</td>
      <td>3.000</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>3.000</td>
      <td>0.000</td>
      <td>100.000</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0</td>
      <td>-15.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0</td>
      <td>0.000</td>
      <td>31,964.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-10.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>100.000</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0</td>
      <td>-10.000</td>
      <td>116,098.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>100.000</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0</td>
      <td>-5.000</td>
      <td>257,037.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>100.000</td>
    </tr>
  </tbody>
</table>
</div>



### id_12 - id_38


```python
plotCategoryRateBar('id_12')
plotCategoryRateBar('id_13',10)
plotCategoryRateBar('id_14',10)
plotCategoryRateBar('id_15')
plotCategoryRateBar('id_16')
plotCategoryRateBar('id_17',10)
plotCategoryRateBar('id_18')
plotCategoryRateBar('id_19',10)
plotCategoryRateBar('id_20',10)
```


![png](notebook_files/notebook_50_0.png)



![png](notebook_files/notebook_50_1.png)



![png](notebook_files/notebook_50_2.png)



![png](notebook_files/notebook_50_3.png)



![png](notebook_files/notebook_50_4.png)



![png](notebook_files/notebook_50_5.png)



![png](notebook_files/notebook_50_6.png)



![png](notebook_files/notebook_50_7.png)



![png](notebook_files/notebook_50_8.png)



```python
plotCategoryRateBar('id_21',20)
plotCategoryRateBar('id_22')
plotCategoryRateBar('id_23',10)
plotCategoryRateBar('id_24')
plotCategoryRateBar('id_25',20)
plotCategoryRateBar('id_26',10)
plotCategoryRateBar('id_27', 15)
plotCategoryRateBar('id_28')
plotCategoryRateBar('id_29')
plotCategoryRateBar('id_30',10)
```


![png](notebook_files/notebook_51_0.png)



![png](notebook_files/notebook_51_1.png)



![png](notebook_files/notebook_51_2.png)



![png](notebook_files/notebook_51_3.png)



![png](notebook_files/notebook_51_4.png)



![png](notebook_files/notebook_51_5.png)



![png](notebook_files/notebook_51_6.png)



![png](notebook_files/notebook_51_7.png)



![png](notebook_files/notebook_51_8.png)



![png](notebook_files/notebook_51_9.png)



```python
plotCategoryRateBar('id_31', 20)

train_id_f0['_id_31_ua'] = train_id_f0['id_31'].apply(lambda x: x.split()[0] if x == x else 'unknown')
train_id_f1['_id_31_ua'] = train_id_f1['id_31'].apply(lambda x: x.split()[0] if x == x else 'unknown')
plotCategoryRateBar('_id_31_ua', 10)
```


![png](notebook_files/notebook_52_0.png)



![png](notebook_files/notebook_52_1.png)



```python
plotCategoryRateBar('id_32')
plotCategoryRateBar('id_33',15)
plotCategoryRateBar('id_34')
plotCategoryRateBar('id_35')
plotCategoryRateBar('id_36')
plotCategoryRateBar('id_37')
plotCategoryRateBar('id_38')
```


![png](notebook_files/notebook_53_0.png)



![png](notebook_files/notebook_53_1.png)



![png](notebook_files/notebook_53_2.png)



![png](notebook_files/notebook_53_3.png)



![png](notebook_files/notebook_53_4.png)



![png](notebook_files/notebook_53_5.png)



![png](notebook_files/notebook_53_6.png)


### DeviceType, DeviceInfo


```python
plotCategoryRateBar('DeviceType')
plotCategoryRateBar('DeviceInfo',10)
```


![png](notebook_files/notebook_55_0.png)



![png](notebook_files/notebook_55_1.png)


## Transaction data

* TransactionDT: timedelta from a given reference datetime (not an actual timestamp)
* TransactionAMT: transaction payment amount in USD
* ProductCD: product code, the product for each transaction
* card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.
* addr: address
* dist: distance
* P_ and (R__) emaildomain: purchaser and recipient email domain
* C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
* D1-D15: timedelta, such as days between previous transaction, etc.
* M1-M9: match, such as names on card and address, etc.
* Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.


```python
ccols = [f'C{i}' for i in range(1,15)]
dcols = [f'D{i}' for i in range(1,16)]
mcols = [f'M{i}' for i in range(1,10)]
vcols = [f'V{i}' for i in range(1,340)]
```


```python
train_trn_f0 = train_trn[train_trn['isFraud'] == 0]
train_trn_f1 = train_trn[train_trn['isFraud'] == 1]
print(train_trn_f0.shape, train_trn_f1.shape)

def plotTrnHistByFraud(col, bins=20, figsize=(8,3)):
    with np.errstate(invalid='ignore'):
        plt.figure(figsize=figsize)
        plt.hist([train_trn_f0[col], train_trn_f1[col]], bins=bins, density=True, color=['royalblue', 'orange'])

def plotTrnLogHistByFraud(col, bins=20, figsize=(8,3)):
    with np.errstate(invalid='ignore'):
        plt.figure(figsize=figsize)
        plt.hist([np.log1p(train_trn_f0[col]), np.log1p(train_trn_f1[col])], bins=bins, density=True, color=['royalblue', 'orange'])
        
def plotTrnCategoryRateBar(col, topN=np.nan, figsize=(8,3)):
    a, b = train_trn_f0, train_trn_f1
    if topN == topN: # isNotNan
        vals = b[col].value_counts(normalize=True).to_frame().iloc[:topN,0]
        subA = a.loc[a[col].isin(vals.index.values), col]
        df = pd.DataFrame({'normal':subA.value_counts(normalize=True), 'fraud':vals})
    else:
        df = pd.DataFrame({'normal':a[col].value_counts(normalize=True), 'fraud':b[col].value_counts(normalize=True)})
    df.sort_values('fraud', ascending=False).plot.bar(figsize=figsize)
```

    (569877, 404) (20663, 404)
    

### TransactionDT


```python
#START_DATE = '2017-11-30'
START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")

train_date = train_trn['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
test_date = test_trn['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

print('train date:', train_date.min(), '-', train_date.max())
print('test  date:', test_date.min(), '-', test_date.max())
```

    train date: 2017-12-02 00:00:00 - 2018-06-01 23:58:51
    test  date: 2018-07-02 00:00:24 - 2018-12-31 23:59:05
    


```python
plt.figure(figsize=(12,4))
train_trn['TransactionDT'].hist(bins=20)
test_trn['TransactionDT'].hist(bins=20)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f01230f6ef0>




![png](notebook_files/notebook_62_1.png)



```python
def appendLagDT(df):
    df = df.assign(_date_lag = df['TransactionDT'] - df.groupby(['card1','card2'])['TransactionDT'].shift(1))
    return df

train_trn = appendLagDT(train_trn)
train_trn_f0 = train_trn[train_trn['isFraud'] == 0]
train_trn_f1 = train_trn[train_trn['isFraud'] == 1]
```


```python
pd.concat([train_trn_f0['_date_lag'].describe(), 
           train_trn_f1['_date_lag'].describe()], axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>_date_lag</th>
      <th>_date_lag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>548,219.000</td>
      <td>19,898.000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>171,647.294</td>
      <td>122,550.929</td>
    </tr>
    <tr>
      <td>std</td>
      <td>707,372.753</td>
      <td>649,306.365</td>
    </tr>
    <tr>
      <td>min</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>1,004.000</td>
      <td>423.000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>6,010.000</td>
      <td>3,511.000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>59,342.000</td>
      <td>29,619.500</td>
    </tr>
    <tr>
      <td>max</td>
      <td>15,599,674.000</td>
      <td>14,447,608.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plotTrnLogHistByFraud('_date_lag')
```


![png](notebook_files/notebook_65_0.png)


### TransactionAmt


```python
plotTrnHistByFraud('TransactionAmt')
plotTrnLogHistByFraud('TransactionAmt')
```


![png](notebook_files/notebook_67_0.png)



![png](notebook_files/notebook_67_1.png)



```python
amt_desc = pd.concat([train_trn_f0['TransactionAmt'].describe(), train_trn_f1['TransactionAmt'].describe()], axis=1)
amt_desc.columns = ['normal','fraud']
amt_desc
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>normal</th>
      <th>fraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>569,877.000</td>
      <td>20,663.000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>nan</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>std</td>
      <td>nan</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>min</td>
      <td>0.251</td>
      <td>0.292</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>43.969</td>
      <td>35.031</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>68.500</td>
      <td>75.000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>120.000</td>
      <td>161.000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>31,936.000</td>
      <td>5,192.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
def appendLagAmt(df):
    df = df.assign(_amt_lag = df['TransactionAmt'] - df.groupby(['card1','card2'])['TransactionAmt'].shift(1))
    df['_amt_lag_sig'] = df['_amt_lag'].apply(lambda x: '0' if np.isnan(x) else '+' if x >=0 else '-')
    return df

train_trn = appendLagAmt(train_trn)
train_trn_f0 = train_trn[train_trn['isFraud'] == 0]
train_trn_f1 = train_trn[train_trn['isFraud'] == 1]
```


```python
plotTrnHistByFraud('_amt_lag')
plotTrnCategoryRateBar('_amt_lag_sig')
```


![png](notebook_files/notebook_70_0.png)



![png](notebook_files/notebook_70_1.png)


### ProductCD


```python
plotTrnCategoryRateBar('ProductCD')
```


![png](notebook_files/notebook_72_0.png)



```python
cols = ['ProductCD','addr1','addr2','dist1','dist2']
train_trn[cols].head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ProductCD</th>
      <th>addr1</th>
      <th>addr2</th>
      <th>dist1</th>
      <th>dist2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>W</td>
      <td>315.000</td>
      <td>87.000</td>
      <td>19.000</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>1</td>
      <td>W</td>
      <td>325.000</td>
      <td>87.000</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>2</td>
      <td>W</td>
      <td>330.000</td>
      <td>87.000</td>
      <td>287.000</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>3</td>
      <td>W</td>
      <td>476.000</td>
      <td>87.000</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>4</td>
      <td>H</td>
      <td>420.000</td>
      <td>87.000</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>5</td>
      <td>W</td>
      <td>272.000</td>
      <td>87.000</td>
      <td>36.000</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>6</td>
      <td>W</td>
      <td>126.000</td>
      <td>87.000</td>
      <td>0.000</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>7</td>
      <td>W</td>
      <td>325.000</td>
      <td>87.000</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>8</td>
      <td>H</td>
      <td>337.000</td>
      <td>87.000</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>9</td>
      <td>W</td>
      <td>204.000</td>
      <td>87.000</td>
      <td>19.000</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>10</td>
      <td>C</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>11</td>
      <td>C</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>30.000</td>
    </tr>
    <tr>
      <td>12</td>
      <td>W</td>
      <td>204.000</td>
      <td>87.000</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>13</td>
      <td>W</td>
      <td>330.000</td>
      <td>87.000</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>14</td>
      <td>W</td>
      <td>226.000</td>
      <td>87.000</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>15</td>
      <td>W</td>
      <td>315.000</td>
      <td>87.000</td>
      <td>3.000</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>16</td>
      <td>H</td>
      <td>170.000</td>
      <td>87.000</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>17</td>
      <td>H</td>
      <td>204.000</td>
      <td>87.000</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>18</td>
      <td>W</td>
      <td>184.000</td>
      <td>87.000</td>
      <td>5.000</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>19</td>
      <td>W</td>
      <td>264.000</td>
      <td>87.000</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols = ['addr1','addr2','dist1','dist2']
for f in cols:
    train_trn[f + '_isna'] = train_trn[f].isna()
```


```python
pd.crosstab(train_trn['ProductCD'], train_trn['addr1_isna'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>addr1_isna</th>
      <th>False</th>
      <th>True</th>
    </tr>
    <tr>
      <th>ProductCD</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>C</td>
      <td>3400</td>
      <td>65119</td>
    </tr>
    <tr>
      <td>H</td>
      <td>32940</td>
      <td>84</td>
    </tr>
    <tr>
      <td>R</td>
      <td>37649</td>
      <td>50</td>
    </tr>
    <tr>
      <td>S</td>
      <td>11419</td>
      <td>209</td>
    </tr>
    <tr>
      <td>W</td>
      <td>439426</td>
      <td>244</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.crosstab(train_trn['ProductCD'], train_trn['dist1_isna'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>dist1_isna</th>
      <th>False</th>
      <th>True</th>
    </tr>
    <tr>
      <th>ProductCD</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>C</td>
      <td>0</td>
      <td>68519</td>
    </tr>
    <tr>
      <td>H</td>
      <td>0</td>
      <td>33024</td>
    </tr>
    <tr>
      <td>R</td>
      <td>0</td>
      <td>37699</td>
    </tr>
    <tr>
      <td>S</td>
      <td>0</td>
      <td>11628</td>
    </tr>
    <tr>
      <td>W</td>
      <td>238269</td>
      <td>201401</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.crosstab(train_trn['ProductCD'], train_trn['dist2_isna'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>dist2_isna</th>
      <th>False</th>
      <th>True</th>
    </tr>
    <tr>
      <th>ProductCD</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>C</td>
      <td>26741</td>
      <td>41778</td>
    </tr>
    <tr>
      <td>H</td>
      <td>901</td>
      <td>32123</td>
    </tr>
    <tr>
      <td>R</td>
      <td>5524</td>
      <td>32175</td>
    </tr>
    <tr>
      <td>S</td>
      <td>4461</td>
      <td>7167</td>
    </tr>
    <tr>
      <td>W</td>
      <td>0</td>
      <td>439670</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_trn[(train_trn['dist1_isna'] == False) & (train_trn['dist2_isna'] == False)][cols]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>addr1</th>
      <th>addr2</th>
      <th>dist1</th>
      <th>dist2</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
train_trn = pd.concat([train_trn, pd.get_dummies(train_trn[['ProductCD']])], axis=1)
train_trn.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionID</th>
      <th>isFraud</th>
      <th>TransactionDT</th>
      <th>TransactionAmt</th>
      <th>ProductCD</th>
      <th>card1</th>
      <th>card2</th>
      <th>card3</th>
      <th>card4</th>
      <th>card5</th>
      <th>...</th>
      <th>_amt_lag_sig</th>
      <th>addr1_isna</th>
      <th>addr2_isna</th>
      <th>dist1_isna</th>
      <th>dist2_isna</th>
      <th>ProductCD_C</th>
      <th>ProductCD_H</th>
      <th>ProductCD_R</th>
      <th>ProductCD_S</th>
      <th>ProductCD_W</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2987000</td>
      <td>0</td>
      <td>86400</td>
      <td>68.500</td>
      <td>W</td>
      <td>13926</td>
      <td>nan</td>
      <td>150.000</td>
      <td>discover</td>
      <td>142.000</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2987001</td>
      <td>0</td>
      <td>86401</td>
      <td>29.000</td>
      <td>W</td>
      <td>2755</td>
      <td>404.000</td>
      <td>150.000</td>
      <td>mastercard</td>
      <td>102.000</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2987002</td>
      <td>0</td>
      <td>86469</td>
      <td>59.000</td>
      <td>W</td>
      <td>4663</td>
      <td>490.000</td>
      <td>150.000</td>
      <td>visa</td>
      <td>166.000</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2987003</td>
      <td>0</td>
      <td>86499</td>
      <td>50.000</td>
      <td>W</td>
      <td>18132</td>
      <td>567.000</td>
      <td>150.000</td>
      <td>mastercard</td>
      <td>117.000</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2987004</td>
      <td>0</td>
      <td>86506</td>
      <td>50.000</td>
      <td>H</td>
      <td>4497</td>
      <td>514.000</td>
      <td>150.000</td>
      <td>mastercard</td>
      <td>102.000</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 416 columns</p>
</div>




```python
cols = ['ProductCD_W','ProductCD_C','ProductCD_H','ProductCD_R','ProductCD_S','dist1_isna','dist2_isna','addr1_isna','addr2_isna']
train_trn[cols].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ProductCD_W</th>
      <th>ProductCD_C</th>
      <th>ProductCD_H</th>
      <th>ProductCD_R</th>
      <th>ProductCD_S</th>
      <th>dist1_isna</th>
      <th>dist2_isna</th>
      <th>addr1_isna</th>
      <th>addr2_isna</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ProductCD_W</td>
      <td>1.000</td>
      <td>-0.618</td>
      <td>-0.415</td>
      <td>-0.446</td>
      <td>-0.242</td>
      <td>-0.482</td>
      <td>0.445</td>
      <td>-0.601</td>
      <td>-0.601</td>
    </tr>
    <tr>
      <td>ProductCD_C</td>
      <td>-0.618</td>
      <td>1.000</td>
      <td>-0.088</td>
      <td>-0.095</td>
      <td>-0.051</td>
      <td>0.298</td>
      <td>-0.484</td>
      <td>0.967</td>
      <td>0.967</td>
    </tr>
    <tr>
      <td>ProductCD_H</td>
      <td>-0.415</td>
      <td>-0.088</td>
      <td>1.000</td>
      <td>-0.064</td>
      <td>-0.034</td>
      <td>0.200</td>
      <td>0.036</td>
      <td>-0.084</td>
      <td>-0.084</td>
    </tr>
    <tr>
      <td>ProductCD_R</td>
      <td>-0.446</td>
      <td>-0.095</td>
      <td>-0.064</td>
      <td>1.000</td>
      <td>-0.037</td>
      <td>0.215</td>
      <td>-0.089</td>
      <td>-0.091</td>
      <td>-0.091</td>
    </tr>
    <tr>
      <td>ProductCD_S</td>
      <td>-0.242</td>
      <td>-0.051</td>
      <td>-0.034</td>
      <td>-0.037</td>
      <td>1.000</td>
      <td>0.117</td>
      <td>-0.186</td>
      <td>-0.042</td>
      <td>-0.042</td>
    </tr>
    <tr>
      <td>dist1_isna</td>
      <td>-0.482</td>
      <td>0.298</td>
      <td>0.200</td>
      <td>0.215</td>
      <td>0.117</td>
      <td>1.000</td>
      <td>-0.215</td>
      <td>0.290</td>
      <td>0.290</td>
    </tr>
    <tr>
      <td>dist2_isna</td>
      <td>0.445</td>
      <td>-0.484</td>
      <td>0.036</td>
      <td>-0.089</td>
      <td>-0.186</td>
      <td>-0.215</td>
      <td>1.000</td>
      <td>-0.462</td>
      <td>-0.462</td>
    </tr>
    <tr>
      <td>addr1_isna</td>
      <td>-0.601</td>
      <td>0.967</td>
      <td>-0.084</td>
      <td>-0.091</td>
      <td>-0.042</td>
      <td>0.290</td>
      <td>-0.462</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>addr2_isna</td>
      <td>-0.601</td>
      <td>0.967</td>
      <td>-0.084</td>
      <td>-0.091</td>
      <td>-0.042</td>
      <td>0.290</td>
      <td>-0.462</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_trn['_amount_max_ProductCD'] = train_trn.groupby(['ProductCD'])['TransactionAmt'].transform('max')
train_trn[['ProductCD','_amount_max_ProductCD']].drop_duplicates().sort_values(by='_amount_max_ProductCD', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ProductCD</th>
      <th>_amount_max_ProductCD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>W</td>
      <td>31,936.000</td>
    </tr>
    <tr>
      <td>99</td>
      <td>R</td>
      <td>1,800.000</td>
    </tr>
    <tr>
      <td>38</td>
      <td>S</td>
      <td>1,550.000</td>
    </tr>
    <tr>
      <td>10</td>
      <td>C</td>
      <td>713.000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>H</td>
      <td>500.000</td>
    </tr>
  </tbody>
</table>
</div>



### card1 - card6


```python
cols = [f'card{n}' for n in range(1,7)]
train_trn[cols].isnull().sum()
```




    card1       0
    card2    8933
    card3    1565
    card4    1577
    card5    4259
    card6    1571
    dtype: int64




```python
train_trn[cols].nunique()
```




    card1    13553
    card2      500
    card3      114
    card4        4
    card5      119
    card6        4
    dtype: int64




```python
train_trn[cols].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>card1</th>
      <th>card2</th>
      <th>card3</th>
      <th>card4</th>
      <th>card5</th>
      <th>card6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>13926</td>
      <td>nan</td>
      <td>150.000</td>
      <td>discover</td>
      <td>142.000</td>
      <td>credit</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2755</td>
      <td>404.000</td>
      <td>150.000</td>
      <td>mastercard</td>
      <td>102.000</td>
      <td>credit</td>
    </tr>
    <tr>
      <td>2</td>
      <td>4663</td>
      <td>490.000</td>
      <td>150.000</td>
      <td>visa</td>
      <td>166.000</td>
      <td>debit</td>
    </tr>
    <tr>
      <td>3</td>
      <td>18132</td>
      <td>567.000</td>
      <td>150.000</td>
      <td>mastercard</td>
      <td>117.000</td>
      <td>debit</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4497</td>
      <td>514.000</td>
      <td>150.000</td>
      <td>mastercard</td>
      <td>102.000</td>
      <td>credit</td>
    </tr>
    <tr>
      <td>5</td>
      <td>5937</td>
      <td>555.000</td>
      <td>150.000</td>
      <td>visa</td>
      <td>226.000</td>
      <td>debit</td>
    </tr>
    <tr>
      <td>6</td>
      <td>12308</td>
      <td>360.000</td>
      <td>150.000</td>
      <td>visa</td>
      <td>166.000</td>
      <td>debit</td>
    </tr>
    <tr>
      <td>7</td>
      <td>12695</td>
      <td>490.000</td>
      <td>150.000</td>
      <td>visa</td>
      <td>226.000</td>
      <td>debit</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2803</td>
      <td>100.000</td>
      <td>150.000</td>
      <td>visa</td>
      <td>226.000</td>
      <td>debit</td>
    </tr>
    <tr>
      <td>9</td>
      <td>17399</td>
      <td>111.000</td>
      <td>150.000</td>
      <td>mastercard</td>
      <td>224.000</td>
      <td>debit</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_trn[train_trn['card4']=='visa']['card1'].hist(bins=50)
train_trn[train_trn['card4']=='mastercard']['card1'].hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f0120a35d68>




![png](notebook_files/notebook_86_1.png)



```python
train_trn[train_trn['card4']=='visa']['card2'].hist(bins=50)
train_trn[train_trn['card4']=='mastercard']['card2'].hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f01203a2080>




![png](notebook_files/notebook_87_1.png)



```python
plotTrnCategoryRateBar('card1', 15)
plotTrnHistByFraud('card1', bins=30)
```


![png](notebook_files/notebook_88_0.png)



![png](notebook_files/notebook_88_1.png)



```python
plotTrnCategoryRateBar('card2', 15)
plotTrnHistByFraud('card2', bins=30)
```


![png](notebook_files/notebook_89_0.png)



![png](notebook_files/notebook_89_1.png)



```python
train_trn_f0['_card1_card2'] = train_trn_f0['card1'].astype(str) + '_' + train_trn_f0['card2'].astype(str)
train_trn_f1['_card1_card2'] = train_trn_f1['card1'].astype(str) + '_' + train_trn_f1['card2'].astype(str)

plotTrnCategoryRateBar('_card1_card2', 50, figsize=(15,3))
```


![png](notebook_files/notebook_90_0.png)



```python
plotTrnCategoryRateBar('card3', 10)
```


![png](notebook_files/notebook_91_0.png)



```python
plotTrnCategoryRateBar('card4')
```


![png](notebook_files/notebook_92_0.png)



```python
plotTrnCategoryRateBar('card5', 10)
```


![png](notebook_files/notebook_93_0.png)



```python
plotTrnCategoryRateBar('card6')
```


![png](notebook_files/notebook_94_0.png)



```python
print(len(train_trn))
print(train_trn['card1'].nunique(), train_trn['card2'].nunique(), train_trn['card3'].nunique(), train_trn['card5'].nunique())

train_trn['card_n'] = (train_trn['card1'].astype(str) + '_' + train_trn['card2'].astype(str) \
       + '_' + train_trn['card3'].astype(str) + '_' + train_trn['card5'].astype(str))
print('unique cards:', train_trn['card_n'].nunique())
```

    590540
    13553 500 114 119
    unique cards: 14845
    


```python
vc = train_trn['card_n'].value_counts()
vc[vc > 3000].plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f012048fcf8>




![png](notebook_files/notebook_96_1.png)



```python
train_trn.groupby(['card_n'])['isFraud'].mean().sort_values(ascending=False)
```




    card_n
    4774_555.0_150.0_226.0    1.000
    5848_555.0_223.0_226.0    1.000
    1509_555.0_150.0_226.0    1.000
    6314_555.0_182.0_102.0    1.000
    14770_142.0_185.0_224.0   1.000
                               ... 
    3783_225.0_150.0_224.0    0.000
    3781_369.0_150.0_102.0    0.000
    3779_555.0_150.0_226.0    0.000
    3776_512.0_150.0_117.0    0.000
    10000_111.0_150.0_117.0   0.000
    Name: isFraud, Length: 14845, dtype: float64



### addr1, addr2


```python
train_trn['addr1'].nunique(), train_trn['addr2'].nunique()
```




    (332, 74)




```python
plotTrnCategoryRateBar('addr1', 20)
plotTrnHistByFraud('addr1', bins=30)
```


![png](notebook_files/notebook_100_0.png)



![png](notebook_files/notebook_100_1.png)



```python
train_trn['addr1'].value_counts(dropna=False).to_frame().iloc[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>addr1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>nan</td>
      <td>65706</td>
    </tr>
    <tr>
      <td>299.000</td>
      <td>46335</td>
    </tr>
    <tr>
      <td>325.000</td>
      <td>42751</td>
    </tr>
    <tr>
      <td>204.000</td>
      <td>42020</td>
    </tr>
    <tr>
      <td>264.000</td>
      <td>39870</td>
    </tr>
    <tr>
      <td>330.000</td>
      <td>26287</td>
    </tr>
    <tr>
      <td>315.000</td>
      <td>23078</td>
    </tr>
    <tr>
      <td>441.000</td>
      <td>20827</td>
    </tr>
    <tr>
      <td>272.000</td>
      <td>20141</td>
    </tr>
    <tr>
      <td>123.000</td>
      <td>16105</td>
    </tr>
  </tbody>
</table>
</div>




```python
plotTrnCategoryRateBar('addr2', 10)
print('addr2 nunique:', train_trn['addr2'].nunique())
```

    addr2 nunique: 74
    


![png](notebook_files/notebook_102_1.png)



```python
train_trn['addr2'].value_counts(dropna=False).to_frame().iloc[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>addr2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>87.000</td>
      <td>520481</td>
    </tr>
    <tr>
      <td>nan</td>
      <td>65706</td>
    </tr>
    <tr>
      <td>60.000</td>
      <td>3084</td>
    </tr>
    <tr>
      <td>96.000</td>
      <td>638</td>
    </tr>
    <tr>
      <td>32.000</td>
      <td>91</td>
    </tr>
    <tr>
      <td>65.000</td>
      <td>82</td>
    </tr>
    <tr>
      <td>16.000</td>
      <td>55</td>
    </tr>
    <tr>
      <td>31.000</td>
      <td>47</td>
    </tr>
    <tr>
      <td>19.000</td>
      <td>33</td>
    </tr>
    <tr>
      <td>26.000</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>



### dist1, dist2


```python
plotTrnCategoryRateBar('dist1', 20)
```


![png](notebook_files/notebook_105_0.png)



```python
plotTrnCategoryRateBar('dist2', 20)
```


![png](notebook_files/notebook_106_0.png)



```python
train_trn_f0['dist3'] = np.where(train_trn_f0['dist1'].isna(), train_trn_f0['dist2'], train_trn_f0['dist1'])
train_trn_f1['dist3'] = np.where(train_trn_f1['dist1'].isna(), train_trn_f1['dist2'], train_trn_f1['dist1'])

plotTrnCategoryRateBar('dist3', 20)
plotTrnLogHistByFraud('dist3')
```


![png](notebook_files/notebook_107_0.png)



![png](notebook_files/notebook_107_1.png)


### P_emaildomain, R_emaildomain


```python
plotTrnCategoryRateBar('P_emaildomain',10)
plotTrnCategoryRateBar('R_emaildomain',10)
```


![png](notebook_files/notebook_109_0.png)



![png](notebook_files/notebook_109_1.png)



```python
train_trn['P_emaildomain'].fillna('unknown',inplace=True)
train_trn['R_emaildomain'].fillna('unknown',inplace=True)

inf = pd.DataFrame([], columns=['P_emaildomain','R_emaildomain','Count','isFraud'])
for n in (train_trn['P_emaildomain'] + ' ' + train_trn['R_emaildomain']).unique():
    p, r = n.split()[0], n.split()[1]
    df = train_trn[(train_trn['P_emaildomain'] == p) & (train_trn['R_emaildomain'] == r)]
    inf = inf.append(pd.DataFrame([p, r, len(df), df['isFraud'].mean()], index=inf.columns).T)

inf.sort_values(by='isFraud', ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>P_emaildomain</th>
      <th>R_emaildomain</th>
      <th>Count</th>
      <th>isFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>yahoo.com</td>
      <td>mail.com</td>
      <td>1</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>0</td>
      <td>anonymous.com</td>
      <td>protonmail.com</td>
      <td>2</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>0</td>
      <td>aol.com</td>
      <td>mail.com</td>
      <td>3</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>0</td>
      <td>unknown</td>
      <td>protonmail.com</td>
      <td>6</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>0</td>
      <td>protonmail.com</td>
      <td>gmail.com</td>
      <td>4</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>0</td>
      <td>protonmail.com</td>
      <td>protonmail.com</td>
      <td>23</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>0</td>
      <td>yahoo.com</td>
      <td>protonmail.com</td>
      <td>1</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>0</td>
      <td>gmail.com</td>
      <td>protonmail.com</td>
      <td>6</td>
      <td>0.833</td>
    </tr>
    <tr>
      <td>0</td>
      <td>mail.com</td>
      <td>hotmail.com</td>
      <td>4</td>
      <td>0.750</td>
    </tr>
    <tr>
      <td>0</td>
      <td>aol.com</td>
      <td>protonmail.com</td>
      <td>3</td>
      <td>0.667</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_trn_f1['P_emaildomain_prefix'] = train_trn_f1['P_emaildomain'].fillna('unknown').apply(lambda x: x.split('.')[0])
pd.crosstab(train_trn_f1['P_emaildomain_prefix'], train_trn_f1['ProductCD']).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>P_emaildomain_prefix</th>
      <th>aim</th>
      <th>anonymous</th>
      <th>aol</th>
      <th>att</th>
      <th>bellsouth</th>
      <th>cableone</th>
      <th>charter</th>
      <th>comcast</th>
      <th>cox</th>
      <th>earthlink</th>
      <th>...</th>
      <th>protonmail</th>
      <th>roadrunner</th>
      <th>rocketmail</th>
      <th>sbcglobal</th>
      <th>sc</th>
      <th>suddenlink</th>
      <th>unknown</th>
      <th>verizon</th>
      <th>yahoo</th>
      <th>ymail</th>
    </tr>
    <tr>
      <th>ProductCD</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>C</td>
      <td>0</td>
      <td>366</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>311</td>
      <td>0</td>
      <td>124</td>
      <td>0</td>
    </tr>
    <tr>
      <td>H</td>
      <td>0</td>
      <td>143</td>
      <td>42</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>0</td>
      <td>6</td>
      <td>...</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>179</td>
      <td>9</td>
    </tr>
    <tr>
      <td>R</td>
      <td>1</td>
      <td>122</td>
      <td>34</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>3</td>
      <td>10</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>7</td>
      <td>218</td>
      <td>3</td>
    </tr>
    <tr>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>686</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>W</td>
      <td>39</td>
      <td>228</td>
      <td>535</td>
      <td>30</td>
      <td>44</td>
      <td>3</td>
      <td>22</td>
      <td>204</td>
      <td>25</td>
      <td>5</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>1791</td>
      <td>13</td>
      <td>1799</td>
      <td>38</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




```python
train_trn['P_emaildomain_prefix'] = train_trn['P_emaildomain'].apply(lambda x: x.split('.')[0])
ct = pd.crosstab(train_trn['P_emaildomain_prefix'], train_trn['ProductCD'])
ct = ct.sort_values(by='W')[-15:]
ct.plot.barh(stacked=True, figsize=(12,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f011dcf06a0>




![png](notebook_files/notebook_112_1.png)


### C1 - C14


```python
for i in range(1,15):
    plotTrnCategoryRateBar(f'C{i}',10)
```


![png](notebook_files/notebook_114_0.png)



![png](notebook_files/notebook_114_1.png)



![png](notebook_files/notebook_114_2.png)



![png](notebook_files/notebook_114_3.png)



![png](notebook_files/notebook_114_4.png)



![png](notebook_files/notebook_114_5.png)



![png](notebook_files/notebook_114_6.png)



![png](notebook_files/notebook_114_7.png)



![png](notebook_files/notebook_114_8.png)



![png](notebook_files/notebook_114_9.png)



![png](notebook_files/notebook_114_10.png)



![png](notebook_files/notebook_114_11.png)



![png](notebook_files/notebook_114_12.png)



![png](notebook_files/notebook_114_13.png)



```python
train_trn[ccols].describe().loc[['count','mean','std','min','max']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C1</th>
      <th>C2</th>
      <th>C3</th>
      <th>C4</th>
      <th>C5</th>
      <th>C6</th>
      <th>C7</th>
      <th>C8</th>
      <th>C9</th>
      <th>C10</th>
      <th>C11</th>
      <th>C12</th>
      <th>C13</th>
      <th>C14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>590,540.000</td>
      <td>590,540.000</td>
      <td>590,540.000</td>
      <td>590,540.000</td>
      <td>590,540.000</td>
      <td>590,540.000</td>
      <td>590,540.000</td>
      <td>590,540.000</td>
      <td>590,540.000</td>
      <td>590,540.000</td>
      <td>590,540.000</td>
      <td>590,540.000</td>
      <td>590,540.000</td>
      <td>590,540.000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.006</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>std</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.151</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>min</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>4,684.000</td>
      <td>5,692.000</td>
      <td>26.000</td>
      <td>2,252.000</td>
      <td>349.000</td>
      <td>2,252.000</td>
      <td>2,256.000</td>
      <td>3,332.000</td>
      <td>210.000</td>
      <td>3,256.000</td>
      <td>3,188.000</td>
      <td>3,188.000</td>
      <td>2,918.000</td>
      <td>1,429.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10,5))

corr = train_trn[['isFraud'] + ccols].corr()
sns.heatmap(corr, annot=True, fmt='.2f')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f011d9ab5c0>




![png](notebook_files/notebook_116_1.png)


### Cx & card


```python
cols = ['TransactionDT','TransactionAmt','isFraud'] + ccols
train_trn[train_trn['card1'] == 9500][cols].head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionDT</th>
      <th>TransactionAmt</th>
      <th>isFraud</th>
      <th>C1</th>
      <th>C2</th>
      <th>C3</th>
      <th>C4</th>
      <th>C5</th>
      <th>C6</th>
      <th>C7</th>
      <th>C8</th>
      <th>C9</th>
      <th>C10</th>
      <th>C11</th>
      <th>C12</th>
      <th>C13</th>
      <th>C14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>107</td>
      <td>88258</td>
      <td>226.000</td>
      <td>0</td>
      <td>3.000</td>
      <td>3.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>2.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>2.000</td>
      <td>0.000</td>
      <td>8.000</td>
      <td>2.000</td>
    </tr>
    <tr>
      <td>122</td>
      <td>88538</td>
      <td>80.000</td>
      <td>0</td>
      <td>21.000</td>
      <td>25.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>33.000</td>
      <td>22.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>17.000</td>
      <td>0.000</td>
      <td>19.000</td>
      <td>0.000</td>
      <td>99.000</td>
      <td>18.000</td>
    </tr>
    <tr>
      <td>161</td>
      <td>89085</td>
      <td>107.938</td>
      <td>0</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>4.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>5.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>174</td>
      <td>89250</td>
      <td>107.938</td>
      <td>0</td>
      <td>3.000</td>
      <td>3.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>7.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>2.000</td>
      <td>0.000</td>
      <td>2.000</td>
      <td>0.000</td>
      <td>31.000</td>
      <td>3.000</td>
    </tr>
    <tr>
      <td>225</td>
      <td>89969</td>
      <td>43.000</td>
      <td>0</td>
      <td>2.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>2.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>7.000</td>
      <td>2.000</td>
    </tr>
    <tr>
      <td>255</td>
      <td>90440</td>
      <td>54.000</td>
      <td>0</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>332</td>
      <td>91758</td>
      <td>100.000</td>
      <td>0</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td>335</td>
      <td>91804</td>
      <td>100.000</td>
      <td>0</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td>336</td>
      <td>91824</td>
      <td>100.000</td>
      <td>0</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td>338</td>
      <td>91884</td>
      <td>100.000</td>
      <td>0</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td>363</td>
      <td>92285</td>
      <td>29.500</td>
      <td>0</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>519</td>
      <td>95335</td>
      <td>57.938</td>
      <td>0</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>1081</td>
      <td>117398</td>
      <td>2,160.000</td>
      <td>1</td>
      <td>3.000</td>
      <td>5.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>3.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3.000</td>
      <td>0.000</td>
      <td>4.000</td>
      <td>0.000</td>
      <td>2.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>1092</td>
      <td>118524</td>
      <td>226.000</td>
      <td>0</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>1118</td>
      <td>120874</td>
      <td>107.938</td>
      <td>0</td>
      <td>137.000</td>
      <td>119.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>136.000</td>
      <td>90.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>73.000</td>
      <td>0.000</td>
      <td>93.000</td>
      <td>0.000</td>
      <td>543.000</td>
      <td>120.000</td>
    </tr>
    <tr>
      <td>1216</td>
      <td>127935</td>
      <td>57.938</td>
      <td>0</td>
      <td>132.000</td>
      <td>107.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>82.000</td>
      <td>86.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>70.000</td>
      <td>0.000</td>
      <td>88.000</td>
      <td>0.000</td>
      <td>424.000</td>
      <td>104.000</td>
    </tr>
    <tr>
      <td>1225</td>
      <td>128307</td>
      <td>97.000</td>
      <td>0</td>
      <td>8.000</td>
      <td>5.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>4.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>2.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>11.000</td>
      <td>5.000</td>
    </tr>
    <tr>
      <td>1250</td>
      <td>129171</td>
      <td>226.000</td>
      <td>1</td>
      <td>2.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>2.000</td>
      <td>2.000</td>
    </tr>
    <tr>
      <td>1258</td>
      <td>129424</td>
      <td>50.000</td>
      <td>0</td>
      <td>107.000</td>
      <td>84.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>82.000</td>
      <td>66.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>55.000</td>
      <td>0.000</td>
      <td>66.000</td>
      <td>0.000</td>
      <td>383.000</td>
      <td>85.000</td>
    </tr>
    <tr>
      <td>1268</td>
      <td>129709</td>
      <td>53.969</td>
      <td>0</td>
      <td>107.000</td>
      <td>84.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>82.000</td>
      <td>67.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>55.000</td>
      <td>0.000</td>
      <td>66.000</td>
      <td>0.000</td>
      <td>383.000</td>
      <td>85.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols = ['TransactionDT','TransactionAmt','isFraud'] + ccols
train_trn[train_trn['card1'] == 4774][cols].head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionDT</th>
      <th>TransactionAmt</th>
      <th>isFraud</th>
      <th>C1</th>
      <th>C2</th>
      <th>C3</th>
      <th>C4</th>
      <th>C5</th>
      <th>C6</th>
      <th>C7</th>
      <th>C8</th>
      <th>C9</th>
      <th>C10</th>
      <th>C11</th>
      <th>C12</th>
      <th>C13</th>
      <th>C14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>389687</td>
      <td>9766493</td>
      <td>445.000</td>
      <td>1</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>393954</td>
      <td>9906065</td>
      <td>445.000</td>
      <td>1</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>393970</td>
      <td>9906367</td>
      <td>445.000</td>
      <td>1</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>394352</td>
      <td>9913683</td>
      <td>445.000</td>
      <td>1</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>394372</td>
      <td>9913975</td>
      <td>445.000</td>
      <td>1</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>2.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>394375</td>
      <td>9914050</td>
      <td>426.000</td>
      <td>1</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>2.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_trn[train_trn['card1'] == 14770][cols].head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionDT</th>
      <th>TransactionAmt</th>
      <th>isFraud</th>
      <th>C1</th>
      <th>C2</th>
      <th>C3</th>
      <th>C4</th>
      <th>C5</th>
      <th>C6</th>
      <th>C7</th>
      <th>C8</th>
      <th>C9</th>
      <th>C10</th>
      <th>C11</th>
      <th>C12</th>
      <th>C13</th>
      <th>C14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>78089</td>
      <td>1706422</td>
      <td>64.688</td>
      <td>1</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>78134</td>
      <td>1706831</td>
      <td>64.688</td>
      <td>1</td>
      <td>1.000</td>
      <td>2.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>



### D1-D15


```python
for i in range(1,16):
    plotTrnCategoryRateBar(f'D{i}',10)
```


![png](notebook_files/notebook_122_0.png)



![png](notebook_files/notebook_122_1.png)



![png](notebook_files/notebook_122_2.png)



![png](notebook_files/notebook_122_3.png)



![png](notebook_files/notebook_122_4.png)



![png](notebook_files/notebook_122_5.png)



![png](notebook_files/notebook_122_6.png)



![png](notebook_files/notebook_122_7.png)



![png](notebook_files/notebook_122_8.png)



![png](notebook_files/notebook_122_9.png)



![png](notebook_files/notebook_122_10.png)



![png](notebook_files/notebook_122_11.png)



![png](notebook_files/notebook_122_12.png)



![png](notebook_files/notebook_122_13.png)



![png](notebook_files/notebook_122_14.png)



```python
train_trn[dcols].describe().loc[['count','mean','std','min','max']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>D1</th>
      <th>D2</th>
      <th>D3</th>
      <th>D4</th>
      <th>D5</th>
      <th>D6</th>
      <th>D7</th>
      <th>D8</th>
      <th>D9</th>
      <th>D10</th>
      <th>D11</th>
      <th>D12</th>
      <th>D13</th>
      <th>D14</th>
      <th>D15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>589,271.000</td>
      <td>309,743.000</td>
      <td>327,662.000</td>
      <td>421,618.000</td>
      <td>280,699.000</td>
      <td>73,187.000</td>
      <td>38,917.000</td>
      <td>74,926.000</td>
      <td>74,926.000</td>
      <td>514,518.000</td>
      <td>311,253.000</td>
      <td>64,717.000</td>
      <td>61,952.000</td>
      <td>62,187.000</td>
      <td>501,427.000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>inf</td>
      <td>nan</td>
      <td>0.560</td>
      <td>nan</td>
      <td>nan</td>
      <td>inf</td>
      <td>inf</td>
      <td>inf</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>std</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>inf</td>
      <td>nan</td>
      <td>0.317</td>
      <td>nan</td>
      <td>nan</td>
      <td>inf</td>
      <td>inf</td>
      <td>inf</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>min</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-122.000</td>
      <td>0.000</td>
      <td>-83.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-53.000</td>
      <td>-83.000</td>
      <td>0.000</td>
      <td>-193.000</td>
      <td>-83.000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>640.000</td>
      <td>640.000</td>
      <td>819.000</td>
      <td>869.000</td>
      <td>819.000</td>
      <td>873.000</td>
      <td>843.000</td>
      <td>1,708.000</td>
      <td>0.958</td>
      <td>876.000</td>
      <td>670.000</td>
      <td>648.000</td>
      <td>847.000</td>
      <td>878.000</td>
      <td>879.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12,4))

plt.scatter(train_trn_f0['TransactionDT'], train_trn_f0['D1'], s=2)
plt.scatter(train_trn_f1['TransactionDT'], train_trn_f1['D1'], s=2, c='r')
plt.scatter(test_trn['TransactionDT'], test_trn['D1'], s=2, c='g')
```




    <matplotlib.collections.PathCollection at 0x7f012048f320>




![png](notebook_files/notebook_124_1.png)



```python
plt.figure(figsize=(12,4))

# ref. https://www.kaggle.com/kyakovlev/ieee-columns-scaling
plt.scatter(train_trn_f0['TransactionDT'], train_trn_f0['D15'], s=2)
plt.scatter(train_trn_f1['TransactionDT'], train_trn_f1['D15'], s=2, c='r')
plt.scatter(test_trn['TransactionDT'], test_trn['D15'], s=2, c='g')
```




    <matplotlib.collections.PathCollection at 0x7f0120496cf8>




![png](notebook_files/notebook_125_1.png)



```python
plt.figure(figsize=(10,5))

corr = train_trn[['isFraud'] + dcols].corr()
sns.heatmap(corr, annot=True, fmt='.2f')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f011d434ac8>




![png](notebook_files/notebook_126_1.png)



```python
fig, ax = plt.subplots(1, 2, figsize=(15, 3))
train_trn.loc[train_trn['isFraud']==0, dcols].isnull().sum(axis=1).to_frame().hist(ax=ax[0], bins=20)
train_trn.loc[train_trn['isFraud']==1, dcols].isnull().sum(axis=1).to_frame().hist(ax=ax[1], bins=20)
```




    array([<matplotlib.axes._subplots.AxesSubplot object at 0x7f011da82b70>],
          dtype=object)




![png](notebook_files/notebook_127_1.png)


### Dx & card


```python
cols = ['TransactionDT','TransactionAmt','isFraud'] + dcols
train_trn[train_trn['card1'] == 9500][cols].head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionDT</th>
      <th>TransactionAmt</th>
      <th>isFraud</th>
      <th>D1</th>
      <th>D2</th>
      <th>D3</th>
      <th>D4</th>
      <th>D5</th>
      <th>D6</th>
      <th>D7</th>
      <th>D8</th>
      <th>D9</th>
      <th>D10</th>
      <th>D11</th>
      <th>D12</th>
      <th>D13</th>
      <th>D14</th>
      <th>D15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>107</td>
      <td>88258</td>
      <td>226.000</td>
      <td>0</td>
      <td>119.000</td>
      <td>119.000</td>
      <td>28.000</td>
      <td>327.000</td>
      <td>28.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>327.000</td>
      <td>226.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>327.000</td>
    </tr>
    <tr>
      <td>122</td>
      <td>88538</td>
      <td>80.000</td>
      <td>0</td>
      <td>416.000</td>
      <td>416.000</td>
      <td>39.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>343.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>416.000</td>
    </tr>
    <tr>
      <td>161</td>
      <td>89085</td>
      <td>107.938</td>
      <td>0</td>
      <td>125.000</td>
      <td>125.000</td>
      <td>59.000</td>
      <td>124.000</td>
      <td>59.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>124.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>124.000</td>
    </tr>
    <tr>
      <td>174</td>
      <td>89250</td>
      <td>107.938</td>
      <td>0</td>
      <td>60.000</td>
      <td>60.000</td>
      <td>15.000</td>
      <td>413.000</td>
      <td>15.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>15.000</td>
      <td>276.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>413.000</td>
    </tr>
    <tr>
      <td>225</td>
      <td>89969</td>
      <td>43.000</td>
      <td>0</td>
      <td>86.000</td>
      <td>86.000</td>
      <td>4.000</td>
      <td>299.000</td>
      <td>4.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>299.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>299.000</td>
    </tr>
    <tr>
      <td>255</td>
      <td>90440</td>
      <td>54.000</td>
      <td>0</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td>332</td>
      <td>91758</td>
      <td>100.000</td>
      <td>0</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>335</td>
      <td>91804</td>
      <td>100.000</td>
      <td>0</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.042</td>
      <td>0.042</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>336</td>
      <td>91824</td>
      <td>100.000</td>
      <td>0</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.042</td>
      <td>0.042</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>338</td>
      <td>91884</td>
      <td>100.000</td>
      <td>0</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.042</td>
      <td>0.042</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>363</td>
      <td>92285</td>
      <td>29.500</td>
      <td>0</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td>519</td>
      <td>95335</td>
      <td>57.938</td>
      <td>0</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td>1081</td>
      <td>117398</td>
      <td>2,160.000</td>
      <td>1</td>
      <td>315.000</td>
      <td>nan</td>
      <td>5.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>315.000</td>
      <td>318.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>225.000</td>
    </tr>
    <tr>
      <td>1092</td>
      <td>118524</td>
      <td>226.000</td>
      <td>0</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>59.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td>1118</td>
      <td>120874</td>
      <td>107.938</td>
      <td>0</td>
      <td>23.000</td>
      <td>23.000</td>
      <td>13.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>340.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>344.000</td>
    </tr>
    <tr>
      <td>1216</td>
      <td>127935</td>
      <td>57.938</td>
      <td>0</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>197.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>197.000</td>
    </tr>
    <tr>
      <td>1225</td>
      <td>128307</td>
      <td>97.000</td>
      <td>0</td>
      <td>291.000</td>
      <td>291.000</td>
      <td>14.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>153.000</td>
      <td>420.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>420.000</td>
    </tr>
    <tr>
      <td>1250</td>
      <td>129171</td>
      <td>226.000</td>
      <td>1</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>87.000</td>
      <td>87.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>87.000</td>
      <td>87.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>87.000</td>
    </tr>
    <tr>
      <td>1258</td>
      <td>129424</td>
      <td>50.000</td>
      <td>0</td>
      <td>0.000</td>
      <td>nan</td>
      <td>0.000</td>
      <td>31.000</td>
      <td>30.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>197.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>197.000</td>
    </tr>
    <tr>
      <td>1268</td>
      <td>129709</td>
      <td>53.969</td>
      <td>0</td>
      <td>0.000</td>
      <td>nan</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>197.000</td>
      <td>296.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>197.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols = ['TransactionDT','TransactionAmt','isFraud'] + dcols
train_trn[train_trn['card1'] == 4774][cols].head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionDT</th>
      <th>TransactionAmt</th>
      <th>isFraud</th>
      <th>D1</th>
      <th>D2</th>
      <th>D3</th>
      <th>D4</th>
      <th>D5</th>
      <th>D6</th>
      <th>D7</th>
      <th>D8</th>
      <th>D9</th>
      <th>D10</th>
      <th>D11</th>
      <th>D12</th>
      <th>D13</th>
      <th>D14</th>
      <th>D15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>389687</td>
      <td>9766493</td>
      <td>445.000</td>
      <td>1</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td>393954</td>
      <td>9906065</td>
      <td>445.000</td>
      <td>1</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>2.000</td>
    </tr>
    <tr>
      <td>393970</td>
      <td>9906367</td>
      <td>445.000</td>
      <td>1</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>2.000</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>2.000</td>
    </tr>
    <tr>
      <td>394352</td>
      <td>9913683</td>
      <td>445.000</td>
      <td>1</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>2.000</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>2.000</td>
    </tr>
    <tr>
      <td>394372</td>
      <td>9913975</td>
      <td>445.000</td>
      <td>1</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>2.000</td>
    </tr>
    <tr>
      <td>394375</td>
      <td>9914050</td>
      <td>426.000</td>
      <td>1</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>2.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_trn[train_trn['card1'] == 14770][cols].head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionDT</th>
      <th>TransactionAmt</th>
      <th>isFraud</th>
      <th>D1</th>
      <th>D2</th>
      <th>D3</th>
      <th>D4</th>
      <th>D5</th>
      <th>D6</th>
      <th>D7</th>
      <th>D8</th>
      <th>D9</th>
      <th>D10</th>
      <th>D11</th>
      <th>D12</th>
      <th>D13</th>
      <th>D14</th>
      <th>D15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>78089</td>
      <td>1706422</td>
      <td>64.688</td>
      <td>1</td>
      <td>0.000</td>
      <td>nan</td>
      <td>nan</td>
      <td>0.000</td>
      <td>nan</td>
      <td>0.000</td>
      <td>nan</td>
      <td>0.750</td>
      <td>0.750</td>
      <td>0.000</td>
      <td>nan</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td>78134</td>
      <td>1706831</td>
      <td>64.688</td>
      <td>1</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.750</td>
      <td>0.750</td>
      <td>0.000</td>
      <td>nan</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>



### M1 - M9


```python
plotTrnCategoryRateBar('M1')
plotTrnCategoryRateBar('M2')
plotTrnCategoryRateBar('M3')
plotTrnCategoryRateBar('M4')
plotTrnCategoryRateBar('M5')
plotTrnCategoryRateBar('M6')
plotTrnCategoryRateBar('M7')
plotTrnCategoryRateBar('M8')
plotTrnCategoryRateBar('M9')
```


![png](notebook_files/notebook_133_0.png)



![png](notebook_files/notebook_133_1.png)



![png](notebook_files/notebook_133_2.png)



![png](notebook_files/notebook_133_3.png)



![png](notebook_files/notebook_133_4.png)



![png](notebook_files/notebook_133_5.png)



![png](notebook_files/notebook_133_6.png)



![png](notebook_files/notebook_133_7.png)



![png](notebook_files/notebook_133_8.png)


### Vxxx


```python
for f in ['V1','V14','V41','V65','V88','V107','V305']:
    plotTrnCategoryRateBar(f)
```


![png](notebook_files/notebook_135_0.png)



![png](notebook_files/notebook_135_1.png)



![png](notebook_files/notebook_135_2.png)



![png](notebook_files/notebook_135_3.png)



![png](notebook_files/notebook_135_4.png)



![png](notebook_files/notebook_135_5.png)



![png](notebook_files/notebook_135_6.png)



```python
vsum0 = train_trn_f0[vcols].sum(axis=1)
vsum1 = train_trn_f1[vcols].sum(axis=1)
plt.scatter(train_trn_f0['_ymd'], vsum0, s=5)
plt.scatter(train_trn_f1['_ymd'], vsum1, s=5, c='r')
```




    <matplotlib.collections.PathCollection at 0x7f011d686080>




![png](notebook_files/notebook_136_1.png)



```python
m = train_trn_f1[vcols].describe().T['max']
m[m >= 10000]
```




    V127    19,860.000
    V128    10,162.000
    V133    15,607.000
    V159    43,552.000
    V160   639,717.438
    V165    17,468.949
    V202    10,900.000
    V203    81,450.000
    V204    37,850.000
    V207    20,080.000
    V212    34,625.000
    V213    16,950.000
    V215    13,725.000
    V264    10,035.000
    V306    11,848.000
    V307    83,258.367
    V308    18,123.957
    V316    11,848.000
    V317    82,130.953
    V318    18,123.957
    Name: max, dtype: float64




```python
plt.scatter(train_trn_f0['_ymd'], train_trn_f0['V160'], s=5)
plt.scatter(train_trn_f1['_ymd'], train_trn_f1['V160'], s=5, c='r')
```




    <matplotlib.collections.PathCollection at 0x7f011d1fed30>




![png](notebook_files/notebook_138_1.png)



```python
vcols_1 = [f'V{i}' for i in range(1,160)]+[f'V{i}' for i in range(161,340)]
vsum0 = train_trn_f0[vcols_1].sum(axis=1)
vsum1 = train_trn_f1[vcols_1].sum(axis=1)
plt.scatter(train_trn_f0['_ymd'], vsum0, s=5)
plt.scatter(train_trn_f1['_ymd'], vsum1, s=5, c='r')
```




    <matplotlib.collections.PathCollection at 0x7f0120766a90>




![png](notebook_files/notebook_139_1.png)



```python
train_trn[vcols].isnull().sum() / len(train_trn)
```




    V1     0.473
    V2     0.473
    V3     0.473
    V4     0.473
    V5     0.473
            ... 
    V335   0.861
    V336   0.861
    V337   0.861
    V338   0.861
    V339   0.861
    Length: 339, dtype: float64




```python
train_trn.loc[train_trn['V1'].isnull(), vcols].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V330</th>
      <th>V331</th>
      <th>V332</th>
      <th>V333</th>
      <th>V334</th>
      <th>V335</th>
      <th>V336</th>
      <th>V337</th>
      <th>V338</th>
      <th>V339</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>3</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>4</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td>7</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>8</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td>10</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>11</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>12</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>13</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>14</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 339 columns</p>
</div>




```python
train_trn.loc[train_trn['V1'].isnull() == False, vcols].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V330</th>
      <th>V331</th>
      <th>V332</th>
      <th>V333</th>
      <th>V334</th>
      <th>V335</th>
      <th>V336</th>
      <th>V337</th>
      <th>V338</th>
      <th>V339</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>15</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>18</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>20</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>23</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>27</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>...</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 339 columns</p>
</div>




```python
fig, ax = plt.subplots(1, 2, figsize=(15, 3))
train_trn.loc[train_trn['isFraud']==0, vcols].isnull().sum(axis=1).to_frame().hist(ax=ax[0], bins=20)
train_trn.loc[train_trn['isFraud']==1, vcols].isnull().sum(axis=1).to_frame().hist(ax=ax[1], bins=20)
```




    array([<matplotlib.axes._subplots.AxesSubplot object at 0x7f011dc38ba8>],
          dtype=object)




![png](notebook_files/notebook_143_1.png)



```python
train_trn[vcols].describe().T[['min','max']].T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V330</th>
      <th>V331</th>
      <th>V332</th>
      <th>V333</th>
      <th>V334</th>
      <th>V335</th>
      <th>V336</th>
      <th>V337</th>
      <th>V338</th>
      <th>V339</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>min</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>1.000</td>
      <td>8.000</td>
      <td>9.000</td>
      <td>6.000</td>
      <td>6.000</td>
      <td>9.000</td>
      <td>9.000</td>
      <td>8.000</td>
      <td>8.000</td>
      <td>4.000</td>
      <td>...</td>
      <td>55.000</td>
      <td>160,000.000</td>
      <td>160,000.000</td>
      <td>160,000.000</td>
      <td>55,136.000</td>
      <td>55,136.000</td>
      <td>55,136.000</td>
      <td>104,060.000</td>
      <td>104,060.000</td>
      <td>104,060.000</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 339 columns</p>
</div>




```python
vcols = [f'V{i}' for i in range(1,340)]

pca = PCA()
pca.fit(train_trn[vcols].fillna(-1))
plt.xlabel('components')
plt.plot(np.add.accumulate(pca.explained_variance_ratio_))
plt.show()

pca = PCA(n_components=0.99)
vcol_pca = pca.fit_transform(train_trn[vcols].fillna(-1))
print(vcol_pca.ndim)
```


![png](notebook_files/notebook_145_0.png)


    2
    


```python
del train_trn_f0,train_trn_f1,train_id_f0,train_id_f1

print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val)) for val in dir()]],
                   index=['name','size']).T.sort_values('size', ascending=False).reset_index(drop=True)[:10])
```

               name        size
    0     train_trn  1243551195
    1      test_trn   820917583
    2  train_id_trn   157073961
    3      train_id   155487526
    4       test_id   146112147
    5         vsum0     6838548
    6    train_date     4724472
    7     test_date     4053680
    8            vc     1951187
    9           _65     1951187
    
