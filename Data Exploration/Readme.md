# Import all the necessary libraries


```python
# Basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as spy
%matplotlib inline
import copy
```


```python
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
```


```python
# Pandas display settings - columns

# Display all columns
pd.set_option("display.max_columns", None)
```

# Data ingestion 


```python
# Load dataset
data = pd.read_excel("InsurancePremiumDefault.xlsx",sheet_name='premium')
```

# **Data Inspection**

**Preview dataset**


```python
# Preview the dataset
# View the first 5, last 5 and random 10 rows
print('First five rows', '--'*55)
display(data.head())

print('Last five rows', '--'*55)
display(data.tail())

print('Random ten rows', '--'*55)
np.random.seed(1)
display(data.sample(n=10))
```

    First five rows --------------------------------------------------------------------------------------------------------------
    


<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>perc_premium_paid_by_cash_credit</th>
      <th>age_in_days</th>
      <th>Income</th>
      <th>Count_3-6_months_late</th>
      <th>Count_6-12_months_late</th>
      <th>Count_more_than_12_months_late</th>
      <th>Marital Status</th>
      <th>Veh_Owned</th>
      <th>No_of_dep</th>
      <th>Accomodation</th>
      <th>risk_score</th>
      <th>no_of_premiums_paid</th>
      <th>sourcing_channel</th>
      <th>residence_area_type</th>
      <th>premium</th>
      <th>default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.317</td>
      <td>11330</td>
      <td>90050</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>98.810</td>
      <td>8</td>
      <td>A</td>
      <td>Rural</td>
      <td>5400</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.000</td>
      <td>30309</td>
      <td>156080</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>99.066</td>
      <td>3</td>
      <td>A</td>
      <td>Urban</td>
      <td>11700</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.015</td>
      <td>16069</td>
      <td>145020</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>99.170</td>
      <td>14</td>
      <td>C</td>
      <td>Urban</td>
      <td>18000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.000</td>
      <td>23733</td>
      <td>187560</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>99.370</td>
      <td>13</td>
      <td>A</td>
      <td>Urban</td>
      <td>13800</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.888</td>
      <td>19360</td>
      <td>103050</td>
      <td>7</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>98.800</td>
      <td>15</td>
      <td>A</td>
      <td>Urban</td>
      <td>7500</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    Last five rows --------------------------------------------------------------------------------------------------------------
    


<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>perc_premium_paid_by_cash_credit</th>
      <th>age_in_days</th>
      <th>Income</th>
      <th>Count_3-6_months_late</th>
      <th>Count_6-12_months_late</th>
      <th>Count_more_than_12_months_late</th>
      <th>Marital Status</th>
      <th>Veh_Owned</th>
      <th>No_of_dep</th>
      <th>Accomodation</th>
      <th>risk_score</th>
      <th>no_of_premiums_paid</th>
      <th>sourcing_channel</th>
      <th>residence_area_type</th>
      <th>premium</th>
      <th>default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>79848</th>
      <td>79849</td>
      <td>0.249</td>
      <td>25555</td>
      <td>64420</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>99.08</td>
      <td>10</td>
      <td>A</td>
      <td>Urban</td>
      <td>5700</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79849</th>
      <td>79850</td>
      <td>0.003</td>
      <td>16797</td>
      <td>660040</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>99.65</td>
      <td>9</td>
      <td>B</td>
      <td>Urban</td>
      <td>28500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79850</th>
      <td>79851</td>
      <td>0.012</td>
      <td>24835</td>
      <td>227760</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>99.66</td>
      <td>11</td>
      <td>A</td>
      <td>Rural</td>
      <td>11700</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79851</th>
      <td>79852</td>
      <td>0.190</td>
      <td>10959</td>
      <td>153060</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>99.46</td>
      <td>24</td>
      <td>A</td>
      <td>Urban</td>
      <td>11700</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79852</th>
      <td>79853</td>
      <td>0.000</td>
      <td>19720</td>
      <td>324030</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>99.80</td>
      <td>7</td>
      <td>D</td>
      <td>Rural</td>
      <td>3300</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    Random ten rows --------------------------------------------------------------------------------------------------------------
    


<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>perc_premium_paid_by_cash_credit</th>
      <th>age_in_days</th>
      <th>Income</th>
      <th>Count_3-6_months_late</th>
      <th>Count_6-12_months_late</th>
      <th>Count_more_than_12_months_late</th>
      <th>Marital Status</th>
      <th>Veh_Owned</th>
      <th>No_of_dep</th>
      <th>Accomodation</th>
      <th>risk_score</th>
      <th>no_of_premiums_paid</th>
      <th>sourcing_channel</th>
      <th>residence_area_type</th>
      <th>premium</th>
      <th>default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>53658</th>
      <td>53659</td>
      <td>0.150</td>
      <td>22643</td>
      <td>171080</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>98.78</td>
      <td>12</td>
      <td>A</td>
      <td>Urban</td>
      <td>11700</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25309</th>
      <td>25310</td>
      <td>0.201</td>
      <td>10232</td>
      <td>75090</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>99.61</td>
      <td>7</td>
      <td>A</td>
      <td>Rural</td>
      <td>1200</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26163</th>
      <td>26164</td>
      <td>0.010</td>
      <td>19719</td>
      <td>144120</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>99.89</td>
      <td>7</td>
      <td>A</td>
      <td>Urban</td>
      <td>11700</td>
      <td>1</td>
    </tr>
    <tr>
      <th>55134</th>
      <td>55135</td>
      <td>0.000</td>
      <td>23734</td>
      <td>350070</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>99.51</td>
      <td>9</td>
      <td>A</td>
      <td>Urban</td>
      <td>1200</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29706</th>
      <td>29707</td>
      <td>0.094</td>
      <td>27023</td>
      <td>69110</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>99.83</td>
      <td>8</td>
      <td>A</td>
      <td>Urban</td>
      <td>5400</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11087</th>
      <td>11088</td>
      <td>0.123</td>
      <td>28844</td>
      <td>180030</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>99.47</td>
      <td>8</td>
      <td>A</td>
      <td>Rural</td>
      <td>9600</td>
      <td>1</td>
    </tr>
    <tr>
      <th>71267</th>
      <td>71268</td>
      <td>0.990</td>
      <td>12785</td>
      <td>55640</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>99.66</td>
      <td>7</td>
      <td>A</td>
      <td>Rural</td>
      <td>3300</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4389</th>
      <td>4390</td>
      <td>0.051</td>
      <td>22276</td>
      <td>171080</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>99.69</td>
      <td>11</td>
      <td>A</td>
      <td>Rural</td>
      <td>7500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47666</th>
      <td>47667</td>
      <td>0.000</td>
      <td>19717</td>
      <td>219430</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>99.27</td>
      <td>11</td>
      <td>D</td>
      <td>Urban</td>
      <td>11700</td>
      <td>1</td>
    </tr>
    <tr>
      <th>31240</th>
      <td>31241</td>
      <td>0.000</td>
      <td>15708</td>
      <td>150110</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>99.71</td>
      <td>4</td>
      <td>C</td>
      <td>Urban</td>
      <td>11700</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


**Initial observations**
- `id` is row identifier, which does not add any value. This variable can be removed later.
- `perc_premium_paid_by_cash_credit` is a continuous,ratio numerical variable.
- `age_in_days`,`Income`,`Veh_Owned`, `No_of_dep`,`No_of_dep`,`no_of_premiums_paid` & `premium`  are discrete,interval numerical variables.
- `Income` is a discrete,interval numerical variable.
- `Count_3-6_months_late`, `Count_6-12_months_late` & `Count_more_than_12_months_late` are interval and discrete numerical variables.
- `Marital Status` is binary (0 - Unmarried, 1 - Married)
- `Accomodation` is binary (0 - Rented, 1 - Owned)
- `risk_score` is a continuous, numerical variable.
- `sourcing_channel` & `residence_area_type` are categorical nominal variables.
- `default` is the Target variable. It is a binary (0 - defaulter, 1 - non_defaulter) variable.

---

## Variable List


```python
# Display list of variables in dataset
variable_list = data.columns.tolist()
print(variable_list)
```

    ['id', 'perc_premium_paid_by_cash_credit', 'age_in_days', 'Income', 'Count_3-6_months_late', 'Count_6-12_months_late', 'Count_more_than_12_months_late', 'Marital Status', 'Veh_Owned', 'No_of_dep', 'Accomodation', 'risk_score', 'no_of_premiums_paid', 'sourcing_channel', 'residence_area_type', 'premium', 'default']
    

---

Let's rename the variables for ease of programming


```python
# Column rename dictionary
renamed_columns = {
    'id': 'ID',
    'perc_premium_paid_by_cash_credit': 'Perc_premium_paid_in_cash',
    'age_in_days': 'Age_in_days',
    'Count_3-6_months_late': 'Late_premium_payment_3-6_months',
    'Count_6-12_months_late': 'Late_premium_payment_6-12_months',
    'Count_more_than_12_months_late': 'Late_premium_payment_>12_months',
    'Marital Status': 'Marital_Status',
    'Veh_Owned': 'Vehicles_Owned',
    'No_of_dep': 'No_of_dependents',
    'risk_score': 'Risk_score',
    'no_of_premiums_paid': 'No_of_premiums_paid',
    'sourcing_channel': 'Sourcing_channel',
    'residence_area_type': 'Customer_demographic',
    'premium': 'Premium_payment',
    'default': 'Default'
}

# Rename dataframe columns names
data = data.rename(columns = renamed_columns)
```


```python
# Check for updated column names
variable_list = data.columns.tolist()
print(variable_list)
```

    ['ID', 'Perc_premium_paid_in_cash', 'Age_in_days', 'Income', 'Late_premium_payment_3-6_months', 'Late_premium_payment_6-12_months', 'Late_premium_payment_>12_months', 'Marital_Status', 'Vehicles_Owned', 'No_of_dependents', 'Accomodation', 'Risk_score', 'No_of_premiums_paid', 'Sourcing_channel', 'Customer_demographic', 'Premium_payment', 'Default']
    

---

## Dataset shape


```python
shape = data.shape
n_rows = shape[0]
n_cols = shape[1]
print(f"The Dataframe consists of '{n_rows}' rows and '{n_cols}' columns")
```

    The Dataframe consists of '79853' rows and '17' columns
    

**Data types**


```python
# Check the data types
data.dtypes
```




    ID                                    int64
    Perc_premium_paid_in_cash           float64
    Age_in_days                           int64
    Income                                int64
    Late_premium_payment_3-6_months       int64
    Late_premium_payment_6-12_months      int64
    Late_premium_payment_>12_months       int64
    Marital_Status                        int64
    Vehicles_Owned                        int64
    No_of_dependents                      int64
    Accomodation                          int64
    Risk_score                          float64
    No_of_premiums_paid                   int64
    Sourcing_channel                     object
    Customer_demographic                 object
    Premium_payment                       int64
    Default                               int64
    dtype: object



**Data info**


```python
# Get info of the dataframe columns
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 79853 entries, 0 to 79852
    Data columns (total 17 columns):
     #   Column                            Non-Null Count  Dtype  
    ---  ------                            --------------  -----  
     0   ID                                79853 non-null  int64  
     1   Perc_premium_paid_in_cash         79853 non-null  float64
     2   Age_in_days                       79853 non-null  int64  
     3   Income                            79853 non-null  int64  
     4   Late_premium_payment_3-6_months   79853 non-null  int64  
     5   Late_premium_payment_6-12_months  79853 non-null  int64  
     6   Late_premium_payment_>12_months   79853 non-null  int64  
     7   Marital_Status                    79853 non-null  int64  
     8   Vehicles_Owned                    79853 non-null  int64  
     9   No_of_dependents                  79853 non-null  int64  
     10  Accomodation                      79853 non-null  int64  
     11  Risk_score                        79853 non-null  float64
     12  No_of_premiums_paid               79853 non-null  int64  
     13  Sourcing_channel                  79853 non-null  object 
     14  Customer_demographic              79853 non-null  object 
     15  Premium_payment                   79853 non-null  int64  
     16  Default                           79853 non-null  int64  
    dtypes: float64(2), int64(13), object(2)
    memory usage: 10.4+ MB
    

- Two (2) variables have been identified as `Panda object` type. These shall be converted to the `category` type.

**Convert Pandas Objects to Category type**


```python
# Convert variables with "object" type to "category" type
for i in data.columns:
    if data[i].dtypes == "object":
        data[i] = data[i].astype("category") 

# Confirm if there no variables with "object" type
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 79853 entries, 0 to 79852
    Data columns (total 17 columns):
     #   Column                            Non-Null Count  Dtype   
    ---  ------                            --------------  -----   
     0   ID                                79853 non-null  int64   
     1   Perc_premium_paid_in_cash         79853 non-null  float64 
     2   Age_in_days                       79853 non-null  int64   
     3   Income                            79853 non-null  int64   
     4   Late_premium_payment_3-6_months   79853 non-null  int64   
     5   Late_premium_payment_6-12_months  79853 non-null  int64   
     6   Late_premium_payment_>12_months   79853 non-null  int64   
     7   Marital_Status                    79853 non-null  int64   
     8   Vehicles_Owned                    79853 non-null  int64   
     9   No_of_dependents                  79853 non-null  int64   
     10  Accomodation                      79853 non-null  int64   
     11  Risk_score                        79853 non-null  float64 
     12  No_of_premiums_paid               79853 non-null  int64   
     13  Sourcing_channel                  79853 non-null  category
     14  Customer_demographic              79853 non-null  category
     15  Premium_payment                   79853 non-null  int64   
     16  Default                           79853 non-null  int64   
    dtypes: category(2), float64(2), int64(13)
    memory usage: 9.3 MB
    

- `The memory usage has decreased from 10.4+ MB to 9.3 MB`

**Missing value summary function**


```python
def missing_val_chk(data):
    """
    This function to checks for missing values 
    and generates a summary.
    """
    if data.isnull().sum().any() == True:
        # Number of missing in each column
        missing_vals = pd.DataFrame(data.isnull().sum().sort_values(
            ascending=False)).rename(columns={0: '# missing'})

        # Create a percentage missing
        missing_vals['percent'] = ((missing_vals['# missing'] / len(data)) *
                                   100).round(decimals=3)

        # Remove rows with 0
        missing_vals = missing_vals[missing_vals['# missing'] != 0].dropna()

        # display missing value dataframe
        print("The missing values summary")
        display(missing_vals)
    else:
        print("There are NO missing values in the dataset")
```

## Missing Values Check


```python
#Applying the missing value summary function
missing_val_chk(data)
```

    There are NO missing values in the dataset
    

***

Before we check the 5 Point numerical summary, let's verify that `ID` is row identifier.

If this is the case then the number of unique values will equal the number of rows.


```python
# Check to see if ID unique values equal number of rows

if data.ID.nunique() == data.shape[0]:
    print("ID is a row identifier, we shall drop ID variable")
else:
    print("ID is a row identifier")
```

    ID is a row identifier, we shall drop ID variable
    

**Dropping ID variable**


```python
data.drop(columns="ID", axis=1, inplace=True)
```

---

## 5 Point Summary

**Numerical type Summary**


```python
# Five point summary of all numerical type variables in the dataset
data.describe().T
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Perc_premium_paid_in_cash</th>
      <td>79853.0</td>
      <td>0.314288</td>
      <td>0.334915</td>
      <td>0.0</td>
      <td>0.034</td>
      <td>0.167</td>
      <td>0.538</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>Age_in_days</th>
      <td>79853.0</td>
      <td>18846.696906</td>
      <td>5208.719136</td>
      <td>7670.0</td>
      <td>14974.000</td>
      <td>18625.000</td>
      <td>22636.000</td>
      <td>37602.00</td>
    </tr>
    <tr>
      <th>Income</th>
      <td>79853.0</td>
      <td>208847.171177</td>
      <td>496582.597257</td>
      <td>24030.0</td>
      <td>108010.000</td>
      <td>166560.000</td>
      <td>252090.000</td>
      <td>90262600.00</td>
    </tr>
    <tr>
      <th>Late_premium_payment_3-6_months</th>
      <td>79853.0</td>
      <td>0.248369</td>
      <td>0.691102</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>13.00</td>
    </tr>
    <tr>
      <th>Late_premium_payment_6-12_months</th>
      <td>79853.0</td>
      <td>0.078093</td>
      <td>0.436251</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>17.00</td>
    </tr>
    <tr>
      <th>Late_premium_payment_&gt;12_months</th>
      <td>79853.0</td>
      <td>0.059935</td>
      <td>0.311840</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>Marital_Status</th>
      <td>79853.0</td>
      <td>0.498679</td>
      <td>0.500001</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>Vehicles_Owned</th>
      <td>79853.0</td>
      <td>1.998009</td>
      <td>0.817248</td>
      <td>1.0</td>
      <td>1.000</td>
      <td>2.000</td>
      <td>3.000</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>No_of_dependents</th>
      <td>79853.0</td>
      <td>2.503012</td>
      <td>1.115901</td>
      <td>1.0</td>
      <td>2.000</td>
      <td>3.000</td>
      <td>3.000</td>
      <td>4.00</td>
    </tr>
    <tr>
      <th>Accomodation</th>
      <td>79853.0</td>
      <td>0.501296</td>
      <td>0.500001</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>Risk_score</th>
      <td>79853.0</td>
      <td>99.067243</td>
      <td>0.725892</td>
      <td>91.9</td>
      <td>98.830</td>
      <td>99.180</td>
      <td>99.520</td>
      <td>99.89</td>
    </tr>
    <tr>
      <th>No_of_premiums_paid</th>
      <td>79853.0</td>
      <td>10.863887</td>
      <td>5.170687</td>
      <td>2.0</td>
      <td>7.000</td>
      <td>10.000</td>
      <td>14.000</td>
      <td>60.00</td>
    </tr>
    <tr>
      <th>Premium_payment</th>
      <td>79853.0</td>
      <td>10924.507533</td>
      <td>9401.676542</td>
      <td>1200.0</td>
      <td>5400.000</td>
      <td>7500.000</td>
      <td>13800.000</td>
      <td>60000.00</td>
    </tr>
    <tr>
      <th>Default</th>
      <td>79853.0</td>
      <td>0.937410</td>
      <td>0.242226</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



- `Perc_premium_paid_in_cash` is highly right skewed as the *mean* is almost twice the *median*. Also the *standard deviation* is greater than the *mean* implying a wide spread.   

- `Age_in_days` is fairly symmetrical with *mean* and *median* being very close but there is some right skew as the difference between Q3 & Q4 is larger than other quartiles.
- `Income` highly right skewed as the *mean* is greater than the *median*. Also the *standard deviation* is more than twice the *mean* implying a wide spread.
- `Late_premium_payment_3-6_months`, `Late_premium_payment_6-12_months` & `Late_premium_payment_>12_months` are categorical ordinal variables.
- `Marital_Status` is a binary variable with approximately 50% of the rows having a value of 1 (50% of the customers are married).
- `Vehicles_Owned` & `No_of_dependents` are categorical ordinal variables.
- `Accomodation` is a binary variable with approximately 50% of the rows having a value of 1 (50% of the customers owned their homes).
- `Risk_score` is fairly symmetrical with *mean* and *median* being very close.
- `No_of_premiums_paid` is fairly symmetrical with *mean* and *median* being very close but there is some right skew as the difference between Q3 & Q4 is larger than other quartiles.
- `Premium_payment` is highly right skewed as the *mean* is greater than the *median*. Also the *standard deviation* is close to the *mean* implying a wide spread.
- `Default` is a binary variable with approximately 94% of the rows having a value of 1 (94% of the customers are non-defaulters). This target variable is heavily imbalanced as only 6% of the rows are defaulters.

**Categorical type Summary**


```python
data.describe(include=['category']).T
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Sourcing_channel</th>
      <td>79853</td>
      <td>5</td>
      <td>A</td>
      <td>43134</td>
    </tr>
    <tr>
      <th>Customer_demographic</th>
      <td>79853</td>
      <td>2</td>
      <td>Urban</td>
      <td>48183</td>
    </tr>
  </tbody>
</table>
</div>



- `Sourcing_channel` there are 5 different states in which "A" is the most frequent.
- `Customer_demographic` there are 2 different states in which "Urban" is the most frequent. 

<font color='red'>**This dataset will be skewed to policy holders sourced from _Channel A_ and _Urban_ residences**

---

**Number of unique states for all variables**


```python
# Check the unique values
data.nunique().to_frame()
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Perc_premium_paid_in_cash</th>
      <td>1001</td>
    </tr>
    <tr>
      <th>Age_in_days</th>
      <td>833</td>
    </tr>
    <tr>
      <th>Income</th>
      <td>24165</td>
    </tr>
    <tr>
      <th>Late_premium_payment_3-6_months</th>
      <td>14</td>
    </tr>
    <tr>
      <th>Late_premium_payment_6-12_months</th>
      <td>17</td>
    </tr>
    <tr>
      <th>Late_premium_payment_&gt;12_months</th>
      <td>10</td>
    </tr>
    <tr>
      <th>Marital_Status</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Vehicles_Owned</th>
      <td>3</td>
    </tr>
    <tr>
      <th>No_of_dependents</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Accomodation</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Risk_score</th>
      <td>673</td>
    </tr>
    <tr>
      <th>No_of_premiums_paid</th>
      <td>57</td>
    </tr>
    <tr>
      <th>Sourcing_channel</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Customer_demographic</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Premium_payment</th>
      <td>30</td>
    </tr>
    <tr>
      <th>Default</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



**Categorical Variable Identification**

Although the following variables are numerical in nature, they represent **categorical** variables:
* `Late_premium_payment_3-6_months`
* `Late_premium_payment_6-12_months`
* `Late_premium_payment_>12_months`
* `Vehicles_Owned`
* `No_of_dependents` 

---

**Create a list of numerical variables**


```python
numerical_vars = [
    'Perc_premium_paid_in_cash', 'Age_in_days', 'Income', 'Risk_score',
    'No_of_premiums_paid', 'Premium_payment'
]
```

**Create a list of categorical variables**


```python
categorical_vars = [
    'Late_premium_payment_3-6_months', 'Late_premium_payment_6-12_months',
    'Late_premium_payment_>12_months', 'Marital_Status', 'Vehicles_Owned',
    'No_of_dependents', 'Accomodation', 'Sourcing_channel',
    'Customer_demographic', 'Default'
]
```

---

## Numerical data


```python
data[numerical_vars].describe().T
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Perc_premium_paid_in_cash</th>
      <td>79853.0</td>
      <td>0.314288</td>
      <td>0.334915</td>
      <td>0.0</td>
      <td>0.034</td>
      <td>0.167</td>
      <td>0.538</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>Age_in_days</th>
      <td>79853.0</td>
      <td>18846.696906</td>
      <td>5208.719136</td>
      <td>7670.0</td>
      <td>14974.000</td>
      <td>18625.000</td>
      <td>22636.000</td>
      <td>37602.00</td>
    </tr>
    <tr>
      <th>Income</th>
      <td>79853.0</td>
      <td>208847.171177</td>
      <td>496582.597257</td>
      <td>24030.0</td>
      <td>108010.000</td>
      <td>166560.000</td>
      <td>252090.000</td>
      <td>90262600.00</td>
    </tr>
    <tr>
      <th>Risk_score</th>
      <td>79853.0</td>
      <td>99.067243</td>
      <td>0.725892</td>
      <td>91.9</td>
      <td>98.830</td>
      <td>99.180</td>
      <td>99.520</td>
      <td>99.89</td>
    </tr>
    <tr>
      <th>No_of_premiums_paid</th>
      <td>79853.0</td>
      <td>10.863887</td>
      <td>5.170687</td>
      <td>2.0</td>
      <td>7.000</td>
      <td>10.000</td>
      <td>14.000</td>
      <td>60.00</td>
    </tr>
    <tr>
      <th>Premium_payment</th>
      <td>79853.0</td>
      <td>10924.507533</td>
      <td>9401.676542</td>
      <td>1200.0</td>
      <td>5400.000</td>
      <td>7500.000</td>
      <td>13800.000</td>
      <td>60000.00</td>
    </tr>
  </tbody>
</table>
</div>



### Skew Summary


```python
# Display the skew summary for the numerical variables
for var in data[numerical_vars].columns:
    var_skew = data[var].skew()
    if var_skew > 1:
        print(f"The '{var}' distribution is highly right skewed.\n")
    elif var_skew < -1:
        print(f"The '{var}' distribution is highly left skewed.\n")
    elif (var_skew > 0.5) & (var_skew < 1):
        print(f"The '{var}' distribution is moderately right skewed.\n")
    elif (var_skew < -0.5) & (var_skew > -1):
        print(f"The '{var}' distribution is moderately left skewed.\n")
    else:
        print(f"The '{var}' distribution is fairly symmetrical.\n")
```

    The 'Perc_premium_paid_in_cash' distribution is moderately right skewed.
    
    The 'Age_in_days' distribution is fairly symmetrical.
    
    The 'Income' distribution is highly right skewed.
    
    The 'Risk_score' distribution is highly left skewed.
    
    The 'No_of_premiums_paid' distribution is highly right skewed.
    
    The 'Premium_payment' distribution is highly right skewed.
    
    

**Outlier check function**


```python
# Outlier check
def outlier_count(data):
    """
    This function checks the lower and upper 
    outliers for all numerical variables.
    
    Outliers are found where data points exists either:
    - Greater than `1.5*IQR` above the 75th percentile
    - Less than `1.5*IQR` below the 25th percentile
    """
    numeric = data.select_dtypes(include=np.number).columns.to_list()
    for i in numeric:
        # Get name of series
        name = data[i].name
        # Calculate the IQR for all values and omit NaNs
        IQR = spy.stats.iqr(data[i], nan_policy="omit")
        # Calculate the boxplot upper fence
        upper_fence = data[i].quantile(0.75) + 1.5 * IQR
        # Calculate the boxplot lower fence
        lower_fence = data[i].quantile(0.25) - 1.5 * IQR
        # Calculate the count of outliers above upper fence
        upper_outliers = data[i][data[i] > upper_fence].count()
        # Calculate the count of outliers below lower fence
        lower_outliers = data[i][data[i] < lower_fence].count()
        # Check if there are no outliers
        if (upper_outliers == 0) & (lower_outliers == 0):
            continue
        print(
            f"The '{name}' distribution has '{lower_outliers}' lower outliers and '{upper_outliers}' upper outliers.\n"
        )
```

### Outlier check


```python
#Applying the Outlier check function for the sub-dataframe of numerical variables
outlier_count(data[numerical_vars])
```

    The 'Age_in_days' distribution has '0' lower outliers and '44' upper outliers.
    
    The 'Income' distribution has '0' lower outliers and '3428' upper outliers.
    
    The 'Risk_score' distribution has '3784' lower outliers and '0' upper outliers.
    
    The 'No_of_premiums_paid' distribution has '0' lower outliers and '1426' upper outliers.
    
    The 'Premium_payment' distribution has '0' lower outliers and '4523' upper outliers.
    
    

### Numerical Variable Summary

| Variable| Skew | Outliers | 
| :-: | :-: | :-: |
| **Perc_premium_paid_in_cash** | Moderately right skewed | No Outliers | 
| **Age_in_days** | Fairly symmetrical | 44 Upper Outliers | 
| **Income** | Highly right skewed | 3428 Upper Outliers |
| **Risk_score** | Highly left skewed | 3784 Lower Outliers |
| **No_of_premiums_paid** | Highly right skewed | 1426 Upper Outliers |
| **Premium_payment** | Highly right skewed | 4523 Upper Outliers |

---

## Categorical data

### Unique states

**Detailed investigation of unique values**


```python
# Display the unique values for all categorical variables
for i in categorical_vars:
    print('Unique values in',i, 'are :')
    print(data[i].value_counts())
    print('--'*55)
```

    Unique values in Late_premium_payment_3-6_months are :
    Late_premium_payment_3-6_months
    0     66898
    1      8826
    2      2519
    3       954
    4       374
    5       168
    6        68
    7        23
    8        15
    9         4
    13        1
    12        1
    10        1
    11        1
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in Late_premium_payment_6-12_months are :
    Late_premium_payment_6-12_months
    0     75928
    1      2680
    2       693
    3       317
    4       130
    5        46
    6        26
    7        11
    8         5
    9         4
    10        4
    11        2
    14        2
    13        2
    15        1
    17        1
    12        1
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in Late_premium_payment_>12_months are :
    Late_premium_payment_>12_months
    0     76135
    1      2996
    2       498
    3       151
    4        48
    5        13
    6         6
    7         3
    8         2
    11        1
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in Marital_Status are :
    Marital_Status
    0    40032
    1    39821
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in Vehicles_Owned are :
    Vehicles_Owned
    1    26746
    3    26587
    2    26520
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in No_of_dependents are :
    No_of_dependents
    3    20215
    2    19902
    4    19896
    1    19840
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in Accomodation are :
    Accomodation
    1    40030
    0    39823
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in Sourcing_channel are :
    Sourcing_channel
    A    43134
    B    16512
    C    12039
    D     7559
    E      609
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in Customer_demographic are :
    Customer_demographic
    Urban    48183
    Rural    31670
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    Unique values in Default are :
    Default
    1    74855
    0     4998
    Name: count, dtype: int64
    --------------------------------------------------------------------------------------------------------------
    

- `Late_premium_payment_6-12_months` - there is discontinuity in the unique values as state 16 is missing in the range. 
-  `Late_premium_payment_>12_months` - there is discontinuity in the unique values as states 9 & 10 are missing in the range.

---

### Categorical Variable Summary

There are categorical variables in the numeric format.

| Variable| Type | Range | 
| :-: | :-: | :-: |
| **Late_premium_payment_3-6_months**| Ordinal | 14 states |
| **Late_premium_payment_6-12_months**| Ordinal | 17 states |
| **Late_premium_payment_>12_months**| Ordinal | 10 states |
| **Marital_Status**| Nominal | 2 states |
| **Vehicles_Owned**| Ordinal | 3 states |
| **No_of_dependents**| Ordinal | 4 states |
| **Accomodation**| Nominal | 2 states |
| **Sourcing_channel**| Nominal | 5 states |
| **Customer_demographic**| Nominal | 2 states |
| **Default**| Nominal | 2 states |

---

## Target Variable

Target variable is **`Default`**


```python
# Checking the distribution of target variable

# Count the different "Default" states
count = data["Default"].value_counts().T
# Calculate the percentage different "Default" states
percentage = data['Default'].value_counts(normalize=True).T * 100
# Join count and percentage series
target_dist = pd.concat([count, percentage], axis=1)
# Set column names
target_dist.columns = ['count', 'percentage']
# Set Index name
target_dist.index.name = "Default"
# Display target distribution dataframe
target_dist
```




<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>percentage</th>
    </tr>
    <tr>
      <th>Default</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>74855</td>
      <td>93.740999</td>
    </tr>
    <tr>
      <th>0</th>
      <td>4998</td>
      <td>6.259001</td>
    </tr>
  </tbody>
</table>
</div>



**Out of the 79854 policy holders, only 6.26% defaulted**

<font color='red'> The Target variable is **Heavily Imbalanced**

---

---
