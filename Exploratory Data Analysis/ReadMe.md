<details>
<summary>Importing Libraries & Data ingestion </summary>
<br>
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
import math
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
    

**Dropping ID variable**


```python
data.drop(columns="ID", axis=1, inplace=True)
```

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

</details>

---

# Exploratory Data Analysis

## Univariate Analysis

### Numerical Variables

#### Histogram Overview

Let's get an overview of the distributions of the numerical variables.


```python
def histogram_overview(data):
    """
    Histogram Overview function
    
    This function below generates a subplots of  `histogram plots` & 
    showing the `distribution of the numerical varible input`
    
    * Generates subplots for each numerical variable in a three column structure.
    * The function takes the Pandas dataframe as the input
    * The function selects the numerical variables from the applied dataframe.
    * It generates a vertical `line` to indicate the `mean`, `median` and `mode` on the histogram
    * `sns.set_style` - sets the Seaborn theme
    * `subplot_nrows` - calculated number of subplot rows
    * `subplot_ncols` - configures the three column structure subplots
    * `figsize` - indicates the size of the plot
    * `sns.reset_defaults()` - resets Seaborn themes and settings to default
    
    """
    num_vars = data.select_dtypes(include=np.number).columns.to_list()
    plt.figure(figsize=(10, 10))
    for i in range(len(num_vars)):
        # Set seaborn theme
        sns.set_style("darkgrid")
        # Subplot no of columns
        subplot_ncols = math.ceil(np.sqrt(len(num_vars)))
        # Subplot no of rows
        subplot_nrows = subplot_ncols

        plt.subplot(subplot_nrows, subplot_ncols, i + 1)

        plt.hist(data[num_vars[i]])
        #Plot vertical line for the mean
        plt.axvline(data[num_vars[i]].mean(),
                    color='green',
                    linestyle='--',
                    label="mean")
        #Plot vertical line for the median
        plt.axvline(data[num_vars[i]].median(),
                    color='red',
                    linestyle='-',
                    label="median")
        #Plot vertical line for the mode
        plt.axvline(data[num_vars[i]].mode()[0],
                    color='black',
                    linestyle='-',
                    label="mode")
        plt.legend()
        plt.tight_layout()
        plt.title(num_vars[i], fontsize=16)
    plt.show()
    # Reset seaborn theme
    sns.reset_defaults()
```


```python
histogram_overview(data[numerical_vars])
```


    
![png](output_25_0.png)
    


**Observation:**
* **Age_in_days** - Somewhat normal as the mean and median are very close.

* All other variables are skewed.

#### Boxplot overview

Let's get a boxplot overview across each numerical variable


```python
def boxplot_overview(data):
    """
    This function below generates a subplots of `box plots` &
    showing the `distribution of the numerical variable input with outliers`.

    * Generates subplots for each numerical variable in a three column structure.
    * The function takes the Pandas dataframe as the input
    * The function selects the numerical variables from the applied dataframe.
    * It shows the `mean` in the boxplot.
    * `sns.set_style` - sets the Seaborn theme
    * `subplot_nrows` - calculated number of subplot rows
    * `subplot_ncols` - configures the three column structure subplots
    * `figsize` - indicates the size of the plot
    * `sns.reset_defaults()` - resets Seaborn themes and settings to default
    """

    num_vars = data.select_dtypes(include=np.number).columns.to_list()
    plt.figure(figsize=(10, 10))
    for i in range(len(num_vars)):
        # Set seaborn theme
        sns.set_style("darkgrid")
        # Subplot no of columns
        subplot_ncols = math.ceil(np.sqrt(len(num_vars)) )  
        # Subplot no of rows
        subplot_nrows = math.ceil(len(num_vars) / subplot_ncols)  
          
        plt.subplot(subplot_nrows, subplot_ncols, i + 1)
        sns.boxplot(y=data[num_vars[i]], width=0.3, showmeans=True)
        plt.tight_layout()
        plt.title(num_vars[i], fontsize=16)
    plt.show()
    # Reset seaborn theme
    sns.reset_defaults()
```


```python
boxplot_overview(data[numerical_vars])
```


    
![png](output_30_0.png)
    


**Observation:**
* **Perc_premium_paid_in_cash** has no outliers.

* **Age_in_days** , **Income**, **No_of_premiums_paid** & **Premium_payment** -  have upper outliers.

* **Risk_score** has lower outliers

#### Histogram Distribution

Let's generate Histograms for each numerical variable and visually identify any its distributions.


```python
def hist_box(data):
    """
    This function below generates a `box plot` & `histogram` 
    showing the `distribution of the numerical varible input`.
    * The function also checks for `outliers` and states the location (`lower`/`upper`)
    * The function also `generates an image file` of the plot.
    * The function takes the Pandas series as the input.
    * It creates a `subplot` with `box plot` and `histogram` distribution
    * It generates a vertical `line` to indicate the `mean`, `median` and `mode` on the histogram
    * It calculates the Inter Quartile Range using `Scipy Stats`
    * `sns.set_style` - sets the Seaborn theme
    * `nrows` - sets the shape of the subplot
    * `gridspec_kw` - configures the ratio of the size of the plots
    * `figsize` - indicates the size of the plot
    * `sns.reset_defaults()` - resets Seaborn themes and settings to default
    
    """

    # Get name of series
    name = data.name
    sns.set_style("darkgrid")
    f, axes = plt.subplots(nrows=2,
                           gridspec_kw={"height_ratios": (1, 3)})
    sns.boxplot(data, showmeans=True, color='m', ax=axes[0])
    sns.distplot(data, bins=15, ax=axes[1], color='deepskyblue', kde=False)
    axes[1].axvline(data.mean(), color='green', linestyle='--', label="mean")
    axes[1].axvline(data.median(), color='red', linestyle='-', label="median")
    axes[1].axvline(data.mode()[0], color='black', linestyle='-', label="mode")
    plt.legend(("mean", "median", "mode"), fontsize=12)
    plt.suptitle("Distribution of {}".format(name), fontsize=22)
    plt.tight_layout()
    plt.show()

    # Outlier check
    IQR = spy.stats.iqr(data, nan_policy="omit")
    upper_fence = data.quantile(0.75) + 1.5 * IQR
    lower_fence = data.quantile(0.25) - 1.5 * IQR
    upper_outliers = data[data > upper_fence].count()
    lower_outliers = data[data < lower_fence].count()
    print(
        f"The '{name}' distribution has '{lower_outliers}' lower outliers and '{upper_outliers}' upper outliers."
    )

    # Line separator
    print('--' * 55)
```

---

**Plot the distribution of all numerical variables**


```python
for each_var in data[numerical_vars].columns:
    hist_box(data=data[each_var])
    plt.figure()
    plt.show()
```


    
![png](output_37_0.png)
    


    The 'Perc_premium_paid_in_cash' distribution has '0' lower outliers and '0' upper outliers.
    --------------------------------------------------------------------------------------------------------------
    


    <Figure size 640x480 with 0 Axes>



    
![png](output_37_3.png)
    


    The 'Age_in_days' distribution has '0' lower outliers and '44' upper outliers.
    --------------------------------------------------------------------------------------------------------------
    


    <Figure size 640x480 with 0 Axes>



    
![png](output_37_6.png)
    


    The 'Income' distribution has '0' lower outliers and '3428' upper outliers.
    --------------------------------------------------------------------------------------------------------------
    


    <Figure size 640x480 with 0 Axes>



    
![png](output_37_9.png)
    


    The 'Risk_score' distribution has '3784' lower outliers and '0' upper outliers.
    --------------------------------------------------------------------------------------------------------------
    


    <Figure size 640x480 with 0 Axes>



    
![png](output_37_12.png)
    


    The 'No_of_premiums_paid' distribution has '0' lower outliers and '1426' upper outliers.
    --------------------------------------------------------------------------------------------------------------
    


    <Figure size 640x480 with 0 Axes>



    
![png](output_37_15.png)
    


    The 'Premium_payment' distribution has '0' lower outliers and '4523' upper outliers.
    --------------------------------------------------------------------------------------------------------------
    


    <Figure size 640x480 with 0 Axes>


**Observation:**
* `Perc_premium_paid_in_cash`
    * The bulk of the policy holders paid between 5%-55% of their policy in cash.
* `Age_in_days`
    * The bulk of the policy holders are between 15,000-22,500 days old (41-62 years old), i.e., middle aged to senior adults.
* `Income`
    * Income is greatly skewed in the dataset with some policy holders making over 10,000,000.
* `Risk_score`
    * The policy holders' risk scores typically range from 98-100% with outliers as far as 92%

* `No_of_premiums_paid`
    * The number of premiums paid by policy holders typically range from 0-24 with the bulk being within 7-14.
    
* `Premium_payment`
    * The bulk of the insurance premiums range from 5,000 to 15,000.
    * There are some policies in excess of 30,000

---

### Categorical Variables

We shall use bar chart to represent the categorical variables.


```python
def bar_chart(data):
    """
    This function below generates a `bar chart` showing
    the `distribution of the categorical varible input`.
    * The function also `generates an image file` of the plot.
    * The function takes the Pandas series as the input.
    * It `computes the frequency of each unique element` and 
      displays the distribution of the elements to in horizontal bars.
    * The `percentage of each bar` is also calculated and placed to 
      the right end of each bar.
    * `sns.despine()` - removes the upper and right border of the chart
    * For each horizontal bar the width is calculated as a percentage of
      the entire quanta of datapoints.
    * The percentage is annotated to the each bar by plotting the cardinal locations.

    """
    
    # Create a horizontal count plot while sorting variables in descending order
    g=sns.countplot(y=data)
    # Remove the top and right spines from plot
    sns.despine()
    # length of the column
    col_length = len(data) 
    for p in g.patches:
        # percentage of each class of the category
        percentage = '{:.1f}%'.format(100 * p.get_width()/col_length)
        # width of the plot
        x = p.get_x() + p.get_width() + 0.02
        # height of the plot
        y = p.get_y() + p.get_height()/2
        # annotate the percentage
        g.annotate(percentage, (x, y), size = 12) 
        plt.title("Distribution of {}".format(data.name),loc="center",fontsize = 22)
    plt.show()
    # Line separator
    print('--'*55)

```

---

**Plot the distribution of all categorical variables**


```python
for each_var in data[categorical_vars].columns:
    plt.figure()
    bar_chart(data=data[each_var])
```


    
![png](output_45_0.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_45_2.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_45_4.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_45_6.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_45_8.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_45_10.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_45_12.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_45_14.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_45_16.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_45_18.png)
    


    --------------------------------------------------------------------------------------------------------------
    

**Observations:**
* Policy holders typically pay their premiums on times as the majority of them were not late on payments.   
This is indicative of the policy holders having an active policy to exercise claims.
* The policy holders in this dataset is more or less evenly balanced across `Marital Status` as appromimately 50% are _Married_ and _Unmarried_
* The distribution of `Number of vehicles owned` across policy holders is also evenly balanced as approximately 33% have  one (1), two (2) and three(3) vehicles.
* The distribution of `Number of dependents` across policy holders is also evenly balanced as approximately 25% have  one (1), two (2) , three(3) and four(4) dependents.
* The policy holders in this dataset is more or less evenly balanced across `Accomodation` as appromimately 50% _owned_ or _rented_ their place of residence.
* The bulk of the policy holders were sourced using `Channel A` (54%) while the least was `Channel E`
* The majority of the policy holders reside in `Urban` residence types (~60%)
* 6.3% of the policy holders defaulted on their premium payments. The data is `heavily imbalanced`

---

## Numerical Correlation Analysis

Let's check to see to see if there are correlations between the numerical variables.

Since it was observed that `Marital Status`, `Number of vehicles owned`,`Number of dependents` & `Accomodation` were evenly balanced across the dataset, these variables will not provide any meaningful correlations when exploring the heatmaps.

Therefore let's create a subset of the variables for numerical correlation analysis.


```python
# Variables to ignore in numerical correlation analysis
variables_to_ignore = [
    'Marital_Status', 'Vehicles_Owned', 'No_of_dependents', 'Accomodation'
]
# Create a new list of columns
num_corr_vars = data.columns.tolist()
for variables in variables_to_ignore:
    num_corr_vars.remove(variables)
print(num_corr_vars)
```

    ['Perc_premium_paid_in_cash', 'Age_in_days', 'Income', 'Late_premium_payment_3-6_months', 'Late_premium_payment_6-12_months', 'Late_premium_payment_>12_months', 'Risk_score', 'No_of_premiums_paid', 'Sourcing_channel', 'Customer_demographic', 'Premium_payment', 'Default']
    

### Heat Map

**Pearson**


```python
onehot = pd.get_dummies(data[num_corr_vars],
                        columns=data.select_dtypes(include=['category']).columns.tolist())


oh_corr = onehot.corr(method='pearson')

annot_kws = {"fontsize": 12}

symmetric_matrix = (oh_corr + oh_corr.T) / 2

# Create a mask for the upper half of the matrix
mask = np.triu(np.ones_like(symmetric_matrix), k=1)

plt.figure(figsize=(16, 12))
sns.heatmap(oh_corr, annot=True, fmt=".2f", mask=mask,
            cmap='coolwarm', square=True, annot_kws=annot_kws)
plt.yticks(rotation=0)
plt.show()
```


    
![png](output_54_0.png)
    


**Observation:**
* There no significantly correlated variables according to the Pearson analysis.

*Since there are many outliers in the data, let's run the Spearman correlation analysis which is not sensitive to outliers.*

**Spearman**


```python
onehot = pd.get_dummies(data[num_corr_vars],
                        columns=data.select_dtypes(include=['category']).columns.tolist())


oh_corr = onehot.corr(method='spearman')

annot_kws = {"fontsize": 12}

symmetric_matrix = (oh_corr + oh_corr.T) / 2

# Create a mask for the upper half of the matrix
mask = np.triu(np.ones_like(symmetric_matrix), k=1)

plt.figure(figsize=(16, 12))
sns.heatmap(oh_corr, annot=True, fmt=".2f", mask=mask,
            cmap='coolwarm', square=True, annot_kws=annot_kws)
plt.yticks(rotation=0)
plt.show()
```


    
![png](output_58_0.png)
    


**Observation:**
* There is a highly positive correlation with **Income** and **Premium_payment**

---

**Observations:**  

The Spearman correlation resulted in only one pair of correlated variables as the data has many outliers.


| Variable1| Variable2 | Correlation | 
| :-: | :-: | :-: |
| **Premium_payment** | **Income** | highly correlated |

---

# Bivariate Analysis

## Bivariate Scatter Plots

Let's generate a pairplot of the numerical variables before we dive into the Numerical Variable Bivariate Analysis


```python
# sns.pairplot(data,corner=True, hue="Default", markers="o");
```

There are no clear linear relationships among all the variables.

Let's use the variables from the numerical correlation analysis


```python
# sns.pairplot(data[num_corr_vars],corner=True, hue="Default", markers="o");
```

---

`Income` and `Premium_payment` was observed to be highly correlated.  
Let's visualize using a scatterplot  

**Income** vs **Premium_payment**


```python
sns.scatterplot(data=data, y='Income', x='Premium_payment',hue='Default')
plt.show();
```


    
![png](output_72_0.png)
    


**Observation:**
* There appears to be a linear relationship which is being masked by the presence of outliers.  
In Part 2 of the Capstone, the outliers will be removed and further analysis will be conducted.

---

## Continuous Variable Exploration

### Numerical - Categorical

Let's define a function to generate numerical and categorical plots


```python
def num_cat_plots(numerical_variable):
    """
    This function creates a list of the 
    categorical variables without the target varible.
    It then generates boxplots for the input
    numerical variable with each categorical variable
    to display the distribution.
    
    """
    cat_vars = data[categorical_vars].columns.to_list()
    cat_vars.remove('Default')
    for cat_var in cat_vars:
        sns.catplot(y=numerical_variable.name,
                    x=cat_var,
                    hue="Default",
                    kind="box",
                    data=data,
                    showmeans=True,
                    height=2,
                    aspect=3)
        plt.show()
        # Line separator
        print('--' * 55)
```

#### Perc_premium_paid_in_cash


```python
num_cat_plots(data.Perc_premium_paid_in_cash)
```


    
![png](output_80_0.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_80_2.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_80_4.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_80_6.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_80_8.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_80_10.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_80_12.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_80_14.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_80_16.png)
    


    --------------------------------------------------------------------------------------------------------------
    

* **Perc_premium_paid_in_cash**
    * Policy holders who default on their insurance premiums tend to pay the bulk of their premium with cash.   
    More data is need to confirm but a reasonable assumption is the policy holders who default, work at jobs which pays their income in cash.

#### Age_in_days


```python
num_cat_plots(data.Age_in_days)
```


    
![png](output_83_0.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_83_2.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_83_4.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_83_6.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_83_8.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_83_10.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_83_12.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_83_14.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_83_16.png)
    


    --------------------------------------------------------------------------------------------------------------
    

* **Age_in_days**
    * The average Age_in_days of policy holders who default on their premium payments is lower than those who dont default.   

#### Income


```python
num_cat_plots(data.Income)
```


    
![png](output_86_0.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_86_2.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_86_4.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_86_6.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_86_8.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_86_10.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_86_12.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_86_14.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_86_16.png)
    


    --------------------------------------------------------------------------------------------------------------
    

* **Income**
    * The presence of many outliers make it visually dificult to discern any differnce across defaulters and non-defaulters across income.  
    In the Capstone Part 2, Outlier treatment will be conducted to rectify and allow the any trends to be visually observed.

#### Risk_score


```python
num_cat_plots(data.Risk_score)
```


    
![png](output_89_0.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_89_2.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_89_4.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_89_6.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_89_8.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_89_10.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_89_12.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_89_14.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_89_16.png)
    


    --------------------------------------------------------------------------------------------------------------
    

* **Risk_score**
    * The average Risk_score of policy holders who default on their premium payments is marginally lower than those who don't default.
    * There is a significant quantity of outliers which is prevently clearer distinguishment of any differences between  defaulters and non-defaulters. This will be rectified in Capstone Part 2.

#### No_of_premiums_paid


```python
num_cat_plots(data.No_of_premiums_paid)
```


    
![png](output_92_0.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_92_2.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_92_4.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_92_6.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_92_8.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_92_10.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_92_12.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_92_14.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_92_16.png)
    


    --------------------------------------------------------------------------------------------------------------
    

* **No_of_premiums_paid**
    * The average No_of_premiums_paid of policy holders who default on their premium payments is marginally lower than those who don't default.
    There is a significant quantity of outliers which is prevently clearer distinguishment of any differences between defaulters and non-defaulters. This will be rectified in Capstone Part 2.

#### Premium_payment


```python
num_cat_plots(data.Premium_payment)
```


    
![png](output_95_0.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_95_2.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_95_4.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_95_6.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_95_8.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_95_10.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_95_12.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_95_14.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_95_16.png)
    


    --------------------------------------------------------------------------------------------------------------
    

* **Premium_payment**
    * The average Premium_payment of policy holders who default on their premium payments is lower than those who don't default.

---

**Default by Age_in_days**


```python
sns.lineplot(x='Age_in_days',y='Default',data=data)
plt.title("Default by Age_in_days");
```

Observations:
* Senior Adults (80+ years) have a higher probability of defaulting on their insurance premiums.

---

 ## Categorical Variable Exploration

Let's define a function to generate categorical variables vs target variable plots


```python
def categ_target_plots(target_variable):
    """
    This function creates a list of the 
    categorical variables without the target varible.
    It then generates countplots for the input
    target variable with each categorical variable
    to display the distribution.
    
    """
    cat_vars = data[categorical_vars].columns.to_list()
    cat_vars.remove(target_variable.name)

    for cat_var in cat_vars:
        sns.catplot(data=data,
                    y=cat_var,
                    hue=target_variable.name,
                    kind="count")
        plt.title("{} by {}".format(cat_var, target_variable.name),
                  loc="center",
                  fontsize=16)
        plt.show()
        # Line separator
        print('--' * 55)
```


```python
categ_target_plots(data.Default)
```


    
![png](output_105_0.png)
    



    
![png](output_105_1.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_105_3.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_105_5.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_105_7.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_105_9.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_105_11.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_105_13.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_105_15.png)
    


    --------------------------------------------------------------------------------------------------------------
    


    
![png](output_105_17.png)
    


    --------------------------------------------------------------------------------------------------------------
    

---


```python
def stacked_plot(x, flag=True):
    sns.set(palette='nipy_spectral')
    table_values = pd.crosstab(x, data['Default'], margins=True)
    if flag == True:
        display(table_values)

    table_norm = pd.crosstab(x, data['Default'], normalize='index')
    table_norm.plot(kind='bar', stacked=True, figsize=(8, 5))
    plt.legend(loc='lower left', frameon=False)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.show()
    print('-'*80)  
```


```python
# Categorical variables
categ_list = data[categorical_vars].columns.to_list()
categ_list.remove('Default')

for each_var in categ_list:
    stacked_plot(data[each_var])
    plt.figure()
    plt.show();
```


<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Default</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Late_premium_payment_3-6_months</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2687</td>
      <td>64211</td>
      <td>66898</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1153</td>
      <td>7673</td>
      <td>8826</td>
    </tr>
    <tr>
      <th>2</th>
      <td>592</td>
      <td>1927</td>
      <td>2519</td>
    </tr>
    <tr>
      <th>3</th>
      <td>288</td>
      <td>666</td>
      <td>954</td>
    </tr>
    <tr>
      <th>4</th>
      <td>158</td>
      <td>216</td>
      <td>374</td>
    </tr>
    <tr>
      <th>5</th>
      <td>67</td>
      <td>101</td>
      <td>168</td>
    </tr>
    <tr>
      <th>6</th>
      <td>31</td>
      <td>37</td>
      <td>68</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10</td>
      <td>13</td>
      <td>23</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6</td>
      <td>9</td>
      <td>15</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>All</th>
      <td>4998</td>
      <td>74855</td>
      <td>79853</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](output_108_1.png)
    


    --------------------------------------------------------------------------------
    


    <Figure size 640x480 with 0 Axes>



<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Default</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Late_premium_payment_6-12_months</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3505</td>
      <td>72423</td>
      <td>75928</td>
    </tr>
    <tr>
      <th>1</th>
      <td>828</td>
      <td>1852</td>
      <td>2680</td>
    </tr>
    <tr>
      <th>2</th>
      <td>334</td>
      <td>359</td>
      <td>693</td>
    </tr>
    <tr>
      <th>3</th>
      <td>185</td>
      <td>132</td>
      <td>317</td>
    </tr>
    <tr>
      <th>4</th>
      <td>85</td>
      <td>45</td>
      <td>130</td>
    </tr>
    <tr>
      <th>5</th>
      <td>30</td>
      <td>16</td>
      <td>46</td>
    </tr>
    <tr>
      <th>6</th>
      <td>13</td>
      <td>13</td>
      <td>26</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>4</td>
      <td>11</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>All</th>
      <td>4998</td>
      <td>74855</td>
      <td>79853</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](output_108_5.png)
    


    --------------------------------------------------------------------------------
    


    <Figure size 640x480 with 0 Axes>



<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Default</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Late_premium_payment_&gt;12_months</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3810</td>
      <td>72325</td>
      <td>76135</td>
    </tr>
    <tr>
      <th>1</th>
      <td>835</td>
      <td>2161</td>
      <td>2996</td>
    </tr>
    <tr>
      <th>2</th>
      <td>228</td>
      <td>270</td>
      <td>498</td>
    </tr>
    <tr>
      <th>3</th>
      <td>85</td>
      <td>66</td>
      <td>151</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25</td>
      <td>23</td>
      <td>48</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>6</td>
      <td>13</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>All</th>
      <td>4998</td>
      <td>74855</td>
      <td>79853</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](output_108_9.png)
    


    --------------------------------------------------------------------------------
    


    <Figure size 640x480 with 0 Axes>



<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Default</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Marital_Status</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2571</td>
      <td>37461</td>
      <td>40032</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2427</td>
      <td>37394</td>
      <td>39821</td>
    </tr>
    <tr>
      <th>All</th>
      <td>4998</td>
      <td>74855</td>
      <td>79853</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](output_108_13.png)
    


    --------------------------------------------------------------------------------
    


    <Figure size 640x480 with 0 Axes>



<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Default</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Vehicles_Owned</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1668</td>
      <td>25078</td>
      <td>26746</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1678</td>
      <td>24842</td>
      <td>26520</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1652</td>
      <td>24935</td>
      <td>26587</td>
    </tr>
    <tr>
      <th>All</th>
      <td>4998</td>
      <td>74855</td>
      <td>79853</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](output_108_17.png)
    


    --------------------------------------------------------------------------------
    


    <Figure size 640x480 with 0 Axes>



<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Default</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>No_of_dependents</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1190</td>
      <td>18650</td>
      <td>19840</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1258</td>
      <td>18644</td>
      <td>19902</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1283</td>
      <td>18932</td>
      <td>20215</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1267</td>
      <td>18629</td>
      <td>19896</td>
    </tr>
    <tr>
      <th>All</th>
      <td>4998</td>
      <td>74855</td>
      <td>79853</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](output_108_21.png)
    


    --------------------------------------------------------------------------------
    


    <Figure size 640x480 with 0 Axes>



<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Default</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Accomodation</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2453</td>
      <td>37370</td>
      <td>39823</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2545</td>
      <td>37485</td>
      <td>40030</td>
    </tr>
    <tr>
      <th>All</th>
      <td>4998</td>
      <td>74855</td>
      <td>79853</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](output_108_25.png)
    


    --------------------------------------------------------------------------------
    


    <Figure size 640x480 with 0 Axes>



<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Default</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Sourcing_channel</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2349</td>
      <td>40785</td>
      <td>43134</td>
    </tr>
    <tr>
      <th>B</th>
      <td>1066</td>
      <td>15446</td>
      <td>16512</td>
    </tr>
    <tr>
      <th>C</th>
      <td>903</td>
      <td>11136</td>
      <td>12039</td>
    </tr>
    <tr>
      <th>D</th>
      <td>634</td>
      <td>6925</td>
      <td>7559</td>
    </tr>
    <tr>
      <th>E</th>
      <td>46</td>
      <td>563</td>
      <td>609</td>
    </tr>
    <tr>
      <th>All</th>
      <td>4998</td>
      <td>74855</td>
      <td>79853</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](output_108_29.png)
    


    --------------------------------------------------------------------------------
    


    <Figure size 640x480 with 0 Axes>



<div>

*Output:*


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Default</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Customer_demographic</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Rural</th>
      <td>1998</td>
      <td>29672</td>
      <td>31670</td>
    </tr>
    <tr>
      <th>Urban</th>
      <td>3000</td>
      <td>45183</td>
      <td>48183</td>
    </tr>
    <tr>
      <th>All</th>
      <td>4998</td>
      <td>74855</td>
      <td>79853</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](output_108_33.png)
    


    --------------------------------------------------------------------------------
    


    <Figure size 640x480 with 0 Axes>


<font color='red'>**The later the insurance premium payment, the higher probability of the policy holder being a defaulter.**

---

