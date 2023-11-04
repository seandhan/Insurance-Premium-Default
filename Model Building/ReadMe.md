# Model Building

Discuss possible analytical approaches for model development


* Modelling Process
* Model comparisons
* Model interpretation
* Business insights and Recommendations

---

## Model Building Criteria 

The premium paid by the customer is the major revenue source for insurance companies.  
Since the default in premium payments incurr significant revenue losses, insurance companies would like to know upfront which type of customers would default premium payments.

In this project, the target variable is heavily imbalanced. Therefore the most appropriate metric for model building will be the F1 score which will provide a balance between the Recall and Precision.

<font color='red'>The metric to be maximized is **F1 Score**.

---

## Model Building Preparation

**Define functions**
* Confusion matrix
* Metric scores(accuracy, recall and precision)


**Confusion Matrix definition:**

| **Parameter**| **Description** | 
| :-: | :-: |
| **True Positive** | Predict policy holder will **default** on premium payment and the policy holder actually **defaulted** on their premium payment|
| **False Positive** | Predict policy holder will **default** on premium payment and the policy holder **did not default**  on their premium payment|
| **True Negative** | Predict policy holder will **not default** on premium payment and the policy holder actually **did not default** on their premium payment|
| **False Negative** | Predict policy holder will **not default** on premium payment and the policy holder **defaulted** on their premium payment|

---


```python
# Function to calculate different metric scores of the model - Accuracy, Recall, Precision and F1 Score
def get_metrics_score(model, train, test, train_y, test_y, flag=True):
    '''
    model : classifier to predict values of X

    '''
    # defining an empty list to store train and test results
    score_list = []

    pred_train = model.predict(train)
    pred_test = model.predict(test)

    train_acc = model.score(train, train_y)
    test_acc = model.score(test, test_y)

    train_recall = metrics.recall_score(train_y, pred_train)
    test_recall = metrics.recall_score(test_y, pred_test)

    train_precision = metrics.precision_score(train_y, pred_train)
    test_precision = metrics.precision_score(test_y, pred_test)

    train_f1 = metrics.f1_score(train_y, pred_train)
    test_f1 = metrics.f1_score(test_y, pred_test)

    score_list.extend((train_acc, test_acc, train_recall, test_recall,
                       train_precision, test_precision, train_f1, test_f1))

    # If the flag is set to True then only the following print statements will be dispayed. The default value is set to True.
    if flag == True:
        print("Accuracy on training set : ", model.score(train, train_y))
        print("Accuracy on test set : ", model.score(test, test_y))
        print("Recall on training set : ",
              metrics.recall_score(train_y, pred_train))
        print("Recall on test set : ", metrics.recall_score(test_y, pred_test))
        print("Precision on training set : ",
              metrics.precision_score(train_y, pred_train))
        print("Precision on test set : ",
              metrics.precision_score(test_y, pred_test))
        print("F1 Score on training set : ",
              metrics.f1_score(train_y, pred_train))
        print("F1 Score on test set : ", metrics.f1_score(test_y, pred_test))

    return score_list  # returning the list with train and test scores
```


```python
def make_confusion_matrix(model, y_actual, labels=[1, 0]):
    '''
    model : classifier to predict values of X
    y_actual : ground truth  
    
    '''
    y_predict = model.predict(X_test)
    cm = metrics.confusion_matrix(y_actual, y_predict, labels=[0, 1])
    df_cm = pd.DataFrame(
        cm,
        index=[i for i in ["Actual - No", "Actual - Yes"]],
        columns=[i for i in ['Predicted - No', 'Predicted - Yes']])
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = [
        "{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)
    ]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    plt.figure(figsize=(5, 5))
    sns.heatmap(df_cm, annot=labels, fmt='')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
```

---

## Data Preparation

The dataframe still has the `ID` variable which will not be used in modelling process. Therefore this will be ignored only for the modelling analysis


```python
# Separating Independent and Dependent variables

variables_to_be_excluded = ['ID', 'Default']

# Independant variables
# Removing 'ID' & 'Default' variables
X = df.drop(columns=variables_to_be_excluded,axis=1)

# Dependent variable
y = df['Default']
# Convert Dependent variable back to binary form
y = y.replace({'Not_Default': 0, 'Default': 1})
```


```python
# Convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first=True)
```

**Let's check if our variables has multicollinearity**


```python
# dataframe with numerical column only
num_feature_set = X.copy()
from statsmodels.tools.tools import add_constant

num_feature_set = add_constant(num_feature_set)
```


```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_series1 = pd.Series([
    variance_inflation_factor(num_feature_set.values, i)
    for i in range(num_feature_set.shape[1])
],
                        index=num_feature_set.columns)
print('Series before feature selection: \n\n{}\n'.format(vif_series1))
```

    Series before feature selection: 
    
    const                                            16054.702130
    Perc_premium_paid_in_cash_0.1 - 0.2                  1.156030
    Perc_premium_paid_in_cash_0.2 - 0.3                  1.125598
    Perc_premium_paid_in_cash_0.3 - 0.4                  1.111702
    Perc_premium_paid_in_cash_0.4 - 0.5                  1.101760
    Perc_premium_paid_in_cash_0.5 - 0.6                  1.091226
    Perc_premium_paid_in_cash_0.6 - 0.7                  1.086973
    Perc_premium_paid_in_cash_0.7 - 0.8                  1.086804
    Perc_premium_paid_in_cash_0.8 - 0.9                  1.096424
    Perc_premium_paid_in_cash_0.9 - 1                    1.404229
    Late_premium_payment_3-6_months_Paid_on_time         1.129826
    Late_premium_payment_6-12_months_Paid_on_time        1.166382
    Late_premium_payment_>12_months_Paid_on_time         1.127558
    Marital_Status_Unmarried                             1.000747
    Vehicles_Owned_Three                                 1.331100
    Vehicles_Owned_Two                                   1.330878
    No_of_dependents_One                                 1.502333
    No_of_dependents_Three                               1.506439
    No_of_dependents_Two                                 1.502780
    Accomodation_Rented                                  1.000482
    Risk_score_92 - 93                                  13.395923
    Risk_score_93 - 94                                  19.586691
    Risk_score_94 - 95                                  33.947152
    Risk_score_95 - 96                                  63.387173
    Risk_score_96 - 97                                 187.675925
    Risk_score_97 - 98                                 660.185032
    Risk_score_98 - 99                                3212.989059
    Risk_score_99 - 100                               3594.736759
    No_of_premiums_paid_5 - 10                           3.079338
    No_of_premiums_paid_10 - 15                          3.205583
    No_of_premiums_paid_15 - 20                          2.219501
    No_of_premiums_paid_>20                              1.610583
    Sourcing_channel_B                                   1.141501
    Sourcing_channel_C                                   1.227626
    Sourcing_channel_D                                   1.194954
    Sourcing_channel_E                                   1.020886
    Customer_demographic_Urban                           1.007598
    Premium_payment_5000 - 10000                         1.918718
    Premium_payment_10000 - 15000                        1.873207
    Premium_payment_15000 - 20000                        1.528267
    Premium_payment_>20000                               2.312909
    Age_in_years_30 - 40                                 3.184162
    Age_in_years_40 - 50                                 4.068588
    Age_in_years_50 - 60                                 3.924534
    Age_in_years_60 - 70                                 3.305031
    Age_in_years_>70                                     2.442670
    Income_'000_60 - 100                                 3.088352
    Income_'000_100 - 140                                3.677406
    Income_'000_140 - 180                                3.519660
    Income_'000_180 - 220                                3.416340
    Income_'000_220 - 260                                2.970099
    Income_'000_260 - 300                                2.429873
    Income_'000_>300                                     4.973013
    dtype: float64
    
    

* Risk score has high VIF scores, let's calculate the p-value using Statsmodels before we consider dropping prior model building


```python
X_train, X_test, y_train, y_test = train_test_split(num_feature_set,
                                                    y,
                                                    test_size=0.30,
                                                   stratify=y)
```


```python
import statsmodels.api as sm
logit = sm.Logit(y_train, X_train)
lg = logit.fit()

lg.summary()
```

    Optimization terminated successfully.
             Current function value: 0.183787
             Iterations 19
    




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>Default</td>     <th>  No. Observations:  </th>  <td> 55897</td> 
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td> 55844</td> 
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>    52</td> 
</tr>
<tr>
  <th>Date:</th>            <td>Thu, 10 Jun 2021</td> <th>  Pseudo R-squ.:     </th>  <td>0.2148</td> 
</tr>
<tr>
  <th>Time:</th>                <td>13:51:18</td>     <th>  Log-Likelihood:    </th> <td> -10273.</td>
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -13083.</td>
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th>  <td> 0.000</td> 
</tr>
</table>
<table class="simpletable">
<tr>
                        <td></td>                           <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                                         <td>-8978.9568</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>
</tr>
<tr>
  <th>Perc_premium_paid_in_cash_0.1 - 0.2</th>           <td>    0.1983</td> <td>    0.088</td> <td>    2.250</td> <td> 0.024</td> <td>    0.026</td> <td>    0.371</td>
</tr>
<tr>
  <th>Perc_premium_paid_in_cash_0.2 - 0.3</th>           <td>    0.4138</td> <td>    0.092</td> <td>    4.478</td> <td> 0.000</td> <td>    0.233</td> <td>    0.595</td>
</tr>
<tr>
  <th>Perc_premium_paid_in_cash_0.3 - 0.4</th>           <td>    0.6823</td> <td>    0.091</td> <td>    7.535</td> <td> 0.000</td> <td>    0.505</td> <td>    0.860</td>
</tr>
<tr>
  <th>Perc_premium_paid_in_cash_0.4 - 0.5</th>           <td>    0.8335</td> <td>    0.090</td> <td>    9.260</td> <td> 0.000</td> <td>    0.657</td> <td>    1.010</td>
</tr>
<tr>
  <th>Perc_premium_paid_in_cash_0.5 - 0.6</th>           <td>    1.0764</td> <td>    0.089</td> <td>   12.094</td> <td> 0.000</td> <td>    0.902</td> <td>    1.251</td>
</tr>
<tr>
  <th>Perc_premium_paid_in_cash_0.6 - 0.7</th>           <td>    1.1531</td> <td>    0.090</td> <td>   12.838</td> <td> 0.000</td> <td>    0.977</td> <td>    1.329</td>
</tr>
<tr>
  <th>Perc_premium_paid_in_cash_0.7 - 0.8</th>           <td>    1.3675</td> <td>    0.087</td> <td>   15.720</td> <td> 0.000</td> <td>    1.197</td> <td>    1.538</td>
</tr>
<tr>
  <th>Perc_premium_paid_in_cash_0.8 - 0.9</th>           <td>    1.4536</td> <td>    0.084</td> <td>   17.376</td> <td> 0.000</td> <td>    1.290</td> <td>    1.618</td>
</tr>
<tr>
  <th>Perc_premium_paid_in_cash_0.9 - 1</th>             <td>    1.6392</td> <td>    0.065</td> <td>   25.049</td> <td> 0.000</td> <td>    1.511</td> <td>    1.768</td>
</tr>
<tr>
  <th>Late_premium_payment_3-6_months_Paid_on_time</th>  <td>   -0.8976</td> <td>    0.042</td> <td>  -21.431</td> <td> 0.000</td> <td>   -0.980</td> <td>   -0.816</td>
</tr>
<tr>
  <th>Late_premium_payment_6-12_months_Paid_on_time</th> <td>   -1.5207</td> <td>    0.053</td> <td>  -28.755</td> <td> 0.000</td> <td>   -1.624</td> <td>   -1.417</td>
</tr>
<tr>
  <th>Late_premium_payment_>12_months_Paid_on_time</th>  <td>   -0.9509</td> <td>    0.056</td> <td>  -16.963</td> <td> 0.000</td> <td>   -1.061</td> <td>   -0.841</td>
</tr>
<tr>
  <th>Marital_Status_Unmarried</th>                      <td>    0.0377</td> <td>    0.038</td> <td>    0.979</td> <td> 0.327</td> <td>   -0.038</td> <td>    0.113</td>
</tr>
<tr>
  <th>Vehicles_Owned_Three</th>                          <td>   -0.0050</td> <td>    0.047</td> <td>   -0.106</td> <td> 0.916</td> <td>   -0.097</td> <td>    0.087</td>
</tr>
<tr>
  <th>Vehicles_Owned_Two</th>                            <td>   -0.0252</td> <td>    0.047</td> <td>   -0.532</td> <td> 0.594</td> <td>   -0.118</td> <td>    0.067</td>
</tr>
<tr>
  <th>No_of_dependents_One</th>                          <td>   -0.0250</td> <td>    0.055</td> <td>   -0.455</td> <td> 0.649</td> <td>   -0.133</td> <td>    0.083</td>
</tr>
<tr>
  <th>No_of_dependents_Three</th>                        <td>    0.0091</td> <td>    0.054</td> <td>    0.167</td> <td> 0.867</td> <td>   -0.097</td> <td>    0.115</td>
</tr>
<tr>
  <th>No_of_dependents_Two</th>                          <td>    0.0170</td> <td>    0.054</td> <td>    0.314</td> <td> 0.754</td> <td>   -0.089</td> <td>    0.123</td>
</tr>
<tr>
  <th>Accomodation_Rented</th>                           <td>   -0.0366</td> <td>    0.038</td> <td>   -0.952</td> <td> 0.341</td> <td>   -0.112</td> <td>    0.039</td>
</tr>
<tr>
  <th>Risk_score_92 - 93</th>                            <td> 8979.0855</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>
</tr>
<tr>
  <th>Risk_score_93 - 94</th>                            <td> 8978.8264</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>
</tr>
<tr>
  <th>Risk_score_94 - 95</th>                            <td> 8979.3270</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>
</tr>
<tr>
  <th>Risk_score_95 - 96</th>                            <td> 8978.7474</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>
</tr>
<tr>
  <th>Risk_score_96 - 97</th>                            <td> 8979.3155</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>
</tr>
<tr>
  <th>Risk_score_97 - 98</th>                            <td> 8979.0962</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>
</tr>
<tr>
  <th>Risk_score_98 - 99</th>                            <td> 8978.8363</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>
</tr>
<tr>
  <th>Risk_score_99 - 100</th>                           <td> 8978.6261</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan</td> <td>      nan</td>
</tr>
<tr>
  <th>No_of_premiums_paid_5 - 10</th>                    <td>   -0.2027</td> <td>    0.062</td> <td>   -3.245</td> <td> 0.001</td> <td>   -0.325</td> <td>   -0.080</td>
</tr>
<tr>
  <th>No_of_premiums_paid_10 - 15</th>                   <td>    0.0761</td> <td>    0.072</td> <td>    1.056</td> <td> 0.291</td> <td>   -0.065</td> <td>    0.217</td>
</tr>
<tr>
  <th>No_of_premiums_paid_15 - 20</th>                   <td>    0.3508</td> <td>    0.086</td> <td>    4.069</td> <td> 0.000</td> <td>    0.182</td> <td>    0.520</td>
</tr>
<tr>
  <th>No_of_premiums_paid_>20</th>                       <td>    0.5170</td> <td>    0.109</td> <td>    4.755</td> <td> 0.000</td> <td>    0.304</td> <td>    0.730</td>
</tr>
<tr>
  <th>Sourcing_channel_B</th>                            <td>    0.0342</td> <td>    0.051</td> <td>    0.668</td> <td> 0.504</td> <td>   -0.066</td> <td>    0.135</td>
</tr>
<tr>
  <th>Sourcing_channel_C</th>                            <td>    0.1505</td> <td>    0.056</td> <td>    2.689</td> <td> 0.007</td> <td>    0.041</td> <td>    0.260</td>
</tr>
<tr>
  <th>Sourcing_channel_D</th>                            <td>    0.2544</td> <td>    0.065</td> <td>    3.907</td> <td> 0.000</td> <td>    0.127</td> <td>    0.382</td>
</tr>
<tr>
  <th>Sourcing_channel_E</th>                            <td>    0.1075</td> <td>    0.208</td> <td>    0.516</td> <td> 0.606</td> <td>   -0.301</td> <td>    0.516</td>
</tr>
<tr>
  <th>Customer_demographic_Urban</th>                    <td>    0.0330</td> <td>    0.040</td> <td>    0.835</td> <td> 0.404</td> <td>   -0.045</td> <td>    0.111</td>
</tr>
<tr>
  <th>Premium_payment_5000 - 10000</th>                  <td>    0.0581</td> <td>    0.052</td> <td>    1.121</td> <td> 0.262</td> <td>   -0.043</td> <td>    0.160</td>
</tr>
<tr>
  <th>Premium_payment_10000 - 15000</th>                 <td>    0.0627</td> <td>    0.073</td> <td>    0.856</td> <td> 0.392</td> <td>   -0.081</td> <td>    0.206</td>
</tr>
<tr>
  <th>Premium_payment_15000 - 20000</th>                 <td>    0.0128</td> <td>    0.100</td> <td>    0.128</td> <td> 0.898</td> <td>   -0.182</td> <td>    0.208</td>
</tr>
<tr>
  <th>Premium_payment_>20000</th>                        <td>    0.1503</td> <td>    0.092</td> <td>    1.637</td> <td> 0.102</td> <td>   -0.030</td> <td>    0.330</td>
</tr>
<tr>
  <th>Age_in_years_30 - 40</th>                          <td>   -0.2010</td> <td>    0.075</td> <td>   -2.667</td> <td> 0.008</td> <td>   -0.349</td> <td>   -0.053</td>
</tr>
<tr>
  <th>Age_in_years_40 - 50</th>                          <td>   -0.2304</td> <td>    0.076</td> <td>   -3.047</td> <td> 0.002</td> <td>   -0.379</td> <td>   -0.082</td>
</tr>
<tr>
  <th>Age_in_years_50 - 60</th>                          <td>   -0.4106</td> <td>    0.078</td> <td>   -5.292</td> <td> 0.000</td> <td>   -0.563</td> <td>   -0.259</td>
</tr>
<tr>
  <th>Age_in_years_60 - 70</th>                          <td>   -0.6374</td> <td>    0.088</td> <td>   -7.284</td> <td> 0.000</td> <td>   -0.809</td> <td>   -0.466</td>
</tr>
<tr>
  <th>Age_in_years_>70</th>                              <td>   -0.7059</td> <td>    0.111</td> <td>   -6.348</td> <td> 0.000</td> <td>   -0.924</td> <td>   -0.488</td>
</tr>
<tr>
  <th>Income_'000_60 - 100</th>                          <td>   -0.0170</td> <td>    0.081</td> <td>   -0.209</td> <td> 0.834</td> <td>   -0.176</td> <td>    0.142</td>
</tr>
<tr>
  <th>Income_'000_100 - 140</th>                         <td>   -0.1463</td> <td>    0.086</td> <td>   -1.707</td> <td> 0.088</td> <td>   -0.314</td> <td>    0.022</td>
</tr>
<tr>
  <th>Income_'000_140 - 180</th>                         <td>   -0.2026</td> <td>    0.092</td> <td>   -2.195</td> <td> 0.028</td> <td>   -0.384</td> <td>   -0.022</td>
</tr>
<tr>
  <th>Income_'000_180 - 220</th>                         <td>   -0.2902</td> <td>    0.099</td> <td>   -2.943</td> <td> 0.003</td> <td>   -0.483</td> <td>   -0.097</td>
</tr>
<tr>
  <th>Income_'000_220 - 260</th>                         <td>   -0.4717</td> <td>    0.112</td> <td>   -4.199</td> <td> 0.000</td> <td>   -0.692</td> <td>   -0.252</td>
</tr>
<tr>
  <th>Income_'000_260 - 300</th>                         <td>   -0.4106</td> <td>    0.126</td> <td>   -3.247</td> <td> 0.001</td> <td>   -0.659</td> <td>   -0.163</td>
</tr>
<tr>
  <th>Income_'000_>300</th>                              <td>   -0.4774</td> <td>    0.114</td> <td>   -4.191</td> <td> 0.000</td> <td>   -0.701</td> <td>   -0.254</td>
</tr>
</table>



* The Risk scores which had high VIF scores produce p-value with NaN. These variables will be removed


```python
vars_to_drop = [
    'Risk_score_92 - 93', 'Risk_score_93 - 94', 'Risk_score_94 - 95',
    'Risk_score_95 - 96', 'Risk_score_96 - 97', 'Risk_score_97 - 98',
    'Risk_score_98 - 99', 'Risk_score_99 - 100'
]
```


```python
X_train1 = X_train.drop(columns=vars_to_drop , axis = 1)
X_test1 = X_test.drop(columns=vars_to_drop, axis = 1)

logit1 = sm.Logit(y_train, X_train1)
lg1 = logit1.fit()

lg1.summary()
```

    Optimization terminated successfully.
             Current function value: 0.184302
             Iterations 8
    




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>Default</td>     <th>  No. Observations:  </th>  <td> 55897</td> 
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td> 55852</td> 
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>    44</td> 
</tr>
<tr>
  <th>Date:</th>            <td>Thu, 10 Jun 2021</td> <th>  Pseudo R-squ.:     </th>  <td>0.2126</td> 
</tr>
<tr>
  <th>Time:</th>                <td>13:51:18</td>     <th>  Log-Likelihood:    </th> <td> -10302.</td>
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -13083.</td>
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th>  <td> 0.000</td> 
</tr>
</table>
<table class="simpletable">
<tr>
                        <td></td>                           <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                                         <td>   -0.3422</td> <td>    0.132</td> <td>   -2.594</td> <td> 0.009</td> <td>   -0.601</td> <td>   -0.084</td>
</tr>
<tr>
  <th>Perc_premium_paid_in_cash_0.1 - 0.2</th>           <td>    0.2136</td> <td>    0.088</td> <td>    2.427</td> <td> 0.015</td> <td>    0.041</td> <td>    0.386</td>
</tr>
<tr>
  <th>Perc_premium_paid_in_cash_0.2 - 0.3</th>           <td>    0.4381</td> <td>    0.092</td> <td>    4.753</td> <td> 0.000</td> <td>    0.257</td> <td>    0.619</td>
</tr>
<tr>
  <th>Perc_premium_paid_in_cash_0.3 - 0.4</th>           <td>    0.7240</td> <td>    0.090</td> <td>    8.029</td> <td> 0.000</td> <td>    0.547</td> <td>    0.901</td>
</tr>
<tr>
  <th>Perc_premium_paid_in_cash_0.4 - 0.5</th>           <td>    0.8802</td> <td>    0.090</td> <td>    9.825</td> <td> 0.000</td> <td>    0.705</td> <td>    1.056</td>
</tr>
<tr>
  <th>Perc_premium_paid_in_cash_0.5 - 0.6</th>           <td>    1.1258</td> <td>    0.089</td> <td>   12.693</td> <td> 0.000</td> <td>    0.952</td> <td>    1.300</td>
</tr>
<tr>
  <th>Perc_premium_paid_in_cash_0.6 - 0.7</th>           <td>    1.2138</td> <td>    0.089</td> <td>   13.604</td> <td> 0.000</td> <td>    1.039</td> <td>    1.389</td>
</tr>
<tr>
  <th>Perc_premium_paid_in_cash_0.7 - 0.8</th>           <td>    1.4275</td> <td>    0.087</td> <td>   16.492</td> <td> 0.000</td> <td>    1.258</td> <td>    1.597</td>
</tr>
<tr>
  <th>Perc_premium_paid_in_cash_0.8 - 0.9</th>           <td>    1.5228</td> <td>    0.083</td> <td>   18.329</td> <td> 0.000</td> <td>    1.360</td> <td>    1.686</td>
</tr>
<tr>
  <th>Perc_premium_paid_in_cash_0.9 - 1</th>             <td>    1.6986</td> <td>    0.065</td> <td>   26.125</td> <td> 0.000</td> <td>    1.571</td> <td>    1.826</td>
</tr>
<tr>
  <th>Late_premium_payment_3-6_months_Paid_on_time</th>  <td>   -0.8984</td> <td>    0.042</td> <td>  -21.476</td> <td> 0.000</td> <td>   -0.980</td> <td>   -0.816</td>
</tr>
<tr>
  <th>Late_premium_payment_6-12_months_Paid_on_time</th> <td>   -1.5092</td> <td>    0.053</td> <td>  -28.652</td> <td> 0.000</td> <td>   -1.612</td> <td>   -1.406</td>
</tr>
<tr>
  <th>Late_premium_payment_>12_months_Paid_on_time</th>  <td>   -0.9482</td> <td>    0.056</td> <td>  -16.961</td> <td> 0.000</td> <td>   -1.058</td> <td>   -0.839</td>
</tr>
<tr>
  <th>Marital_Status_Unmarried</th>                      <td>    0.0397</td> <td>    0.038</td> <td>    1.033</td> <td> 0.302</td> <td>   -0.036</td> <td>    0.115</td>
</tr>
<tr>
  <th>Vehicles_Owned_Three</th>                          <td>   -0.0050</td> <td>    0.047</td> <td>   -0.106</td> <td> 0.915</td> <td>   -0.097</td> <td>    0.087</td>
</tr>
<tr>
  <th>Vehicles_Owned_Two</th>                            <td>   -0.0244</td> <td>    0.047</td> <td>   -0.517</td> <td> 0.605</td> <td>   -0.117</td> <td>    0.068</td>
</tr>
<tr>
  <th>No_of_dependents_One</th>                          <td>   -0.0293</td> <td>    0.055</td> <td>   -0.533</td> <td> 0.594</td> <td>   -0.137</td> <td>    0.078</td>
</tr>
<tr>
  <th>No_of_dependents_Three</th>                        <td>    0.0069</td> <td>    0.054</td> <td>    0.128</td> <td> 0.898</td> <td>   -0.099</td> <td>    0.113</td>
</tr>
<tr>
  <th>No_of_dependents_Two</th>                          <td>    0.0121</td> <td>    0.054</td> <td>    0.223</td> <td> 0.823</td> <td>   -0.094</td> <td>    0.118</td>
</tr>
<tr>
  <th>Accomodation_Rented</th>                           <td>   -0.0354</td> <td>    0.038</td> <td>   -0.920</td> <td> 0.357</td> <td>   -0.111</td> <td>    0.040</td>
</tr>
<tr>
  <th>No_of_premiums_paid_5 - 10</th>                    <td>   -0.1056</td> <td>    0.061</td> <td>   -1.738</td> <td> 0.082</td> <td>   -0.225</td> <td>    0.013</td>
</tr>
<tr>
  <th>No_of_premiums_paid_10 - 15</th>                   <td>    0.2437</td> <td>    0.068</td> <td>    3.593</td> <td> 0.000</td> <td>    0.111</td> <td>    0.377</td>
</tr>
<tr>
  <th>No_of_premiums_paid_15 - 20</th>                   <td>    0.5578</td> <td>    0.081</td> <td>    6.902</td> <td> 0.000</td> <td>    0.399</td> <td>    0.716</td>
</tr>
<tr>
  <th>No_of_premiums_paid_>20</th>                       <td>    0.7506</td> <td>    0.103</td> <td>    7.265</td> <td> 0.000</td> <td>    0.548</td> <td>    0.953</td>
</tr>
<tr>
  <th>Sourcing_channel_B</th>                            <td>    0.0386</td> <td>    0.051</td> <td>    0.755</td> <td> 0.450</td> <td>   -0.062</td> <td>    0.139</td>
</tr>
<tr>
  <th>Sourcing_channel_C</th>                            <td>    0.1666</td> <td>    0.056</td> <td>    2.984</td> <td> 0.003</td> <td>    0.057</td> <td>    0.276</td>
</tr>
<tr>
  <th>Sourcing_channel_D</th>                            <td>    0.2637</td> <td>    0.065</td> <td>    4.056</td> <td> 0.000</td> <td>    0.136</td> <td>    0.391</td>
</tr>
<tr>
  <th>Sourcing_channel_E</th>                            <td>    0.1089</td> <td>    0.207</td> <td>    0.525</td> <td> 0.600</td> <td>   -0.298</td> <td>    0.516</td>
</tr>
<tr>
  <th>Customer_demographic_Urban</th>                    <td>    0.0347</td> <td>    0.039</td> <td>    0.879</td> <td> 0.379</td> <td>   -0.043</td> <td>    0.112</td>
</tr>
<tr>
  <th>Premium_payment_5000 - 10000</th>                  <td>    0.0538</td> <td>    0.052</td> <td>    1.042</td> <td> 0.297</td> <td>   -0.047</td> <td>    0.155</td>
</tr>
<tr>
  <th>Premium_payment_10000 - 15000</th>                 <td>    0.0544</td> <td>    0.073</td> <td>    0.745</td> <td> 0.456</td> <td>   -0.089</td> <td>    0.198</td>
</tr>
<tr>
  <th>Premium_payment_15000 - 20000</th>                 <td>    0.0076</td> <td>    0.099</td> <td>    0.076</td> <td> 0.939</td> <td>   -0.187</td> <td>    0.202</td>
</tr>
<tr>
  <th>Premium_payment_>20000</th>                        <td>    0.1381</td> <td>    0.092</td> <td>    1.507</td> <td> 0.132</td> <td>   -0.042</td> <td>    0.318</td>
</tr>
<tr>
  <th>Age_in_years_30 - 40</th>                          <td>   -0.1408</td> <td>    0.075</td> <td>   -1.882</td> <td> 0.060</td> <td>   -0.287</td> <td>    0.006</td>
</tr>
<tr>
  <th>Age_in_years_40 - 50</th>                          <td>   -0.1469</td> <td>    0.075</td> <td>   -1.969</td> <td> 0.049</td> <td>   -0.293</td> <td>   -0.001</td>
</tr>
<tr>
  <th>Age_in_years_50 - 60</th>                          <td>   -0.3302</td> <td>    0.077</td> <td>   -4.308</td> <td> 0.000</td> <td>   -0.480</td> <td>   -0.180</td>
</tr>
<tr>
  <th>Age_in_years_60 - 70</th>                          <td>   -0.5645</td> <td>    0.087</td> <td>   -6.508</td> <td> 0.000</td> <td>   -0.734</td> <td>   -0.394</td>
</tr>
<tr>
  <th>Age_in_years_>70</th>                              <td>   -0.6642</td> <td>    0.111</td> <td>   -5.993</td> <td> 0.000</td> <td>   -0.881</td> <td>   -0.447</td>
</tr>
<tr>
  <th>Income_'000_60 - 100</th>                          <td>   -0.0752</td> <td>    0.080</td> <td>   -0.939</td> <td> 0.348</td> <td>   -0.232</td> <td>    0.082</td>
</tr>
<tr>
  <th>Income_'000_100 - 140</th>                         <td>   -0.2365</td> <td>    0.084</td> <td>   -2.826</td> <td> 0.005</td> <td>   -0.400</td> <td>   -0.072</td>
</tr>
<tr>
  <th>Income_'000_140 - 180</th>                         <td>   -0.3241</td> <td>    0.090</td> <td>   -3.617</td> <td> 0.000</td> <td>   -0.500</td> <td>   -0.148</td>
</tr>
<tr>
  <th>Income_'000_180 - 220</th>                         <td>   -0.4303</td> <td>    0.096</td> <td>   -4.501</td> <td> 0.000</td> <td>   -0.618</td> <td>   -0.243</td>
</tr>
<tr>
  <th>Income_'000_220 - 260</th>                         <td>   -0.6320</td> <td>    0.109</td> <td>   -5.789</td> <td> 0.000</td> <td>   -0.846</td> <td>   -0.418</td>
</tr>
<tr>
  <th>Income_'000_260 - 300</th>                         <td>   -0.5815</td> <td>    0.123</td> <td>   -4.715</td> <td> 0.000</td> <td>   -0.823</td> <td>   -0.340</td>
</tr>
<tr>
  <th>Income_'000_>300</th>                              <td>   -0.6769</td> <td>    0.110</td> <td>   -6.174</td> <td> 0.000</td> <td>   -0.892</td> <td>   -0.462</td>
</tr>
</table>



---

The dataframe still has the `ID` variable which will not be used in modelling process. Therefore this will be ignored only for the modelling analysis

**Separating Independent and Dependent variables**


```python
# Separating Independent and Dependent variables

variables_to_be_excluded = ['ID', 'Default', 'Risk_score']

# Independant variables
# Removing 'ID' & 'Default' variables
X = df.drop(columns=variables_to_be_excluded,axis=1)

# Dependent variable
y = df['Default']
# Convert Dependent variable back to binary form
y = y.replace({'Not_Default': 0, 'Default': 1})
```

---

**Convert categorical variables to dummy variables**


```python
# Convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first=True)
```

---

**Split Data into Train & Test Sets**


```python
# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.30,
                                                    stratify=y,
                                                    random_state=1)
```

---

## Preliminary Model Selection

Prior to considering models for evaluation, let's conduct AUC and PR analysis to see which model perform satisfactorily without tuning.

**ROC curves**


```python
plt.figure(figsize=(12, 12))

# Add the models to the list that you want to view on the ROC plot
models = [{
    'label': 'Logistic Regression',
    'model': LogisticRegression()
}, {
    'label': 'XGBoost',
    'model': XGBClassifier(eval_metric='logloss')
}, {
    'label': 'Decision Tree',
    'model': DecisionTreeClassifier()
}, {
    'label': 'Bagging Classifier',
    'model': BaggingClassifier()
}, {
    'label': 'Random Forest',
    'model': RandomForestClassifier()
}, {
    'label': 'AdaBoost',
    'model': AdaBoostClassifier()
}, {
    'label': 'Gradient Boosting',
    'model': GradientBoostingClassifier(),
}]

# Below for loop iterates through your models list
for m in models:
    model = m['model']  # select the model
    model.fit(X_train, y_train)  # train the model
    y_pred = model.predict(X_test)  # predict the test data
    # Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test,
                                             model.predict_proba(X_test)[:, 1])
    # Calculate Area under the curve to display on the plot
    auc_ = metrics.roc_auc_score(y_test, model.predict_proba(X_test)[::,1])
    # Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (Area = %0.2f)' % (m['label'], auc_))
# Custom settings for the plot
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# Display
plt.show() 
```


    
![png](output_39_0.png)
    


**Precision Recall Curves**


```python
# Define the function that will display every Precision-Recall curve


def plot_PR_curve(models):
    for m in models:
        model = m['model']
        model.fit(X_train, y_train)
        print('{}'.format(m['label']))
        # predict probabilities
        probs = model.predict_proba(X_test)
        # keep probabilities for the positive outcome only
        probs = probs[:, 1]
        # predict class values
        y_hat = model.predict(X_test)
        #calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, probs)
        # calculate F1 score
        f1 = f1_score(y_test, y_hat)
        #calculate precision-recall AUC
        the_auc = auc(recall, precision)
        # Calculate average precision score
        ap = average_precision_score(y_test, probs)
        print('F1 score=%.3f AUC=%.3f Average Precision=%.3f' %
              (f1, the_auc, ap))

        plt.figure(figsize=[8, 7])
        # plot no skill (the blue line)
        plt.plot([0, 1], [0.1, 0.1], linestyle='--')
        # plot the precision-recall curve for the model
        plt.plot(recall,
                 precision,
                 marker='.',
                 label='%s ROC (Area = %0.2f)' % (m['label'], the_auc))
        plt.legend(loc='best')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall Curve')
        plt.show()
```


```python
models = [
    {'label': 'Logistic Regression','model': LogisticRegression()}, 
    {'label': 'XGBoost','model': XGBClassifier(eval_metric='logloss')}, 
    {'label': 'Decision Tree','model': DecisionTreeClassifier()}, 
    {'label': 'Bagging Classifier','model': BaggingClassifier()}, 
    {'label': 'Random Forest','model': RandomForestClassifier()}, 
    {'label': 'AdaBoost','model': AdaBoostClassifier()}, 
    {'label': 'Gradient Boosting','model': GradientBoostingClassifier()}
]

plot_PR_curve(models)
```

    Logistic Regression
    F1 score=0.219 AUC=0.329 Average Precision=0.329
    


    
![png](output_42_1.png)
    


    XGBoost
    F1 score=0.191 AUC=0.270 Average Precision=0.271
    


    
![png](output_42_3.png)
    


    Decision Tree
    F1 score=0.206 AUC=0.230 Average Precision=0.092
    


    
![png](output_42_5.png)
    


    Bagging Classifier
    F1 score=0.193 AUC=0.202 Average Precision=0.180
    


    
![png](output_42_7.png)
    


    Random Forest
    F1 score=0.128 AUC=0.246 Average Precision=0.243
    


    
![png](output_42_9.png)
    


    AdaBoost
    F1 score=0.240 AUC=0.328 Average Precision=0.328
    


    
![png](output_42_11.png)
    


    Gradient Boosting
    F1 score=0.198 AUC=0.320 Average Precision=0.321
    


    
![png](output_42_13.png)
    


From the above we have seen the performance of the basic predictive performance on the data. Let's progress using the following classifiers:
* Logistic Regression
* AdaBoost
* Gradient Boosting Methom (GBM)
* XGBoost

---

## Logistic Regression Model fitting


```python
lr = LogisticRegression()
lr.fit(X_train, y_train)

#Calculating different metrics
get_metrics_score(lr, X_train, X_test, y_train, y_test)

# creating confusion matrix
make_confusion_matrix(lr, y_test)
```

    Accuracy on training set :  0.9381898849669928
    Accuracy on test set :  0.9398063115712139
    Recall on training set :  0.12660760217204917
    Recall on test set :  0.13475650433622416
    Precision on training set :  0.5261282660332541
    Precision on test set :  0.5821325648414986
    F1 Score on training set :  0.20410043768716887
    F1 Score on test set :  0.21885157096424704
    


    
![png](output_46_1.png)
    


* The LR model generalizes well and performs marginally better on the test data.

---

### LR Hyperparameter tuning


```python
#Choose the type of classifier.
lr_estimator = LogisticRegression()

LRparam_grid = {
    'C': [1, 10, 100, 1000],
    'penalty': ['l1', 'l2'],
    'max_iter': list(range(100, 800, 100)),
    'solver': ['newton-cg', 'lbfgs', 'liblinear']
}

LR_search = GridSearchCV(estimator=lr_estimator,
                         param_grid=LRparam_grid,
                         scoring='f1',
                         refit=True,
                         cv=5)

# fitting the model for grid search
LR_result = LR_search.fit(X_train, y_train)

LR_result.best_estimator_
```




    LogisticRegression(C=1, penalty='l1', solver='liblinear')




```python
LR_result = LR_result.best_estimator_
```


```python
#Calculating different metrics
get_metrics_score(LR_result, X_train, X_test, y_train, y_test)

# creating confusion matrix
make_confusion_matrix(LR_result, y_test)
```

    Accuracy on training set :  0.93815410487146
    Accuracy on test set :  0.9396810819836366
    Recall on training set :  0.12775078593883968
    Recall on test set :  0.13475650433622416
    Precision on training set :  0.5246478873239436
    Precision on test set :  0.5771428571428572
    F1 Score on training set :  0.20547000689496667
    F1 Score on test set :  0.21849648458626283
    


    
![png](output_52_1.png)
    


---

### LR AUC ROC curve


```python
plt.figure(figsize=(12, 12))

# Add the models to the list that you want to view on the ROC plot
models = [{
    'label': 'Logistic Regression',
    'model': lr_estimator
}, {
    'label': 'Tuned Logistic Regression',
    'model': LR_result
}]

# Below for loop iterates through your models list
for m in models:
    model = m['model']  # select the model
    model.fit(X_train, y_train)  # train the model
    y_pred = model.predict(X_test)  # predict the test data
    # Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test,
                                             model.predict_proba(X_test)[:, 1])
    # Calculate Area under the curve to display on the plot
    auc_ = metrics.roc_auc_score(y_test, model.predict_proba(X_test)[::,1])
    # Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (Area = %0.2f)' % (m['label'], auc_))
# Custom settings for the plot
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# Display
plt.show() 
```


    
![png](output_55_0.png)
    


---

## AdaBoost Classifier


```python
#Fitting the model
ab_classifier = AdaBoostClassifier(random_state=1)
ab_classifier.fit(X_train,y_train)

#Calculating different metrics
get_metrics_score(ab_classifier, X_train, X_test, y_train, y_test)

#Creating confusion matrix
make_confusion_matrix(ab_classifier,y_test)
```

    Accuracy on training set :  0.93815410487146
    Accuracy on test set :  0.9392219068291868
    Recall on training set :  0.14375535867390682
    Recall on test set :  0.1534356237491661
    Precision on training set :  0.5217842323651453
    Precision on test set :  0.5515587529976019
    F1 Score on training set :  0.22540891776831729
    F1 Score on test set :  0.24008350730688938
    


    
![png](output_58_1.png)
    


* The AdaBoost model generalizes well and performs marginally better on the test data.

### Hyperparameter Tuning


```python
%%time

# Choose the type of classifier.
abc = AdaBoostClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {
    #Let's try different max_depth for base_estimator
    "base_estimator": [
        DecisionTreeClassifier(max_depth=1),
        DecisionTreeClassifier(max_depth=2),
        DecisionTreeClassifier(max_depth=3)
    ],
    "n_estimators":
    np.arange(10, 110, 10),
    "learning_rate":
    np.arange(0.1, 2, 0.1)
}

# Type of scoring used to compare parameter  combinations
acc_scorer = metrics.make_scorer(metrics.f1_score)

#Calling RandomizedSearchCV
abc_randomcv = RandomizedSearchCV(estimator=abc,
                               param_distributions=parameters,
                               n_iter=50,
                               scoring=acc_scorer,
                               cv=5,
                               refit=True,
                               random_state=1)

#Fitting parameters in RandomizedSearchCV
abc_tuned = abc_randomcv.fit(X_train, y_train)

print("Best parameters are {} with CV score={}:".format(
    abc_tuned.best_params_, abc_tuned.best_score_))

abc_tuned.best_estimator_
```

    Best parameters are {'n_estimators': 10, 'learning_rate': 1.3000000000000003, 'base_estimator': DecisionTreeClassifier(max_depth=1)} with CV score=0.23909410533277936:
    Wall time: 10min 18s
    




    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),
                       learning_rate=1.3000000000000003, n_estimators=10,
                       random_state=1)




```python
abc_tuned=abc_tuned.best_estimator_
```


```python
#Calculating different metrics
get_metrics_score(abc_tuned, X_train, X_test, y_train, y_test)

# creating confusion matrix
make_confusion_matrix(abc_tuned, y_test)
```

    Accuracy on training set :  0.9367229010501458
    Accuracy on test set :  0.9385122724995826
    Recall on training set :  0.15718776793369535
    Recall on test set :  0.1581054036024016
    Precision on training set :  0.4833040421792619
    Precision on test set :  0.5290178571428571
    F1 Score on training set :  0.2372223420314859
    F1 Score on test set :  0.24345146379044685
    


    
![png](output_63_1.png)
    


---

### Adaboost AUC ROC curve


```python
plt.figure(figsize=(12, 12))

# Add the models to the list that you want to view on the ROC plot
models = [{
    'label': 'AdaBoost',
    'model': AdaBoostClassifier()
}, {
    'label': 'Tuned AdaBoost',
    'model': abc_tuned
}]

# Below for loop iterates through your models list
for m in models:
    model = m['model']  # select the model
    model.fit(X_train, y_train)  # train the model
    y_pred = model.predict(X_test)  # predict the test data
    # Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test,
                                             model.predict_proba(X_test)[:, 1])
    # Calculate Area under the curve to display on the plot
    auc_ = metrics.roc_auc_score(y_test, model.predict_proba(X_test)[::,1])
    # Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (Area = %0.2f)' % (m['label'], auc_))
# Custom settings for the plot
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# Display
plt.show() 
```


    
![png](output_66_0.png)
    


---

## Gradient Boosting Classifier


```python
#Fitting the model
gb_classifier = GradientBoostingClassifier(random_state=1)
gb_classifier.fit(X_train,y_train)

#Calculating different metrics
get_metrics_score(gb_classifier, X_train, X_test, y_train, y_test)

#Creating confusion matrix
make_confusion_matrix(gb_classifier,y_test)
```

    Accuracy on training set :  0.9392990679285114
    Accuracy on test set :  0.939848054767073
    Recall on training set :  0.11431837667905116
    Recall on test set :  0.11874583055370247
    Precision on training set :  0.5763688760806917
    Precision on test set :  0.5973154362416108
    F1 Score on training set :  0.1907941807774863
    F1 Score on test set :  0.19810795770728992
    


    
![png](output_69_1.png)
    


* The GBM model generalizes well and performs marginally better on the test data.

### Hyperparameter Tuning


```python
%%time

# Choose the type of classifier. 
gbc = GradientBoostingClassifier(init=AdaBoostClassifier(random_state=1),random_state=1)

# Grid of parameters to choose from
parameters = {
    "n_estimators": [100,150,200],
    "subsample":[0.8,0.9,1],
    "max_features":[0.7,0.8,0.9,1]
}

# Type of scoring used to compare parameter combinations
acc_scorer = metrics.make_scorer(metrics.f1_score)

#Calling RandomizedSearchCV
gbc_randomcv = RandomizedSearchCV(estimator=gbc,
                               param_distributions=parameters,
                               n_iter=50,
                               scoring=acc_scorer,
                               cv=5,
                               refit = True,
                               random_state=1)

#Fitting parameters in RandomizedSearchCV
gbc_tuned = gbc_randomcv.fit(X_train, y_train)

print("Best parameters are {} with CV score={}:".format(
    gbc_tuned.best_params_, gbc_tuned.best_score_))


gbc_tuned.best_estimator_
```

    Best parameters are {'subsample': 0.9, 'n_estimators': 150, 'max_features': 0.8} with CV score=0.179019153336666:
    Wall time: 17min 49s
    




    GradientBoostingClassifier(init=AdaBoostClassifier(random_state=1),
                               max_features=0.8, n_estimators=150, random_state=1,
                               subsample=0.9)




```python
gbc_tuned = gbc_tuned.best_estimator_
```


```python
#Calculating different metrics
get_metrics_score(gbc_tuned, X_train, X_test, y_train, y_test)

#Creating confusion matrix
make_confusion_matrix(gbc_tuned,y_test)
```

    Accuracy on training set :  0.939531638549475
    Accuracy on test set :  0.9396810819836366
    Recall on training set :  0.12060588739639898
    Recall on test set :  0.12274849899933289
    Precision on training set :  0.5820689655172414
    Precision on test set :  0.5859872611464968
    F1 Score on training set :  0.19981060606060608
    F1 Score on test set :  0.2029784886927744
    


    
![png](output_74_1.png)
    


---

### GBM AUC ROC curve


```python
plt.figure(figsize=(12, 12))

# Add the models to the list that you want to view on the ROC plot
models = [{
    'label': 'Gradient Boosting',
    'model': gb_classifier
}, {
    'label': 'Tuned Gradient Boosting',
    'model': gbc_tuned
}]

# Below for loop iterates through your models list
for m in models:
    model = m['model']  # select the model
    model.fit(X_train, y_train)  # train the model
    y_pred = model.predict(X_test)  # predict the test data
    # Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test,
                                             model.predict_proba(X_test)[:, 1])
    # Calculate Area under the curve to display on the plot
    auc_ = metrics.roc_auc_score(y_test, model.predict_proba(X_test)[::,1])
    # Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (Area = %0.2f)' % (m['label'], auc_))
# Custom settings for the plot
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# Display
plt.show() 
```


    
![png](output_77_0.png)
    


---

## XGBoost Classifier


```python
#Fitting the model
xgb_classifier = XGBClassifier(random_state=1,eval_metric= 'logloss')
xgb_classifier.fit(X_train,y_train)

#Calculating different metrics
get_metrics_score(xgb_classifier, X_train, X_test, y_train, y_test)

#Creating confusion matrix
make_confusion_matrix(xgb_classifier,y_test)
```

    Accuracy on training set :  0.9525377032756678
    Accuracy on test set :  0.9350475872432793
    Recall on training set :  0.2797942269219777
    Recall on test set :  0.12274849899933289
    Precision on training set :  0.8803956834532374
    Precision on test set :  0.4329411764705882
    F1 Score on training set :  0.42463673823465625
    F1 Score on test set :  0.19126819126819128
    


    
![png](output_80_1.png)
    


* The XGBoost model performs worst on the test data.

### Hyperparameter Tuning


```python
%%time

# Choose the type of classifier.
xgb = XGBClassifier(random_state=1, eval_metric='logloss')

# Grid of parameters to choose from
parameters = {
    "n_estimators": np.arange(10, 100, 20),
    "scale_pos_weight": [0, 1, 2, 5],
    "subsample": [0.5, 0.7, 0.9],
    "learning_rate": [0.01, 0.05, 0.1]
}

# Type of scoring used to compare parameter combinations
acc_scorer = metrics.make_scorer(metrics.f1_score)

# # Run the grid search
# grid_obj = GridSearchCV(xgb_tuned, parameters, scoring=acc_scorer, cv=5)
# grid_obj = grid_obj.fit(X_train, y_train)

# # Set the clf to the best combination of parameters
# xgb_tuned = grid_obj.best_estimator_

# # Fit the best algorithm to the data.
# xgb_tuned.fit(X_train, y_train)

#Calling RandomizedSearchCV
xgb_randomcv = RandomizedSearchCV(estimator=xgb,
                               param_distributions=parameters,
                               n_iter=50,
                               scoring=acc_scorer,
                               refit=True,
                               cv=5,
                               random_state=1)

#Fitting parameters in RandomizedSearchCV
xgb_tuned = xgb_randomcv.fit(X_train, y_train)

print("Best parameters are {} with CV score={}:".format(
    xgb_tuned.best_params_, xgb_tuned.best_score_))
```

    Best parameters are {'subsample': 0.7, 'scale_pos_weight': 5, 'n_estimators': 90, 'learning_rate': 0.05} with CV score=0.37135784385418896:
    Wall time: 5min 8s
    


```python
#Calculating different metrics
get_metrics_score(xgb_tuned, X_train, X_test, y_train, y_test)

#Creating confusion matrix
make_confusion_matrix(xgb_tuned,y_test)
```

    Accuracy on training set :  0.4288726280540741
    Accuracy on test set :  0.3880510440835267
    Recall on training set :  0.4941411831951986
    Recall on test set :  0.44629753168779185
    Precision on training set :  0.37883435582822084
    Precision on test set :  0.34325295023088764
    F1 Score on training set :  0.4288726280540741
    F1 Score on test set :  0.3880510440835267
    


    
![png](output_84_1.png)
    


---

### XGB AUC ROC curve


```python
plt.figure(figsize=(12, 12))

# Add the models to the list that you want to view on the ROC plot
models = [{
    'label': 'XGBoost',
    'model': xgb_classifier
}, {
    'label': 'Tuned XGBoost',
    'model': xgb_tuned
}]

# Below for loop iterates through your models list
for m in models:
    model = m['model']  # select the model
    model.fit(X_train, y_train)  # train the model
    y_pred = model.predict(X_test)  # predict the test data
    # Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test,
                                             model.predict_proba(X_test)[:, 1])
    # Calculate Area under the curve to display on the plot
    auc_ = metrics.roc_auc_score(y_test, model.predict_proba(X_test)[::,1])
    # Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (Area = %0.2f)' % (m['label'], auc_))
# Custom settings for the plot
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# Display
plt.show() 
```


    
![png](output_87_0.png)
    


---

## Comparing all models

### AUC Curves


```python
plt.figure(figsize=(12, 12))

# Add the models to the list that you want to view on the ROC plot
models = [{
    'label': 'Logistic Regression',
    'model': lr
}, {
    'label': 'Tuned Logistic Regression',
    'model': LR_result
}, {
    'label': 'AdaBoost Classifier',
    'model': ab_classifier
}, {
    'label': 'Tuned AdaBoost Classifier',
    'model': abc_tuned
}, {
    'label': 'Gradient Boosting Classifier',
    'model': gb_classifier
}, {
    'label': 'Tuned Gradient Boosting Classifier',
    'model': gbc_tuned
}, {
    'label': 'XGBoost Classifier',
    'model': xgb_classifier
}, {
    'label': 'Tuned XGBoost Classifier',
    'model': xgb_tuned
}]

# Below for loop iterates through your models list
for m in models:
    model = m['model']  # select the model
    model.fit(X_train, y_train)  # train the model
    y_pred = model.predict(X_test)  # predict the test data
    # Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test,
                                             model.predict_proba(X_test)[:, 1])
    # Calculate Area under the curve to display on the plot
    auc_ = metrics.roc_auc_score(y_test, model.predict_proba(X_test)[::,1])
    # Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (Area = %0.2f)' % (m['label'], auc_))
# Custom settings for the plot
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# Display
plt.show() 
```


    
![png](output_91_0.png)
    


### Precision Recall Curves


```python
models = [{
    'label': 'Logistic Regression',
    'model': lr
}, {
    'label': 'Tuned Logistic Regression',
    'model': LR_result
}, {
    'label': 'AdaBoost Classifier',
    'model': ab_classifier
}, {
    'label': 'Tuned AdaBoost Classifier',
    'model': abc_tuned
}, {
    'label': 'Gradient Boosting Classifier',
    'model': gb_classifier
}, {
    'label': 'Tuned Gradient Boosting Classifier',
    'model': gbc_tuned
}, {
    'label': 'XGBoost Classifier',
    'model': xgb_classifier
}, {
    'label': 'Tuned XGBoost Classifier',
    'model': xgb_tuned
}]

from sklearn.metrics import auc


plot_PR_curve(models)
```

    Logistic Regression
    F1 score=0.219 AUC=0.329 Average Precision=0.329
    


    
![png](output_93_1.png)
    


    Tuned Logistic Regression
    F1 score=0.218 AUC=0.329 Average Precision=0.329
    


    
![png](output_93_3.png)
    


    AdaBoost Classifier
    F1 score=0.240 AUC=0.328 Average Precision=0.328
    


    
![png](output_93_5.png)
    


    Tuned AdaBoost Classifier
    F1 score=0.243 AUC=0.327 Average Precision=0.305
    


    
![png](output_93_7.png)
    


    Gradient Boosting Classifier
    F1 score=0.198 AUC=0.320 Average Precision=0.321
    


    
![png](output_93_9.png)
    


    Tuned Gradient Boosting Classifier
    F1 score=0.203 AUC=0.321 Average Precision=0.322
    


    
![png](output_93_11.png)
    


    XGBoost Classifier
    F1 score=0.191 AUC=0.270 Average Precision=0.271
    


    
![png](output_93_13.png)
    


    Tuned XGBoost Classifier
    F1 score=0.388 AUC=0.315 Average Precision=0.316
    


    
![png](output_93_15.png)
    


### Tabular 


```python
# defining list of models
models = [
    lr, LR_result, ab_classifier, abc_tuned, gb_classifier, gbc_tuned,
    xgb_classifier, xgb_tuned
]

#           , pruned_dtree_model, dtree_estimator,rf_estimator, rf_tuned, bagging_classifier,bagging_estimator_tuned,
#           ab_classifier, abc_tuned, gb_classifier, gbc_tuned, xgb_classifier,xgb_tuned, stacking_classifier]

# defining empty lists to add train and test results
acc_train = []
acc_test = []
recall_train = []
recall_test = []
precision_train = []
precision_test = []
f1_score_train = []
f1_score_test = []

# looping through all the models to get the metrics score - Accuracy, Recall Precision and F1-Score
for model in models:

    j = get_metrics_score(model, X_train, X_test, y_train, y_test, False)
    acc_train.append(j[0])
    acc_test.append(j[1])
    recall_train.append(j[2])
    recall_test.append(j[3])
    precision_train.append(j[4])
    precision_test.append(j[5])
    f1_score_train.append(j[6])
    f1_score_test.append(j[7])
```


```python
comparison_frame = pd.DataFrame({
    'Model': [
        'Logistic Regression', 'Tuned Logistic Regression',
        'AdaBoost Classifier', 'Tuned AdaBoost Classifier',
        'Gradient Boosting Classifier', 'Tuned Gradient Boosting Classifier',
        'XGBoost Classifier', 'Tuned XGBoost Classifier'
    ],
    'Train_Accuracy':
    acc_train,
    'Test_Accuracy':
    acc_test,
    'Train_Recall':
    recall_train,
    'Test_Recall':
    recall_test,
    'Train_Precision':
    precision_train,
    'Test_Precision':
    precision_test,
    'Train_F1_Score':
    f1_score_train,
    'Test_F1_Score':
    f1_score_test,
})

#Sorting models in decreasing order of test recall
comparison_frame.sort_values(by='Test_F1_Score', ascending=False)
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
      <th>Model</th>
      <th>Train_Accuracy</th>
      <th>Test_Accuracy</th>
      <th>Train_Recall</th>
      <th>Test_Recall</th>
      <th>Train_Precision</th>
      <th>Test_Precision</th>
      <th>Train_F1_Score</th>
      <th>Test_F1_Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>Tuned XGBoost Classifier</td>
      <td>0.428873</td>
      <td>0.388051</td>
      <td>0.494141</td>
      <td>0.446298</td>
      <td>0.378834</td>
      <td>0.343253</td>
      <td>0.428873</td>
      <td>0.388051</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tuned AdaBoost Classifier</td>
      <td>0.936723</td>
      <td>0.938512</td>
      <td>0.157188</td>
      <td>0.158105</td>
      <td>0.483304</td>
      <td>0.529018</td>
      <td>0.237222</td>
      <td>0.243451</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AdaBoost Classifier</td>
      <td>0.938154</td>
      <td>0.939222</td>
      <td>0.143755</td>
      <td>0.153436</td>
      <td>0.521784</td>
      <td>0.551559</td>
      <td>0.225409</td>
      <td>0.240084</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>0.938190</td>
      <td>0.939806</td>
      <td>0.126608</td>
      <td>0.134757</td>
      <td>0.526128</td>
      <td>0.582133</td>
      <td>0.204100</td>
      <td>0.218852</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tuned Logistic Regression</td>
      <td>0.938154</td>
      <td>0.939681</td>
      <td>0.127751</td>
      <td>0.134757</td>
      <td>0.524648</td>
      <td>0.577143</td>
      <td>0.205470</td>
      <td>0.218496</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tuned Gradient Boosting Classifier</td>
      <td>0.939532</td>
      <td>0.939681</td>
      <td>0.120606</td>
      <td>0.122748</td>
      <td>0.582069</td>
      <td>0.585987</td>
      <td>0.199811</td>
      <td>0.202978</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Gradient Boosting Classifier</td>
      <td>0.939299</td>
      <td>0.939848</td>
      <td>0.114318</td>
      <td>0.118746</td>
      <td>0.576369</td>
      <td>0.597315</td>
      <td>0.190794</td>
      <td>0.198108</td>
    </tr>
    <tr>
      <th>6</th>
      <td>XGBoost Classifier</td>
      <td>0.952538</td>
      <td>0.935048</td>
      <td>0.279794</td>
      <td>0.122748</td>
      <td>0.880396</td>
      <td>0.432941</td>
      <td>0.424637</td>
      <td>0.191268</td>
    </tr>
  </tbody>
</table>
</div>



---


```python
models = [{
    'label': 'Logistic Regression',
    'model': lr
}, {
    'label': 'Tuned Logistic Regression',
    'model': LR_result
}, {
    'label': 'AdaBoost Classifier',
    'model': ab_classifier
}, {
    'label': 'Tuned AdaBoost Classifier',
    'model': abc_tuned
}, {
    'label': 'Gradient Boosting Classifier',
    'model': gb_classifier
}, {
    'label': 'Tuned Gradient Boosting Classifier',
    'model': gbc_tuned
}, {
    'label': 'XGBoost Classifier',
    'model': xgb_classifier
}, {
    'label': 'Tuned XGBoost Classifier',
    'model': xgb_tuned
}]

# defining empty lists to add test results
test_precision = []
test_recall = []
test_f1_score = []
test_AUC = []

# looping through all the models to get the metrics scores
for m in models:
    model = m['model']
    model.fit(X_train, y_train)
    # predict probabilities
    probs = model.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    probs = probs[:, 1]
    # predict class values
    y_hat = model.predict(X_test)
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    # Calculate F1 score
    f1 = f1_score(y_test, y_hat)
    # Calculate precision-recall AUC
    the_auc = auc(recall, precision)
    # appending test value results to empty list
    test_precision.append(precision)
    test_recall.append(recall)
    test_f1_score.append(f1)
    test_AUC.append(the_auc)
```


```python
comp_frame = pd.DataFrame({
    'Model': [
        'Logistic Regression', 'Tuned Logistic Regression',
        'AdaBoost Classifier', 'Tuned AdaBoost Classifier',
        'Gradient Boosting Classifier', 'Tuned Gradient Boosting Classifier',
        'XGBoost Classifier', 'Tuned XGBoost Classifier'
    ],

    'Test_F1_Score':
    test_f1_score,
    'Test_AUC_Score':
    test_AUC,
})

#Sorting models in decreasing order of test recall
comp_frame.sort_values(by='Test_AUC_Score', ascending=False)
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
      <th>Model</th>
      <th>Test_F1_Score</th>
      <th>Test_AUC_Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>0.218852</td>
      <td>0.328717</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tuned Logistic Regression</td>
      <td>0.218496</td>
      <td>0.328658</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AdaBoost Classifier</td>
      <td>0.240084</td>
      <td>0.327749</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tuned AdaBoost Classifier</td>
      <td>0.243451</td>
      <td>0.327109</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tuned Gradient Boosting Classifier</td>
      <td>0.202978</td>
      <td>0.321295</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Gradient Boosting Classifier</td>
      <td>0.198108</td>
      <td>0.320212</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Tuned XGBoost Classifier</td>
      <td>0.388051</td>
      <td>0.315049</td>
    </tr>
    <tr>
      <th>6</th>
      <td>XGBoost Classifier</td>
      <td>0.191268</td>
      <td>0.270132</td>
    </tr>
  </tbody>
</table>
</div>



---

## Model Selection - Logistic Regression



From the models, above we have seen that the Logistic Regression model produced the highest AUC scores

Let's find the optimal probability cut off and adjust the model to get the best confusion matrix


```python
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

lr.fit(X_train, y_train)
# predict probabilities
yhat = lr.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = yhat[:, 1]
# define thresholds
thresholds = np.arange(0, 1, 0.001)
# evaluate each threshold
scores = [f1_score(y_test, to_labels(probs, t)) for t in thresholds]
# get best threshold
ix = np.argmax(scores)
Opt_threshold = thresholds[ix]
print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
```

    Threshold=0.173, F-Score=0.38645
    

Since the data was heavily imbalanced, the Threshold value was found to be **0.173**

Lets modify the probability cutoff to match the optimal threshold


```python
# the probability of being y=1
train_pred=lr.predict_proba(X_train)[:,1]
test_pred=lr.predict_proba(X_test)[:,1]

opt_train_pred=[1 if i > Opt_threshold else 0 for i in train_pred]
opt_test_pred=[1 if i > Opt_threshold else 0 for i in test_pred]
```

---

**Define simple confusion matric visualization**


```python
def conf_matrix(y_actual, y_predict, labels=[1, 0]):
    '''
    y_predict: prediction of class
    y_actual : ground truth  
    '''
    cm = confusion_matrix(y_actual, y_predict, labels=[0, 1])
    df_cm = pd.DataFrame(
        cm,
        index=[i for i in ["Actual - No", "Actual - Yes"]],
        columns=[i for i in ['Predicted - No', 'Predicted - Yes']])

    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = [
        "{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)
    ]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(2, 2)
    plt.figure(figsize=(5, 5))
    sns.heatmap(df_cm, annot=labels, fmt='')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
```


```python
print('Accuracy on train data:', accuracy_score(y_train, opt_train_pred))
print('Recall on train data:', recall_score(y_train, opt_train_pred))
print('Precision on train data:', precision_score(y_train, opt_train_pred))
print('f1 score on train data:', f1_score(y_train, opt_train_pred))

# let us make confusion matrix on train set
conf_matrix(y_train,opt_train_pred)
```

    Accuracy on train data: 0.9132690484283593
    Recall on train data: 0.4161188911117462
    Precision on train data: 0.34170382539310024
    f1 score on train data: 0.3752577319587629
    


    
![png](output_111_1.png)
    



```python
print('Accuracy on test data:', accuracy_score(y_test, opt_test_pred))
print('Recall on test data:', recall_score(y_test, opt_test_pred))
print('Precision on test data:', precision_score(y_test, opt_test_pred))
print('f1 score on test data:', f1_score(y_test, opt_test_pred))

# let us make confusion matrix on test set
conf_matrix(y_test, opt_test_pred)
```

    Accuracy on test data: 0.9160961763232593
    Recall on test data: 0.42228152101400934
    Precision on test data: 0.3562183455261677
    f1 score on test data: 0.38644688644688646
    


    
![png](output_112_1.png)
    


* The F1 Score from the confusion matrix matches the F1 score when calculating the optimal threshold.

---

## Feature importances


```python
## Coefficients of all the attributes with Name and Values as separate column
Coeff_Col_df = pd.DataFrame()
Coeff_Col_df['Col'] = X_train.columns
Coeff_Col_df['Coeff'] = np.round(abs(lr.coef_[0]), 2)
print("Shape of Coeff DF-->", Coeff_Col_df.shape)
print('')
print("Coefficients of all the attributes with Name and Values:")
Coeff_Col_df.sort_values(by='Coeff', ascending=False)
```

    Shape of Coeff DF--> (44, 2)
    
    Coefficients of all the attributes with Name and Values:
    




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
      <th>Col</th>
      <th>Coeff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>Perc_premium_paid_in_cash_0.9 - 1</td>
      <td>1.62</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Late_premium_payment_6-12_months_Paid_on_time</td>
      <td>1.49</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Perc_premium_paid_in_cash_0.8 - 0.9</td>
      <td>1.46</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Perc_premium_paid_in_cash_0.7 - 0.8</td>
      <td>1.37</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Perc_premium_paid_in_cash_0.6 - 0.7</td>
      <td>1.18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Perc_premium_paid_in_cash_0.5 - 0.6</td>
      <td>1.04</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Late_premium_payment_&gt;12_months_Paid_on_time</td>
      <td>1.04</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Late_premium_payment_3-6_months_Paid_on_time</td>
      <td>0.90</td>
    </tr>
    <tr>
      <th>22</th>
      <td>No_of_premiums_paid_&gt;20</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Perc_premium_paid_in_cash_0.4 - 0.5</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>21</th>
      <td>No_of_premiums_paid_15 - 20</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Age_in_years_&gt;70</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Age_in_years_60 - 70</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Income_'000_&gt;300</td>
      <td>0.61</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Perc_premium_paid_in_cash_0.3 - 0.4</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Income_'000_260 - 300</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Income_'000_220 - 260</td>
      <td>0.47</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Income_'000_180 - 220</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Perc_premium_paid_in_cash_0.2 - 0.3</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Age_in_years_50 - 60</td>
      <td>0.30</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Income_'000_140 - 180</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>20</th>
      <td>No_of_premiums_paid_10 - 15</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Sourcing_channel_D</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Age_in_years_30 - 40</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Income_'000_100 - 140</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Age_in_years_40 - 50</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Sourcing_channel_C</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Sourcing_channel_E</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Perc_premium_paid_in_cash_0.1 - 0.2</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Premium_payment_&gt;20000</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>19</th>
      <td>No_of_premiums_paid_5 - 10</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>15</th>
      <td>No_of_dependents_One</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Marital_Status_Unmarried</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Sourcing_channel_B</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Premium_payment_5000 - 10000</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>17</th>
      <td>No_of_dependents_Two</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Income_'000_60 - 100</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Vehicles_Owned_Two</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Customer_demographic_Urban</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Premium_payment_10000 - 15000</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Accomodation_Rented</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Premium_payment_15000 - 20000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Vehicles_Owned_Three</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>No_of_dependents_Three</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Top 15 dummy variables which influence Loan selection
Coeff_Col_df.sort_values(by='Coeff', ascending=False).head(15)
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
      <th>Col</th>
      <th>Coeff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>Perc_premium_paid_in_cash_0.9 - 1</td>
      <td>1.62</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Late_premium_payment_6-12_months_Paid_on_time</td>
      <td>1.49</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Perc_premium_paid_in_cash_0.8 - 0.9</td>
      <td>1.46</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Perc_premium_paid_in_cash_0.7 - 0.8</td>
      <td>1.37</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Perc_premium_paid_in_cash_0.6 - 0.7</td>
      <td>1.18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Perc_premium_paid_in_cash_0.5 - 0.6</td>
      <td>1.04</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Late_premium_payment_&gt;12_months_Paid_on_time</td>
      <td>1.04</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Late_premium_payment_3-6_months_Paid_on_time</td>
      <td>0.90</td>
    </tr>
    <tr>
      <th>22</th>
      <td>No_of_premiums_paid_&gt;20</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Perc_premium_paid_in_cash_0.4 - 0.5</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>21</th>
      <td>No_of_premiums_paid_15 - 20</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Age_in_years_&gt;70</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Age_in_years_60 - 70</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Income_'000_&gt;300</td>
      <td>0.61</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Perc_premium_paid_in_cash_0.3 - 0.4</td>
      <td>0.59</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Last 10 dummy variables which influence Loan selection
Coeff_Col_df.sort_values(by='Coeff', ascending=False).tail(10)
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
      <th>Col</th>
      <th>Coeff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>Premium_payment_5000 - 10000</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>17</th>
      <td>No_of_dependents_Two</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Income_'000_60 - 100</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Vehicles_Owned_Two</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Customer_demographic_Urban</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Premium_payment_10000 - 15000</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Accomodation_Rented</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Premium_payment_15000 - 20000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Vehicles_Owned_Three</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>No_of_dependents_Three</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



The following variables were found the greatest predictors of policy holders defaulting:
* **Perc_premium_paid_in_cash**
    * The greater the percentage of premium paid in cash, the greater the defaulting potential
* **Late_premium_payment_3-6_months_Paid_on_time**
* **Late_premium_payment_6-12_months_Paid_on_time**
* **Late_premium_payment_>12_months_Paid_on_time**
* **No_of_premiums_paid**
    * The greater the number of premium paid , the greater the defaulting potential
* **Age**
    * The oldest policy holders had the greatest potential to default
* **Income**
    * The highest income earners had the greatest potential to default 


---


---

## SMOTE

Since the data was heavily imbalanced. Let's run SMOTE and observe if it would produce a better model

### Oversampling train data using SMOTE


```python
pip install imblearn
```


```python
from imblearn.over_sampling import SMOTE
```


```python
print("Before UpSampling, counts of label 'Default': {}".format(sum(y_train == 1)))
print("Before UpSampling, counts of label 'Not_Default': {} \n".format(sum(y_train == 0)))

# Synthetic Minority Over Sampling Technique
sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)
X_train_over, y_train_over = sm.fit_resample(X_train, y_train)


print("After UpSampling, counts of label 'Default': {}".format(sum(y_train_over == 1)))
print("After UpSampling, counts of label 'Not_Default': {} \n".format(sum(y_train_over == 0)))


print('After UpSampling, the shape of train_X: {}'.format(X_train_over.shape))
print('After UpSampling, the shape of train_y: {} \n'.format(y_train_over.shape))
```

    Before UpSampling, counts of label 'Default': 3499
    Before UpSampling, counts of label 'Not_Default': 52398 
    
    After UpSampling, counts of label 'Default': 52398
    After UpSampling, counts of label 'Not_Default': 52398 
    
    After UpSampling, the shape of train_X: (104796, 44)
    After UpSampling, the shape of train_y: (104796,) 
    
    

---

#### Logistic Regression on oversampled data


```python
log_reg_over = LogisticRegression(random_state = 1)

# Training the basic logistic regression model with training set 
log_reg_over.fit(X_train_over,y_train_over)
```




    LogisticRegression(random_state=1)




```python
# Calculating different metrics
get_metrics_score(log_reg_over, X_train_over, X_test, y_train_over, y_test)

# creating confusion matrix
make_confusion_matrix(log_reg_over, y_test)
```

    Accuracy on training set :  0.933938318256422
    Accuracy on test set :  0.9307062948739355
    Recall on training set :  0.8776861712279095
    Recall on test set :  0.038692461641094064
    Precision on training set :  0.9889469496591617
    Precision on test set :  0.20938628158844766
    F1 Score on training set :  0.9300007077784855
    F1 Score on test set :  0.06531531531531533
    


    
![png](output_131_1.png)
    


* With the application of SMOTE oversampling, the training performance is significantly better but the testing performance is abismal.

Let's explore Regularization and Undersampling to see its performance will be better or worse.

---

### Undersampling train data using SMOTE


```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=1)
X_train_un, y_train_un = rus.fit_resample(X_train, y_train)
```


```python
print("Before Under Sampling, counts of label 'Default': {}".format(sum(y_train == 1)))
print("Before Under Sampling, counts of label 'Not_Default': {} \n".format(sum(y_train == 0)))

print("After Under Sampling, counts of label 'Default': {}".format(sum(y_train_un == 1)))
print("After Under Sampling, counts of label 'Not_Default': {} \n".format(sum(y_train_un == 0)))

print('After Under Sampling, the shape of train_X: {}'.format(X_train_un.shape))
print('After Under Sampling, the shape of train_y: {} \n'.format(y_train_un.shape))
```

    Before Under Sampling, counts of label 'Default': 3499
    Before Under Sampling, counts of label 'Not_Default': 52398 
    
    After Under Sampling, counts of label 'Default': 3499
    After Under Sampling, counts of label 'Not_Default': 3499 
    
    After Under Sampling, the shape of train_X: (6998, 44)
    After Under Sampling, the shape of train_y: (6998,) 
    
    

---

#### Logistic Regression on undersampled data


```python
log_reg_under = LogisticRegression(random_state=1)
log_reg_under.fit(X_train_un, y_train_un)
```




    LogisticRegression(random_state=1)




```python
#Calculating different metrics
get_metrics_score(log_reg_under, X_train_un, X_test, y_train_un, y_test)

# creating confusion matrix
make_confusion_matrix(log_reg_under, y_test)
```

    Accuracy on training set :  0.7655044298370963
    Accuracy on test set :  0.7735849056603774
    Recall on training set :  0.7356387539296942
    Recall on test set :  0.7184789859906604
    Precision on training set :  0.782370820668693
    Precision on test set :  0.17716729725283764
    F1 Score on training set :  0.758285461776403
    F1 Score on test set :  0.2842438638163104
    


    
![png](output_140_1.png)
    


* The Undersampled model generalizes on Accuracy and Recall performs better on the training data.

Neither SMOTE Oversampling nor Undersampling performed better than the LR model which was adjusted for the optimal threshold.

---

---

---

---
