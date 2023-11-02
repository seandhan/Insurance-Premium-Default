<p align="center">
  <img src="https://forthebadge.com/images/badges/made-with-python.svg" />&nbsp;&nbsp;&nbsp;
  <img src="https://forthebadge.com/images/badges/made-with-markdown.svg" />&nbsp;&nbsp;&nbsp;
  <img src="https://forthebadge.com/images/badges/powered-by-oxygen.svg" />&nbsp;&nbsp;
</p>




<h1 align="center">Insurance Premium Default Propensity</h1>

<p align="center">The primary objective of this project is to develop a model that can effectively predict the likelihood of a customer defaulting on insurance premium payments.</p>

---

## 📝 Table of Contents

- [🤓 Description](#description)
- [💻 Dataset Overview](#dataset-overview)
- [🛠️ Feature Engineering](#feature-engineering)
- [📊 Exploratory Data Analysis](#exploratory-data-analysis)
- [🏗️ Model Building](#model-building)
- [✨ Recommendations](#recommendations)
- [📗 Notebooks](#notebooks)
- [📧 Contact Information](#contact-information)

## 🤓 Description <a name = "description"></a>

As the premium paid by customers is the major revenue source for insurance companies, defaulting on these payments results in significant revenue losses. Hence, insurance companies would like to know upfront which types of customers would default on premium payments.

The objectives of this project are to:
1. Build a model that can predict the likelihood of a customer defaulting on premium payments.
2. Identify the factors that drive higher default rates.
3. Propose a strategy for reducing default rates by using the model and other insights from the analysis.



---

## 💻 Dataset Overview <a name = "dataset-overview"></a>

The dataset source file can found through the following link:
### Click to view 👇:

[![Data_link]()]()

The used cars database contains 17 variables. The data dictionary below explains each variable:

<details>
<summary>Data Dictionary</summary>
<br>

1. **id**: Unique customer ID
2. **perc_premium_paid_by_cash_credit**: What % of the premium was paid by cash payments?
3. **age_in_days**: age of the customer in days 
4. **Income**: Income of the customer 
5. **Marital Status**: Married/Unmarried, Married (1), unmarried (0)
6. **Veh_owned**: Number of vehicles owned (1-3)
7. **Count_3-6_months_late**: Number of times premium was paid 3-6 months late 
8. **Count_6-12_months_late**: Number of times premium was paid 6-12 months late 
9. **Count_more_than_12_months_late**: Number of times premium was paid more than 12 months late 
10. **Risk_score**: Risk score of customer (similar to credit score)
11.	**No_of_dep**: Number of dependents in the family of the customer (1-4) 
12.	**Accommodation**: Owned (1), Rented (0)
13.	**no_of_premiums_paid**: Number of premiums paid till date 
14.	**sourcing_channel**: Channel through which customer was sourced 
15.	**residence_area_type**: Residence type of the customer
16.	**premium** : Total premium amount paid till now
17.	**default**: (Y variable) - 0 indicates that customer has defaulted the premium and 1 indicates that customer has not defaulted the premium

</details>

### Click to view 👇:

[![Data Exploration](https://github.com/seandhan/image_database/blob/main/Solution-Dataset%20Exploration-.svg)](https://github.com/seandhan/Used-Car-Price-Prediction/blob/main/Dataset%20Exploration/ReadMe.md)

There was a significant amount of data pre-processing required prior data visualization. These steps can be seen in the following section.

----

## 🛠️ Feature Engineering <a name = "feature-engineering"></a>

The step by step data cleaning and wrangling can be observed in this section

### Click to view 👇:

[![Feature Engineering](https://github.com/seandhan/image_database/blob/main/Solution-Feature%20Engineering-.svg)](https://github.com/seandhan/Used-Car-Price-Prediction/blob/main/Feature%20Engineering/ReadME.MD)


----

## 📊 Exploratory Data Analysis <a name = "exploratory-data-analysis"></a>

The Univariate and Bivariate analysis can be seen here.

### Click to view 👇:

[![Exploratory Data Analysis](https://github.com/seandhan/image_database/blob/main/Solution-Exploratory%20Data%20Analysis-.svg)](https://github.com/seandhan/Used-Car-Price-Prediction/blob/main/Exploratory%20Data%20Analysis/ReadME.MD)


----

## 🏗️ Model Building <a name = "model-building"></a>

The data model preparation and linear regression steps can be seen here.

### Click to view 👇:

[![Model Building](https://github.com/seandhan/image_database/blob/main/Solution-Model%20Building-.svg)](https://github.com/seandhan/Used-Car-Price-Prediction/blob/main/Model%20Building/README.MD)


----


## ✨ Recommendations <a name = "recommendations"></a>

Based on the analysis the following recommendations can be made to further ensure than insurance companies maintain receiving their insurance premium payments:
1. Improve the accessibility to non-cash payment services. This can be achieved by placing special emphasis on payment options in all marketing campaigns.
2. Include additional no-claim discounts and services to customers who pay their premiums via non-cash methods.
3. Notify policy holders via phone and mail with policy expiration dates and renewal from four months prior and continuously send electronic reminders every month until due date. On date of policy expiration, insurance agents should contact policy holders to further remind them.
4. The Insurance companies should amalgamate insurance packages by policy holder to reduce the number of policies to each individual client. Discounts should also be given to clients who hold many policies, to reduce the hesitance of clients from completing their premium payments.
5. Since older customers tend to default more than younger ones, if applicable, then their registered next of kin should be contacted and marketed about the policies available.
6. Since the highest income earners churn the fastest, then insurance companies should contact those clients on a regular basis possibly offering additional trial services to influence them to remain loyal.


----

## 📗 Notebooks <a name = "notebooks"></a>

The Notebook for the "Data Exploration" can be accessed below:

### Click to view 👇:

[![DataExp Notebook](https://github.com/seandhan/image_database/blob/main/Notebook-Dataset%20Exploration-.svg)](https://github.com/seandhan/Used-Car-Price-Prediction/blob/main/Notebooks/Data%20exploration.ipynb)

The Notebook for the "Feature Engineering" can be accessed below:

### Click to view 👇:

[![Feature Engineering Notebook](https://github.com/seandhan/image_database/blob/main/Notebook-Feature%20engineering-.svg)](https://github.com/seandhan/Used-Car-Price-Prediction/blob/main/Notebooks/Feature_engineering.ipynb)

The Notebook for the "Exploratory Data Analysis" can be accessed below:

### Click to view 👇:

[![EDA Notebook](https://github.com/seandhan/image_database/blob/main/Notebook-Exploratory%20Data%20analysis-.svg)](https://github.com/seandhan/Used-Car-Price-Prediction/blob/main/Notebooks/Exploratory%20Data%20Analysis.ipynb)

The Notebook for the "Model Building" can be accessed below:

### Click to view 👇:

[![Model Building Notebook](https://github.com/seandhan/image_database/blob/main/Notebook-Model%20Building-.svg)](https://github.com/seandhan/Used-Car-Price-Prediction/blob/main/Notebooks/Model_Building.ipynb)

----



## 📧 Contact Information <a name = "contact-information"></a>

- Email: [sean_dhanasar@msn.com](mailto:sean_dhanasar@msn.com)
- LinkedIn: [Sean Dhanasar](https://www.linkedin.com/in/sdhanasar)

## Analysis
Click [HERE](https://github.com/seandhan/Insurance-Premium-Default/blob/main/InsurancePremiumDefault.ipynb) to view the entire analysis.
