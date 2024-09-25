
# DataScienceAssignents

This repository consists of Data Science assignment questions and solutions for learning various methods in data science to excel as a beginner

## 01 - Basic Statistics
Descriptive Analytics and Data Preprocessing on Sales & Discounts Dataset
### Introduction
 • To perform descriptive analytics, visualize data distributions, and preprocess the dataset for further analysis.
### Descriptive Analytics for Numerical Columns
• Objective: To compute and analyze basic statistical measures for numerical columns in the dataset.

• Steps:
    
        ◦ Load the dataset into a data analysis tool or programming environment (e.g., Python with pandas library).
        ◦ Identify numerical columns in the dataset.
        ◦ Calculate the mean, median, mode, and standard deviation for these columns.
        ◦ Provide a brief interpretation of these statistics.
### Data Visualization
 • Objective: To visualize the distribution and relationship of numerical and categorical variables in the dataset.
    
    • Histograms:
        ◦ Plot histograms for each numerical column.
        ◦ Analyze the distribution (e.g., skewness, presence of outliers) and provide inferences.
    • Boxplots:
        ◦ Create boxplots for numerical variables to identify outliers and the interquartile range.
        ◦ Discuss any findings, such as extreme values or unusual distributions.
    • Bar Chart Analysis for Categorical Column:
        ◦ Identify categorical columns in the dataset.
        ◦ Create bar charts to visualize the frequency or count of each category.
        ◦ Analyze the distribution of categories and provide insights.
### Standardization of Numerical Variables
• Objective: To scale numerical variables for uniformity, improving the dataset’s suitability for analytical models.
    
• Steps:
       
        ◦ Explain the concept of standardization (z-score normalization).
        ◦ Standardize the numerical columns using the formula: z=x-mu/sigma 
        ◦ ​Show before and after comparisons of the data distributions.
### Conversion of Categorical Data into Dummy Variables
• Objective: To transform categorical variables into a format that can be provided to ML algorithms.

• Steps:
       
        ◦ Discuss the need for converting categorical data into dummy variables (one-hot encoding).
        ◦ Apply one-hot encoding to the categorical columns, creating binary (0 or 1) columns for each category.
        ◦ Display a portion of the transformed dataset.
### Conclusion
• Summarize the key findings from the descriptive analytics and data visualizations.
    
• Reflect on the importance of data preprocessing steps like standardization and one-hot encoding in data analysis and machine learning.

#### Dataset
[sales_data_with_discounts](https://github.com/aravindajai/DataScienceAssignments/blob/Datasets/sales_data_with_discounts.csv)
## 02 - Basic Statistics

Estimation And Confidence Intervals

### Background
In quality control processes, especially when dealing with high-value items, destructive sampling is a necessary but costly method to ensure product quality. The test to determine whether an item meets the quality standards destroys the item, leading to the requirement of small sample sizes due to cost constraints.
### Scenario
A manufacturer of print-heads for personal computers is interested in estimating the mean durability of their print-heads in terms of the number of characters printed before failure. To assess this, the manufacturer conducts a study on a small sample of print-heads due to the destructive nature of the testing process.
### Data
A total of 15 print-heads were randomly selected and tested until failure. The durability of each print-head (in millions of characters) was recorded as follows:
1.13, 1.55, 1.43, 0.92, 1.25, 1.36, 1.32, 0.85, 1.07, 1.48, 1.20, 1.33, 1.18, 1.22, 1.29
### Assignment Tasks
    • a. Build 99% Confidence Interval Using Sample Standard Deviation. Assuming the sample is representative of the population,
         construct a 99% confidence interval for the mean number of characters printed before the print-head fails using the sample
         standard deviation. Explain the steps you take and the rationale behind using the t-distribution for this task.
    • b. Build 99% Confidence Interval Using Known Population Standard Deviation. If it were known that the population standard deviation
         is 0.2 million characters, construct a 99% confidence interval for the mean number of characters printed before failure.


### Task A: 99% Confidence Interval Using Sample Standard Deviation
    1. Given Data:
        ◦ Durability data (in millions of characters): [1.13, 1.55, 1.43, 0.92, 1.25, 1.36, 1.32, 0.85, 1.07, 1.48, 1.20, 1.33, 1.18, 1.22, 1.29]
        ◦ Sample size (nnn) = 15
        ◦ Confidence level = 99%
    2. Sample Mean Calculation:
     Sample Mean(xˉ)=∑xin=1.13+1.55+⋯+1.2915=1.2467 million        
     characters\text{Sample Mean} (\bar{x}) = \frac{\sum x_i}{n} = 
     \frac{1.13 + 1.55 + \dots + 1.29}{15} = 1.2467 \text{ million 
     characters}Sample Mean(xˉ)=n∑xi​​=151.13+1.55+⋯+1.29​=1.2467 
     million characters
    3. Sample Standard Deviation Calculation:
     s=∑(xi−xˉ)2n−1=(1.13−1.2467)2+⋯+(1.29−1.2467)214≈0.2026 
     million characterss = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n-1}
     } = \sqrt{\frac{(1.13 - 1.2467)^2 + \dots + (1.29 - 1.2467)^2}
     {14}} \approx 0.2026 \text{ million characters}s=n−1∑(xi​−xˉ)
     2​​=14(1.13−1.2467)2+⋯+(1.29−1.2467)2​​≈0.2026 million 
     characters
    4. t-Critical Value Calculation:
        ◦ Degrees of freedom (dfdfdf) = n−1=14n - 1 = 14n−1=14
        ◦ For a 99% confidence level and df=14df = 14df=14, the 
        t-critical value t0.005t_{0.005}t0.005​ (two-tailed) is 
        approximately 2.977.
    5. Margin of Error Calculation:
     Margin of Error=t×sn=2.977×0.202615≈0.1556 million 
     characters\text{Margin of Error} = t \times \frac{s}{\sqrt{n}
     } = 2.977 \times \frac{0.2026}{\sqrt{15}} \approx 0.1556 \text
     { million characters}Margin of Error=t×n​s​=2.977×15​0.
     2026​≈0.1556 million characters
    6. Confidence Interval Calculation:
     Confidence Interval=xˉ±Margin of Error=1.2467±0.1556\text
     {Confidence Interval} = \bar{x} \pm \text{Margin of Error} = 
     1.2467 \pm 0.1556Confidence Interval=xˉ±Margin of Error=1.
     2467±0.1556
        ◦ Lower bound: 1.2467−0.1556=1.09111.2467 - 0.1556 = 1.
        09111.2467−0.1556=1.0911
        ◦ Upper bound: 1.2467+0.1556=1.40231.2467 + 0.1556 = 1.
        40231.2467+0.1556=1.4023
### Result: 
The 99% confidence interval using the sample standard deviation is [1.0911, 1.4023] million characters.
### Task B: 99% Confidence Interval Using Known Population Standard Deviation
    1. Given Data:
        ◦ Population standard deviation (σ\sigmaσ) = 0.2 million 
        characters
        ◦ Confidence level = 99%
    2. z-Critical Value Calculation:
        ◦ For a 99% confidence level, the z-critical value z0.005z_
        {0.005}z0.005​ (two-tailed) is approximately 2.576.
    3. Margin of Error Calculation:
     Margin of Error=z×σn=2.576×0.215≈0.1331 million 
     characters\text{Margin of Error} = z \times \frac{\sigma}
     {\sqrt{n}} = 2.576 \times \frac{0.2}{\sqrt{15}} \approx 0.
     1331 \text{ million characters}Margin of Error=z×n​σ​=2.
     576×15​0.2​≈0.1331 million characters
    4. Confidence Interval Calculation:
     Confidence Interval=xˉ±Margin of Error=1.2467±0.1331\text
     {Confidence Interval} = \bar{x} \pm \text{Margin of Error} = 
     1.2467 \pm 0.1331Confidence Interval=xˉ±Margin of Error=1.
     2467±0.1331
        ◦ Lower bound: 1.2467−0.1331=1.11361.2467 - 0.1331 = 1.
        11361.2467−0.1331=1.1136
        ◦ Upper bound: 1.2467+0.1331=1.37981.2467 + 0.1331 = 1.
        37981.2467+0.1331=1.3798
### Result: 
The 99% confidence interval using the known population standard deviation is [1.1136, 1.3798] million characters.

## 03 - Basics Of Python

Coding Exercises
### Exercise 1: Prime Numbers
Write a Python program that checks whether a given number is prime or not. A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.
### Exercise 2: Product of Random Numbers
Develop a Python program that generates two random numbers and asks the user to enter the product of these numbers. The program should then check if the user's answer is correct and display an appropriate message.
### Exercise 3: Squares of Even/Odd Numbers
Create a Python script that prints the squares of all even or odd numbers within the range of 100 to 200. Choose either even or odd numbers and document your choice in the code.
### Exercise 4: Word counter
Write a program to count the number of words in a given text.
example:
input_text = "This is a sample text. This text will be used to demonstrate the word counter."
Expected output:
'This': 2 
'is': 1
'a': 1
'sample': 1
'text.': 1

### Exercise 5: Check for Palindrome
Write a Python function called is_palindrome that takes a string as input and returns True if the string is a palindrome, and False otherwise. A palindrome is a word, phrase, number, or other sequence of characters that reads the same forward and backward, ignoring spaces, punctuation, and capitalization.
Example:
Input: "racecar"
Expected Output: True
## 04 - Hypothesis Testing

### Background:
Bombay hospitality Ltd. operates a franchise model for producing exotic Norwegian dinners throughout New England. The operating cost for a franchise in a week (W) is given by the equation W = $1,000 + $5X, where X represents the number of units produced in a week. Recent feedback from restaurant owners suggests that this cost model may no longer be accurate, as their observed weekly operating costs are higher.
### Objective:
To investigate the restaurant owners' claim about the increase in weekly operating costs using hypothesis testing.
### Data Provided:
    • The theoretical weekly operating cost model: W = $1,000 + $5X
    • Sample of 25 restaurants with a mean weekly cost of Rs. 3,050
    • Number of units produced in a week (X) follows a normal distribution with a mean (μ) of 600 units and a standard deviation (σ) of 25 units
### Assignment Tasks:
1. State the Hypotheses statement:
2. Calculate the Test Statistic:
Use the following formula to calculate the test statistic (t):
where:

    • ˉxˉ = sample mean weekly cost (Rs. 3,050)
    • μ = theoretical mean weekly cost according to the cost model (W = $1,000 + $5X for X = 600 units)
    • σ = 5*25 units
    • n = sample size (25 restaurants)

3. Determine the Critical Value:
Using the alpha level of 5% (α = 0.05), determine the critical value from the standard normal (Z) distribution table.

4. Make a Decision:
Compare the test statistic with the critical value to decide whether to reject the null hypothesis.

5. Conclusion:
Based on the decision in step 4, conclude whether there is strong evidence to support the restaurant owners' claim that the weekly operating costs are higher than the model suggests.
## 04 - Chi Square Testing

### Background:
Mizzare Corporation has collected data on customer satisfaction levels for two types of smart home devices: Smart Thermostats and Smart Lights. They want to determine if there's a significant association between the type of device purchased and the customer's satisfaction level.
### Data Provided:
The data is summarized in a contingency table showing the counts of customers in each satisfaction level for both types of devices:

                    Satisfaction                   Smart Thermostat                    Smart Light                       Total    
     
                   Very Satisfied                        50                                 70                            120

                     Satisfied                           80                                 100                           180

                      Neutral                            60                                 90                            150

                    Unsatisfied                          30                                 50                            80                                

                  Very Unsatisfied                       20                                 50                            70

                       Total                             240                                360                           600
### Objective:
To use the Chi-Square test for independence to determine if there's a significant association between the type of smart home device purchased (Smart Thermostats vs. Smart Lights) and the customer satisfaction level.
### Assignment Tasks:
1. State the Hypotheses:
2. Compute the Chi-Square Statistic:
3. Determine the Critical Value: Using the significance level (alpha) of 0.05 and the degrees of freedom (which is the number of categories minus 1).
4. Make a Decision: Compare the Chi-Square statistic with the critical value to decide whether to reject the null hypothesis.

## 05 - EDA 1

### Objective:
The main goal of this assignment is to conduct a thorough exploratory analysis of the "cardiographic.csv" dataset to uncover insights, identify patterns, and understand the dataset's underlying structure. You will use statistical summaries, visualizations, and data manipulation techniques to explore the dataset comprehensively.
### Dataset:
    1. LB - Likely stands for "Baseline Fetal Heart Rate (FHR)" which represents the average fetal heart rate over a period.
    2. AC - Could represent "Accelerations" in the FHR. Accelerations are usually a sign of fetal well-being.
    3. FM - May indicate "Fetal Movements" detected by the monitor.
    4. UC - Likely denotes "Uterine Contractions", which can impact the FHR pattern.
    5. DL - Could stand for "Decelerations Late" with respect to uterine contractions, which can be a sign of fetal distress.
    6. DS - May represent "Decelerations Short" or decelerations of brief duration.
    7. DP - Could indicate "Decelerations Prolonged", or long-lasting decelerations.
    8. ASTV - Might refer to "Percentage of Time with Abnormal Short Term Variability" in the FHR.
    9. MSTV - Likely stands for "Mean Value of Short Term Variability" in the FHR.
    10. ALTV - Could represent "Percentage of Time with Abnormal Long Term Variability" in the FHR.
    11. MLTV - Might indicate "Mean Value of Long Term Variability" in the FHR.


### Tools and Libraries:
  • Python or R programming language
  
  • Data manipulation libraries 
  
  • Data visualization libraries (Matplotlib and Seaborn in Python)
  
  • Jupyter Notebook for documenting your analysis


### Tasks:
  1. Data Cleaning and Preparation:
        
        ◦ Load the dataset into a DataFrame or equivalent data structure.
        
        ◦ Handle missing values appropriately (e.g., imputation, deletion).
        
        ◦ Identify and correct any inconsistencies in data types (e.g., numerical values stored as strings).
        ◦ Detect and treat outliers if necessary.
   2. Statistical Summary:
        
        ◦ Provide a statistical summary for each variable in the dataset, including measures of central tendency (mean, median) and
           dispersion (standard deviation, interquartile range).
        
        ◦ Highlight any interesting findings from this summary.
   3. Data Visualization:
        
        ◦ Create histograms or boxplots to visualize the distributions of various numerical variables.
        
        ◦ Use bar charts or pie charts to display the frequency of categories for categorical variables.
        
        ◦ Generate scatter plots or correlation heatmaps to explore relationships between pairs of variables.
        
        ◦ Employ advanced visualization techniques like pair plots, or violin plots for deeper insights.
  4. Pattern Recognition and Insights:
        
        ◦ Identify any correlations between variables and discuss their potential implications.
        
        ◦ Look for trends or patterns over time if temporal data is available.
  5. Conclusion:
        
        ◦ Summarize the key insights and patterns discovered through your exploratory analysis.
        
        ◦ Discuss how these findings could impact decision-making or further analyses.


#### Dataset
[Cardiotocographic](https://github.com/aravindajai/DataScienceAssignments/blob/Datasets/Cardiotocographic.csv)
## 06 - MLR

### Assignment Task:
Your task is to perform a multiple linear regression analysis to predict the price of Toyota corolla based on the given attributes.
### Dataset Description:
The dataset consists of the following variables:

    Age: Age in years
    KM: Accumulated Kilometers on odometer
    FuelType: Fuel Type (Petrol, Diesel, CNG)
    HP: Horse Power
    Automatic: Automatic ( (Yes=1, No=0)
    CC: Cylinder Volume in cubic centimeters
    Doors: Number of doors
    Weight: Weight in Kilograms
    Quarterly_Tax: 
    Price: Offer Price in EUROs
### Tasks:
1.Perform exploratory data analysis (EDA) to gain insights into the dataset. Provide visualizations and summary statistics of the variables. Pre process the data to apply the MLR.

2.Split the dataset into training and testing sets (e.g., 80% training, 20% testing).

3.Build a multiple linear regression model using the training dataset. Interpret the coefficients of the model. Build minimum of 3 different models.

4.Evaluate the performance of the model using appropriate evaluation metrics on the testing dataset.
5.Apply Lasso and Ridge methods on the model.

#### Dataset
[ToyotaCorolla - MLR](https://github.com/aravindajai/DataScienceAssignments/blob/Datasets/ToyotaCorolla%20-%20MLR.csv)
## 07 - Logistic Regression

### Data Exploration:
a. Load the dataset and perform exploratory data analysis (EDA).

b. Examine the features, their types, and summary statistics.

c. Create visualizations such as histograms, box plots, or pair plots to visualize the distributions and relationships between features.

d. Analyze any patterns or correlations observed in the data.
###  Data Preprocessing:
a. Handle missing values (e.g., imputation).

b. Encode categorical variables.
### Model Building:
a. Build a logistic regression model using appropriate libraries (e.g., scikit-learn).

b. Train the model using the training data.
### Model Evaluation:
a. Evaluate the performance of the model on the testing data using accuracy, precision, recall, F1-score, and ROC-AUC score.
Visualize the ROC curve.
### Interpretation:
a. Interpret the coefficients of the logistic regression model.

b. Discuss the significance of features in predicting the target variable (survival probability in this case).
### Deployment with Streamlit:
In this task, you will deploy your logistic regression model using Streamlit. The deployment can be done locally or online via Streamlit Share. Your task includes creating a Streamlit app in Python that involves loading your trained model and setting up user inputs for predictions. 

(optional)For online deployment, use Streamlit Community Cloud, which supports deployment from GitHub repositories. 
Detailed deployment instructions are available in the Streamlit Documentation.
https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app 

#### Dataset
[Titanic_test](https://github.com/aravindajai/DataScienceAssignments/blob/Datasets/Titanic_test%20LR.csv)

[Titanic_train](https://github.com/aravindajai/DataScienceAssignments/blob/Datasets/Titanic_train%20LR.csv)
## 08 - Clustering

Understanding and Implementing K-Means, Hierarchical, and DBSCAN Algorithms

### Objective:
The objective of this assignment is to introduce to various clustering algorithms, including K-Means, hierarchical, and DBSCAN, and provide hands-on experience in applying these techniques to a real-world dataset.
### Datasets :
### Data Preprocessing:
 1. Preprocess the dataset to handle missing values, remove outliers, and scale the features if necessary.

   2. Perform exploratory data analysis (EDA) to gain insights into the distribution of data and identify potential clusters.
 3. Use multiple visualizations to understand the hidden patterns in the dataset
### Implementing Clustering Algorithms:
 • Implement the K-Means, hierarchical, and DBSCAN algorithms using a programming language such as Python with libraries like scikit-learn or MATLAB.

• Apply each clustering algorithm to the pre-processed dataset to identify clusters within the data.

• Experiment with different parameter settings for hierarchical clustering (e.g., linkage criteria), K-means (Elbow curve for different K values) and DBSCAN (e.g., epsilon, minPts) and evaluate the clustering results.

### Cluster Analysis and Interpretation:
• Analyse the clusters generated by each clustering algorithm and interpret the characteristics of each cluster. Write you insights in few comments.

### Visualization:
Visualize the clustering results using scatter plots or other suitable visualization techniques.

Plot the clusters with different colours to visualize the separation of data points belonging to different clusters.
### Evaluation and Performance Metrics:
Evaluate the quality of clustering using internal evaluation metrics such as silhouette score for K-Means and DBSCAN.

#### Dataset
[EastWestAirlines](https://github.com/aravindajai/DataScienceAssignments/blob/Datasets/EastWestAirlines.xlsx)
## 09 - PCA 

### Task 1: Exploratory Data Analysis (EDA):
 1. Load the dataset and perform basic data exploration.
 2. Examine the distribution of features using histograms, box plots, or density plots.
 3. Investigate correlations between features to understand relationships within the data.
### Task 2: Dimensionality Reduction with PCA:
 1. Standardize the features to ensure they have a mean of 0 and a standard deviation of Implement PCA to reduce the dimensionality of the dataset.
 2. Determine the optimal number of principal components using techniques like scree plot or cumulative explained variance.
 3. Transform the original dataset into the principal components.
### Task 3: Clustering with Original Data:
 1. Apply a clustering algorithm (e.g., K-means) to the original dataset.
 2. Visualize the clustering results using appropriate plots.
 3. Evaluate the clustering performance using metrics such as silhouette score or Davies–Bouldin index.
### Task 4: Clustering with PCA Data:
 1. Apply the same clustering algorithm to the PCA-transformed dataset.
 2. Visualize the clustering results obtained from PCA-transformed data.
 3. Compare the clustering results from PCA-transformed data with those from the original dataset.
### Task 5: Comparison and Analysis:
 1. Compare the clustering results obtained from the original dataset and PCA-transformed data.
 2. Discuss any similarities or differences observed in the clustering results.
 3. Reflect on the impact of dimensionality reduction on clustering performance.
 4. Analyze the trade-offs between using PCA and clustering directly on the original dataset.
### Task 6: Conclusion and Insights

 1. Summarize the key findings and insights from the assignment.
 2. Discuss the practical implications of using PCA and clustering in data analysis.
 3. Provide recommendations for when to use each technique based on the analysis conducted.

 #### Dataset
 [wine](https://github.com/aravindajai/DataScienceAssignments/blob/Datasets/wine.csv)
## 10 - Association Rules

The Objective of this assignment is to introduce students to rule mining techniques, particularly focusing on market basket analysis and provide hands on experience.
### Dataset:
Use the Online retail dataset to apply the association rules.
### Data Preprocessing:
Pre-process the dataset to ensure it is suitable for Association rules, this may include handling missing values, removing duplicates, and converting the data to appropriate format.  
### Association Rule Mining:
    • Implement an Apriori algorithm using tool like python with libraries such as Pandas and Mlxtend etc.
    • Apply association rule mining techniques to the pre-processed dataset to discover interesting relationships between products purchased together.
    • Set appropriate threshold for support, confidence and lift to extract meaning full rules.
### Analysis and Interpretation:
 • Analyse the generated rules to identify interesting patterns and relationships between the products.
 
 • Interpret the results and provide insights into customer purchasing behaviour based on the discovered rules.

 #### Dataset
[Online retail](https://github.com/aravindajai/DataScienceAssignments/blob/Datasets/Online%20retail.xlsx)
## 11 - Recommendation System

### Data Description:

    Unique ID of each anime.
    Anime title.
    Anime broadcast type, such as TV, OVA, etc.
    anime genre.
    The number of episodes of each anime.
    The average rating for each anime compared to the number of users who gave ratings.
    Number of community members for each anime.
### Objective:
The objective of this assignment is to implement a recommendation system using cosine similarity on an anime dataset. 
Dataset:
Use the Anime Dataset which contains information about various anime, including their titles, genres,No.of episodes and user ratings etc.

### Tasks:

#### Data Preprocessing:

Load the dataset into a suitable data structure (e.g., pandas DataFrame).
Handle missing values, if any.
Explore the dataset to understand its structure and attributes.

#### Feature Extraction:

Decide on the features that will be used for computing similarity (e.g., genres, user ratings).
Convert categorical features into numerical representations if necessary.
Normalize numerical features if required.

#### Recommendation System:

Design a function to recommend anime based on cosine similarity.
Given a target anime, recommend a list of similar anime based on cosine similarity scores.
Experiment with different threshold values for similarity scores to adjust the recommendation list size.

### Evaluation:

Split the dataset into training and testing sets.
Evaluate the recommendation system using appropriate metrics such as precision, recall, and F1-score.
Analyze the performance of the recommendation system and identify areas of improvement.

#### Dataset
[anime](https://github.com/aravindajai/DataScienceAssignments/blob/Datasets/anime.csv)
## 12 - EDA 2

### Objective:
This assignment aims to equip you with practical skills in data preprocessing, feature engineering, and feature selection techniques, which are crucial for building efficient machine learning models. You will work with a provided dataset to apply various techniques such as scaling, encoding, and feature selection methods including isolation forest and PPS score analysis.
#### Dataset:
Given "Adult" dataset, which predicts whether income exceeds $50K/yr based on census data.

### Tasks:
#### 1. Data Exploration and Preprocessing:
   • Load the dataset and conduct basic data exploration (summary statistics, missing values, data types).
   
   • Handle missing values as per the best practices (imputation, removal, etc.).
 
 • Apply scaling techniques to numerical features:
    
    ◦ Standard Scaling
    ◦ Min-Max Scaling
  
  • Discuss the scenarios where each scaling technique is preferred and why.
#### 2. Encoding Techniques:
 • Apply One-Hot Encoding to categorical variables with less than 5 categories.
 
 • Use Label Encoding for categorical variables with more than 5 categories.
 
 • Discuss the pros and cons of One-Hot Encoding and Label Encoding.
#### 3. Feature Engineering:
 • Create at least 2 new features that could be beneficial for the model. Explain the rationale behind your choices.
 
 • Apply a transformation (e.g., log transformation) to at least one skewed numerical feature and justify your choice.
#### 4. Feature Selection:
 • Use the Isolation Forest algorithm to identify and remove outliers. Discuss how outliers can affect model performance.
 
 • Apply the PPS (Predictive Power Score) to find and discuss the relationships between features. Compare its findings with the correlation matrix.

 #### Dataset
 [adult_with_headers](https://github.com/aravindajai/DataScienceAssignments/blob/Datasets/adult_with_headers.csv)
## 13 - Decision Tree

### Objective:
The objective of this assignment is to apply Decision Tree Classification to a given dataset, analyse the performance of the model, and interpret the results.
### Tasks:
#### 1. Data Preparation:
Load the dataset into your preferred data analysis environment (e.g., Python with libraries like Pandas and NumPy).
#### 2. Exploratory Data Analysis (EDA):
Perform exploratory data analysis to understand the structure of the dataset.
Check for missing values, outliers, and inconsistencies in the data.
Visualize the distribution of features, including histograms, box plots, and correlation matrices.
#### 3. Feature Engineering:
If necessary, perform feature engineering techniques such as encoding categorical variables, scaling numerical features, or handling missing values.
#### 4. Decision Tree Classification:
Split the dataset into training and testing sets (e.g., using an 80-20 split).
Implement a Decision Tree Classification model using a library like scikit-learn.
Train the model on the training set and evaluate its performance on the testing set using appropriate evaluation metrics

    (e.g., accuracy, precision, recall, F1-score, ROC-AUC).
#### 5. Hyperparameter Tuning:
Perform hyperparameter tuning to optimize the Decision Tree model. Experiment with different hyperparameters such as maximum depth, minimum samples split, and criterion.
#### 6. Model Evaluation and Analysis:
Analyse the performance of the Decision Tree model using the evaluation metrics obtained.
Visualize the decision tree structure to understand the rules learned by the model and identify important features.

#### Dataset
[heart_disease](https://github.com/aravindajai/DataScienceAssignments/blob/Datasets/heart_disease.xlsx)
## 14 - Random Forest

### Dataset Description:

Use the Glass dataset and apply the Random forest model.

### 1. Exploratory Data Analysis (EDA):

Perform exploratory data analysis to understand the structure of the dataset.
Check for missing values, outliers, inconsistencies in the data.

### 2: Data Visualization:

Create visualizations such as histograms, box plots, or pair plots to visualize the distributions and relationships between features.
Analyze any patterns or correlations observed in the data.

### 3: Data Preprocessing

1. Check for missing values in the dataset and decide on a strategy for handling them.Implement the chosen strategy (e.g., imputation or removal) and explain your reasoning.
2. If there are categorical variables, apply encoding techniques like one-hot encoding to convert them into numerical format.
3. Apply feature scaling techniques such as standardization or normalization to ensure that all features are on a similar scale. Handling the imbalance data.

### 4: Random Forest Model Implementation
1. Divide the data into train and test split.
2. Implement a Random Forest classifier using Python and a machine learning library like scikit-learn.
3. Train the model on the train dataset. Evaluate the performance on test data using metrics like accuracy, precision, recall, and F1-score.

### 5: Bagging and Boosting Methods
Apply the Bagging and Boosting methods and compare the results.

#### Dataset 
[glass](https://github.com/aravindajai/DataScienceAssignments/blob/Datasets/glass.xlsx)
## 15 -XGBM & LGBM

### Objective:
The objective of this assignment is to compare the performance of Light GBM and XG Boost algorithms using the Titanic dataset. 
### Exploratory Data Analysis (EDA):
  1. Load the Titanic dataset using Python's pandas library.
   2. Check for missing values.
   3. Explore data distributions using histograms and box plots.
   4. Visualize relationships between features and survival using scatter plots and bar plots.
### Data Preprocessing:
   1. Impute missing values.
   2. Encode categorical variables using one-hot encoding or label encoding. 
   3. If needed you can apply more preprocessing methods on the given dataset.
### Building Predictive Models:
   1. Split the preprocessed dataset into training and testing sets.
   2. Choose appropriate evaluation metrics (e.g., accuracy, precision, recall, F1-score) for model evaluation.
   3. Build predictive models using LightGBM and XGBoost algorithms.
   4. Train the models on the training set and evaluate their performance on the testing set.
   5. Use techniques like cross-validation and hyperparameter tuning to optimize model performance.
### Comparative Analysis:
   1. Compare the performance metrics (e.g., accuracy, precision, recall) of LightGBM and XGBoost models.
   2. Visualize and interpret the results to identify the strengths and weaknesses of each algorithm.

#### Dataset 
[Titanic_test](https://github.com/aravindajai/DataScienceAssignments/blob/Datasets/Titanic_test%20XGBM%20%26%20LGBM.csv)

[Titanic_train](https://github.com/aravindajai/DataScienceAssignments/blob/Datasets/Titanic_train%20XGBM%20%26%20LGBM.csv)
## 16 - KNN 

### Objective: 
The objective of this assignment is to implement and evaluate the K-Nearest Neighbours algorithm for classification using the given datasets
### Dataset:
Need to Classify the animal type
### Tasks:
1. Analyse the data using the visualizations
2. Preprocess the data by handling missing values & Outliers, if any.
3. Split the dataset into training and testing sets (80% training, 20% testing).
4. Implement the K-Nearest Neighbours algorithm using a machine learning library like scikit-learn On training dataset
5. Choose an appropriate distance metric and value for K.
6. Evaluate the classifier's performance on the testing set using accuracy, precision, recall, and F1-score metrics.
7. Visualize the decision boundaries of the classifier.

#### Dataset 
[Zoo](https://github.com/aravindajai/DataScienceAssignments/blob/Datasets/Zoo.csv)
## 17 - SVM

### Dataset Selection:
For this assignment, we'll utilize the widely recognized Mushroom Dataset
### Task 1: Exploratory Data Analysis (EDA)
   1. Load the Mushroom dataset and perform fundamental data exploration.
   2. Utilize histograms, box plots, or density plots to understand feature distributions.
   3. Investigate feature correlations to discern relationships within the data.
### Task 2: Data Preprocessing
   1. Encode categorical variables if necessary.
   2. Split the dataset into training and testing sets.
### Task 3: Data Visualization
   1. Employ scatter plots, pair plots, or relevant visualizations to comprehend feature distributions and relationships.
   2. Visualize class distributions to gauge dataset balance or imbalance.
### Task 4: SVM Implementation
   1. Implement a basic SVM classifier using Python libraries like scikit-learn.
   2. Train the SVM model on the training data.
   3. Evaluate model performance on the testing data using appropriate metrics (e.g., accuracy, precision, recall, F1-score).
### Task 5: Visualization of SVM Results
   1. Visualize classification results on the testing data.
### Task 6: Parameter Tuning and Optimization
   1. Experiment with different SVM hyperparameters (e.g., kernel type, regularization parameter) to optimize performance.
### Task 7: Comparison and Analysis
   1. Compare SVM performance with various kernels (e.g., linear, polynomial, radial basis function).
   2. Analyze SVM strengths and weaknesses for the Mushroom dataset based on EDA and visualization results.
   3. Discuss practical implications of SVM in real-world classification tasks.

#### Dataset 
[mushroom](https://github.com/aravindajai/DataScienceAssignments/blob/Datasets/mushroom.csv)
## 18 - Neural Networks

Classification Using Artificial Neural Networks with Hyperparameter Tuning on Alphabets Data
### Overview
In this assignment, you will be tasked with developing a classification model using Artificial Neural Networks (ANNs) to classify data points from the "Alphabets_data.csv" dataset into predefined categories of alphabets. This exercise aims to deepen your understanding of ANNs and the significant role hyperparameter tuning plays in enhancing model performance.
### Dataset: "Alphabets_data.csv"
The dataset provided, "Alphabets_data.csv", consists of labeled data suitable for a classification task aimed at identifying different alphabets. Before using this data in your model, you'll need to preprocess it to ensure optimal performance.
### Tasks
1. Data Exploration and Preprocessing
   
   • Begin by loading and exploring the "Alphabets_data.csv" dataset. Summarize its key features such as the number of samples, features, and classes.

   • Execute necessary data preprocessing steps including data normalization, managing missing values.
2. Model Implementation
   
   • Construct a basic ANN model using your chosen high-level neural network library. Ensure your model includes at least one hidden layer.

   • Divide the dataset into training and test sets.

   • Train your model on the training set and then use it to make predictions on the test set.
3. Hyperparameter Tuning
  • Modify various hyperparameters, such as the number of hidden layers, neurons per hidden layer, activation functions, and learning rate, to observe their impact on model performance.
    
  • Adopt a structured approach like grid search or random search for hyperparameter tuning, documenting your methodology thoroughly.

4. Evaluation
  • Employ suitable metrics such as accuracy, precision, recall, and F1-score to evaluate your model's performance.
  
  • Discuss the performance differences between the model with default hyperparameters and the tuned model, emphasizing the effects of hyperparameter tuning.
### Evaluation Criteria
   • Accuracy and completeness of the implementation.

   • Proficiency in data preprocessing and model development.

   • Systematic approach and thoroughness in hyperparameter tuning.

   • Depth of evaluation and discussion.

   • Overall quality of the report.
#### Additional Resources
    • TensorFlow Documentation
    • Keras Documentation
#### Dataset 
[Alphabets_data](https://github.com/aravindajai/DataScienceAssignments/blob/Datasets/Alphabets_data.csv)


## 19 - Naive Bayes and Text Mining

### Overview
In this assignment, you will work on the "blogs_categories.csv" dataset, which contains blog posts categorized into various themes. Your task will be to build a text classification model using the Naive Bayes algorithm to categorize the blog posts accurately. Furthermore, you will perform sentiment analysis to understand the general sentiment (positive, negative, neutral) expressed in these posts. This assignment will enhance your understanding of text classification, sentiment analysis, and the practical application of the Naive Bayes algorithm in Natural Language Processing (NLP).
### Dataset
The provided dataset, "blogs_categories.csv", consists of blog posts along with their associated categories. Each row represents a blog post with the following columns:
    
    • Text: The content of the blog post. Column name: Data
    • Category: The category to which the blog post belongs. Column name: Labels
### Tasks
1. Data Exploration and Preprocessing
    
    • Load the "blogs_categories.csv" dataset and perform an exploratory data analysis to understand its structure and content.
    
    • Preprocess the data by cleaning the text (removing punctuation, converting to lowercase, etc.), tokenizing, and removing stopwords.
    
    • Perform feature extraction to convert text data into a format that can be used by the Naive Bayes model, using techniques such as TF-IDF.
2. Naive Bayes Model for Text Classification
    
    • Split the data into training and test sets.
    
    • Implement a Naive Bayes classifier to categorize the blog posts into their respective categories. You can use libraries like scikit-learn for this purpose.
    
    • Train the model on the training set and make predictions on the test set.
3. Sentiment Analysis
    
    • Choose a suitable library or method for performing sentiment analysis on the blog post texts.
    
    • Analyze the sentiments expressed in the blog posts and categorize them as positive, negative, or neutral. Consider only the Data column and get the sentiment for each blog.
    
    • Examine the distribution of sentiments across different categories and summarize your findings.
4. Evaluation
    
    • Evaluate the performance of your Naive Bayes classifier using metrics such as accuracy, precision, recall, and F1-score.
    
    • Discuss the performance of the model and any challenges encountered during the classification process.
    
    • Reflect on the sentiment analysis results and their implications regarding the content of the blog posts.

#### Dataset 
[blogs.csv](https://github.com/aravindajai/DataScienceAssignments/blob/Datasets/blogs.csv)
## 20 - Timeseries

### Objective:
Leverage ARIMA and Exponential Smoothing techniques to forecast future exchange rates based on historical data provided in the exchange_rate.csv dataset. 
### Dataset:
The dataset contains historical exchange rate with each column representing a different currency rate over time. The first column indicates the date, and second column represent exchange rates USD to Australian Dollar.
### Part 1: Data Preparation and Exploration
   1. Data Loading: Load the exchange_rate.csv dataset and parse the date column appropriately.
   2. Initial Exploration: Plot the time series for currency to understand their trends, seasonality, and any anomalies.
   3. Data Preprocessing: Handle any missing values or anomalies identified during the exploration phase.
### Part 2: Model Building - ARIMA
   1. Parameter Selection for ARIMA: Utilize ACF and PACF plots to estimate initial parameters (p, d, q) for the ARIMA model for one or more currency time series.
   2. Model Fitting: Fit the ARIMA model with the selected parameters to the preprocessed time series.
   3. Diagnostics: Analyze the residuals to ensure there are no patterns that might indicate model inadequacies.
   4. Forecasting: Perform out-of-sample forecasting and visualize the predicted values against the actual values.
### Part 3: Model Building - Exponential Smoothing
   1. Model Selection: Depending on the time series characteristics, choose an appropriate Exponential Smoothing model (Simple, Holt’s Linear, or Holt-Winters).
   2. Parameter Optimization: Use techniques such as grid search or AIC to find the optimal parameters for the smoothing levels and components.
   3. Model Fitting and Forecasting: Fit the chosen Exponential Smoothing model and forecast future values. Compare these forecasts visually with the actual data.
### Part 4: Evaluation and Comparison
   1. Compute Error Metrics: Use metrics such as MAE, RMSE, and MAPE to evaluate the forecasts from both models.
   2. Model Comparison: Discuss the performance, advantages, and limitations of each model based on the observed results and error metrics.
   3. Conclusion: Summarize the findings and provide insights on which model(s) yielded the best performance for forecasting exchange rates in this dataset.

#### Dataset
[exchange_rate](https://github.com/aravindajai/DataScienceAssignments/blob/Datasets/exchange_rate.csv)
