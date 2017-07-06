-   [Tuning Parameters](#tuning-parameters)
-   [Dataset](#dataset)
    -   [Summary](#summary)
    -   [Skewness](#skewness)
    -   [Outliers](#outliers)
    -   [Correlation & Collinearity](#correlation-collinearity)
        -   [Correlation](#correlation)
        -   [Collinearity Removal](#collinearity-removal)
    -   [Graphs](#graphs)
        -   [checking\_balance](#checking_balance)
        -   [months\_loan\_duration](#months_loan_duration)
        -   [credit\_history](#credit_history)
        -   [purpose](#purpose)
        -   [amount](#amount)
        -   [savings\_balance](#savings_balance)
        -   [employment\_duration](#employment_duration)
        -   [percent\_of\_income](#percent_of_income)
        -   [years\_at\_residence](#years_at_residence)
        -   [age](#age)
        -   [other\_credit](#other_credit)
        -   [housing](#housing)
        -   [existing\_loans\_count](#existing_loans_count)
        -   [job](#job)
        -   [dependents](#dependents)
        -   [phone](#phone)
-   [Spot-Check](#spot-check)
    -   [Configuration](#configuration)
        -   [Class Balance](#class-balance)
        -   [Training Data](#training-data)
        -   [Test](#test)
        -   [Testing train\_classification](#testing-train_classification)
    -   [Models](#models)
        -   [glm\_no\_pre\_process](#glm_no_pre_process)
        -   [glm\_basic\_processing](#glm_basic_processing)
        -   [glm\_remove\_collinearity\_caret](#glm_remove_collinearity_caret)
        -   [glm\_remove\_collinearity\_custom](#glm_remove_collinearity_custom)
        -   [glm\_yeojohnson](#glm_yeojohnson)
        -   [logistic\_regression\_stepwise\_backward](#logistic_regression_stepwise_backward)
        -   [linear\_discriminant\_analsysis](#linear_discriminant_analsysis)
        -   [linear\_discriminant\_analsysis\_remove\_collinear](#linear_discriminant_analsysis_remove_collinear)
        -   [linear\_discriminant\_analsysis\_remove\_collinear\_skew](#linear_discriminant_analsysis_remove_collinear_skew)
        -   [partial\_least\_squares\_discriminant\_analysis](#partial_least_squares_discriminant_analysis)
        -   [partial\_least\_squares\_discriminant\_analysis\_skew](#partial_least_squares_discriminant_analysis_skew)
        -   [glmnet\_lasso\_ridge](#glmnet_lasso_ridge)
        -   [sparse\_lda](#sparse_lda)
        -   [nearest\_shrunken\_centroids](#nearest_shrunken_centroids)
        -   [regularized\_discriminant\_analysis](#regularized_discriminant_analysis)
        -   [regularized\_discriminant\_analysis\_rc](#regularized_discriminant_analysis_rc)
        -   [mixture\_discriminant\_analysis](#mixture_discriminant_analysis)
        -   [mixture\_discriminant\_analysis\_rc](#mixture_discriminant_analysis_rc)
        -   [neural\_network\_spatial\_rc](#neural_network_spatial_rc)
        -   [neural\_network\_spatial\_rc\_skew](#neural_network_spatial_rc_skew)
        -   [flexible\_discriminant\_analsysis](#flexible_discriminant_analsysis)
        -   [svm\_linear](#svm_linear)
        -   [svm\_polynomial](#svm_polynomial)
        -   [svm\_radial](#svm_radial)
        -   [k\_nearest\_neighbors](#k_nearest_neighbors)
        -   [naive\_bayes](#naive_bayes)
        -   [rpart\_independent\_categories](#rpart_independent_categories)
        -   [rpart\_grouped\_categories](#rpart_grouped_categories)
        -   [treebag\_independent\_categories](#treebag_independent_categories)
        -   [treebag\_grouped\_categories](#treebag_grouped_categories)
        -   [c50\_model\_independent\_categories](#c50_model_independent_categories)
        -   [c50\_model\_grouped\_categories](#c50_model_grouped_categories)
        -   [c50\_rules\_model\_independent\_categories](#c50_rules_model_independent_categories)
        -   [c50\_rules\_model\_grouped\_categories](#c50_rules_model_grouped_categories)
        -   [rf\_independent\_categories](#rf_independent_categories)
        -   [rf\_grouped\_categories](#rf_grouped_categories)
        -   [adaboost\_independent\_categories](#adaboost_independent_categories)
        -   [adaboost\_grouped\_categories](#adaboost_grouped_categories)
        -   [adabag\_independent\_categories](#adabag_independent_categories)
        -   [adabag\_grouped\_categories](#adabag_grouped_categories)
        -   [gbm\_independent\_categories (stochastic gradient boosting)](#gbm_independent_categories-stochastic-gradient-boosting)
        -   [gbm\_grouped\_categories (stochastic gradient boosting)](#gbm_grouped_categories-stochastic-gradient-boosting)
        -   [All Models on Page 550 that are classification or both regression and classification](#all-models-on-page-550-that-are-classification-or-both-regression-and-classification)
        -   [Models used for spot-check.Rmd](#models-used-for-spot-check.rmd)
-   [Resamples & Top Models](#resamples-top-models)
    -   [Resamples](#resamples)
    -   [Train Top Models on Entire Training Dataset & Predict on Test Set](#train-top-models-on-entire-training-dataset-predict-on-test-set)
        -   [Random Forest (rf\_grouped\_categories)](#random-forest-rf_grouped_categories)
        -   [Stochastic Gradient Boosting (gbm\_grouped\_categories)](#stochastic-gradient-boosting-gbm_grouped_categories)
        -   [Stochastic Gradient Boosting (gbm\_independent\_categories)](#stochastic-gradient-boosting-gbm_independent_categories)
        -   [Regularized Discriminant Analysis (regularized\_discriminant\_analysis)](#regularized-discriminant-analysis-regularized_discriminant_analysis)
        -   [Random Forest (rf\_independent\_categories)](#random-forest-rf_independent_categories)

Tuning Parameters
=================

``` r
# train/test set
training_percentage <- 0.90

# cross validation
cross_validation_num_folds <- 10
cross_validation_num_repeats <- 3

# tuning parameters
tuning_number_of_latent_variables_to_retain <- 1:10

tuning_glmnet_alpha <- seq(from = 0, to = 1, length = 5) # alpha = 0 is pure ridge regression, and alpha = 1 is pure lasso regression.
tuning_glmnet_lambda <- seq(from = 0.0001, to = 1, length = 50) # lambda values control the amount of penalization in the model.

tuning_nearest_shrunken_centroids_shrinkage_threshold <- data.frame(threshold = 0:25)

tuning_mda_subclasses <- 1:8

tuning_rda_lambda <- seq(from = 0, to = 1, by = 0.2)
tuning_rda_gamma <- seq(from = 0, to = 1, by = 0.2)

tuning_nnet_size <- 1:10
tuning_nnet_decay <- c(0, 0.1, 1, 2)
parameter_nnet_linout <- FALSE
parameter_nnet_max_iterations <- 2000

tuning_svm_linear_num_costs <- 5
tuning_svm_poly_num_costs <- 3
tuning_svm_radial_num_costs <- 6

tuning_knn_tuning_grid <- data.frame(k = c(4 * (0:5) + 1, 20 * (2:5) + 1, 50 * (3:9) + 1))

tuning_naive_bayes_laplace_correction <- c(0, 0.5, 1, 2)
tuning_naive_bayes_distribution_type <- c(TRUE, FALSE)
tuning_naive_bayes_bandwidth_adjustment <- c(0, 0.5, 1.0)

tuning_gbm_shrinkage <- c(0.01, 0.10, 0.50)
tuning_gbm_num_boosting_iterations <- floor(seq(from = 50, to = 5000, length.out = 3))
tuning_gbm_max_tree_depth <- c(1, 5, 9)
tuning_gbm_min_terminal_node_size <- c(5, 15, 25)
```

Dataset
=======

> Assumes the dataset has factors for strings; logical for TRUE/FALSE; `target` for outcome variable

Summary
-------

> Total predictors: `16`

> Total data-points/rows: `1000`

> Number of training data-points: `900`

Rule of thumbs for dimensions (Probabilistic and Statistical Modeling in Computer Science; pg 430):

> r &lt; sqrt(n); where r is the number of predictors and sqrt(n) is the square root of the sample size (`32`): `TRUE`

> r &lt; sqrt(n\_t); where r is the number of predictors and sqrt(n\_t) is the square root of the training set size (`30`): `TRUE`

    ##  target      checking_balance months_loan_duration   credit_history                 purpose        amount           savings_balance  employment_duration percent_of_income
    ##  yes:300   < 0 DM    :274     Min.   : 4.0         critical :293    business            : 97   Min.   :  250   < 100 DM     :603    < 1 year   :172      Min.   :1.000    
    ##  no :700   > 200 DM  : 63     1st Qu.:12.0         good     :530    car                 :337   1st Qu.: 1366   > 1000 DM    : 48    > 7 years  :253      1st Qu.:2.000    
    ##            1 - 200 DM:269     Median :18.0         perfect  : 40    car0                : 12   Median : 2320   100 - 500 DM :103    1 - 4 years:339      Median :3.000    
    ##            unknown   :394     Mean   :20.9         poor     : 88    education           : 59   Mean   : 3271   500 - 1000 DM: 63    4 - 7 years:174      Mean   :2.973    
    ##                               3rd Qu.:24.0         very good: 49    furniture/appliances:473   3rd Qu.: 3972   unknown      :183    unemployed : 62      3rd Qu.:4.000    
    ##                               Max.   :72.0                          renovations         : 22   Max.   :18424                                             Max.   :4.000    
    ##  years_at_residence      age        other_credit  housing    existing_loans_count         job        dependents      phone    
    ##  Min.   :1.000      Min.   :19.00   bank :139    other:108   Min.   :1.000        management:148   Min.   :1.000   FALSE:596  
    ##  1st Qu.:2.000      1st Qu.:27.00   none :814    own  :713   1st Qu.:1.000        skilled   :630   1st Qu.:1.000   TRUE :404  
    ##  Median :3.000      Median :33.00   store: 47    rent :179   Median :1.000        unemployed: 22   Median :1.000              
    ##  Mean   :2.845      Mean   :35.55                            Mean   :1.407        unskilled :200   Mean   :1.155              
    ##  3rd Qu.:4.000      3rd Qu.:42.00                            3rd Qu.:2.000                         3rd Qu.:1.000              
    ##  Max.   :4.000      Max.   :75.00                            Max.   :4.000                         Max.   :2.000

Skewness
--------

Note: `Box-Cox` can only be applied to sets (i.e. predictors) where all values are `> 0`. So some/most/all? `NA`s will be from that limiation.

| column                 |  boxcox\_skewness|
|:-----------------------|-----------------:|
| target                 |                NA|
| checking\_balance      |                NA|
| months\_loan\_duration |         1.0909038|
| credit\_history        |                NA|
| purpose                |                NA|
| amount                 |         1.9437827|
| savings\_balance       |                NA|
| employment\_duration   |                NA|
| percent\_of\_income    |        -0.5297551|
| years\_at\_residence   |        -0.2717526|
| age                    |         1.0176791|
| other\_credit          |                NA|
| housing                |                NA|
| existing\_loans\_count |         1.2687608|
| job                    |                NA|
| dependents             |         1.9037202|
| phone                  |                NA|

Outliers
--------

| columns                | lower\_outlier\_count | upper\_outlier\_count |
|:-----------------------|:----------------------|:----------------------|
| months\_loan\_duration | 0                     | 16                    |
| amount                 | 0                     | 50                    |
| percent\_of\_income    | 0                     | 0                     |
| years\_at\_residence   | 0                     | 0                     |
| age                    | 0                     | 6                     |
| existing\_loans\_count | 0                     | 0                     |
| dependents             | 0                     | 155                   |

Correlation & Collinearity
--------------------------

### Correlation

<img src="predictive_analysis_classification_files/figure-markdown_github/correlation-1.png" width="750px" />

### Collinearity Removal

#### Caret's `findCorrelation`

Shows caret's recommendation of removing collinear columns based on correlation threshold of `0.9`

> columns recommended for removal: \`\`

> final columns recommended: `target, checking_balance, months_loan_duration, credit_history, purpose, amount, savings_balance, employment_duration, percent_of_income, years_at_residence, age, other_credit, housing, existing_loans_count, job, dependents, phone`

#### Heuristic

This method is described in APM pg 47 as the following steps

-   calculate the correlation matrix of predictors
-   determine the two predictors associated with the largest absolute pairwise correlation (call them predictors `A` and `B`)
-   Determine the average correlation between `A` and the other variables.
    -   Do the same for `B`
-   If `A` has a larger average correlation, remove it; otherwise, remove predcitor `B`
-   Repeat until no absolute correlations are above the threshold (`0.9`)

> columns recommended for removal: \`\`

> final columns recommended: `months_loan_duration, amount, percent_of_income, years_at_residence, age, existing_loans_count, dependents, target, checking_balance, credit_history, purpose, savings_balance, employment_duration, other_credit, housing, job, phone`

Graphs
------

### checking\_balance

<img src="predictive_analysis_classification_files/figure-markdown_github/graphs-1.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-2.png" width="750px" />

> Chi-Square p-value: `0`

### months\_loan\_duration

<img src="predictive_analysis_classification_files/figure-markdown_github/graphs-3.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-4.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-5.png" width="750px" />

statistically different means (check assumptions for t-test): TRUE

The Wilcoxon-Matt-Whitney test (or Wilcoxon rank sum test, or Mann-Whitney U-test) is used when is asked to compare the means of two groups that do not follow a normal distribution: it is a non-parametrical test. (<https://www.r-bloggers.com/wilcoxon-mann-whitney-rank-sum-test-or-test-u/>)

statistically different means (Wilcoxon-Matt-Whitney): TRUE

### credit\_history

<img src="predictive_analysis_classification_files/figure-markdown_github/graphs-6.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-7.png" width="750px" />

> Chi-Square p-value: `0`

### purpose

<img src="predictive_analysis_classification_files/figure-markdown_github/graphs-8.png" width="750px" />

    Warning in chisq.test(count_table): Chi-squared approximation may be incorrect

<img src="predictive_analysis_classification_files/figure-markdown_github/graphs-9.png" width="750px" />

> Chi-Square p-value: `0.145`

### amount

<img src="predictive_analysis_classification_files/figure-markdown_github/graphs-10.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-11.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-12.png" width="750px" />

statistically different means (check assumptions for t-test): TRUE

The Wilcoxon-Matt-Whitney test (or Wilcoxon rank sum test, or Mann-Whitney U-test) is used when is asked to compare the means of two groups that do not follow a normal distribution: it is a non-parametrical test. (<https://www.r-bloggers.com/wilcoxon-mann-whitney-rank-sum-test-or-test-u/>)

statistically different means (Wilcoxon-Matt-Whitney): TRUE

### savings\_balance

<img src="predictive_analysis_classification_files/figure-markdown_github/graphs-13.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-14.png" width="750px" />

> Chi-Square p-value: `0`

### employment\_duration

<img src="predictive_analysis_classification_files/figure-markdown_github/graphs-15.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-16.png" width="750px" />

> Chi-Square p-value: `0.001`

### percent\_of\_income

<img src="predictive_analysis_classification_files/figure-markdown_github/graphs-17.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-18.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-19.png" width="750px" />

statistically different means (check assumptions for t-test): TRUE

The Wilcoxon-Matt-Whitney test (or Wilcoxon rank sum test, or Mann-Whitney U-test) is used when is asked to compare the means of two groups that do not follow a normal distribution: it is a non-parametrical test. (<https://www.r-bloggers.com/wilcoxon-mann-whitney-rank-sum-test-or-test-u/>)

statistically different means (Wilcoxon-Matt-Whitney): TRUE

### years\_at\_residence

<img src="predictive_analysis_classification_files/figure-markdown_github/graphs-20.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-21.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-22.png" width="750px" />

statistically different means (check assumptions for t-test): FALSE

The Wilcoxon-Matt-Whitney test (or Wilcoxon rank sum test, or Mann-Whitney U-test) is used when is asked to compare the means of two groups that do not follow a normal distribution: it is a non-parametrical test. (<https://www.r-bloggers.com/wilcoxon-mann-whitney-rank-sum-test-or-test-u/>)

statistically different means (Wilcoxon-Matt-Whitney): FALSE

### age

<img src="predictive_analysis_classification_files/figure-markdown_github/graphs-23.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-24.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-25.png" width="750px" />

statistically different means (check assumptions for t-test): TRUE

The Wilcoxon-Matt-Whitney test (or Wilcoxon rank sum test, or Mann-Whitney U-test) is used when is asked to compare the means of two groups that do not follow a normal distribution: it is a non-parametrical test. (<https://www.r-bloggers.com/wilcoxon-mann-whitney-rank-sum-test-or-test-u/>)

statistically different means (Wilcoxon-Matt-Whitney): TRUE

### other\_credit

<img src="predictive_analysis_classification_files/figure-markdown_github/graphs-26.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-27.png" width="750px" />

> Chi-Square p-value: `0.002`

### housing

<img src="predictive_analysis_classification_files/figure-markdown_github/graphs-28.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-29.png" width="750px" />

> Chi-Square p-value: `0`

### existing\_loans\_count

<img src="predictive_analysis_classification_files/figure-markdown_github/graphs-30.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-31.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-32.png" width="750px" />

statistically different means (check assumptions for t-test): FALSE

The Wilcoxon-Matt-Whitney test (or Wilcoxon rank sum test, or Mann-Whitney U-test) is used when is asked to compare the means of two groups that do not follow a normal distribution: it is a non-parametrical test. (<https://www.r-bloggers.com/wilcoxon-mann-whitney-rank-sum-test-or-test-u/>)

statistically different means (Wilcoxon-Matt-Whitney): FALSE

### job

<img src="predictive_analysis_classification_files/figure-markdown_github/graphs-33.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-34.png" width="750px" />

> Chi-Square p-value: `0.597`

### dependents

<img src="predictive_analysis_classification_files/figure-markdown_github/graphs-35.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-36.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-37.png" width="750px" />

statistically different means (check assumptions for t-test): FALSE

The Wilcoxon-Matt-Whitney test (or Wilcoxon rank sum test, or Mann-Whitney U-test) is used when is asked to compare the means of two groups that do not follow a normal distribution: it is a non-parametrical test. (<https://www.r-bloggers.com/wilcoxon-mann-whitney-rank-sum-test-or-test-u/>)

statistically different means (Wilcoxon-Matt-Whitney): FALSE

### phone

<img src="predictive_analysis_classification_files/figure-markdown_github/graphs-38.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/graphs-39.png" width="750px" />

> Chi-Square p-value: `0.279`

Spot-Check
==========

Configuration
-------------

### Class Balance

Make sure class balance is even amount training/test datasets.

### Training Data

    ## 
    ## yes  no 
    ## 0.3 0.7

### Test

    ## 
    ## yes  no 
    ## 0.3 0.7

> Using `10`-fold cross-validation with `3` repeats, using the `ROC` statistic to evaluate each model.

> used `90%` of data for `training` set (`900`), and `10%` for `test` set (`100`).

### Testing train\_classification

> NOTE that for logistic regression (GLM), caret's `train()` (because of `glm()`) uses the second-level factor value as the success/postive event but `resamples()` uses the first-level as the success event. The result is either the `sensitivity` and `specificity` for `resamples()` will be reversed (and so I would be unable to compare apples to apples with other models), or I need to keep the first-level factor as the positive event (the default approach), which will mean that THE COEFFICIENTS WILL BE REVERSED, MAKIN THE MODEL RELATIVE TO THE NEGATIVE EVENT. I chose the latter, in order to compare models below, but this means that when using the logistic model to explain the data, the reader needs to mentally reverse the direction/sign of the coefficients, or correct the problem in the final stages of model building.

> NOTE: "Logistic regression does not make many of the key assumptions of linear regression and general linear models that are based on ordinary least squares algorithms – particularly regarding linearity, normality, homoscedasticity, and measurement level." [link](http://www.statisticssolutions.com/assumptions-of-logistic-regression/)

Using formula in `train()`

#### Model Summary


    Call:
    NULL

    Deviance Residuals: 
        Min       1Q   Median       3Q      Max  
    -2.5895  -0.8436   0.4207   0.7751   2.0452  

    Coefficients:
                                      Estimate Std. Error z value             Pr(>|z|)    
    (Intercept)                       1.147190   0.094081  12.194 < 0.0000000000000002 ***
    `checking_balance> 200 DM`        0.238392   0.090948   2.621             0.008763 ** 
    `checking_balance1 - 200 DM`      0.169782   0.093537   1.815             0.069503 .  
    checking_balanceunknown           0.864977   0.112309   7.702   0.0000000000000134 ***
    months_loan_duration             -0.243480   0.109843  -2.217             0.026649 *  
    credit_historygood               -0.296156   0.119650  -2.475             0.013317 *  
    credit_historypoor               -0.109460   0.090283  -1.212             0.225354    
    `credit_historyvery good`        -0.241627   0.089351  -2.704             0.006846 ** 
    purposecar                        0.034689   0.130295   0.266             0.790058    
    purposeeducation                 -0.088180   0.099152  -0.889             0.373819    
    `purposefurniture/appliances`     0.180758   0.135377   1.335             0.181803    
    amount                           -0.400086   0.120926  -3.309             0.000938 ***
    `savings_balance100 - 500 DM`     0.015658   0.085541   0.183             0.854758    
    `savings_balance500 - 1000 DM`    0.054567   0.098771   0.552             0.580635    
    savings_balanceunknown            0.329990   0.100635   3.279             0.001041 ** 
    `employment_duration> 7 years`    0.255840   0.126875   2.016             0.043749 *  
    `employment_duration1 - 4 years`  0.099657   0.111016   0.898             0.369353    
    `employment_duration4 - 7 years`  0.371148   0.112825   3.290             0.001003 ** 
    employment_durationunemployed     0.050138   0.097707   0.513             0.607846    
    percent_of_income                -0.383384   0.097775  -3.921   0.0000881507786376 ***
    years_at_residence                0.005518   0.095818   0.058             0.954075    
    age                               0.128240   0.104123   1.232             0.218093    
    other_creditnone                  0.187232   0.083529   2.242             0.024992 *  
    housingown                        0.151433   0.135406   1.118             0.263413    
    housingrent                      -0.084313   0.130666  -0.645             0.518761    
    existing_loans_count             -0.178839   0.108980  -1.641             0.100790    
    jobskilled                        0.007380   0.130238   0.057             0.954813    
    jobunskilled                      0.044701   0.131371   0.340             0.733655    
    dependents                       -0.041299   0.086738  -0.476             0.633977    
    phoneTRUE                         0.222771   0.099246   2.245             0.024791 *  
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    (Dispersion parameter for binomial family taken to be 1)

        Null deviance: 1099.56  on 899  degrees of freedom
    Residual deviance:  872.53  on 870  degrees of freedom
    AIC: 932.53

    Number of Fisher Scoring iterations: 5

#### Model Predictors

     [1] "checking_balance> 200 DM"       "checking_balance1 - 200 DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historypoor"             "credit_historyvery good"        "purposecar"                     "purposeeducation"               "purposefurniture/appliances"   
    [11] "amount"                         "savings_balance100 - 500 DM"    "savings_balance500 - 1000 DM"   "savings_balanceunknown"         "employment_duration> 7 years"  
    [16] "employment_duration1 - 4 years" "employment_duration4 - 7 years" "employment_durationunemployed"  "percent_of_income"              "years_at_residence"            
    [21] "age"                            "other_creditnone"               "housingown"                     "housingrent"                    "existing_loans_count"          
    [26] "jobskilled"                     "jobunskilled"                   "dependents"                     "phoneTRUE"                     

#### Variable Importance

    glm variable importance

      only 20 most important variables shown (out of 29)

     Overall
      7.7017
      3.9211
      3.3085
      3.2896
      3.2791
      2.7042
      2.6212
      2.4752
      2.2446
      2.2415
      2.2166
      2.0165
      1.8151
      1.6410
      1.3352
      1.2316
      1.2124
      1.1184
      0.8977
      0.8893

Models
------

### glm\_no\_pre\_process

#### Model Summary


    Call:
    NULL

    Deviance Residuals: 
        Min       1Q   Median       3Q      Max  
    -2.6364  -0.7956   0.4122   0.7664   1.8864  

    Coefficients:
                                        Estimate  Std. Error z value          Pr(>|z|)    
    (Intercept)                       1.67233497  0.93888105   1.781          0.074880 .  
    `checking_balance> 200 DM`        0.91202032  0.36959975   2.468          0.013603 *  
    `checking_balance1 - 200 DM`      0.36146669  0.21675128   1.668          0.095384 .  
    checking_balanceunknown           1.69919542  0.23318347   7.287 0.000000000000317 ***
    months_loan_duration             -0.01909034  0.00930934  -2.051          0.040300 *  
    credit_historygood               -0.83854132  0.26057700  -3.218          0.001291 ** 
    credit_historyperfect            -1.18300647  0.42783664  -2.765          0.005691 ** 
    credit_historypoor               -0.70792254  0.34560901  -2.048          0.040527 *  
    `credit_historyvery good`        -1.43539610  0.42810208  -3.353          0.000800 ***
    purposecar                       -0.14052327  0.32598325  -0.431          0.666414    
    purposecar0                       0.63233919  0.81457907   0.776          0.437585    
    purposeeducation                 -0.58632030  0.43971583  -1.333          0.182398    
    `purposefurniture/appliances`     0.16610147  0.31881865   0.521          0.602373    
    purposerenovations               -0.68269967  0.60731250  -1.124          0.260957    
    amount                           -0.00013829  0.00004389  -3.151          0.001627 ** 
    `savings_balance> 1000 DM`        1.03432320  0.51321912   2.015          0.043867 *  
    `savings_balance100 - 500 DM`     0.13185558  0.28429005   0.464          0.642786    
    `savings_balance500 - 1000 DM`    0.27415120  0.41264732   0.664          0.506452    
    savings_balanceunknown            0.90758459  0.26502755   3.424          0.000616 ***
    `employment_duration> 7 years`    0.51216659  0.29605002   1.730          0.083630 .  
    `employment_duration1 - 4 years`  0.16207344  0.23846600   0.680          0.496726    
    `employment_duration4 - 7 years`  0.92790647  0.30112909   3.081          0.002060 ** 
    employment_durationunemployed     0.14840842  0.43655991   0.340          0.733894    
    percent_of_income                -0.34774866  0.08869354  -3.921 0.000088259512881 ***
    years_at_residence               -0.00385951  0.08729784  -0.044          0.964736    
    age                               0.01108220  0.00927862   1.194          0.232329    
    other_creditnone                  0.52544326  0.24108458   2.179          0.029295 *  
    other_creditstore                 0.12816587  0.42389741   0.302          0.762384    
    housingown                        0.27205220  0.30231677   0.900          0.368178    
    housingrent                      -0.25445634  0.34509987  -0.737          0.460915    
    existing_loans_count             -0.33507655  0.19199533  -1.745          0.080944 .  
    jobskilled                        0.04584693  0.28901415   0.159          0.873959    
    jobunemployed                     0.09476193  0.65453976   0.145          0.884887    
    jobunskilled                      0.14669504  0.35145618   0.417          0.676392    
    dependents                       -0.11052559  0.24712936  -0.447          0.654703    
    phoneTRUE                         0.41866313  0.20925782   2.001          0.045424 *  
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    (Dispersion parameter for binomial family taken to be 1)

        Null deviance: 1099.56  on 899  degrees of freedom
    Residual deviance:  857.01  on 864  degrees of freedom
    AIC: 929.01

    Number of Fisher Scoring iterations: 5

#### Model Predictors

     [1] "checking_balance> 200 DM"       "checking_balance1 - 200 DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historyperfect"          "credit_historypoor"             "credit_historyvery good"        "purposecar"                     "purposecar0"                   
    [11] "purposeeducation"               "purposefurniture/appliances"    "purposerenovations"             "amount"                         "savings_balance> 1000 DM"      
    [16] "savings_balance100 - 500 DM"    "savings_balance500 - 1000 DM"   "savings_balanceunknown"         "employment_duration> 7 years"   "employment_duration1 - 4 years"
    [21] "employment_duration4 - 7 years" "employment_durationunemployed"  "percent_of_income"              "years_at_residence"             "age"                           
    [26] "other_creditnone"               "other_creditstore"              "housingown"                     "housingrent"                    "existing_loans_count"          
    [31] "jobskilled"                     "jobunemployed"                  "jobunskilled"                   "dependents"                     "phoneTRUE"                     

#### Variable Importance

    glm variable importance

      only 20 most important variables shown (out of 35)

     Overall
       7.287
       3.921
       3.424
       3.353
       3.218
       3.151
       3.081
       2.765
       2.468
       2.179
       2.051
       2.048
       2.015
       2.001
       1.745
       1.730
       1.668
       1.333
       1.194
       1.124

### glm\_basic\_processing

#### Model Summary


    Call:
    NULL

    Deviance Residuals: 
        Min       1Q   Median       3Q      Max  
    -2.5895  -0.8436   0.4207   0.7751   2.0452  

    Coefficients:
                                      Estimate Std. Error z value             Pr(>|z|)    
    (Intercept)                       1.147190   0.094081  12.194 < 0.0000000000000002 ***
    `checking_balance> 200 DM`        0.238392   0.090948   2.621             0.008763 ** 
    `checking_balance1 - 200 DM`      0.169782   0.093537   1.815             0.069503 .  
    checking_balanceunknown           0.864977   0.112309   7.702   0.0000000000000134 ***
    months_loan_duration             -0.243480   0.109843  -2.217             0.026649 *  
    credit_historygood               -0.296156   0.119650  -2.475             0.013317 *  
    credit_historypoor               -0.109460   0.090283  -1.212             0.225354    
    `credit_historyvery good`        -0.241627   0.089351  -2.704             0.006846 ** 
    purposecar                        0.034689   0.130295   0.266             0.790058    
    purposeeducation                 -0.088180   0.099152  -0.889             0.373819    
    `purposefurniture/appliances`     0.180758   0.135377   1.335             0.181803    
    amount                           -0.400086   0.120926  -3.309             0.000938 ***
    `savings_balance100 - 500 DM`     0.015658   0.085541   0.183             0.854758    
    `savings_balance500 - 1000 DM`    0.054567   0.098771   0.552             0.580635    
    savings_balanceunknown            0.329990   0.100635   3.279             0.001041 ** 
    `employment_duration> 7 years`    0.255840   0.126875   2.016             0.043749 *  
    `employment_duration1 - 4 years`  0.099657   0.111016   0.898             0.369353    
    `employment_duration4 - 7 years`  0.371148   0.112825   3.290             0.001003 ** 
    employment_durationunemployed     0.050138   0.097707   0.513             0.607846    
    percent_of_income                -0.383384   0.097775  -3.921   0.0000881507786376 ***
    years_at_residence                0.005518   0.095818   0.058             0.954075    
    age                               0.128240   0.104123   1.232             0.218093    
    other_creditnone                  0.187232   0.083529   2.242             0.024992 *  
    housingown                        0.151433   0.135406   1.118             0.263413    
    housingrent                      -0.084313   0.130666  -0.645             0.518761    
    existing_loans_count             -0.178839   0.108980  -1.641             0.100790    
    jobskilled                        0.007380   0.130238   0.057             0.954813    
    jobunskilled                      0.044701   0.131371   0.340             0.733655    
    dependents                       -0.041299   0.086738  -0.476             0.633977    
    phoneTRUE                         0.222771   0.099246   2.245             0.024791 *  
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    (Dispersion parameter for binomial family taken to be 1)

        Null deviance: 1099.56  on 899  degrees of freedom
    Residual deviance:  872.53  on 870  degrees of freedom
    AIC: 932.53

    Number of Fisher Scoring iterations: 5

#### Model Predictors

     [1] "checking_balance> 200 DM"       "checking_balance1 - 200 DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historypoor"             "credit_historyvery good"        "purposecar"                     "purposeeducation"               "purposefurniture/appliances"   
    [11] "amount"                         "savings_balance100 - 500 DM"    "savings_balance500 - 1000 DM"   "savings_balanceunknown"         "employment_duration> 7 years"  
    [16] "employment_duration1 - 4 years" "employment_duration4 - 7 years" "employment_durationunemployed"  "percent_of_income"              "years_at_residence"            
    [21] "age"                            "other_creditnone"               "housingown"                     "housingrent"                    "existing_loans_count"          
    [26] "jobskilled"                     "jobunskilled"                   "dependents"                     "phoneTRUE"                     

#### Variable Importance

    glm variable importance

      only 20 most important variables shown (out of 29)

     Overall
      7.7017
      3.9211
      3.3085
      3.2896
      3.2791
      2.7042
      2.6212
      2.4752
      2.2446
      2.2415
      2.2166
      2.0165
      1.8151
      1.6410
      1.3352
      1.2316
      1.2124
      1.1184
      0.8977
      0.8893

### glm\_remove\_collinearity\_caret

> No collinear columns removed... skipping.

### glm\_remove\_collinearity\_custom

> No collinear columns removed... skipping.

### glm\_yeojohnson

#### Model Summary


    Call:
    NULL

    Deviance Residuals: 
        Min       1Q   Median       3Q      Max  
    -2.6109  -0.8652   0.4351   0.7762   1.9022  

    Coefficients:
                                     Estimate Std. Error z value             Pr(>|z|)    
    (Intercept)                       1.14412    0.09387  12.188 < 0.0000000000000002 ***
    `checking_balance> 200 DM`        0.24800    0.09058   2.738              0.00618 ** 
    `checking_balance1 - 200 DM`      0.14181    0.09266   1.530              0.12592    
    checking_balanceunknown           0.84683    0.11174   7.578    0.000000000000035 ***
    months_loan_duration             -0.37111    0.12392  -2.995              0.00275 ** 
    credit_historygood               -0.28423    0.12404  -2.292              0.02193 *  
    credit_historypoor               -0.09885    0.09005  -1.098              0.27234    
    `credit_historyvery good`        -0.23044    0.09013  -2.557              0.01057 *  
    purposecar                        0.02868    0.12795   0.224              0.82264    
    purposeeducation                 -0.09452    0.09774  -0.967              0.33355    
    `purposefurniture/appliances`     0.20303    0.13313   1.525              0.12725    
    amount                           -0.17853    0.13343  -1.338              0.18090    
    `savings_balance100 - 500 DM`     0.02200    0.08474   0.260              0.79519    
    `savings_balance500 - 1000 DM`    0.06394    0.09920   0.645              0.51919    
    savings_balanceunknown            0.31708    0.10004   3.169              0.00153 ** 
    `employment_duration> 7 years`    0.25067    0.12583   1.992              0.04636 *  
    `employment_duration1 - 4 years`  0.09564    0.11026   0.867              0.38571    
    `employment_duration4 - 7 years`  0.35903    0.11203   3.205              0.00135 ** 
    employment_durationunemployed     0.05352    0.09701   0.552              0.58113    
    percent_of_income                -0.29189    0.10029  -2.911              0.00361 ** 
    years_at_residence                0.01564    0.09497   0.165              0.86919    
    age                               0.13466    0.10272   1.311              0.18988    
    other_creditnone                  0.18621    0.08298   2.244              0.02482 *  
    housingown                        0.16967    0.13346   1.271              0.20361    
    housingrent                      -0.05022    0.12909  -0.389              0.69728    
    existing_loans_count             -0.16304    0.11190  -1.457              0.14513    
    jobskilled                        0.07133    0.12760   0.559              0.57614    
    jobunskilled                      0.08496    0.13031   0.652              0.51440    
    dependents                       -0.05138    0.08726  -0.589              0.55599    
    phoneTRUE                         0.18234    0.09788   1.863              0.06247 .  
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    (Dispersion parameter for binomial family taken to be 1)

        Null deviance: 1099.56  on 899  degrees of freedom
    Residual deviance:  881.96  on 870  degrees of freedom
    AIC: 941.96

    Number of Fisher Scoring iterations: 5

#### Model Predictors

     [1] "checking_balance> 200 DM"       "checking_balance1 - 200 DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historypoor"             "credit_historyvery good"        "purposecar"                     "purposeeducation"               "purposefurniture/appliances"   
    [11] "amount"                         "savings_balance100 - 500 DM"    "savings_balance500 - 1000 DM"   "savings_balanceunknown"         "employment_duration> 7 years"  
    [16] "employment_duration1 - 4 years" "employment_duration4 - 7 years" "employment_durationunemployed"  "percent_of_income"              "years_at_residence"            
    [21] "age"                            "other_creditnone"               "housingown"                     "housingrent"                    "existing_loans_count"          
    [26] "jobskilled"                     "jobunskilled"                   "dependents"                     "phoneTRUE"                     

#### Variable Importance

    glm variable importance

      only 20 most important variables shown (out of 29)

     Overall
      7.5785
      3.2046
      3.1695
      2.9948
      2.9106
      2.7380
      2.5567
      2.2915
      2.2441
      1.9921
      1.8629
      1.5304
      1.5250
      1.4570
      1.3380
      1.3109
      1.2713
      1.0977
      0.9670
      0.8674

### logistic\_regression\_stepwise\_backward

#### Model Summary


    Call:
    NULL

    Deviance Residuals: 
        Min       1Q   Median       3Q      Max  
    -2.5554  -0.8800   0.4347   0.7691   2.0064  

    Coefficients:
                                      Estimate Std. Error z value             Pr(>|z|)    
    (Intercept)                       1.131640   0.092838  12.189 < 0.0000000000000002 ***
    `checking_balance> 200 DM`        0.245993   0.089350   2.753             0.005903 ** 
    `checking_balance1 - 200 DM`      0.165115   0.091617   1.802             0.071508 .  
    checking_balanceunknown           0.859391   0.111245   7.725   0.0000000000000112 ***
    credit_historygood               -0.283603   0.117919  -2.405             0.016169 *  
    credit_historypoor               -0.120046   0.088977  -1.349             0.177278    
    `credit_historyvery good`        -0.246000   0.088881  -2.768             0.005645 ** 
    `savings_balance100 - 500 DM`     0.001352   0.084900   0.016             0.987294    
    `savings_balance500 - 1000 DM`    0.070428   0.098308   0.716             0.473743    
    savings_balanceunknown            0.312059   0.098696   3.162             0.001568 ** 
    `employment_duration> 7 years`    0.295568   0.116962   2.527             0.011502 *  
    `employment_duration1 - 4 years`  0.103760   0.109846   0.945             0.344866    
    `employment_duration4 - 7 years`  0.372998   0.111061   3.359             0.000784 ***
    employment_durationunemployed     0.059406   0.089472   0.664             0.506715    
    other_creditnone                  0.183429   0.082681   2.219             0.026519 *  
    housingown                        0.178234   0.123668   1.441             0.149520    
    housingrent                      -0.081868   0.122738  -0.667             0.504761    
    phoneTRUE                         0.210035   0.091554   2.294             0.021785 *  
    months_loan_duration             -0.251555   0.107446  -2.341             0.019221 *  
    amount                           -0.409812   0.117708  -3.482             0.000498 ***
    percent_of_income                -0.372375   0.096322  -3.866             0.000111 ***
    existing_loans_count             -0.180763   0.106837  -1.692             0.090654 .  
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    (Dispersion parameter for binomial family taken to be 1)

        Null deviance: 1099.56  on 899  degrees of freedom
    Residual deviance:  879.81  on 878  degrees of freedom
    AIC: 923.81

    Number of Fisher Scoring iterations: 5

#### Model Predictors

     [1] "checking_balance> 200 DM"       "checking_balance1 - 200 DM"     "checking_balanceunknown"        "credit_historygood"             "credit_historypoor"            
     [6] "credit_historyvery good"        "savings_balance100 - 500 DM"    "savings_balance500 - 1000 DM"   "savings_balanceunknown"         "employment_duration> 7 years"  
    [11] "employment_duration1 - 4 years" "employment_duration4 - 7 years" "employment_durationunemployed"  "other_creditnone"               "housingown"                    
    [16] "housingrent"                    "phoneTRUE"                      "months_loan_duration"           "amount"                         "percent_of_income"             
    [21] "existing_loans_count"          

#### Variable Importance

    glm variable importance

      only 20 most important variables shown (out of 21)

     Overall
      7.7252
      3.8659
      3.4816
      3.3585
      3.1618
      2.7677
      2.7531
      2.5271
      2.4051
      2.3412
      2.2941
      2.2185
      1.8022
      1.6920
      1.4412
      1.3492
      0.9446
      0.7164
      0.6670
      0.6640

### linear\_discriminant\_analsysis

#### Model Summary

                Length Class      Mode     
    prior        2     -none-     numeric  
    counts       2     -none-     numeric  
    means       58     -none-     numeric  
    scaling     29     -none-     numeric  
    lev          2     -none-     character
    svd          1     -none-     numeric  
    N            1     -none-     numeric  
    call         3     -none-     call     
    xNames      29     -none-     character
    problemType  1     -none-     character
    tuneValue    1     data.frame list     
    obsLevels    2     -none-     character
    param        0     -none-     list     

#### Model Predictors

     [1] "checking_balance> 200 DM"       "checking_balance1 - 200 DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historypoor"             "credit_historyvery good"        "purposecar"                     "purposeeducation"               "purposefurniture/appliances"   
    [11] "amount"                         "savings_balance100 - 500 DM"    "savings_balance500 - 1000 DM"   "savings_balanceunknown"         "employment_duration> 7 years"  
    [16] "employment_duration1 - 4 years" "employment_duration4 - 7 years" "employment_durationunemployed"  "percent_of_income"              "years_at_residence"            
    [21] "age"                            "other_creditnone"               "housingown"                     "housingrent"                    "existing_loans_count"          
    [26] "jobskilled"                     "jobunskilled"                   "dependents"                     "phoneTRUE"                     

#### Variable Importance

    ROC curve variable importance

     Importance
         0.6906
         0.6195
         0.6146
         0.5906
         0.5700
         0.5533
         0.5470
         0.5279
         0.5275
         0.5258
         0.5252
         0.5204
         0.5193
         0.5105
         0.5065
         0.5063

### linear\_discriminant\_analsysis\_remove\_collinear

> No collinear columns removed... skipping.

### linear\_discriminant\_analsysis\_remove\_collinear\_skew

#### Model Summary

                Length Class      Mode     
    prior        2     -none-     numeric  
    counts       2     -none-     numeric  
    means       58     -none-     numeric  
    scaling     29     -none-     numeric  
    lev          2     -none-     character
    svd          1     -none-     numeric  
    N            1     -none-     numeric  
    call         3     -none-     call     
    xNames      29     -none-     character
    problemType  1     -none-     character
    tuneValue    1     data.frame list     
    obsLevels    2     -none-     character
    param        0     -none-     list     

#### Model Predictors

     [1] "checking_balance> 200 DM"       "checking_balance1 - 200 DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historypoor"             "credit_historyvery good"        "purposecar"                     "purposeeducation"               "purposefurniture/appliances"   
    [11] "amount"                         "savings_balance100 - 500 DM"    "savings_balance500 - 1000 DM"   "savings_balanceunknown"         "employment_duration> 7 years"  
    [16] "employment_duration1 - 4 years" "employment_duration4 - 7 years" "employment_durationunemployed"  "percent_of_income"              "years_at_residence"            
    [21] "age"                            "other_creditnone"               "housingown"                     "housingrent"                    "existing_loans_count"          
    [26] "jobskilled"                     "jobunskilled"                   "dependents"                     "phoneTRUE"                     

#### Variable Importance

    ROC curve variable importance

     Importance
         0.6906
         0.6195
         0.6146
         0.5906
         0.5700
         0.5533
         0.5470
         0.5279
         0.5275
         0.5258
         0.5252
         0.5204
         0.5193
         0.5105
         0.5065
         0.5063

### partial\_least\_squares\_discriminant\_analysis

Data: X dimension: 900 29 Y dimension: 900 2 Fit method: oscorespls Number of components considered: 4 TRAINING: % variance explained

#### Model Predictors

     [1] "checking_balance> 200 DM"       "checking_balance1 - 200 DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historypoor"             "credit_historyvery good"        "purposecar"                     "purposeeducation"               "purposefurniture/appliances"   
    [11] "amount"                         "savings_balance100 - 500 DM"    "savings_balance500 - 1000 DM"   "savings_balanceunknown"         "employment_duration> 7 years"  
    [16] "employment_duration1 - 4 years" "employment_duration4 - 7 years" "employment_durationunemployed"  "percent_of_income"              "years_at_residence"            
    [21] "age"                            "other_creditnone"               "housingown"                     "housingrent"                    "existing_loans_count"          
    [26] "jobskilled"                     "jobunskilled"                   "dependents"                     "phoneTRUE"                     

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/partial_least_squares_discriminant_analysis-1.png" width="750px" />

#### Variable Importance

    pls variable importance

      only 20 most important variables shown (out of 29)

     Overall
     0.07640
     0.04441
     0.03569
     0.03420
     0.03307
     0.03221
     0.02983
     0.02795
     0.02299
     0.02299
     0.02162
     0.02091
     0.01912
     0.01669
     0.01606
     0.01457
     0.01339
     0.01202
     0.01105
     0.01049

### partial\_least\_squares\_discriminant\_analysis\_skew

Data: X dimension: 900 29 Y dimension: 900 2 Fit method: oscorespls Number of components considered: 4 TRAINING: % variance explained

#### Model Predictors

     [1] "checking_balance> 200 DM"       "checking_balance1 - 200 DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historypoor"             "credit_historyvery good"        "purposecar"                     "purposeeducation"               "purposefurniture/appliances"   
    [11] "amount"                         "savings_balance100 - 500 DM"    "savings_balance500 - 1000 DM"   "savings_balanceunknown"         "employment_duration> 7 years"  
    [16] "employment_duration1 - 4 years" "employment_duration4 - 7 years" "employment_durationunemployed"  "percent_of_income"              "years_at_residence"            
    [21] "age"                            "other_creditnone"               "housingown"                     "housingrent"                    "existing_loans_count"          
    [26] "jobskilled"                     "jobunskilled"                   "dependents"                     "phoneTRUE"                     

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/partial_least_squares_discriminant_analysis_skew-1.png" width="750px" />

#### Variable Importance

    pls variable importance

      only 20 most important variables shown (out of 29)

     Overall
     0.07757
     0.04637
     0.03396
     0.03391
     0.03246
     0.03047
     0.02846
     0.02615
     0.02488
     0.02391
     0.02239
     0.02073
     0.01960
     0.01679
     0.01644
     0.01405
     0.01357
     0.01241
     0.01114
     0.01100

### glmnet\_lasso\_ridge

#### Model Summary

                Length Class      Mode     
    a0            66   -none-     numeric  
    beta        1914   dgCMatrix  S4       
    df            66   -none-     numeric  
    dim            2   -none-     numeric  
    lambda        66   -none-     numeric  
    dev.ratio     66   -none-     numeric  
    nulldev        1   -none-     numeric  
    npasses        1   -none-     numeric  
    jerr           1   -none-     numeric  
    offset         1   -none-     logical  
    classnames     2   -none-     character
    call           5   -none-     call     
    nobs           1   -none-     numeric  
    lambdaOpt      1   -none-     numeric  
    xNames        29   -none-     character
    problemType    1   -none-     character
    tuneValue      2   data.frame list     
    obsLevels      2   -none-     character
    param          0   -none-     list     

#### Model Predictors

     [1] "checking_balance> 200 DM"       "checking_balance1 - 200 DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historyvery good"        "purposeeducation"               "purposefurniture/appliances"    "amount"                         "savings_balance500 - 1000 DM"  
    [11] "savings_balanceunknown"         "employment_duration> 7 years"   "employment_duration4 - 7 years" "percent_of_income"              "age"                           
    [16] "other_creditnone"               "housingown"                     "housingrent"                    "phoneTRUE"                     

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/glmnet_lasso_ridge-1.png" width="750px" />

#### Variable Importance

    glmnet variable importance

      only 20 most important variables shown (out of 29)

     Overall
     0.65776
     0.22805
     0.22683
     0.21889
     0.21305
     0.17986
     0.14292
     0.13728
     0.13057
     0.12062
     0.10410
     0.10060
     0.08770
     0.07662
     0.05505
     0.04112
     0.03713
     0.01972
     0.01924
     0.00000

### sparse\_lda

#### Model Summary

                Length Class      Mode     
    call         5     -none-     call     
    beta        22     -none-     numeric  
    theta        2     -none-     numeric  
    varNames    22     -none-     character
    varIndex    22     -none-     numeric  
    origP        1     -none-     numeric  
    rss          1     -none-     numeric  
    fit          8     lda        list     
    classes      2     -none-     character
    lambda       1     -none-     numeric  
    stop         1     -none-     numeric  
    xNames      29     -none-     character
    problemType  1     -none-     character
    tuneValue    2     data.frame list     
    obsLevels    2     -none-     character
    param        0     -none-     list     

#### Model Predictors

     [1] "checking_balance> 200 DM"       "checking_balance1 - 200 DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historypoor"             "credit_historyvery good"        "purposeeducation"               "purposefurniture/appliances"    "amount"                        
    [11] "savings_balance500 - 1000 DM"   "savings_balanceunknown"         "employment_duration> 7 years"   "employment_duration4 - 7 years" "percent_of_income"             
    [16] "age"                            "other_creditnone"               "housingown"                     "housingrent"                    "existing_loans_count"          
    [21] "dependents"                     "phoneTRUE"                     

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/sparse_lda-1.png" width="750px" />

#### Variable Importance

    ROC curve variable importance

     Importance
         0.6906
         0.6195
         0.6146
         0.5906
         0.5700
         0.5533
         0.5470
         0.5279
         0.5275
         0.5258
         0.5252
         0.5204
         0.5193
         0.5105
         0.5065
         0.5063

### nearest\_shrunken\_centroids

> was causing an error, turned off

### regularized\_discriminant\_analysis

#### Model Summary

                   Length Class      Mode     
    call              5   -none-     call     
    regularization    2   -none-     numeric  
    classes           2   -none-     character
    prior             2   -none-     numeric  
    error.rate        1   -none-     numeric  
    varnames         29   -none-     character
    means            58   -none-     numeric  
    covariances    1682   -none-     numeric  
    covpooled       841   -none-     numeric  
    converged         1   -none-     logical  
    iter              1   -none-     numeric  
    xNames           29   -none-     character
    problemType       1   -none-     character
    tuneValue         2   data.frame list     
    obsLevels         2   -none-     character
    param             0   -none-     list     

#### Model Predictors

     [1] "checking_balance> 200 DM"       "checking_balance1 - 200 DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historypoor"             "credit_historyvery good"        "purposecar"                     "purposeeducation"               "purposefurniture/appliances"   
    [11] "amount"                         "savings_balance100 - 500 DM"    "savings_balance500 - 1000 DM"   "savings_balanceunknown"         "employment_duration> 7 years"  
    [16] "employment_duration1 - 4 years" "employment_duration4 - 7 years" "employment_durationunemployed"  "percent_of_income"              "years_at_residence"            
    [21] "age"                            "other_creditnone"               "housingown"                     "housingrent"                    "existing_loans_count"          
    [26] "jobskilled"                     "jobunskilled"                   "dependents"                     "phoneTRUE"                     

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/regularized_discriminant_analysis-1.png" width="750px" />

#### Variable Importance

    ROC curve variable importance

     Importance
         0.6906
         0.6195
         0.6146
         0.5906
         0.5700
         0.5533
         0.5470
         0.5279
         0.5275
         0.5258
         0.5252
         0.5204
         0.5193
         0.5105
         0.5065
         0.5063

### regularized\_discriminant\_analysis\_rc

> No collinear columns removed... skipping.

### mixture\_discriminant\_analysis

#### Model Summary

                      Length Class      Mode     
    percent.explained  1     -none-     numeric  
    values             1     -none-     numeric  
    means              2     -none-     numeric  
    theta.mod          1     -none-     numeric  
    dimension          1     -none-     numeric  
    sub.prior          2     -none-     list     
    fit                5     polyreg    list     
    call               4     -none-     call     
    weights            2     -none-     list     
    prior              2     table      numeric  
    assign.theta       2     -none-     list     
    deviance           1     -none-     numeric  
    confusion          4     table      numeric  
    terms              3     terms      call     
    xNames            29     -none-     character
    problemType        1     -none-     character
    tuneValue          1     data.frame list     
    obsLevels          2     -none-     character
    param              0     -none-     list     

#### Model Predictors

     [1] "checking_balance> 200 DM"       "checking_balance1 - 200 DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historypoor"             "credit_historyvery good"        "purposecar"                     "purposeeducation"               "purposefurniture/appliances"   
    [11] "amount"                         "savings_balance100 - 500 DM"    "savings_balance500 - 1000 DM"   "savings_balanceunknown"         "employment_duration> 7 years"  
    [16] "employment_duration1 - 4 years" "employment_duration4 - 7 years" "employment_durationunemployed"  "percent_of_income"              "years_at_residence"            
    [21] "age"                            "other_creditnone"               "housingown"                     "housingrent"                    "existing_loans_count"          
    [26] "jobskilled"                     "jobunskilled"                   "dependents"                     "phoneTRUE"                     

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/mixture_discriminant_analysis-1.png" width="750px" />

#### Variable Importance

    ROC curve variable importance

     Importance
         0.6906
         0.6195
         0.6146
         0.5906
         0.5700
         0.5533
         0.5470
         0.5279
         0.5275
         0.5258
         0.5252
         0.5204
         0.5193
         0.5105
         0.5065
         0.5063

### mixture\_discriminant\_analysis\_rc

> No collinear columns removed... skipping.

### neural\_network\_spatial\_rc

#### Model Summary

    a 29-1-1 network with 32 weights
    options were - entropy fitting  decay=1
      b->h1  i1->h1  i2->h1  i3->h1  i4->h1  i5->h1  i6->h1  i7->h1  i8->h1  i9->h1 i10->h1 i11->h1 i12->h1 i13->h1 i14->h1 i15->h1 i16->h1 i17->h1 i18->h1 i19->h1 i20->h1 i21->h1 
      -0.05    0.71    0.11    2.67   -1.18   -0.54   -0.31   -1.04    0.04   -0.44    0.52   -1.13    0.08    0.34    0.97    0.63    0.03    0.83   -0.32   -0.96    0.03    0.50 
    i22->h1 i23->h1 i24->h1 i25->h1 i26->h1 i27->h1 i28->h1 i29->h1 
       0.78    0.72   -0.38   -0.11    0.02    0.18   -0.10    0.60 
     b->o h1->o 
    -1.18  4.39 

#### Model Predictors

    Loading required package: nnet

     [1] "checking_balance> 200 DM"       "checking_balance1 - 200 DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historypoor"             "credit_historyvery good"        "purposecar"                     "purposeeducation"               "purposefurniture/appliances"   
    [11] "amount"                         "savings_balance100 - 500 DM"    "savings_balance500 - 1000 DM"   "savings_balanceunknown"         "employment_duration> 7 years"  
    [16] "employment_duration1 - 4 years" "employment_duration4 - 7 years" "employment_durationunemployed"  "percent_of_income"              "years_at_residence"            
    [21] "age"                            "other_creditnone"               "housingown"                     "housingrent"                    "existing_loans_count"          
    [26] "jobskilled"                     "jobunskilled"                   "dependents"                     "phoneTRUE"                     

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/neural_network_spatial_rc-1.png" width="750px" />

#### Variable Importance

    nnet variable importance

      only 20 most important variables shown (out of 29)

     Overall
      16.370
       7.252
       6.960
       6.389
       5.955
       5.869
       5.108
       4.761
       4.415
       4.384
       3.841
       3.698
       3.331
       3.192
       3.087
       2.709
       2.360
       2.090
       1.972
       1.925

### neural\_network\_spatial\_rc\_skew

#### Model Summary

    a 29-1-1 network with 32 weights
    options were - entropy fitting  decay=1
      b->h1  i1->h1  i2->h1  i3->h1  i4->h1  i5->h1  i6->h1  i7->h1  i8->h1  i9->h1 i10->h1 i11->h1 i12->h1 i13->h1 i14->h1 i15->h1 i16->h1 i17->h1 i18->h1 i19->h1 i20->h1 i21->h1 
      -0.08    0.74    0.01    2.69   -1.40   -0.52   -0.31   -1.03    0.02   -0.46    0.59   -0.60    0.10    0.34    0.96    0.60    0.00    0.80   -0.30   -0.81    0.08    0.58 
    i22->h1 i23->h1 i24->h1 i25->h1 i26->h1 i27->h1 i28->h1 i29->h1 
       0.81    0.77   -0.30   -0.07    0.13    0.23   -0.14    0.51 
     b->o h1->o 
    -1.11  4.32 

#### Model Predictors

     [1] "checking_balance> 200 DM"       "checking_balance1 - 200 DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historypoor"             "credit_historyvery good"        "purposecar"                     "purposeeducation"               "purposefurniture/appliances"   
    [11] "amount"                         "savings_balance100 - 500 DM"    "savings_balance500 - 1000 DM"   "savings_balanceunknown"         "employment_duration> 7 years"  
    [16] "employment_duration1 - 4 years" "employment_duration4 - 7 years" "employment_durationunemployed"  "percent_of_income"              "years_at_residence"            
    [21] "age"                            "other_creditnone"               "housingown"                     "housingrent"                    "existing_loans_count"          
    [26] "jobskilled"                     "jobunskilled"                   "dependents"                     "phoneTRUE"                     

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/neural_network_spatial_rc_skew-1.png" width="750px" />

#### Variable Importance

    nnet variable importance

      only 20 most important variables shown (out of 29)

     Overall
      16.945
       8.837
       6.512
       6.060
       5.101
       5.091
       5.020
       4.845
       4.631
       3.756
       3.753
       3.694
       3.671
       3.250
       3.201
       2.877
       2.148
       1.966
       1.913
       1.875

### flexible\_discriminant\_analsysis

#### Model Summary

                      Length Class      Mode     
    percent.explained  1     -none-     numeric  
    values             1     -none-     numeric  
    means              2     -none-     numeric  
    theta.mod          1     -none-     numeric  
    dimension          1     -none-     numeric  
    prior              2     table      numeric  
    fit               29     earth      list     
    call               7     -none-     call     
    terms              3     terms      call     
    confusion          4     table      numeric  
    xNames            29     -none-     character
    problemType        1     -none-     character
    tuneValue          2     data.frame list     
    obsLevels          2     -none-     character
    param              0     -none-     list     

#### Model Predictors

    Loading required package: earth

    Loading required package: plotmo

    Loading required package: plotrix


    Attaching package: 'plotrix'

    The following object is masked from 'package:gplots':

        plotCI

    The following object is masked from 'package:scales':

        rescale

    The following object is masked from 'package:psych':

        rescale

    Loading required package: TeachingDemos


    Attaching package: 'TeachingDemos'

    The following object is masked from 'package:klaR':

        triplot

    The following objects are masked from 'package:Hmisc':

        cnvrt.coords, subplot

     [1] "checking_balanceunknown"       "months_loan_duration"          "amount"                        "savings_balanceunknown"        "`checking_balance>200DM`"     
     [6] "housingown"                    "`credit_historyverygood`"      "`employment_duration4-7years`" "`checking_balance1-200DM`"     "percent_of_income"            
    [11] "`employment_duration>7years`"  "phoneTRUE"                    

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/flexible_discriminant_analsysis-1.png" width="750px" />

#### Variable Importance

    fda variable importance

      only 20 most important variables shown (out of 29)

     Overall
      100.00
       64.00
       54.82
       52.24
       48.51
       47.77
       41.48
       35.00
       23.89
       21.20
       18.69
       15.11
        0.00
        0.00
        0.00
        0.00
        0.00
        0.00
        0.00
        0.00

### svm\_linear

#### Model Summary


    Call:
    svm.default(x = as.matrix(x), y = y, kernel = "linear", cost = param$cost, probability = classProbs)


    Parameters:
       SVM-Type:  C-classification 
     SVM-Kernel:  linear 
           cost:  0.5 
          gamma:  0.02857143 

    Number of Support Vectors:  500

     ( 256 244 )


    Number of Classes:  2 

    Levels: 
     yes no

#### Model Predictors

     [1] "checking_balance> 200 DM"       "checking_balance1 - 200 DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historyperfect"          "credit_historypoor"             "credit_historyvery good"        "purposecar"                     "purposecar0"                   
    [11] "purposeeducation"               "purposefurniture/appliances"    "purposerenovations"             "amount"                         "savings_balance> 1000 DM"      
    [16] "savings_balance100 - 500 DM"    "savings_balance500 - 1000 DM"   "savings_balanceunknown"         "employment_duration> 7 years"   "employment_duration1 - 4 years"
    [21] "employment_duration4 - 7 years" "employment_durationunemployed"  "percent_of_income"              "years_at_residence"             "age"                           
    [26] "other_creditnone"               "other_creditstore"              "housingown"                     "housingrent"                    "existing_loans_count"          
    [31] "jobskilled"                     "jobunemployed"                  "jobunskilled"                   "dependents"                     "phoneTRUE"                     

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/svm_linear-1.png" width="750px" />

#### Variable Importance

    ROC curve variable importance

     Importance
         0.6906
         0.6195
         0.6146
         0.5906
         0.5700
         0.5533
         0.5470
         0.5279
         0.5275
         0.5258
         0.5252
         0.5204
         0.5193
         0.5105
         0.5065
         0.5063

### svm\_polynomial

#### Model Summary

    Length  Class   Mode 
         1   ksvm     S4 

#### Model Predictors

    Loading required package: kernlab


    Attaching package: 'kernlab'

    The following object is masked from 'package:scales':

        alpha

    The following object is masked from 'package:psych':

        alpha

    The following object is masked from 'package:ggplot2':

        alpha

    [1] NA

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/svm_polynomial-1.png" width="750px" />

#### Variable Importance

    ROC curve variable importance

     Importance
         0.6906
         0.6195
         0.6146
         0.5906
         0.5700
         0.5533
         0.5470
         0.5279
         0.5275
         0.5258
         0.5252
         0.5204
         0.5193
         0.5105
         0.5065
         0.5063

### svm\_radial

#### Model Summary

    Length  Class   Mode 
         1   ksvm     S4 

#### Model Predictors

     [1] "checking_balance..200.DM"       "checking_balance1...200.DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historyperfect"          "credit_historypoor"             "credit_historyvery.good"        "purposecar"                     "purposecar0"                   
    [11] "purposeeducation"               "purposefurniture.appliances"    "purposerenovations"             "amount"                         "savings_balance..1000.DM"      
    [16] "savings_balance100...500.DM"    "savings_balance500...1000.DM"   "savings_balanceunknown"         "employment_duration..7.years"   "employment_duration1...4.years"
    [21] "employment_duration4...7.years" "employment_durationunemployed"  "percent_of_income"              "years_at_residence"             "age"                           
    [26] "other_creditnone"               "other_creditstore"              "housingown"                     "housingrent"                    "existing_loans_count"          
    [31] "jobskilled"                     "jobunemployed"                  "jobunskilled"                   "dependents"                     "phoneTRUE"                     

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/svm_radial-1.png" width="750px" />

#### Variable Importance

    ROC curve variable importance

     Importance
         0.6906
         0.6195
         0.6146
         0.5906
         0.5700
         0.5533
         0.5470
         0.5279
         0.5275
         0.5258
         0.5252
         0.5204
         0.5193
         0.5105
         0.5065
         0.5063

### k\_nearest\_neighbors

#### Model Summary

                Length Class      Mode     
    learn        2     -none-     list     
    k            1     -none-     numeric  
    theDots      0     -none-     list     
    xNames      35     -none-     character
    problemType  1     -none-     character
    tuneValue    1     data.frame list     
    obsLevels    2     -none-     character
    param        0     -none-     list     

#### Model Predictors

     [1] "checking_balance> 200 DM"       "checking_balance1 - 200 DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historyperfect"          "credit_historypoor"             "credit_historyvery good"        "purposecar"                     "purposecar0"                   
    [11] "purposeeducation"               "purposefurniture/appliances"    "purposerenovations"             "amount"                         "savings_balance> 1000 DM"      
    [16] "savings_balance100 - 500 DM"    "savings_balance500 - 1000 DM"   "savings_balanceunknown"         "employment_duration> 7 years"   "employment_duration1 - 4 years"
    [21] "employment_duration4 - 7 years" "employment_durationunemployed"  "percent_of_income"              "years_at_residence"             "age"                           
    [26] "other_creditnone"               "other_creditstore"              "housingown"                     "housingrent"                    "existing_loans_count"          
    [31] "jobskilled"                     "jobunemployed"                  "jobunskilled"                   "dependents"                     "phoneTRUE"                     

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/k_nearest_neighbors-1.png" width="750px" />

#### Variable Importance

    ROC curve variable importance

     Importance
         0.6906
         0.6195
         0.6146
         0.5906
         0.5700
         0.5533
         0.5470
         0.5279
         0.5275
         0.5258
         0.5252
         0.5204
         0.5193
         0.5105
         0.5065
         0.5063

### naive\_bayes

#### Model Summary

                Length Class      Mode     
    apriori      2     table      numeric  
    tables      29     -none-     list     
    levels       2     -none-     character
    call         6     -none-     call     
    x           29     data.frame list     
    usekernel    1     -none-     logical  
    varnames    29     -none-     character
    xNames      29     -none-     character
    problemType  1     -none-     character
    tuneValue    3     data.frame list     
    obsLevels    2     -none-     character
    param        0     -none-     list     

#### Model Predictors

     [1] "checking_balance..200.DM"       "checking_balance1...200.DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historypoor"             "credit_historyvery.good"        "purposecar"                     "purposeeducation"               "purposefurniture.appliances"   
    [11] "amount"                         "savings_balance100...500.DM"    "savings_balance500...1000.DM"   "savings_balanceunknown"         "employment_duration..7.years"  
    [16] "employment_duration1...4.years" "employment_duration4...7.years" "employment_durationunemployed"  "percent_of_income"              "years_at_residence"            
    [21] "age"                            "other_creditnone"               "housingown"                     "housingrent"                    "existing_loans_count"          
    [26] "jobskilled"                     "jobunskilled"                   "dependents"                     "phoneTRUE"                     

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/naive_bayes-1.png" width="750px" />

#### Variable Importance

    ROC curve variable importance

     Importance
         0.6906
         0.6195
         0.6146
         0.5906
         0.5700
         0.5533
         0.5470
         0.5279
         0.5275
         0.5258
         0.5252
         0.5204
         0.5193
         0.5105
         0.5065
         0.5063

### rpart\_independent\_categories

> See APM pg 373/405 for descriptions on independent categories (binary dummy variables) vs grouped categories
>
> When you use the formula interface, most modeling functions (including train, lm, glm, etc) internally run model.matrix to process the data set. This will create dummy variables from any factor variables. The non-formula interface does not \[1\]. <https://stackoverflow.com/questions/22200923/different-results-with-formula-and-non-formula-for-caret-training>

#### Model Summary

    CART 

    900 samples
     16 predictor
      2 classes: 'yes', 'no' 

    No pre-processing
    Resampling: Cross-Validated (10 fold, repeated 3 times) 
    Summary of sample sizes: 810, 810, 810, 810, 810, 810, ... 
    Resampling results across tuning parameters:

      cp           ROC        Sens        Spec     
      0.000000000  0.6996865  0.42962963  0.7984127
      0.001404853  0.6968352  0.42222222  0.8042328
      0.002809706  0.6966588  0.40493827  0.8142857
      0.004214559  0.7011268  0.37777778  0.8301587
      0.005619413  0.7051342  0.36913580  0.8412698
      0.007024266  0.7053302  0.36049383  0.8465608
      0.008429119  0.7082500  0.35308642  0.8576720
      0.009833972  0.7082109  0.35432099  0.8576720
      0.011238825  0.7050656  0.34444444  0.8640212
      0.012643678  0.7096610  0.33703704  0.8703704
      0.014048531  0.7099157  0.33456790  0.8730159
      0.015453384  0.7083970  0.32962963  0.8756614
      0.016858238  0.7068097  0.32962963  0.8708995
      0.018263091  0.7068097  0.32962963  0.8708995
      0.019667944  0.7031550  0.31481481  0.8740741
      0.021072797  0.6947874  0.28148148  0.8820106
      0.022477650  0.6869684  0.26913580  0.8873016
      0.023882503  0.6792279  0.24567901  0.9010582
      0.025287356  0.6668234  0.20864198  0.9190476
      0.026692209  0.6590927  0.20246914  0.9211640
      0.028097063  0.6596904  0.19382716  0.9264550
      0.029501916  0.6578189  0.18888889  0.9259259
      0.030906769  0.6431119  0.16296296  0.9349206
      0.032311622  0.6431119  0.16296296  0.9349206
      0.033716475  0.6187929  0.14074074  0.9402116
      0.035121328  0.5844503  0.10123457  0.9539683
      0.036526181  0.5844503  0.10123457  0.9539683
      0.037931034  0.5715168  0.08641975  0.9582011
      0.039335888  0.5572996  0.06790123  0.9661376
      0.040740741  0.5572996  0.06790123  0.9661376

    ROC was used to select the optimal model using  the largest value.
    The final value used for the model was cp = 0.01404853.
    NULL

#### Model Predictors

    Loading required package: rpart

     [1] "checking_balanceunknown"       "months_loan_duration"          "employment_durationunemployed" "years_at_residence"            "amount"                       
     [6] "credit_historyvery good"       "purposecar"                    "checking_balance1 - 200 DM"    "savings_balance500 - 1000 DM"  "savings_balanceunknown"       
    [11] "savings_balance> 1000 DM"      "purposecar0"                   "age"                           "purposeeducation"              "existing_loans_count"         
    [16] "purposefurniture/appliances"   "credit_historyperfect"         "percent_of_income"            

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/rpart_independent_categories-1.png" width="750px" />

#### Variable Importance

    rpart variable importance

      only 20 most important variables shown (out of 39)

     Overall
      48.515
      38.796
      35.218
      23.962
      19.750
      12.775
      12.345
       7.802
       5.558
       5.554
       5.024
       4.283
       4.128
       3.874
       2.962
       2.949
       2.495
       2.420
       0.000
       0.000

<img src="predictive_analysis_classification_files/figure-markdown_github/rpart_independent_categories-2.png" width="750px" />

### rpart\_grouped\_categories

#### Model Summary

    CART 

    900 samples
     16 predictor
      2 classes: 'yes', 'no' 

    No pre-processing
    Resampling: Cross-Validated (10 fold, repeated 3 times) 
    Summary of sample sizes: 810, 810, 810, 810, 810, 810, ... 
    Resampling results across tuning parameters:

      cp           ROC        Sens       Spec     
      0.000000000  0.7124633  0.4271605  0.8407407
      0.001660281  0.7126984  0.4234568  0.8455026
      0.003320562  0.7123163  0.4172840  0.8507937
      0.004980843  0.7158044  0.4012346  0.8513228
      0.006641124  0.7103860  0.4000000  0.8613757
      0.008301405  0.6988830  0.3901235  0.8629630
      0.009961686  0.6997452  0.3839506  0.8656085
      0.011621967  0.7066725  0.3802469  0.8730159
      0.013282248  0.7080443  0.3814815  0.8767196
      0.014942529  0.7114540  0.3790123  0.8777778
      0.016602810  0.7134039  0.3950617  0.8735450
      0.018263091  0.7134039  0.3950617  0.8735450
      0.019923372  0.7128650  0.3925926  0.8708995
      0.021583653  0.7141289  0.3876543  0.8756614
      0.023243934  0.7119048  0.3691358  0.8804233
      0.024904215  0.7127278  0.3432099  0.8920635
      0.026564496  0.7131981  0.3407407  0.8925926
      0.028224777  0.7131981  0.3407407  0.8925926
      0.029885057  0.7124437  0.3407407  0.8899471
      0.031545338  0.7139918  0.3382716  0.8931217
      0.033205619  0.6895356  0.2975309  0.9042328
      0.034865900  0.6895356  0.2975309  0.9042328
      0.036526181  0.6895356  0.2975309  0.9042328
      0.038186462  0.6894964  0.2925926  0.9063492
      0.039846743  0.6894964  0.2925926  0.9063492
      0.041507024  0.6807858  0.2753086  0.9084656
      0.043167305  0.6807858  0.2753086  0.9084656
      0.044827586  0.6672154  0.2555556  0.9158730
      0.046487867  0.6519988  0.2185185  0.9232804
      0.048148148  0.6170096  0.1703704  0.9407407

    ROC was used to select the optimal model using  the largest value.
    The final value used for the model was cp = 0.004980843.
    NULL

#### Model Predictors

     [1] "checking_balance"     "months_loan_duration" "employment_duration"  "savings_balance"      "age"                  "percent_of_income"    "credit_history"      
     [8] "housing"              "amount"               "purpose"              "existing_loans_count" "years_at_residence"   "other_credit"         "job"                 
    [15] "dependents"           "phone"               

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/rpart_grouped_categories-1.png" width="750px" />

#### Variable Importance

    rpart variable importance

     Overall
      82.021
      53.922
      51.051
      49.945
      46.431
      45.612
      42.355
      30.746
      18.387
      14.663
      14.032
      11.601
       9.531
       8.122
       7.171
       0.000

<img src="predictive_analysis_classification_files/figure-markdown_github/rpart_grouped_categories-2.png" width="750px" />

### treebag\_independent\_categories

#### Model Summary

    Bagged CART 

    900 samples
     16 predictor
      2 classes: 'yes', 'no' 

    No pre-processing
    Resampling: Cross-Validated (10 fold, repeated 3 times) 
    Summary of sample sizes: 810, 810, 810, 810, 810, 810, ... 
    Resampling results:

      ROC        Sens       Spec     
      0.7522438  0.4703704  0.8465608

    NULL

#### Model Predictors

    Loading required package: ipred

     [1] "checking_balanceunknown"        "months_loan_duration"           "phoneTRUE"                      "percent_of_income"              "housingown"                    
     [6] "age"                            "amount"                         "existing_loans_count"           "checking_balance1 - 200 DM"     "savings_balance> 1000 DM"      
    [11] "years_at_residence"             "employment_duration> 7 years"   "jobskilled"                     "employment_duration4 - 7 years" "employment_duration1 - 4 years"
    [16] "housingrent"                    "jobunskilled"                   "purposefurniture/appliances"    "credit_historygood"             "checking_balance> 200 DM"      
    [21] "credit_historyperfect"          "purposecar"                     "credit_historyvery good"        "purposeeducation"               "dependents"                    
    [26] "credit_historypoor"             "purposerenovations"             "other_creditnone"               "savings_balanceunknown"         "savings_balance100 - 500 DM"   
    [31] "savings_balance500 - 1000 DM"   "purposecar0"                    "employment_durationunemployed"  "jobunemployed"                  "other_creditstore"             

#### Variable Importance

    treebag variable importance

      only 20 most important variables shown (out of 45)

     Overall
      196.14
      136.57
      133.42
       64.97
       59.69
       39.99
       36.03
       35.23
       32.05
       30.71
       27.57
       26.32
       25.32
       24.81
       24.64
       24.50
       24.06
       23.87
       23.27
       22.49

### treebag\_grouped\_categories

#### Model Summary

    Bagged CART 

    900 samples
     16 predictor
      2 classes: 'yes', 'no' 

    No pre-processing
    Resampling: Cross-Validated (10 fold, repeated 3 times) 
    Summary of sample sizes: 810, 810, 810, 810, 810, 810, ... 
    Resampling results:

      ROC        Sens       Spec    
      0.7624535  0.4740741  0.852381

    NULL

#### Model Predictors

     [1] "checking_balance"     "months_loan_duration" "savings_balance"      "employment_duration"  "purpose"              "percent_of_income"    "age"                 
     [8] "amount"               "credit_history"       "job"                  "years_at_residence"   "existing_loans_count" "housing"              "phone"               
    [15] "other_credit"         "dependents"          

#### Variable Importance

    treebag variable importance

     Overall
      185.55
      134.89
      131.73
       97.96
       97.16
       90.85
       85.83
       79.24
       60.97
       59.13
       50.89
       45.28
       35.36
       26.27
       19.84
       13.43

### c50\_model\_independent\_categories

#### Model Summary

                 Length Class      Mode     
    names          1    -none-     character
    cost           1    -none-     character
    costMatrix     0    -none-     NULL     
    caseWeights    1    -none-     logical  
    control       11    -none-     list     
    trials         2    -none-     numeric  
    rbm            1    -none-     logical  
    boostResults   5    data.frame list     
    size         100    -none-     numeric  
    dims           2    -none-     numeric  
    call           7    -none-     call     
    levels         2    -none-     character
    output         1    -none-     character
    tree           1    -none-     character
    predictors    35    -none-     character
    rules          1    -none-     character
    xNames        35    -none-     character
    problemType    1    -none-     character
    tuneValue      3    data.frame list     
    obsLevels      2    -none-     character
    param          0    -none-     list     

#### Model Predictors

    Loading required package: C50

     [1] "checking_balanceunknown"       "months_loan_duration"          "amount"                        "savings_balanceunknown"        "employment_duration4"         
     [6] "age"                           "percent_of_income"             "credit_historyvery"            "checking_balance>"             "dependents"                   
    [11] "credit_historypoor"            "purposeeducation"              "savings_balance>"              "credit_historyperfect"         "savings_balance100"           
    [16] "years_at_residence"            "employment_duration>"          "credit_historygood"            "other_creditnone"              "employment_durationunemployed"
    [21] "existing_loans_count"          "housingrent"                   "checking_balance1"             "housingown"                    "purposecar"                   
    [26] "jobunskilled"                  "phoneTRUE"                     "jobskilled"                    "savings_balance500"            "purposefurniture/appliances"  
    [31] "other_creditstore"             "employment_duration1"          "purposerenovations"            "purposecar0"                   "jobunemployed"                

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/c50_model_independent_categories-1.png" width="750px" />

#### Variable Importance

    C5.0 variable importance

      only 20 most important variables shown (out of 44)

     Overall
         100
         100
         100
         100
         100
         100
         100
         100
         100
         100
         100
         100
         100
         100
         100
         100
         100
         100
         100
         100

### c50\_model\_grouped\_categories

#### Model Summary


    Call:
    C5.0.default(x = structure(list(checking_balance = structure(c(1L, 3L, 4L, 1L, 1L, 4L, 4L, 3L, 4L, 3L, 1L, 3L, 1L, 1L, 4L, 1L, 3L, 4L, 4L, 1L, 1L, 3L, 4L, 1L, 4L, 2L, 3L, 3L,
     0, earlyStopping = TRUE, label = "outcome", seed = 3515L), .Names = c("subset", "bands", "winnow", "noGlobalPruning", "CF", "minCases", "fuzzyThreshold",
     "sample", "earlyStopping", "label", "seed")))


    C5.0 [Release 2.07 GPL Edition]     Tue Jun 27 21:54:06 2017
    -------------------------------

    Class specified by attribute `outcome'

    Read 900 cases (17 attributes) from undefined.data

    -----  Trial 0:  -----

    Rules:

    Rule 0/1: (10, lift 3.1)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 36
        amount <= 7980
        employment_duration in {< 1 year, > 7 years}
        ->  class yes  [0.917]

    Rule 0/2: (10, lift 3.1)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 36
        credit_history in {critical, good}
        employment_duration = 1 - 4 years
        ->  class yes  [0.917]

    Rule 0/3: (8, lift 3.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {critical, good}
        purpose = car
        other_credit in {bank, store}
        phone = FALSE
        ->  class yes  [0.900]

    Rule 0/4: (7, lift 3.0)
        credit_history in {good, poor}
        amount <= 2101
        other_credit = store
        phone = FALSE
        ->  class yes  [0.889]

    Rule 0/5: (6, lift 2.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 11
        purpose = car
        amount <= 5179
        savings_balance = < 100 DM
        employment_duration = > 7 years
        age <= 50
        phone = FALSE
        ->  class yes  [0.875]

    Rule 0/6: (6, lift 2.9)
        checking_balance = unknown
        purpose = business
        employment_duration in {< 1 year, 1 - 4 years, unemployed}
        percent_of_income > 1
        other_credit in {bank, store}
        ->  class yes  [0.875]

    Rule 0/7: (6, lift 2.9)
        checking_balance in {> 200 DM, unknown}
        months_loan_duration > 16
        purpose = car
        percent_of_income > 1
        other_credit = bank
        phone = FALSE
        ->  class yes  [0.875]

    Rule 0/8: (27/3, lift 2.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {critical, good}
        amount > 7980
        ->  class yes  [0.862]

    Rule 0/9: (5, lift 2.9)
        checking_balance = < 0 DM
        credit_history in {good, poor}
        purpose = education
        phone = FALSE
        ->  class yes  [0.857]

    Rule 0/10: (12/1, lift 2.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 16
        purpose = furniture/appliances
        savings_balance = < 100 DM
        percent_of_income > 1
        years_at_residence <= 3
        other_credit = none
        existing_loans_count <= 1
        job = skilled
        phone = FALSE
        ->  class yes  [0.857]

    Rule 0/11: (5, lift 2.9)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        employment_duration in {1 - 4 years, 4 - 7 years}
        years_at_residence > 3
        housing = rent
        job = skilled
        ->  class yes  [0.857]

    Rule 0/12: (5, lift 2.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 36
        savings_balance = < 100 DM
        job = unemployed
        phone = FALSE
        ->  class yes  [0.857]

    Rule 0/13: (5, lift 2.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = poor
        savings_balance = < 100 DM
        percent_of_income > 3
        phone = TRUE
        ->  class yes  [0.857]

    Rule 0/14: (17/2, lift 2.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 36
        credit_history in {critical, good, poor}
        purpose in {business, car0, education, furniture/appliances}
        amount <= 7980
        age <= 37
        ->  class yes  [0.842]

    Rule 0/15: (4, lift 2.8)
        checking_balance = < 0 DM
        credit_history = critical
        savings_balance = < 100 DM
        housing = rent
        phone = TRUE
        ->  class yes  [0.833]

    Rule 0/16: (15/2, lift 2.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 11
        months_loan_duration <= 36
        purpose = car
        amount <= 5179
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 4 - 7 years}
        phone = FALSE
        ->  class yes  [0.824]

    Rule 0/17: (9/1, lift 2.7)
        checking_balance = < 0 DM
        credit_history = good
        savings_balance = unknown
        job = skilled
        phone = FALSE
        ->  class yes  [0.818]

    Rule 0/18: (8/1, lift 2.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 11
        amount > 1829
        age > 30
        other_credit = none
        job = unskilled
        phone = FALSE
        ->  class yes  [0.800]

    Rule 0/19: (12/2, lift 2.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        phone = FALSE
        ->  class yes  [0.786]

    Rule 0/20: (7/1, lift 2.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 11
        credit_history = good
        savings_balance = < 100 DM
        percent_of_income > 2
        percent_of_income <= 3
        phone = TRUE
        ->  class yes  [0.778]

    Rule 0/21: (46/11, lift 2.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {perfect, very good}
        savings_balance in {< 100 DM, > 1000 DM}
        ->  class yes  [0.750]

    Rule 0/22: (9/2, lift 2.4)
        checking_balance = 1 - 200 DM
        credit_history in {perfect, very good}
        savings_balance = 100 - 500 DM
        ->  class yes  [0.727]

    Rule 0/23: (481/267, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        ->  class yes  [0.445]

    Rule 0/24: (100/6, lift 1.3)
        checking_balance in {> 200 DM, unknown}
        age > 44
        ->  class no  [0.931]

    Rule 0/25: (62/4, lift 1.3)
        checking_balance in {> 200 DM, unknown}
        months_loan_duration <= 16
        purpose = car
        ->  class no  [0.922]

    Rule 0/26: (20/1, lift 1.3)
        checking_balance = 1 - 200 DM
        savings_balance = unknown
        housing in {other, own}
        phone = FALSE
        ->  class no  [0.909]

    Rule 0/27: (19/1, lift 1.3)
        months_loan_duration > 16
        amount <= 7980
        savings_balance = 100 - 500 DM
        phone = TRUE
        ->  class no  [0.905]

    Rule 0/28: (349/35, lift 1.3)
        checking_balance in {> 200 DM, unknown}
        other_credit = none
        ->  class no  [0.897]

    Rule 0/29: (147/18, lift 1.2)
        months_loan_duration <= 11
        credit_history in {critical, good, poor}
        amount <= 7980
        ->  class no  [0.872]

    Rule 0/30: (740/244, lift 1.0)
        months_loan_duration > 11
        ->  class no  [0.670]

    Default class: no

    -----  Trial 1:  -----

    Rules:

    Rule 1/1: (7.4, lift 2.2)
        checking_balance = unknown
        credit_history in {good, poor}
        employment_duration in {< 1 year, 1 - 4 years}
        housing = rent
        existing_loans_count > 1
        ->  class yes  [0.893]

    Rule 1/2: (7.3, lift 2.2)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        employment_duration = 1 - 4 years
        percent_of_income <= 3
        housing = other
        ->  class yes  [0.892]

    Rule 1/3: (6.5, lift 2.2)
        employment_duration = unemployed
        years_at_residence <= 1
        ->  class yes  [0.882]

    Rule 1/4: (11.3/0.8, lift 2.1)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 8
        amount <= 1569
        employment_duration = < 1 year
        percent_of_income <= 3
        ->  class yes  [0.866]

    Rule 1/5: (4.9, lift 2.1)
        checking_balance = unknown
        credit_history in {good, poor}
        percent_of_income <= 1
        existing_loans_count > 1
        job = skilled
        ->  class yes  [0.855]

    Rule 1/6: (17.1/2.4, lift 2.0)
        checking_balance = unknown
        credit_history in {good, poor}
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years, unemployed}
        age <= 24
        phone = FALSE
        ->  class yes  [0.824]

    Rule 1/7: (20.1/3.1, lift 2.0)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        employment_duration = < 1 year
        percent_of_income <= 3
        housing = rent
        ->  class yes  [0.813]

    Rule 1/8: (15.3/3.1, lift 1.9)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 8
        employment_duration = > 7 years
        percent_of_income <= 3
        dependents > 1
        ->  class yes  [0.761]

    Rule 1/9: (11.8/2.4, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 42
        employment_duration = 1 - 4 years
        ->  class yes  [0.757]

    Rule 1/10: (126.8/37, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 11
        savings_balance = < 100 DM
        percent_of_income > 3
        years_at_residence > 1
        ->  class yes  [0.705]

    Rule 1/11: (25/7.1, lift 1.7)
        employment_duration = > 7 years
        percent_of_income > 1
        age <= 41
        other_credit in {bank, store}
        ->  class yes  [0.700]

    Rule 1/12: (21.6/6.3, lift 1.7)
        checking_balance = unknown
        employment_duration in {1 - 4 years, unemployed}
        percent_of_income > 1
        other_credit in {bank, store}
        ->  class yes  [0.691]

    Rule 1/13: (129.3/64.8, lift 1.2)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        employment_duration = > 7 years
        ->  class yes  [0.499]

    Rule 1/14: (674.9/388.9, lift 1.0)
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years, unemployed}
        ->  class yes  [0.424]

    Rule 1/15: (43.4/2.5, lift 1.6)
        checking_balance = unknown
        employment_duration = > 7 years
        age > 41
        ->  class no  [0.924]

    Rule 1/16: (102.6/7.4, lift 1.6)
        checking_balance = unknown
        credit_history in {critical, perfect}
        other_credit = none
        ->  class no  [0.920]

    Rule 1/17: (8, lift 1.5)
        months_loan_duration > 8
        months_loan_duration <= 11
        savings_balance = < 100 DM
        percent_of_income > 3
        years_at_residence > 3
        ->  class no  [0.900]

    Rule 1/18: (63.9/8.8, lift 1.4)
        months_loan_duration <= 8
        credit_history in {critical, good, poor}
        ->  class no  [0.851]

    Rule 1/19: (88.8/19.6, lift 1.3)
        checking_balance = unknown
        employment_duration in {< 1 year, 4 - 7 years}
        percent_of_income > 1
        ->  class no  [0.774]

    Rule 1/20: (73/19.5, lift 1.2)
        credit_history in {critical, good, poor, very good}
        savings_balance in {> 1000 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        percent_of_income > 3
        years_at_residence > 3
        ->  class no  [0.727]

    Rule 1/21: (172.2/52.4, lift 1.2)
        employment_duration = > 7 years
        dependents <= 1
        ->  class no  [0.694]

    Rule 1/22: (461.8/166, lift 1.1)
        percent_of_income <= 3
        ->  class no  [0.640]

    Default class: no

    -----  Trial 2:  -----

    Rules:

    Rule 2/1: (11.2, lift 2.2)
        checking_balance in {> 200 DM, unknown}
        credit_history = poor
        purpose in {business, furniture/appliances, renovations}
        savings_balance in {< 100 DM, > 1000 DM}
        percent_of_income > 3
        ->  class yes  [0.924]

    Rule 2/2: (13.7/0.7, lift 2.1)
        checking_balance in {> 200 DM, unknown}
        purpose in {business, renovations}
        employment_duration in {< 1 year, unemployed}
        ->  class yes  [0.895]

    Rule 2/3: (13.4/0.7, lift 2.1)
        checking_balance in {> 200 DM, unknown}
        months_loan_duration > 28
        employment_duration in {< 1 year, unemployed}
        ->  class yes  [0.893]

    Rule 2/4: (13.3/1.3, lift 2.0)
        checking_balance in {> 200 DM, unknown}
        months_loan_duration > 8
        age <= 34
        other_credit = none
        existing_loans_count <= 1
        job = unskilled
        ->  class yes  [0.849]

    Rule 2/5: (13/1.3, lift 2.0)
        checking_balance in {> 200 DM, unknown}
        months_loan_duration > 8
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        percent_of_income > 1
        age > 32
        age <= 44
        other_credit = bank
        ->  class yes  [0.847]

    Rule 2/6: (17.5/2.6, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        savings_balance = < 100 DM
        percent_of_income <= 3
        years_at_residence > 1
        years_at_residence <= 3
        housing = own
        dependents <= 1
        ->  class yes  [0.814]

    Rule 2/7: (27.4/5.2, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        savings_balance = 100 - 500 DM
        employment_duration in {< 1 year, 1 - 4 years, unemployed}
        years_at_residence > 1
        ->  class yes  [0.788]

    Rule 2/8: (11.7/2.6, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        savings_balance = 500 - 1000 DM
        years_at_residence > 1
        phone = FALSE
        ->  class yes  [0.737]

    Rule 2/9: (34.8/8.7, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        savings_balance = < 100 DM
        years_at_residence > 1
        other_credit = bank
        housing in {own, rent}
        ->  class yes  [0.737]

    Rule 2/10: (24.9/7.3, lift 1.7)
        checking_balance = < 0 DM
        credit_history in {good, very good}
        savings_balance = unknown
        ->  class yes  [0.693]

    Rule 2/11: (20.7/6.5, lift 1.6)
        checking_balance = unknown
        credit_history = good
        other_credit = none
        existing_loans_count > 1
        ->  class yes  [0.667]

    Rule 2/12: (22.7/7.2, lift 1.6)
        checking_balance in {> 200 DM, unknown}
        credit_history = poor
        percent_of_income > 3
        ->  class yes  [0.667]

    Rule 2/13: (54.8/18.4, lift 1.6)
        savings_balance = < 100 DM
        percent_of_income <= 3
        housing = rent
        dependents <= 1
        ->  class yes  [0.658]

    Rule 2/14: (53.9/19.8, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        savings_balance = < 100 DM
        years_at_residence > 1
        phone = TRUE
        ->  class yes  [0.629]

    Rule 2/15: (101.8/46.7, lift 1.3)
        purpose in {education, furniture/appliances}
        employment_duration in {< 1 year, unemployed}
        other_credit = none
        ->  class yes  [0.540]

    Rule 2/16: (527.1/264.3, lift 1.2)
        checking_balance in {< 0 DM, 1 - 200 DM}
        ->  class yes  [0.499]

    Rule 2/17: (17.7, lift 1.6)
        checking_balance in {> 200 DM, unknown}
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        percent_of_income <= 1
        age <= 44
        ->  class no  [0.949]

    Rule 2/18: (17, lift 1.6)
        checking_balance in {> 200 DM, unknown}
        months_loan_duration <= 28
        purpose in {car, car0}
        employment_duration in {< 1 year, unemployed}
        ->  class no  [0.947]

    Rule 2/19: (16.4, lift 1.6)
        credit_history = poor
        purpose in {business, furniture/appliances}
        savings_balance in {100 - 500 DM, 500 - 1000 DM, unknown}
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        ->  class no  [0.946]

    Rule 2/20: (14.4, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {critical, perfect, poor}
        savings_balance = unknown
        ->  class no  [0.939]

    Rule 2/21: (12.4, lift 1.6)
        checking_balance in {> 200 DM, unknown}
        credit_history = poor
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        percent_of_income <= 3
        ->  class no  [0.931]

    Rule 2/22: (49.2/2.6, lift 1.6)
        credit_history = critical
        purpose in {car, furniture/appliances, renovations}
        amount <= 3812
        savings_balance = < 100 DM
        percent_of_income > 3
        other_credit = none
        ->  class no  [0.929]

    Rule 2/23: (8.5, lift 1.6)
        checking_balance in {> 200 DM, unknown}
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        age > 44
        other_credit = bank
        ->  class no  [0.905]

    Rule 2/24: (36.2/2.7, lift 1.6)
        checking_balance in {> 200 DM, unknown}
        months_loan_duration <= 28
        credit_history = good
        purpose in {education, furniture/appliances}
        percent_of_income > 3
        job = skilled
        ->  class no  [0.903]

    Rule 2/25: (19.7/1.3, lift 1.5)
        credit_history = critical
        amount <= 7308
        savings_balance = < 100 DM
        dependents > 1
        ->  class no  [0.894]

    Rule 2/26: (7.2, lift 1.5)
        credit_history = poor
        purpose = car
        employment_duration in {> 7 years, 1 - 4 years}
        percent_of_income > 3
        other_credit = none
        ->  class no  [0.892]

    Rule 2/27: (67.3/7.1, lift 1.5)
        checking_balance in {> 200 DM, unknown}
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        age > 34
        other_credit = none
        existing_loans_count <= 1
        ->  class no  [0.884]

    Rule 2/28: (91.2/10.1, lift 1.5)
        checking_balance in {> 200 DM, unknown}
        credit_history in {critical, perfect, very good}
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        other_credit = none
        ->  class no  [0.881]

    Rule 2/29: (15.5/1.3, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose in {business, car, furniture/appliances}
        savings_balance = > 1000 DM
        ->  class no  [0.867]

    Rule 2/30: (82.9/12.1, lift 1.5)
        checking_balance in {> 200 DM, unknown}
        credit_history = good
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        other_credit = none
        existing_loans_count <= 1
        job in {management, skilled}
        ->  class no  [0.846]

    Rule 2/31: (20.8/2.6, lift 1.4)
        checking_balance in {< 0 DM, 1 - 200 DM}
        savings_balance = 100 - 500 DM
        employment_duration in {> 7 years, 4 - 7 years}
        age <= 54
        existing_loans_count <= 3
        ->  class no  [0.841]

    Rule 2/32: (29/6.1, lift 1.3)
        checking_balance = 1 - 200 DM
        savings_balance = unknown
        years_at_residence > 1
        ->  class no  [0.772]

    Rule 2/33: (135.5/49.4, lift 1.1)
        years_at_residence <= 1
        ->  class no  [0.634]

    Rule 2/34: (408.3/153.7, lift 1.1)
        months_loan_duration <= 42
        purpose = furniture/appliances
        ->  class no  [0.623]

    Default class: yes

    -----  Trial 3:  -----

    Rules:

    Rule 3/1: (10.3, lift 2.2)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = perfect
        percent_of_income > 3
        ->  class yes  [0.919]

    Rule 3/2: (6.1, lift 2.1)
        credit_history = critical
        employment_duration = 1 - 4 years
        years_at_residence > 1
        age <= 24
        ->  class yes  [0.876]

    Rule 3/3: (25.6/2.7, lift 2.1)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration <= 27
        credit_history = very good
        amount > 409
        age > 23
        ->  class yes  [0.865]

    Rule 3/4: (4.9, lift 2.1)
        checking_balance in {< 0 DM, > 200 DM}
        credit_history = critical
        purpose in {furniture/appliances, renovations}
        housing = other
        ->  class yes  [0.855]

    Rule 3/5: (4.5, lift 2.0)
        checking_balance = > 200 DM
        credit_history = critical
        savings_balance in {< 100 DM, 500 - 1000 DM}
        age <= 35
        ->  class yes  [0.846]

    Rule 3/6: (23.5/3.2, lift 2.0)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 7
        credit_history = good
        amount <= 3939
        savings_balance in {< 100 DM, 100 - 500 DM}
        dependents > 1
        phone = FALSE
        ->  class yes  [0.835]

    Rule 3/7: (13/1.6, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        years_at_residence > 1
        age <= 35
        job in {management, unskilled}
        ->  class yes  [0.826]

    Rule 3/8: (41.2/8.8, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {critical, good, perfect, poor}
        amount > 7980
        ->  class yes  [0.774]

    Rule 3/9: (11.4/2.1, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = good
        employment_duration = > 7 years
        years_at_residence <= 2
        phone = FALSE
        ->  class yes  [0.765]

    Rule 3/10: (41/10.7, lift 1.8)
        checking_balance = unknown
        amount > 4153
        employment_duration in {< 1 year, 1 - 4 years, unemployed}
        dependents <= 1
        ->  class yes  [0.728]

    Rule 3/11: (31.8/9, lift 1.7)
        employment_duration in {< 1 year, 1 - 4 years, unemployed}
        percent_of_income > 1
        dependents > 1
        phone = FALSE
        ->  class yes  [0.705]

    Rule 3/12: (40.9/11.8, lift 1.7)
        employment_duration in {< 1 year, 1 - 4 years, unemployed}
        percent_of_income > 1
        age <= 32
        housing = rent
        phone = FALSE
        ->  class yes  [0.702]

    Rule 3/13: (25.2/7.1, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = poor
        savings_balance in {< 100 DM, > 1000 DM}
        ->  class yes  [0.701]

    Rule 3/14: (67.9/19.9, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 7
        credit_history = good
        purpose in {car, education, furniture/appliances, renovations}
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        employment_duration = 1 - 4 years
        years_at_residence > 1
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.700]

    Rule 3/15: (25.9/8.2, lift 1.6)
        purpose in {education, renovations}
        employment_duration in {< 1 year, 1 - 4 years, unemployed}
        percent_of_income > 1
        phone = FALSE
        ->  class yes  [0.669]

    Rule 3/16: (35/12.5, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        savings_balance in {100 - 500 DM, unknown}
        other_credit = none
        phone = TRUE
        ->  class yes  [0.635]

    Rule 3/17: (60.1/4.3, lift 1.6)
        credit_history = critical
        purpose in {business, car, car0, education}
        amount <= 7980
        age > 35
        ->  class no  [0.915]

    Rule 3/18: (17.9/1.1, lift 1.5)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration <= 7
        credit_history = good
        amount <= 7980
        ->  class no  [0.896]

    Rule 3/19: (39.5/3.8, lift 1.5)
        months_loan_duration > 7
        purpose = car
        amount <= 7980
        savings_balance = < 100 DM
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years, unemployed}
        other_credit = none
        phone = TRUE
        ->  class no  [0.885]

    Rule 3/20: (121.8/21.6, lift 1.4)
        checking_balance = unknown
        employment_duration in {> 7 years, 4 - 7 years}
        ->  class no  [0.817]

    Rule 3/21: (70.3/17.3, lift 1.3)
        credit_history = good
        amount <= 7980
        employment_duration = 4 - 7 years
        ->  class no  [0.747]

    Rule 3/22: (835.7/332.6, lift 1.0)
        amount <= 7980
        ->  class no  [0.602]

    Default class: no

    -----  Trial 4:  -----

    Rules:

    Rule 4/1: (9.3, lift 2.1)
        checking_balance = < 0 DM
        months_loan_duration > 30
        purpose = car
        percent_of_income > 1
        dependents <= 1
        ->  class yes  [0.911]

    Rule 4/2: (9.2, lift 2.1)
        checking_balance = unknown
        purpose in {business, car, education}
        savings_balance = < 100 DM
        percent_of_income > 1
        age <= 44
        other_credit = bank
        ->  class yes  [0.911]

    Rule 4/3: (19.9/1.7, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 42
        savings_balance = < 100 DM
        job = skilled
        ->  class yes  [0.876]

    Rule 4/4: (5.4, lift 2.0)
        credit_history = critical
        amount > 11816
        other_credit = none
        ->  class yes  [0.865]

    Rule 4/5: (11.4/1.2, lift 1.9)
        checking_balance = unknown
        credit_history = good
        percent_of_income <= 2
        other_credit = none
        existing_loans_count > 1
        ->  class yes  [0.838]

    Rule 4/6: (14.4/2.2, lift 1.8)
        checking_balance = unknown
        credit_history = poor
        purpose in {business, furniture/appliances, renovations}
        percent_of_income > 3
        ->  class yes  [0.806]

    Rule 4/7: (19.7/3.9, lift 1.8)
        checking_balance in {> 200 DM, 1 - 200 DM}
        credit_history in {good, perfect, poor, very good}
        purpose = car
        savings_balance = 100 - 500 DM
        ->  class yes  [0.774]

    Rule 4/8: (36.3/9.8, lift 1.6)
        checking_balance = 1 - 200 DM
        months_loan_duration > 8
        savings_balance = < 100 DM
        years_at_residence <= 2
        job = skilled
        ->  class yes  [0.718]

    Rule 4/9: (17/4.8, lift 1.6)
        checking_balance = unknown
        credit_history = poor
        percent_of_income > 3
        ->  class yes  [0.695]

    Rule 4/10: (628.7/322.6, lift 1.1)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        ->  class yes  [0.487]

    Rule 4/11: (14.4, lift 1.7)
        checking_balance = 1 - 200 DM
        savings_balance = < 100 DM
        percent_of_income <= 2
        years_at_residence > 2
        job = skilled
        dependents <= 1
        ->  class no  [0.939]

    Rule 4/12: (13.5, lift 1.7)
        months_loan_duration <= 8
        savings_balance = < 100 DM
        years_at_residence <= 2
        ->  class no  [0.935]

    Rule 4/13: (10.6, lift 1.6)
        checking_balance = < 0 DM
        months_loan_duration > 22
        months_loan_duration <= 42
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        ->  class no  [0.921]

    Rule 4/14: (9.6, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {perfect, poor}
        savings_balance = unknown
        ->  class no  [0.914]

    Rule 4/15: (8.5, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose in {business, education}
        savings_balance = unknown
        job = skilled
        ->  class no  [0.904]

    Rule 4/16: (8, lift 1.6)
        checking_balance = 1 - 200 DM
        amount <= 1221
        savings_balance = < 100 DM
        years_at_residence > 2
        job = skilled
        dependents <= 1
        ->  class no  [0.900]

    Rule 4/17: (18.2/1.4, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration <= 11
        savings_balance = < 100 DM
        job = unskilled
        ->  class no  [0.883]

    Rule 4/18: (6.5, lift 1.6)
        savings_balance = < 100 DM
        years_at_residence <= 1
        job = management
        ->  class no  [0.883]

    Rule 4/19: (6.1, lift 1.6)
        checking_balance = unknown
        credit_history = poor
        purpose = car
        ->  class no  [0.876]

    Rule 4/20: (5.4, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        purpose = car
        savings_balance = 100 - 500 DM
        ->  class no  [0.864]

    Rule 4/21: (61.2/15.3, lift 1.3)
        amount > 1597
        savings_balance = unknown
        job = skilled
        ->  class no  [0.742]

    Rule 4/22: (79.6/23.1, lift 1.3)
        savings_balance = < 100 DM
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        job = unskilled
        ->  class no  [0.704]

    Rule 4/23: (271.3/87.3, lift 1.2)
        checking_balance = unknown
        ->  class no  [0.677]

    Rule 4/24: (33.2/10.6, lift 1.2)
        savings_balance = > 1000 DM
        ->  class no  [0.670]

    Rule 4/25: (121.6/44.1, lift 1.1)
        percent_of_income <= 1
        ->  class no  [0.635]

    Default class: no

    -----  Trial 5:  -----

    Rules:

    Rule 5/1: (9.3, lift 2.2)
        credit_history = good
        percent_of_income <= 1
        existing_loans_count > 1
        ->  class yes  [0.912]

    Rule 5/2: (10.1/0.4, lift 2.1)
        purpose = furniture/appliances
        employment_duration = 4 - 7 years
        age <= 23
        existing_loans_count <= 1
        ->  class yes  [0.887]

    Rule 5/3: (6.2, lift 2.1)
        credit_history = good
        amount > 1352
        existing_loans_count > 1
        job = management
        dependents <= 1
        ->  class yes  [0.878]

    Rule 5/4: (21.2/1.9, lift 2.1)
        checking_balance in {< 0 DM, unknown}
        credit_history = good
        purpose in {car, furniture/appliances}
        amount > 1352
        existing_loans_count > 1
        job = skilled
        dependents <= 1
        ->  class yes  [0.877]

    Rule 5/5: (17.4/1.5, lift 2.1)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        purpose = car
        amount > 7393
        existing_loans_count <= 1
        job in {management, skilled, unemployed}
        ->  class yes  [0.873]

    Rule 5/6: (5.5, lift 2.1)
        credit_history = critical
        savings_balance = 500 - 1000 DM
        age <= 34
        other_credit = none
        ->  class yes  [0.867]

    Rule 5/7: (10.7/0.7, lift 2.0)
        credit_history = perfect
        housing in {other, rent}
        ->  class yes  [0.863]

    Rule 5/8: (19.9/3.2, lift 1.9)
        credit_history = critical
        purpose in {car, education, furniture/appliances}
        age <= 46
        other_credit = bank
        ->  class yes  [0.808]

    Rule 5/9: (6.5/0.7, lift 1.9)
        credit_history = very good
        existing_loans_count > 1
        ->  class yes  [0.794]

    Rule 5/10: (2.4, lift 1.8)
        credit_history = perfect
        other_credit = store
        ->  class yes  [0.773]

    Rule 5/11: (42/9.1, lift 1.8)
        months_loan_duration > 7
        credit_history = good
        purpose = car
        amount <= 1388
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        ->  class yes  [0.770]

    Rule 5/12: (8.4/1.5, lift 1.8)
        credit_history = very good
        years_at_residence <= 1
        ->  class yes  [0.756]

    Rule 5/13: (25.4/6.6, lift 1.7)
        months_loan_duration > 16
        credit_history = poor
        savings_balance = < 100 DM
        ->  class yes  [0.724]

    Rule 5/14: (14.1/3.5, lift 1.7)
        credit_history = good
        purpose = furniture/appliances
        employment_duration = unemployed
        ->  class yes  [0.720]

    Rule 5/15: (23.9/7.3, lift 1.6)
        checking_balance = < 0 DM
        credit_history = good
        employment_duration = < 1 year
        years_at_residence > 2
        ->  class yes  [0.679]

    Rule 5/16: (545.4/299.8, lift 1.1)
        age <= 34
        ->  class yes  [0.451]

    Rule 5/17: (578.8/320.2, lift 1.1)
        savings_balance = < 100 DM
        ->  class yes  [0.447]

    Rule 5/18: (35.4, lift 1.7)
        credit_history = good
        purpose = car
        amount > 1388
        amount <= 7393
        percent_of_income <= 2
        existing_loans_count <= 1
        ->  class no  [0.973]

    Rule 5/19: (24.4, lift 1.7)
        checking_balance = unknown
        credit_history = critical
        amount <= 6850
        savings_balance = < 100 DM
        other_credit = none
        ->  class no  [0.962]

    Rule 5/20: (18.6, lift 1.6)
        credit_history = good
        purpose = car
        amount > 4370
        amount <= 7393
        existing_loans_count <= 1
        ->  class no  [0.952]

    Rule 5/21: (13.4, lift 1.6)
        months_loan_duration <= 18
        credit_history = good
        purpose = business
        ->  class no  [0.935]

    Rule 5/22: (13, lift 1.6)
        checking_balance = < 0 DM
        credit_history = critical
        amount <= 2122
        age <= 34
        ->  class no  [0.933]

    Rule 5/23: (12.5, lift 1.6)
        checking_balance = < 0 DM
        credit_history = good
        purpose = furniture/appliances
        employment_duration = < 1 year
        years_at_residence <= 2
        ->  class no  [0.931]

    Rule 5/24: (10.9, lift 1.6)
        months_loan_duration <= 13
        purpose = car
        amount > 1388
        percent_of_income > 2
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.923]

    Rule 5/25: (17.5/1.1, lift 1.5)
        checking_balance = < 0 DM
        months_loan_duration <= 15
        credit_history = good
        purpose = furniture/appliances
        employment_duration = 1 - 4 years
        ->  class no  [0.891]

    Rule 5/26: (6, lift 1.5)
        credit_history = critical
        purpose in {business, car0}
        other_credit = bank
        ->  class no  [0.874]

    Rule 5/27: (16/1.6, lift 1.5)
        months_loan_duration <= 7
        credit_history = good
        amount <= 1388
        ->  class no  [0.859]

    Rule 5/28: (11.3/1, lift 1.5)
        purpose = furniture/appliances
        employment_duration = < 1 year
        existing_loans_count <= 1
        job in {management, unemployed}
        ->  class no  [0.850]

    Rule 5/29: (90.8/14.4, lift 1.4)
        credit_history = critical
        age > 34
        other_credit = none
        ->  class no  [0.834]

    Rule 5/30: (36.8/5.5, lift 1.4)
        credit_history = good
        amount <= 7393
        savings_balance in {< 100 DM, 500 - 1000 DM}
        job = management
        ->  class no  [0.833]

    Rule 5/31: (36.7/6.2, lift 1.4)
        credit_history = good
        purpose = furniture/appliances
        amount <= 5711
        employment_duration = > 7 years
        other_credit in {bank, none}
        existing_loans_count <= 1
        ->  class no  [0.815]

    Rule 5/32: (26.9/5.2, lift 1.4)
        credit_history = poor
        savings_balance in {100 - 500 DM, 500 - 1000 DM, unknown}
        ->  class no  [0.785]

    Rule 5/33: (68.4/14.3, lift 1.4)
        checking_balance in {> 200 DM, unknown}
        purpose = furniture/appliances
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.783]

    Rule 5/34: (34/8.5, lift 1.3)
        existing_loans_count > 1
        job = unskilled
        dependents <= 1
        ->  class no  [0.735]

    Rule 5/35: (76.5/20, lift 1.3)
        employment_duration = 4 - 7 years
        age > 23
        existing_loans_count <= 1
        ->  class no  [0.732]

    Rule 5/36: (629.8/253.4, lift 1.0)
        housing = own
        ->  class no  [0.597]

    Default class: no

    -----  Trial 6:  -----

    Rules:

    Rule 6/1: (21.9, lift 2.3)
        checking_balance = < 0 DM
        amount > 2171
        savings_balance = < 100 DM
        percent_of_income > 2
        age > 28
        job = skilled
        phone = FALSE
        ->  class yes  [0.958]

    Rule 6/2: (17.4, lift 2.2)
        checking_balance = < 0 DM
        amount > 3123
        savings_balance = < 100 DM
        job = skilled
        phone = TRUE
        ->  class yes  [0.948]

    Rule 6/3: (11.6, lift 2.2)
        checking_balance = < 0 DM
        credit_history in {perfect, poor, very good}
        percent_of_income > 2
        job = skilled
        phone = FALSE
        ->  class yes  [0.926]

    Rule 6/4: (8.8, lift 2.1)
        checking_balance = unknown
        purpose = business
        employment_duration in {< 1 year, unemployed}
        ->  class yes  [0.908]

    Rule 6/5: (8, lift 2.1)
        checking_balance = < 0 DM
        months_loan_duration > 27
        savings_balance = unknown
        other_credit = none
        ->  class yes  [0.900]

    Rule 6/6: (8.8/0.3, lift 2.1)
        amount > 7596
        savings_balance = < 100 DM
        percent_of_income > 2
        ->  class yes  [0.880]

    Rule 6/7: (19.5/1.7, lift 2.1)
        checking_balance = < 0 DM
        credit_history in {good, poor, very good}
        savings_balance = < 100 DM
        job = skilled
        phone = TRUE
        ->  class yes  [0.872]

    Rule 6/8: (9.8/0.6, lift 2.0)
        credit_history = good
        amount <= 1386
        savings_balance = < 100 DM
        percent_of_income <= 2
        job = skilled
        phone = FALSE
        ->  class yes  [0.863]

    Rule 6/9: (5.1, lift 2.0)
        checking_balance = < 0 DM
        percent_of_income <= 1
        job = management
        ->  class yes  [0.860]

    Rule 6/10: (8.4/0.7, lift 2.0)
        purpose = car
        amount > 11816
        other_credit = none
        ->  class yes  [0.837]

    Rule 6/11: (13.6/1.8, lift 1.9)
        credit_history = critical
        purpose = car
        other_credit in {bank, store}
        ->  class yes  [0.819]

    Rule 6/12: (10.4/1.4, lift 1.9)
        checking_balance = 1 - 200 DM
        amount <= 10366
        savings_balance = 100 - 500 DM
        housing = rent
        ->  class yes  [0.807]

    Rule 6/13: (10.7/1.5, lift 1.9)
        checking_balance = unknown
        credit_history = good
        purpose = car
        employment_duration = 1 - 4 years
        percent_of_income > 1
        age <= 28
        ->  class yes  [0.803]

    Rule 6/14: (7.8/1, lift 1.9)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 18
        savings_balance = unknown
        phone = TRUE
        ->  class yes  [0.795]

    Rule 6/15: (7.5/1.2, lift 1.8)
        checking_balance = > 200 DM
        dependents > 1
        ->  class yes  [0.766]

    Rule 6/16: (34.7/7.8, lift 1.8)
        checking_balance = 1 - 200 DM
        savings_balance = < 100 DM
        age > 35
        age <= 62
        other_credit = none
        ->  class yes  [0.761]

    Rule 6/17: (16.1/3.4, lift 1.8)
        checking_balance = 1 - 200 DM
        amount > 10366
        ->  class yes  [0.759]

    Rule 6/18: (18.4/3.9, lift 1.8)
        checking_balance = unknown
        purpose = education
        savings_balance in {< 100 DM, 100 - 500 DM}
        percent_of_income > 2
        ->  class yes  [0.759]

    Rule 6/19: (19.9/5.7, lift 1.6)
        checking_balance = 1 - 200 DM
        savings_balance = 100 - 500 DM
        job in {management, unskilled}
        ->  class yes  [0.694]

    Rule 6/20: (66.2/26, lift 1.4)
        checking_balance = < 0 DM
        months_loan_duration > 22
        purpose = furniture/appliances
        ->  class yes  [0.604]

    Rule 6/21: (50.5/24.4, lift 1.2)
        other_credit = store
        ->  class yes  [0.516]

    Rule 6/22: (304.5/151.9, lift 1.2)
        checking_balance = < 0 DM
        ->  class yes  [0.501]

    Rule 6/23: (9.8, lift 1.6)
        checking_balance = < 0 DM
        percent_of_income > 1
        percent_of_income <= 2
        job = management
        ->  class no  [0.915]

    Rule 6/24: (39.8/5.3, lift 1.5)
        amount > 1386
        savings_balance = < 100 DM
        percent_of_income > 1
        percent_of_income <= 2
        age > 26
        other_credit = none
        job = skilled
        ->  class no  [0.849]

    Rule 6/25: (40.6/6.9, lift 1.4)
        checking_balance = unknown
        credit_history in {critical, perfect, poor, very good}
        purpose = furniture/appliances
        ->  class no  [0.814]

    Rule 6/26: (19.5/3.9, lift 1.3)
        savings_balance = < 100 DM
        age > 62
        ->  class no  [0.771]

    Rule 6/27: (58.6/15.6, lift 1.3)
        checking_balance = > 200 DM
        dependents <= 1
        ->  class no  [0.727]

    Rule 6/28: (187.5/65.2, lift 1.1)
        savings_balance = < 100 DM
        age <= 28
        phone = FALSE
        ->  class no  [0.651]

    Rule 6/29: (865.7/360.2, lift 1.0)
        amount <= 10366
        ->  class no  [0.584]

    Default class: no

    -----  Trial 7:  -----

    Rules:

    Rule 7/1: (18, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 15
        months_loan_duration <= 21
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        job = skilled
        phone = FALSE
        ->  class yes  [0.950]

    Rule 7/2: (17.9/0.5, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 36
        purpose = furniture/appliances
        dependents <= 1
        ->  class yes  [0.924]

    Rule 7/3: (6.7, lift 1.8)
        employment_duration = unemployed
        years_at_residence <= 1
        ->  class yes  [0.885]

    Rule 7/4: (8.6/0.6, lift 1.8)
        purpose = furniture/appliances
        amount > 3357
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        phone = TRUE
        ->  class yes  [0.851]

    Rule 7/5: (15/1.8, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        amount <= 1574
        percent_of_income <= 2
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.838]

    Rule 7/6: (46.7/7.2, lift 1.7)
        percent_of_income > 3
        dependents > 1
        phone = FALSE
        ->  class yes  [0.832]

    Rule 7/7: (20.2/3.4, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        amount <= 1347
        employment_duration = 1 - 4 years
        job = skilled
        phone = FALSE
        ->  class yes  [0.801]

    Rule 7/8: (12.7/2, lift 1.7)
        employment_duration = < 1 year
        existing_loans_count > 1
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.798]

    Rule 7/9: (766.7/389, lift 1.0)
        months_loan_duration > 11
        ->  class yes  [0.493]

    Rule 7/10: (14.2, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = business
        housing = own
        dependents <= 1
        phone = FALSE
        ->  class no  [0.938]

    Rule 7/11: (12.3, lift 1.8)
        savings_balance in {100 - 500 DM, unknown}
        employment_duration = 4 - 7 years
        job = skilled
        phone = TRUE
        ->  class no  [0.930]

    Rule 7/12: (12, lift 1.8)
        checking_balance = unknown
        employment_duration = 4 - 7 years
        phone = TRUE
        ->  class no  [0.929]

    Rule 7/13: (11.9, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 36
        purpose = furniture/appliances
        savings_balance = < 100 DM
        age <= 29
        other_credit = none
        job = unskilled
        phone = FALSE
        ->  class no  [0.928]

    Rule 7/14: (11.8, lift 1.8)
        checking_balance in {> 200 DM, unknown}
        savings_balance in {> 1000 DM, unknown}
        age <= 38
        other_credit = none
        job = skilled
        phone = FALSE
        ->  class no  [0.927]

    Rule 7/15: (11.5, lift 1.8)
        checking_balance = < 0 DM
        months_loan_duration > 21
        months_loan_duration <= 36
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        phone = FALSE
        ->  class no  [0.926]

    Rule 7/16: (10.4, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 11
        purpose = furniture/appliances
        job = unskilled
        ->  class no  [0.919]

    Rule 7/17: (13.5/0.3, lift 1.8)
        employment_duration = 4 - 7 years
        housing in {other, rent}
        phone = TRUE
        ->  class no  [0.919]

    Rule 7/18: (9.2, lift 1.8)
        months_loan_duration <= 36
        purpose = furniture/appliances
        savings_balance = > 1000 DM
        ->  class no  [0.911]

    Rule 7/19: (8.5, lift 1.7)
        savings_balance in {> 1000 DM, 500 - 1000 DM}
        employment_duration = 1 - 4 years
        existing_loans_count <= 2
        phone = TRUE
        ->  class no  [0.905]

    Rule 7/20: (8.3, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        credit_history = good
        years_at_residence <= 1
        age <= 38
        dependents <= 1
        ->  class no  [0.903]

    Rule 7/21: (7.1, lift 1.7)
        purpose = car
        amount <= 2225
        percent_of_income > 3
        years_at_residence <= 1
        ->  class no  [0.890]

    Rule 7/22: (6.9, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        employment_duration = < 1 year
        existing_loans_count <= 1
        phone = TRUE
        ->  class no  [0.888]

    Rule 7/23: (11.3/0.6, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        months_loan_duration > 16
        job = unskilled
        dependents <= 1
        ->  class no  [0.882]

    Rule 7/24: (17.8/1.3, lift 1.7)
        months_loan_duration <= 15
        percent_of_income <= 3
        dependents > 1
        phone = FALSE
        ->  class no  [0.882]

    Rule 7/25: (29/2.7, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        age > 38
        dependents <= 1
        phone = FALSE
        ->  class no  [0.879]

    Rule 7/26: (19.2/1.9, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        amount > 1574
        percent_of_income <= 2
        dependents <= 1
        phone = FALSE
        ->  class no  [0.864]

    Rule 7/27: (9/0.5, lift 1.7)
        employment_duration = 4 - 7 years
        percent_of_income <= 3
        dependents > 1
        ->  class no  [0.863]

    Rule 7/28: (5.1, lift 1.7)
        purpose = furniture/appliances
        years_at_residence <= 1
        job = unskilled
        dependents <= 1
        ->  class no  [0.858]

    Rule 7/29: (4.9, lift 1.6)
        checking_balance in {> 200 DM, unknown}
        years_at_residence > 1
        job in {management, unemployed}
        phone = FALSE
        ->  class no  [0.855]

    Rule 7/30: (12.7/1.3, lift 1.6)
        percent_of_income <= 1
        dependents > 1
        phone = FALSE
        ->  class no  [0.841]

    Rule 7/31: (15.4/1.8, lift 1.6)
        purpose = furniture/appliances
        amount <= 3357
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        phone = TRUE
        ->  class no  [0.840]

    Rule 7/32: (15.2/1.8, lift 1.6)
        employment_duration = 4 - 7 years
        job in {management, unskilled}
        phone = TRUE
        ->  class no  [0.838]

    Rule 7/33: (3.9, lift 1.6)
        employment_duration = > 7 years
        job = unskilled
        phone = TRUE
        ->  class no  [0.831]

    Rule 7/34: (3.6, lift 1.6)
        employment_duration = < 1 year
        dependents > 1
        phone = TRUE
        ->  class no  [0.821]

    Rule 7/35: (13.6/1.8, lift 1.6)
        employment_duration = 1 - 4 years
        other_credit = bank
        phone = TRUE
        ->  class no  [0.819]

    Rule 7/36: (22.5/3.8, lift 1.6)
        months_loan_duration <= 36
        credit_history = critical
        purpose = furniture/appliances
        savings_balance = < 100 DM
        age <= 42
        job = skilled
        phone = FALSE
        ->  class no  [0.806]

    Rule 7/37: (19.4/3.3, lift 1.5)
        purpose = car
        employment_duration = 1 - 4 years
        other_credit = none
        job in {management, skilled}
        phone = TRUE
        ->  class no  [0.800]

    Rule 7/38: (2.7, lift 1.5)
        purpose = car0
        phone = FALSE
        ->  class no  [0.787]

    Rule 7/39: (55.3/12.7, lift 1.5)
        checking_balance in {> 200 DM, unknown}
        credit_history in {critical, perfect, poor}
        dependents <= 1
        phone = FALSE
        ->  class no  [0.761]

    Rule 7/40: (34.5/8, lift 1.5)
        employment_duration = > 7 years
        other_credit in {none, store}
        job = skilled
        phone = TRUE
        ->  class no  [0.753]

    Rule 7/41: (38.6/9.7, lift 1.4)
        months_loan_duration <= 15
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        job = skilled
        ->  class no  [0.737]

    Rule 7/42: (31.2/9, lift 1.3)
        employment_duration = unemployed
        years_at_residence > 1
        phone = TRUE
        ->  class no  [0.698]

    Rule 7/43: (55.7/19.6, lift 1.2)
        savings_balance = unknown
        percent_of_income <= 3
        ->  class no  [0.643]

    Default class: yes

    -----  Trial 8:  -----

    Rules:

    Rule 8/1: (9.8, lift 2.1)
        checking_balance = < 0 DM
        months_loan_duration <= 18
        credit_history = critical
        amount > 2122
        job = skilled
        dependents <= 1
        ->  class yes  [0.915]

    Rule 8/2: (13/0.8, lift 2.0)
        checking_balance = unknown
        months_loan_duration > 9
        credit_history in {critical, good}
        purpose = car
        employment_duration = 1 - 4 years
        percent_of_income > 1
        percent_of_income <= 3
        phone = FALSE
        ->  class yes  [0.880]

    Rule 8/3: (39.6/4.4, lift 2.0)
        checking_balance = < 0 DM
        months_loan_duration > 16
        months_loan_duration <= 36
        credit_history = good
        savings_balance in {< 100 DM, unknown}
        percent_of_income > 1
        age <= 48
        other_credit in {bank, none}
        housing in {own, rent}
        job = skilled
        ->  class yes  [0.870]

    Rule 8/4: (4.3, lift 1.9)
        checking_balance = 1 - 200 DM
        employment_duration = > 7 years
        age > 53
        ->  class yes  [0.841]

    Rule 8/5: (4.2, lift 1.9)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 8
        employment_duration = unemployed
        ->  class yes  [0.839]

    Rule 8/6: (17.4/2.2, lift 1.9)
        checking_balance = 1 - 200 DM
        savings_balance in {< 100 DM, > 1000 DM, 100 - 500 DM}
        years_at_residence > 1
        age > 31
        dependents > 1
        ->  class yes  [0.834]

    Rule 8/7: (10.6/1.2, lift 1.9)
        employment_duration = 4 - 7 years
        age <= 22
        ->  class yes  [0.828]

    Rule 8/8: (3.4, lift 1.9)
        checking_balance = < 0 DM
        percent_of_income <= 1
        job = management
        ->  class yes  [0.816]

    Rule 8/9: (17.2/2.7, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration > 36
        job = skilled
        ->  class yes  [0.809]

    Rule 8/10: (10.9/2, lift 1.8)
        checking_balance = unknown
        purpose = business
        percent_of_income > 1
        other_credit in {bank, store}
        ->  class yes  [0.771]

    Rule 8/11: (19.6/4.6, lift 1.7)
        checking_balance = 1 - 200 DM
        employment_duration = < 1 year
        housing = rent
        ->  class yes  [0.740]

    Rule 8/12: (26.1/7.2, lift 1.6)
        checking_balance = < 0 DM
        credit_history in {perfect, poor, very good}
        savings_balance in {< 100 DM, 100 - 500 DM, unknown}
        job = skilled
        ->  class yes  [0.708]

    Rule 8/13: (29.1/8.8, lift 1.6)
        checking_balance = unknown
        amount > 1512
        employment_duration = < 1 year
        other_credit = none
        ->  class yes  [0.684]

    Rule 8/14: (35.6/11.3, lift 1.6)
        checking_balance = 1 - 200 DM
        purpose = furniture/appliances
        amount > 759
        employment_duration = 1 - 4 years
        other_credit = none
        dependents <= 1
        ->  class yes  [0.673]

    Rule 8/15: (31.2/11.9, lift 1.4)
        checking_balance = < 0 DM
        savings_balance = unknown
        ->  class yes  [0.610]

    Rule 8/16: (44.5/17.7, lift 1.4)
        employment_duration = < 1 year
        existing_loans_count > 1
        ->  class yes  [0.599]

    Rule 8/17: (113.4/53.9, lift 1.2)
        checking_balance = < 0 DM
        purpose = car
        ->  class yes  [0.524]

    Rule 8/18: (695/408.9, lift 1.0)
        amount <= 4308
        ->  class yes  [0.412]

    Rule 8/19: (11.7, lift 1.6)
        checking_balance = 1 - 200 DM
        employment_duration = 4 - 7 years
        housing = rent
        ->  class no  [0.927]

    Rule 8/20: (10, lift 1.6)
        checking_balance = unknown
        employment_duration = 1 - 4 years
        percent_of_income <= 1
        ->  class no  [0.917]

    Rule 8/21: (9.3, lift 1.6)
        checking_balance = unknown
        months_loan_duration <= 9
        employment_duration = 1 - 4 years
        ->  class no  [0.912]

    Rule 8/22: (23.6/1.3, lift 1.6)
        checking_balance = unknown
        purpose = furniture/appliances
        employment_duration = 1 - 4 years
        job in {management, skilled}
        ->  class no  [0.909]

    Rule 8/23: (8.9, lift 1.6)
        checking_balance = > 200 DM
        age <= 24
        ->  class no  [0.908]

    Rule 8/24: (8.6, lift 1.6)
        checking_balance = unknown
        amount <= 1512
        employment_duration = < 1 year
        ->  class no  [0.906]

    Rule 8/25: (8, lift 1.6)
        purpose = furniture/appliances
        amount <= 759
        employment_duration = 1 - 4 years
        other_credit = none
        ->  class no  [0.900]

    Rule 8/26: (25.1/1.8, lift 1.6)
        checking_balance = unknown
        employment_duration = 4 - 7 years
        age > 22
        ->  class no  [0.896]

    Rule 8/27: (6.9, lift 1.6)
        checking_balance = > 200 DM
        amount > 4308
        ->  class no  [0.888]

    Rule 8/28: (23.9/2.7, lift 1.5)
        checking_balance = < 0 DM
        credit_history = critical
        amount <= 2122
        dependents <= 1
        ->  class no  [0.857]

    Rule 8/29: (12.3/1.3, lift 1.5)
        checking_balance = < 0 DM
        savings_balance in {> 1000 DM, 500 - 1000 DM}
        ->  class no  [0.838]

    Rule 8/30: (46.1/9.4, lift 1.4)
        checking_balance = unknown
        months_loan_duration > 13
        employment_duration = > 7 years
        ->  class no  [0.784]

    Rule 8/31: (32.7/6.5, lift 1.4)
        checking_balance = < 0 DM
        savings_balance in {< 100 DM, 100 - 500 DM}
        percent_of_income > 1
        job = management
        ->  class no  [0.784]

    Rule 8/32: (33.5/6.8, lift 1.4)
        checking_balance = unknown
        purpose = car
        phone = TRUE
        ->  class no  [0.781]

    Rule 8/33: (158.6/51.8, lift 1.2)
        employment_duration = > 7 years
        dependents <= 1
        ->  class no  [0.671]

    Rule 8/34: (621.6/258.8, lift 1.0)
        housing = own
        ->  class no  [0.583]

    Default class: no

    -----  Trial 9:  -----

    Rules:

    Rule 9/1: (18.1/0.2, lift 2.1)
        months_loan_duration > 8
        amount <= 1264
        dependents > 1
        phone = FALSE
        ->  class yes  [0.942]

    Rule 9/2: (12.7, lift 2.1)
        credit_history in {good, poor}
        amount <= 2241
        other_credit = store
        phone = FALSE
        ->  class yes  [0.932]

    Rule 9/3: (16.2/0.8, lift 2.0)
        checking_balance = < 0 DM
        months_loan_duration > 16
        amount <= 3552
        employment_duration = 1 - 4 years
        phone = FALSE
        ->  class yes  [0.899]

    Rule 9/4: (6, lift 2.0)
        months_loan_duration > 27
        amount <= 3552
        employment_duration = < 1 year
        ->  class yes  [0.875]

    Rule 9/5: (6, lift 2.0)
        checking_balance = > 200 DM
        months_loan_duration > 8
        amount <= 3913
        dependents > 1
        ->  class yes  [0.875]

    Rule 9/6: (20.4/2.1, lift 2.0)
        months_loan_duration > 8
        credit_history in {perfect, very good}
        amount <= 3552
        age <= 31
        phone = FALSE
        ->  class yes  [0.861]

    Rule 9/7: (6.4/0.2, lift 2.0)
        months_loan_duration > 33
        amount <= 3552
        age <= 43
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.860]

    Rule 9/8: (4.9, lift 1.9)
        purpose = car
        amount <= 3552
        employment_duration = < 1 year
        age <= 25
        ->  class yes  [0.856]

    Rule 9/9: (12.3/1.3, lift 1.9)
        months_loan_duration > 16
        credit_history = critical
        amount <= 3552
        employment_duration = 1 - 4 years
        other_credit in {bank, none}
        phone = FALSE
        ->  class yes  [0.836]

    Rule 9/10: (14.4/2.2, lift 1.8)
        checking_balance in {> 200 DM, 1 - 200 DM}
        credit_history = critical
        amount <= 3552
        age <= 43
        existing_loans_count > 1
        phone = TRUE
        ->  class yes  [0.802]

    Rule 9/11: (12.9/2.5, lift 1.7)
        months_loan_duration > 8
        employment_duration = 4 - 7 years
        age <= 23
        phone = FALSE
        ->  class yes  [0.765]

    Rule 9/12: (33.9/8, lift 1.7)
        credit_history in {critical, good, very good}
        purpose = furniture/appliances
        amount > 3913
        percent_of_income <= 2
        other_credit in {bank, none}
        ->  class yes  [0.750]

    Rule 9/13: (27.2/6.6, lift 1.7)
        credit_history in {good, perfect}
        age <= 43
        existing_loans_count > 1
        phone = TRUE
        ->  class yes  [0.740]

    Rule 9/14: (24.2/6.2, lift 1.6)
        purpose = business
        amount > 3913
        savings_balance in {< 100 DM, > 1000 DM}
        ->  class yes  [0.725]

    Rule 9/15: (40/11.4, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 8
        credit_history = good
        percent_of_income > 1
        years_at_residence <= 3
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.704]

    Rule 9/16: (29.6/10, lift 1.5)
        credit_history in {perfect, poor, very good}
        age > 43
        ->  class yes  [0.650]

    Rule 9/17: (108.6/39.2, lift 1.4)
        amount > 3913
        percent_of_income > 2
        ->  class yes  [0.636]

    Rule 9/18: (64.6/24.3, lift 1.4)
        purpose = car
        amount > 6419
        ->  class yes  [0.620]

    Rule 9/19: (20.3, lift 1.7)
        checking_balance in {> 200 DM, 1 - 200 DM, unknown}
        amount <= 3552
        employment_duration = > 7 years
        housing = own
        dependents <= 1
        phone = FALSE
        ->  class no  [0.955]

    Rule 9/20: (20.5, lift 1.7)
        amount <= 3913
        dependents > 1
        phone = TRUE
        ->  class no  [0.955]

    Rule 9/21: (48.6/4.4, lift 1.6)
        amount > 3552
        amount <= 3913
        ->  class no  [0.893]

    Rule 9/22: (32.4/3.1, lift 1.6)
        credit_history in {critical, good, poor}
        purpose in {business, education, furniture/appliances}
        amount <= 3913
        employment_duration = 4 - 7 years
        age > 23
        ->  class no  [0.880]

    Rule 9/23: (19.6/1.7, lift 1.6)
        purpose = car
        amount > 3913
        amount <= 6419
        percent_of_income <= 2
        ->  class no  [0.875]

    Rule 9/24: (5.2, lift 1.5)
        purpose = business
        savings_balance in {100 - 500 DM, unknown}
        percent_of_income <= 2
        ->  class no  [0.861]

    Rule 9/25: (11.2/1.4, lift 1.5)
        credit_history in {perfect, poor}
        purpose = furniture/appliances
        percent_of_income <= 2
        ->  class no  [0.817]

    Rule 9/26: (741.5/312.3, lift 1.0)
        months_loan_duration <= 33
        ->  class no  [0.579]

    Default class: no

    -----  Trial 10:  -----

    Rules:

    Rule 10/1: (13.5, lift 2.1)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = education
        amount > 385
        savings_balance in {< 100 DM, > 1000 DM}
        percent_of_income > 3
        phone = FALSE
        ->  class yes  [0.935]

    Rule 10/2: (9.6, lift 2.1)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 36
        purpose = furniture/appliances
        savings_balance = < 100 DM
        housing = own
        ->  class yes  [0.914]

    Rule 10/3: (11.9/0.4, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        amount <= 1386
        savings_balance = < 100 DM
        percent_of_income <= 3
        housing in {own, rent}
        dependents <= 1
        ->  class yes  [0.901]

    Rule 10/4: (8.1, lift 2.0)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        employment_duration = unemployed
        percent_of_income > 3
        phone = FALSE
        ->  class yes  [0.901]

    Rule 10/5: (7.8, lift 2.0)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        savings_balance = unknown
        percent_of_income <= 3
        housing = other
        ->  class yes  [0.898]

    Rule 10/6: (7.2, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        savings_balance = < 100 DM
        percent_of_income <= 3
        age > 41
        housing in {own, rent}
        dependents <= 1
        ->  class yes  [0.892]

    Rule 10/7: (5.9, lift 2.0)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = perfect
        percent_of_income > 3
        ->  class yes  [0.873]

    Rule 10/8: (25.4/2.6, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {critical, good, perfect}
        amount > 10722
        ->  class yes  [0.869]

    Rule 10/9: (16.6/2.6, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        savings_balance in {100 - 500 DM, 500 - 1000 DM}
        percent_of_income > 3
        phone = FALSE
        ->  class yes  [0.806]

    Rule 10/10: (13.9/2.9, lift 1.7)
        purpose = furniture/appliances
        percent_of_income > 3
        dependents > 1
        phone = FALSE
        ->  class yes  [0.752]

    Rule 10/11: (14.7/3.2, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = business
        savings_balance = < 100 DM
        years_at_residence > 3
        ->  class yes  [0.749]

    Rule 10/12: (37.9/9.6, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 7
        amount <= 3384
        savings_balance = < 100 DM
        employment_duration in {< 1 year, > 7 years, unemployed}
        percent_of_income <= 3
        job in {skilled, unskilled}
        ->  class yes  [0.735]

    Rule 10/13: (28.7/7.2, lift 1.7)
        checking_balance = unknown
        amount > 709
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, 4 - 7 years}
        percent_of_income > 1
        age <= 44
        other_credit = bank
        ->  class yes  [0.733]

    Rule 10/14: (12.9/3.1, lift 1.6)
        checking_balance = unknown
        employment_duration = unemployed
        ->  class yes  [0.721]

    Rule 10/15: (57.9/19, lift 1.5)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = car
        percent_of_income > 3
        age <= 46
        phone = FALSE
        ->  class yes  [0.665]

    Rule 10/16: (43.1/15.1, lift 1.5)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        employment_duration in {< 1 year, > 7 years}
        percent_of_income > 3
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.644]

    Rule 10/17: (436/226.8, lift 1.1)
        percent_of_income > 3
        ->  class yes  [0.480]

    Rule 10/18: (6.7, lift 1.6)
        amount <= 385
        ->  class no  [0.885]

    Rule 10/19: (6, lift 1.6)
        credit_history in {poor, very good}
        amount > 10722
        ->  class no  [0.875]

    Rule 10/20: (18/2, lift 1.5)
        percent_of_income > 3
        dependents > 1
        phone = TRUE
        ->  class no  [0.850]

    Rule 10/21: (861/368.1, lift 1.0)
        amount <= 10722
        ->  class no  [0.572]

    Default class: no

    -----  Trial 11:  -----

    Rules:

    Rule 11/1: (9.9, lift 1.9)
        checking_balance = 1 - 200 DM
        credit_history in {good, perfect, very good}
        savings_balance = 100 - 500 DM
        housing = rent
        ->  class yes  [0.916]

    Rule 11/2: (9.6, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = perfect
        housing in {other, rent}
        ->  class yes  [0.914]

    Rule 11/3: (8.5, lift 1.9)
        purpose in {education, renovations}
        savings_balance = > 1000 DM
        ->  class yes  [0.905]

    Rule 11/4: (6.5, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        amount > 7678
        savings_balance = < 100 DM
        ->  class yes  [0.882]

    Rule 11/5: (6.3, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        purpose = car
        savings_balance = < 100 DM
        age > 43
        other_credit = none
        ->  class yes  [0.880]

    Rule 11/6: (13/1.3, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 20
        credit_history = poor
        savings_balance = < 100 DM
        ->  class yes  [0.848]

    Rule 11/7: (29.8/4.1, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history in {good, perfect, very good}
        purpose in {car, furniture/appliances}
        savings_balance = 100 - 500 DM
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years, unemployed}
        dependents <= 1
        ->  class yes  [0.840]

    Rule 11/8: (4.2, lift 1.8)
        checking_balance = < 0 DM
        credit_history = good
        savings_balance = < 100 DM
        job = unemployed
        ->  class yes  [0.839]

    Rule 11/9: (6.5/0.4, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history in {good, perfect, very good}
        savings_balance = 100 - 500 DM
        dependents > 1
        ->  class yes  [0.834]

    Rule 11/10: (3.2, lift 1.7)
        savings_balance = 100 - 500 DM
        existing_loans_count > 3
        ->  class yes  [0.809]

    Rule 11/11: (24.9/5, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = very good
        savings_balance = < 100 DM
        percent_of_income > 1
        ->  class yes  [0.775]

    Rule 11/12: (32.9/9.3, lift 1.5)
        purpose = car
        savings_balance = < 100 DM
        other_credit in {bank, store}
        ->  class yes  [0.704]

    Rule 11/13: (27.3/8.4, lift 1.4)
        credit_history = good
        purpose in {education, renovations}
        savings_balance = < 100 DM
        ->  class yes  [0.678]

    Rule 11/14: (46.9/17.4, lift 1.3)
        other_credit = store
        ->  class yes  [0.624]

    Rule 11/15: (634.7/322.7, lift 1.0)
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years}
        ->  class yes  [0.492]

    Rule 11/16: (17.2, lift 1.8)
        checking_balance = unknown
        credit_history = good
        purpose = car
        employment_duration in {< 1 year, > 7 years, 4 - 7 years}
        ->  class no  [0.948]

    Rule 11/17: (15.4, lift 1.8)
        checking_balance = unknown
        purpose = furniture/appliances
        existing_loans_count <= 1
        phone = TRUE
        ->  class no  [0.942]

    Rule 11/18: (10.3, lift 1.7)
        checking_balance = unknown
        purpose = car
        percent_of_income <= 1
        ->  class no  [0.919]

    Rule 11/19: (9.5, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 7
        credit_history = good
        purpose = furniture/appliances
        ->  class no  [0.913]

    Rule 11/20: (9.5, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {perfect, poor}
        savings_balance = unknown
        ->  class no  [0.913]

    Rule 11/21: (9.5, lift 1.7)
        checking_balance = unknown
        credit_history in {critical, perfect, poor}
        purpose = car
        other_credit = none
        phone = FALSE
        ->  class no  [0.913]

    Rule 11/22: (30.6/1.9, lift 1.7)
        checking_balance = unknown
        purpose = car
        other_credit = none
        phone = TRUE
        ->  class no  [0.911]

    Rule 11/23: (8.5, lift 1.7)
        savings_balance = 500 - 1000 DM
        employment_duration in {> 7 years, unemployed}
        other_credit = none
        ->  class no  [0.904]

    Rule 11/24: (50.9/4.8, lift 1.7)
        months_loan_duration <= 42
        credit_history = critical
        purpose = car
        amount <= 7678
        age > 29
        other_credit = none
        ->  class no  [0.890]

    Rule 11/25: (14.1/0.8, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        age <= 36
        job = management
        ->  class no  [0.887]

    Rule 11/26: (6.4, lift 1.7)
        purpose in {business, renovations}
        savings_balance = 100 - 500 DM
        housing = own
        dependents <= 1
        ->  class no  [0.880]

    Rule 11/27: (14.6/1.3, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {critical, poor}
        savings_balance = 100 - 500 DM
        existing_loans_count <= 3
        ->  class no  [0.860]

    Rule 11/28: (4.9, lift 1.6)
        savings_balance = 500 - 1000 DM
        years_at_residence <= 1
        ->  class no  [0.855]

    Rule 11/29: (13.1/1.2, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose in {business, car, furniture/appliances}
        savings_balance = > 1000 DM
        ->  class no  [0.852]

    Rule 11/30: (7.8/0.6, lift 1.6)
        purpose = education
        percent_of_income > 2
        age > 43
        ->  class no  [0.839]

    Rule 11/31: (3.5, lift 1.5)
        credit_history = very good
        percent_of_income <= 1
        other_credit = bank
        ->  class no  [0.817]

    Rule 11/32: (29.6/5.4, lift 1.5)
        credit_history = good
        purpose = car
        amount > 1386
        savings_balance = < 100 DM
        percent_of_income <= 2
        age <= 43
        ->  class no  [0.799]

    Rule 11/33: (25.9/4.7, lift 1.5)
        checking_balance = unknown
        purpose = furniture/appliances
        amount <= 3518
        existing_loans_count > 1
        ->  class no  [0.794]

    Rule 11/34: (15.8/2.9, lift 1.5)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        savings_balance = unknown
        job in {unemployed, unskilled}
        ->  class no  [0.782]

    Rule 11/35: (38.9/10.6, lift 1.4)
        checking_balance = unknown
        purpose = furniture/appliances
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.717]

    Rule 11/36: (170.9/68.8, lift 1.1)
        employment_duration = > 7 years
        housing in {other, own}
        ->  class no  [0.597]

    Rule 11/37: (130.1/53.9, lift 1.1)
        employment_duration = 4 - 7 years
        ->  class no  [0.584]

    Default class: no

    -----  Trial 12:  -----

    Rules:

    Rule 12/1: (15.1, lift 2.1)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 33
        credit_history = good
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, 4 - 7 years}
        years_at_residence > 2
        age > 25
        dependents <= 1
        ->  class yes  [0.942]

    Rule 12/2: (12, lift 2.0)
        months_loan_duration > 15
        months_loan_duration <= 21
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        existing_loans_count <= 1
        job = skilled
        ->  class yes  [0.922]

    Rule 12/3: (9.8, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        percent_of_income > 2
        age <= 29
        existing_loans_count > 1
        ->  class yes  [0.915]

    Rule 12/4: (9.5, lift 2.0)
        checking_balance = < 0 DM
        purpose = car
        amount > 3878
        percent_of_income > 2
        existing_loans_count > 1
        ->  class yes  [0.913]

    Rule 12/5: (9, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = education
        savings_balance in {< 100 DM, > 1000 DM}
        percent_of_income > 3
        job = skilled
        ->  class yes  [0.909]

    Rule 12/6: (8.6, lift 2.0)
        credit_history = very good
        purpose = car
        amount <= 11054
        other_credit = none
        ->  class yes  [0.906]

    Rule 12/7: (12.7/0.6, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 18
        purpose = business
        savings_balance = < 100 DM
        years_at_residence > 1
        other_credit = none
        ->  class yes  [0.893]

    Rule 12/8: (8/0.2, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        amount > 1808
        amount <= 2292
        savings_balance = < 100 DM
        existing_loans_count > 1
        ->  class yes  [0.880]

    Rule 12/9: (5, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        other_credit = store
        ->  class yes  [0.858]

    Rule 12/10: (8.9/0.7, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = poor
        age > 45
        ->  class yes  [0.841]

    Rule 12/11: (10.8/1.1, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        amount > 11054
        ->  class yes  [0.839]

    Rule 12/12: (18.5/2.4, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 36
        purpose = furniture/appliances
        ->  class yes  [0.833]

    Rule 12/13: (4, lift 1.9)
        purpose = business
        employment_duration = 1 - 4 years
        existing_loans_count > 2
        ->  class yes  [0.832]

    Rule 12/14: (15/2.1, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        amount > 2613
        savings_balance in {< 100 DM, unknown}
        employment_duration = > 7 years
        existing_loans_count <= 1
        ->  class yes  [0.819]

    Rule 12/15: (13.2/1.8, lift 1.8)
        checking_balance in {> 200 DM, unknown}
        credit_history = good
        purpose = furniture/appliances
        age <= 44
        existing_loans_count <= 1
        job = unskilled
        phone = FALSE
        ->  class yes  [0.818]

    Rule 12/16: (20.4/3.2, lift 1.8)
        checking_balance = < 0 DM
        credit_history = good
        purpose = car
        savings_balance = < 100 DM
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, 4 - 7 years}
        years_at_residence <= 2
        ->  class yes  [0.811]

    Rule 12/17: (8.8/1.1, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = education
        job in {management, unemployed}
        ->  class yes  [0.810]

    Rule 12/18: (15.3/2.6, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        amount <= 1316
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        existing_loans_count <= 1
        job = skilled
        ->  class yes  [0.794]

    Rule 12/19: (13.6/2.3, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = renovations
        age > 26
        ->  class yes  [0.790]

    Rule 12/20: (11.9/2.1, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        purpose = business
        employment_duration in {< 1 year, unemployed}
        ->  class yes  [0.777]

    Rule 12/21: (18.3/3.7, lift 1.7)
        checking_balance = unknown
        credit_history = good
        purpose = furniture/appliances
        age <= 44
        existing_loans_count > 1
        ->  class yes  [0.769]

    Rule 12/22: (16.8/3.8, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        ->  class yes  [0.743]

    Rule 12/23: (11.3/2.7, lift 1.6)
        credit_history = critical
        purpose = car
        other_credit = bank
        ->  class yes  [0.717]

    Rule 12/24: (51.3/18.5, lift 1.4)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        employment_duration in {4 - 7 years, unemployed}
        ->  class yes  [0.633]

    Rule 12/25: (77.9/31.8, lift 1.3)
        purpose = car
        amount <= 2708
        age <= 33
        ->  class yes  [0.590]

    Rule 12/26: (585.6/298.2, lift 1.1)
        checking_balance in {< 0 DM, 1 - 200 DM}
        ->  class yes  [0.491]

    Rule 12/27: (11.2/0.3, lift 1.6)
        checking_balance in {> 200 DM, unknown}
        credit_history = good
        purpose = car
        employment_duration in {< 1 year, 4 - 7 years, unemployed}
        ->  class no  [0.902]

    Rule 12/28: (35.8/9.3, lift 1.3)
        purpose = business
        employment_duration = 1 - 4 years
        existing_loans_count <= 2
        ->  class no  [0.729]

    Rule 12/29: (92.4/25.2, lift 1.3)
        months_loan_duration > 7
        months_loan_duration <= 15
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        ->  class no  [0.722]

    Rule 12/30: (49.8/16.7, lift 1.2)
        months_loan_duration <= 7
        ->  class no  [0.658]

    Rule 12/31: (871.6/383.9, lift 1.0)
        amount <= 11054
        ->  class no  [0.559]

    Default class: yes

    -----  Trial 13:  -----

    Rules:

    Rule 13/1: (9.2, lift 1.9)
        checking_balance = 1 - 200 DM
        credit_history = good
        amount > 888
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        age <= 30
        phone = FALSE
        ->  class yes  [0.911]

    Rule 13/2: (7.7, lift 1.9)
        credit_history = good
        percent_of_income <= 1
        existing_loans_count > 1
        ->  class yes  [0.897]

    Rule 13/3: (19.3/1.3, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        purpose = car
        amount > 7308
        existing_loans_count <= 1
        job in {management, skilled, unemployed}
        ->  class yes  [0.891]

    Rule 13/4: (16.9/1.3, lift 1.9)
        purpose = furniture/appliances
        employment_duration = < 1 year
        years_at_residence > 2
        age > 26
        existing_loans_count <= 1
        job in {skilled, unskilled}
        ->  class yes  [0.880]

    Rule 13/5: (10.3/0.5, lift 1.9)
        months_loan_duration <= 21
        credit_history = good
        purpose = furniture/appliances
        employment_duration = unemployed
        ->  class yes  [0.878]

    Rule 13/6: (15.4/1.1, lift 1.9)
        credit_history = poor
        savings_balance = < 100 DM
        percent_of_income > 2
        phone = TRUE
        ->  class yes  [0.878]

    Rule 13/7: (11.8/0.7, lift 1.9)
        credit_history = poor
        savings_balance = < 100 DM
        years_at_residence <= 1
        ->  class yes  [0.877]

    Rule 13/8: (27.5/3.5, lift 1.8)
        credit_history = good
        amount > 1352
        employment_duration in {> 7 years, 1 - 4 years, unemployed}
        existing_loans_count > 1
        dependents <= 1
        ->  class yes  [0.847]

    Rule 13/9: (3.6, lift 1.7)
        credit_history = perfect
        other_credit = store
        ->  class yes  [0.822]

    Rule 13/10: (19.2/3.1, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM, unknown}
        credit_history = good
        purpose = education
        age <= 48
        existing_loans_count <= 1
        ->  class yes  [0.808]

    Rule 13/11: (6/0.7, lift 1.7)
        credit_history = poor
        savings_balance = > 1000 DM
        ->  class yes  [0.788]

    Rule 13/12: (19.8/4.4, lift 1.6)
        credit_history = critical
        purpose in {car, education, furniture/appliances}
        age <= 60
        other_credit = bank
        ->  class yes  [0.752]

    Rule 13/13: (49.1/12.7, lift 1.6)
        credit_history = very good
        amount <= 7629
        age > 23
        ->  class yes  [0.732]

    Rule 13/14: (655.6/333.6, lift 1.0)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        ->  class yes  [0.491]

    Rule 13/15: (20.7, lift 1.8)
        checking_balance = unknown
        credit_history = good
        purpose = car
        age > 27
        existing_loans_count <= 1
        ->  class no  [0.956]

    Rule 13/16: (12.5, lift 1.8)
        credit_history = critical
        age > 60
        ->  class no  [0.931]

    Rule 13/17: (11.9, lift 1.8)
        months_loan_duration <= 18
        credit_history = good
        purpose = business
        ->  class no  [0.928]

    Rule 13/18: (8.1, lift 1.7)
        credit_history = good
        savings_balance = > 1000 DM
        existing_loans_count <= 1
        ->  class no  [0.901]

    Rule 13/19: (7.9, lift 1.7)
        credit_history = good
        purpose = furniture/appliances
        amount <= 888
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        ->  class no  [0.899]

    Rule 13/20: (7, lift 1.7)
        months_loan_duration <= 11
        credit_history = critical
        savings_balance = < 100 DM
        job = skilled
        ->  class no  [0.889]

    Rule 13/21: (18.8/1.3, lift 1.7)
        credit_history = good
        purpose = car
        amount <= 7308
        job in {management, unemployed}
        ->  class no  [0.889]

    Rule 13/22: (6, lift 1.7)
        credit_history = critical
        purpose in {business, car0}
        other_credit = bank
        ->  class no  [0.875]

    Rule 13/23: (9.9/0.5, lift 1.6)
        credit_history = good
        amount <= 1352
        percent_of_income > 1
        existing_loans_count > 1
        ->  class no  [0.872]

    Rule 13/24: (9.2/0.5, lift 1.6)
        purpose = furniture/appliances
        employment_duration = < 1 year
        existing_loans_count <= 1
        job in {management, unemployed}
        ->  class no  [0.869]

    Rule 13/25: (12.8/1, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        savings_balance in {> 1000 DM, 100 - 500 DM, unknown}
        other_credit = none
        ->  class no  [0.867]

    Rule 13/26: (5.3, lift 1.6)
        credit_history = very good
        age <= 23
        ->  class no  [0.863]

    Rule 13/27: (5, lift 1.6)
        credit_history = very good
        amount > 7629
        ->  class no  [0.857]

    Rule 13/28: (4.9, lift 1.6)
        credit_history = good
        purpose = education
        age > 48
        ->  class no  [0.854]

    Rule 13/29: (17.6/1.9, lift 1.6)
        purpose = furniture/appliances
        employment_duration = 4 - 7 years
        years_at_residence > 3
        other_credit = none
        existing_loans_count <= 1
        ->  class no  [0.851]

    Rule 13/30: (14.5/1.8, lift 1.6)
        credit_history = critical
        savings_balance = < 100 DM
        other_credit = none
        job = unskilled
        ->  class no  [0.831]

    Rule 13/31: (16.9/2.4, lift 1.6)
        checking_balance in {> 200 DM, 1 - 200 DM, unknown}
        credit_history = good
        purpose = furniture/appliances
        employment_duration = > 7 years
        existing_loans_count <= 1
        ->  class no  [0.821]

    Rule 13/32: (48.6/9.2, lift 1.5)
        checking_balance = unknown
        credit_history = critical
        other_credit = none
        ->  class no  [0.798]

    Rule 13/33: (34.8/7.6, lift 1.5)
        credit_history = good
        purpose = furniture/appliances
        employment_duration = < 1 year
        years_at_residence <= 2
        existing_loans_count <= 1
        dependents <= 1
        ->  class no  [0.767]

    Rule 13/34: (23.4/5.4, lift 1.4)
        credit_history = poor
        savings_balance in {100 - 500 DM, 500 - 1000 DM, unknown}
        ->  class no  [0.750]

    Rule 13/35: (41.6/11.5, lift 1.3)
        credit_history = critical
        purpose = furniture/appliances
        savings_balance = < 100 DM
        age <= 42
        ->  class no  [0.714]

    Rule 13/36: (76.1/22.4, lift 1.3)
        purpose = car
        amount > 1103
        amount <= 7308
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.700]

    Rule 13/37: (299.9/123.4, lift 1.1)
        percent_of_income <= 3
        housing = own
        ->  class no  [0.588]

    Default class: yes

    -----  Trial 14:  -----

    Rules:

    Rule 14/1: (14.1, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 15
        months_loan_duration <= 21
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        job = skilled
        phone = FALSE
        ->  class yes  [0.938]

    Rule 14/2: (8.4, lift 2.0)
        checking_balance = 1 - 200 DM
        credit_history = good
        savings_balance = 100 - 500 DM
        housing = rent
        ->  class yes  [0.903]

    Rule 14/3: (8.3, lift 2.0)
        checking_balance = unknown
        purpose = furniture/appliances
        amount > 4594
        employment_duration = < 1 year
        ->  class yes  [0.903]

    Rule 14/4: (8, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 42
        purpose = furniture/appliances
        savings_balance = < 100 DM
        ->  class yes  [0.900]

    Rule 14/5: (7.8, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        savings_balance = 100 - 500 DM
        age <= 24
        ->  class yes  [0.898]

    Rule 14/6: (6.7, lift 1.9)
        checking_balance = unknown
        employment_duration = 4 - 7 years
        years_at_residence <= 2
        age <= 23
        ->  class yes  [0.885]

    Rule 14/7: (3.5, lift 1.8)
        credit_history = poor
        existing_loans_count > 3
        ->  class yes  [0.820]

    Rule 14/8: (3.2, lift 1.8)
        checking_balance = < 0 DM
        credit_history = very good
        existing_loans_count > 1
        ->  class yes  [0.809]

    Rule 14/9: (18.8/3.5, lift 1.7)
        checking_balance = < 0 DM
        credit_history = good
        purpose in {car, furniture/appliances}
        savings_balance = unknown
        employment_duration in {< 1 year, > 7 years, 1 - 4 years}
        job = skilled
        ->  class yes  [0.783]

    Rule 14/10: (44.6/13, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        savings_balance = < 100 DM
        years_at_residence > 1
        other_credit = bank
        ->  class yes  [0.700]

    Rule 14/11: (31.1/9.4, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = perfect
        ->  class yes  [0.685]

    Rule 14/12: (43.9/14.9, lift 1.4)
        purpose = car
        percent_of_income > 1
        percent_of_income <= 3
        job = skilled
        phone = FALSE
        ->  class yes  [0.652]

    Rule 14/13: (42/14.6, lift 1.4)
        checking_balance in {< 0 DM, 1 - 200 DM}
        amount <= 1680
        employment_duration = < 1 year
        ->  class yes  [0.646]

    Rule 14/14: (45.5/16.6, lift 1.4)
        employment_duration = > 7 years
        existing_loans_count <= 2
        dependents > 1
        ->  class yes  [0.630]

    Rule 14/15: (591.2/307.5, lift 1.0)
        savings_balance = < 100 DM
        ->  class yes  [0.480]

    Rule 14/16: (16, lift 1.7)
        amount <= 7678
        employment_duration = > 7 years
        age <= 30
        other_credit = none
        dependents <= 1
        ->  class no  [0.944]

    Rule 14/17: (13.4, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        months_loan_duration <= 18
        employment_duration = > 7 years
        dependents <= 1
        ->  class no  [0.935]

    Rule 14/18: (11.1, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        months_loan_duration > 33
        employment_duration = > 7 years
        ->  class no  [0.924]

    Rule 14/19: (11.1, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        employment_duration = 1 - 4 years
        percent_of_income <= 1
        ->  class no  [0.924]

    Rule 14/20: (22.8/0.9, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        employment_duration = 4 - 7 years
        years_at_residence > 2
        ->  class no  [0.924]

    Rule 14/21: (7.1, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        years_at_residence <= 1
        ->  class no  [0.891]

    Rule 14/22: (11.9/0.9, lift 1.6)
        amount <= 7678
        years_at_residence > 1
        job = unemployed
        dependents <= 1
        ->  class no  [0.863]

    Rule 14/23: (10.3/0.9, lift 1.6)
        checking_balance in {> 200 DM, unknown}
        purpose = car
        percent_of_income > 3
        job = skilled
        phone = FALSE
        ->  class no  [0.846]

    Rule 14/24: (11.1/1.2, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = poor
        savings_balance in {100 - 500 DM, unknown}
        existing_loans_count <= 3
        ->  class no  [0.835]

    Rule 14/25: (14.5/2.2, lift 1.5)
        checking_balance in {> 200 DM, unknown}
        purpose in {car, education}
        employment_duration = < 1 year
        ->  class no  [0.809]

    Rule 14/26: (72.7/18.9, lift 1.4)
        checking_balance in {> 200 DM, unknown}
        months_loan_duration > 18
        months_loan_duration <= 33
        existing_loans_count <= 2
        dependents <= 1
        ->  class no  [0.733]

    Rule 14/27: (58.1/16.5, lift 1.3)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 22
        other_credit = none
        job = unskilled
        ->  class no  [0.710]

    Rule 14/28: (151.6/50, lift 1.2)
        checking_balance in {> 200 DM, unknown}
        purpose = furniture/appliances
        ->  class no  [0.668]

    Rule 14/29: (139.7/53.7, lift 1.1)
        savings_balance = unknown
        ->  class no  [0.614]

    Rule 14/30: (438.4/191.1, lift 1.0)
        purpose = furniture/appliances
        ->  class no  [0.564]

    Default class: no

    -----  Trial 15:  -----

    Rules:

    Rule 15/1: (16.7, lift 2.0)
        months_loan_duration > 8
        amount <= 1223
        dependents > 1
        phone = FALSE
        ->  class yes  [0.946]

    Rule 15/2: (14.4, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 36
        purpose = furniture/appliances
        years_at_residence > 1
        dependents <= 1
        ->  class yes  [0.939]

    Rule 15/3: (11.9, lift 2.0)
        checking_balance = < 0 DM
        months_loan_duration > 28
        months_loan_duration <= 36
        credit_history = good
        purpose = furniture/appliances
        percent_of_income > 1
        phone = FALSE
        ->  class yes  [0.928]

    Rule 15/4: (8.6, lift 2.0)
        credit_history in {critical, good}
        purpose = car
        savings_balance = < 100 DM
        percent_of_income > 1
        other_credit in {bank, store}
        phone = FALSE
        ->  class yes  [0.905]

    Rule 15/5: (9.1, lift 2.0)
        months_loan_duration > 24
        amount <= 2284
        phone = TRUE
        ->  class yes  [0.905]

    Rule 15/6: (20.2/1.5, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        purpose = furniture/appliances
        amount > 1835
        percent_of_income > 1
        phone = FALSE
        ->  class yes  [0.889]

    Rule 15/7: (6.5, lift 1.9)
        months_loan_duration > 11
        credit_history = very good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        phone = FALSE
        ->  class yes  [0.883]

    Rule 15/8: (30.1/3.2, lift 1.9)
        months_loan_duration > 16
        amount <= 4686
        percent_of_income > 1
        dependents > 1
        phone = FALSE
        ->  class yes  [0.870]

    Rule 15/9: (8.8/0.4, lift 1.9)
        purpose = car
        amount <= 1459
        savings_balance = unknown
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.868]

    Rule 15/10: (7.2/0.4, lift 1.8)
        months_loan_duration <= 24
        amount > 7596
        years_at_residence > 3
        phone = TRUE
        ->  class yes  [0.844]

    Rule 15/11: (15.3/1.7, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 8
        purpose = furniture/appliances
        savings_balance in {100 - 500 DM, 500 - 1000 DM}
        percent_of_income > 1
        years_at_residence > 1
        phone = FALSE
        ->  class yes  [0.843]

    Rule 15/12: (28.7/3.9, lift 1.8)
        months_loan_duration > 24
        age <= 29
        phone = TRUE
        ->  class yes  [0.839]

    Rule 15/13: (11.3/1.2, lift 1.8)
        months_loan_duration > 8
        amount <= 2073
        percent_of_income <= 1
        phone = FALSE
        ->  class yes  [0.836]

    Rule 15/14: (9.5/1, lift 1.8)
        months_loan_duration > 8
        age > 50
        dependents > 1
        ->  class yes  [0.826]

    Rule 15/15: (9.1/1, lift 1.8)
        months_loan_duration <= 8
        amount > 3380
        existing_loans_count <= 1
        ->  class yes  [0.823]

    Rule 15/16: (12/1.6, lift 1.8)
        housing in {own, rent}
        existing_loans_count > 2
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.815]

    Rule 15/17: (18.2/3.6, lift 1.7)
        months_loan_duration > 8
        credit_history in {poor, very good}
        percent_of_income > 1
        years_at_residence <= 1
        ->  class yes  [0.772]

    Rule 15/18: (318.3/146.3, lift 1.2)
        months_loan_duration > 8
        months_loan_duration <= 24
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, unemployed}
        years_at_residence <= 3
        ->  class yes  [0.540]

    Rule 15/19: (842.6/442.1, lift 1.0)
        months_loan_duration > 8
        ->  class yes  [0.475]

    Rule 15/20: (18.8, lift 1.8)
        checking_balance in {> 200 DM, unknown}
        purpose = furniture/appliances
        existing_loans_count <= 1
        phone = TRUE
        ->  class no  [0.952]

    Rule 15/21: (14.2, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        credit_history in {critical, perfect, poor, very good}
        purpose = furniture/appliances
        years_at_residence > 1
        dependents <= 1
        phone = FALSE
        ->  class no  [0.938]

    Rule 15/22: (26.9/1.2, lift 1.7)
        months_loan_duration <= 16
        amount > 1223
        percent_of_income > 1
        age <= 52
        dependents > 1
        ->  class no  [0.925]

    Rule 15/23: (15/0.5, lift 1.7)
        purpose = business
        percent_of_income > 1
        percent_of_income <= 3
        dependents <= 1
        phone = FALSE
        ->  class no  [0.910]

    Rule 15/24: (8.1, lift 1.7)
        purpose = furniture/appliances
        savings_balance = unknown
        existing_loans_count > 1
        phone = FALSE
        ->  class no  [0.901]

    Rule 15/25: (17.8/1.2, lift 1.7)
        months_loan_duration > 10
        months_loan_duration <= 24
        credit_history = critical
        years_at_residence <= 3
        existing_loans_count <= 2
        dependents <= 1
        phone = TRUE
        ->  class no  [0.890]

    Rule 15/26: (11.7/0.7, lift 1.6)
        purpose = car
        amount > 1459
        savings_balance = unknown
        existing_loans_count <= 1
        phone = FALSE
        ->  class no  [0.874]

    Rule 15/27: (4.4, lift 1.6)
        credit_history = poor
        purpose = car
        dependents <= 1
        phone = FALSE
        ->  class no  [0.843]

    Rule 15/28: (28.5/4.4, lift 1.5)
        months_loan_duration > 9
        credit_history in {critical, good}
        purpose = furniture/appliances
        percent_of_income > 1
        years_at_residence <= 1
        dependents <= 1
        phone = FALSE
        ->  class no  [0.822]

    Rule 15/29: (23.6/3.6, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 11
        purpose = furniture/appliances
        percent_of_income > 1
        years_at_residence > 1
        phone = FALSE
        ->  class no  [0.821]

    Rule 15/30: (42.3/9.5, lift 1.4)
        checking_balance in {> 200 DM, 1 - 200 DM, unknown}
        credit_history in {critical, good}
        purpose = car
        savings_balance = < 100 DM
        percent_of_income > 1
        other_credit = none
        ->  class no  [0.763]

    Rule 15/31: (36.6/8.6, lift 1.4)
        months_loan_duration > 8
        months_loan_duration <= 24
        percent_of_income <= 1
        existing_loans_count <= 1
        dependents <= 1
        ->  class no  [0.752]

    Rule 15/32: (57.4/15.1, lift 1.4)
        months_loan_duration <= 8
        ->  class no  [0.728]

    Rule 15/33: (842.6/400.5, lift 1.0)
        months_loan_duration > 8
        ->  class no  [0.525]

    Default class: no

    -----  Trial 16:  -----

    Rules:

    Rule 16/1: (12.5, lift 2.0)
        checking_balance = 1 - 200 DM
        credit_history = good
        amount > 3485
        percent_of_income > 2
        years_at_residence > 1
        housing in {other, own}
        ->  class yes  [0.931]

    Rule 16/2: (12, lift 2.0)
        months_loan_duration <= 33
        amount > 10127
        ->  class yes  [0.929]

    Rule 16/3: (22.6/1.1, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 22
        credit_history = good
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        percent_of_income > 2
        other_credit = none
        housing = rent
        ->  class yes  [0.913]

    Rule 16/4: (8.6, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = perfect
        percent_of_income > 3
        ->  class yes  [0.906]

    Rule 16/5: (7.7, lift 1.9)
        checking_balance = < 0 DM
        amount > 2445
        percent_of_income > 2
        job = unskilled
        ->  class yes  [0.897]

    Rule 16/6: (6.3, lift 1.9)
        checking_balance = unknown
        purpose = furniture/appliances
        amount > 4594
        employment_duration = < 1 year
        ->  class yes  [0.879]

    Rule 16/7: (7.8/0.5, lift 1.8)
        credit_history = critical
        amount > 7057
        savings_balance = < 100 DM
        other_credit = none
        ->  class yes  [0.848]

    Rule 16/8: (20.8/2.9, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = perfect
        years_at_residence > 2
        ->  class yes  [0.827]

    Rule 16/9: (14.2/1.9, lift 1.7)
        credit_history = critical
        amount > 1922
        savings_balance = < 100 DM
        housing = rent
        dependents <= 1
        ->  class yes  [0.819]

    Rule 16/10: (13.6/1.9, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = very good
        other_credit = none
        ->  class yes  [0.814]

    Rule 16/11: (13.3/2.2, lift 1.7)
        checking_balance = unknown
        employment_duration = unemployed
        percent_of_income > 2
        ->  class yes  [0.790]

    Rule 16/12: (8.6/1.2, lift 1.7)
        employment_duration = 4 - 7 years
        years_at_residence <= 3
        age <= 23
        ->  class yes  [0.788]

    Rule 16/13: (30.5/6.2, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 36
        savings_balance = < 100 DM
        dependents <= 1
        ->  class yes  [0.779]

    Rule 16/14: (12.5/2.3, lift 1.6)
        checking_balance in {> 200 DM, unknown}
        purpose in {business, renovations}
        employment_duration = < 1 year
        ->  class yes  [0.774]

    Rule 16/15: (64.4/16.1, lift 1.6)
        checking_balance = < 0 DM
        credit_history = good
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        percent_of_income > 2
        other_credit = none
        job in {skilled, unemployed}
        ->  class yes  [0.742]

    Rule 16/16: (28.1/7.1, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        amount <= 1549
        savings_balance in {< 100 DM, 500 - 1000 DM, unknown}
        percent_of_income <= 2
        ->  class yes  [0.730]

    Rule 16/17: (47.8/14.4, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 30
        credit_history = good
        years_at_residence > 1
        ->  class yes  [0.691]

    Rule 16/18: (35.8/11.8, lift 1.4)
        checking_balance in {< 0 DM, 1 - 200 DM}
        years_at_residence > 1
        other_credit = bank
        phone = FALSE
        ->  class yes  [0.663]

    Rule 16/19: (31.7/10.9, lift 1.4)
        checking_balance in {< 0 DM, 1 - 200 DM}
        other_credit = store
        ->  class yes  [0.647]

    Rule 16/20: (138.9/63, lift 1.2)
        employment_duration = 1 - 4 years
        years_at_residence > 2
        ->  class yes  [0.546]

    Rule 16/21: (570.2/291.8, lift 1.0)
        phone = FALSE
        ->  class yes  [0.488]

    Rule 16/22: (12.1, lift 1.8)
        checking_balance in {> 200 DM, unknown}
        employment_duration = 4 - 7 years
        years_at_residence > 3
        ->  class no  [0.929]

    Rule 16/23: (6.7, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = poor
        savings_balance = unknown
        ->  class no  [0.885]

    Rule 16/24: (9.8/0.7, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        savings_balance in {< 100 DM, 100 - 500 DM, unknown}
        other_credit = bank
        phone = TRUE
        ->  class no  [0.858]

    Rule 16/25: (19.1/2.3, lift 1.6)
        credit_history = critical
        savings_balance = < 100 DM
        dependents > 1
        ->  class no  [0.843]

    Rule 16/26: (11.6/1.3, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        savings_balance = > 1000 DM
        ->  class no  [0.830]

    Rule 16/27: (46.9/7.8, lift 1.6)
        credit_history = critical
        amount <= 1922
        savings_balance = < 100 DM
        ->  class no  [0.820]

    Rule 16/28: (41/8.9, lift 1.5)
        amount <= 2445
        savings_balance = < 100 DM
        percent_of_income > 2
        other_credit = none
        housing in {other, own}
        job = unskilled
        ->  class no  [0.770]

    Rule 16/29: (37.9/9.1, lift 1.4)
        months_loan_duration <= 22
        credit_history = good
        amount > 1603
        other_credit = none
        housing = rent
        ->  class no  [0.747]

    Rule 16/30: (75.7/19.9, lift 1.4)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        amount > 1549
        amount <= 11054
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        percent_of_income <= 2
        age > 20
        ->  class no  [0.730]

    Rule 16/31: (64.2/20.4, lift 1.3)
        checking_balance in {> 200 DM, unknown}
        employment_duration = > 7 years
        ->  class no  [0.677]

    Rule 16/32: (246.2/87.9, lift 1.2)
        checking_balance in {> 200 DM, unknown}
        amount <= 4594
        ->  class no  [0.642]

    Rule 16/33: (694.2/308.4, lift 1.1)
        amount <= 11054
        housing in {other, own}
        ->  class no  [0.556]

    Default class: no

    -----  Trial 17:  -----

    Rules:

    Rule 17/1: (13.8/0.2, lift 1.9)
        checking_balance = unknown
        credit_history = good
        percent_of_income <= 2
        other_credit = none
        existing_loans_count > 1
        ->  class yes  [0.926]

    Rule 17/2: (9.3, lift 1.9)
        checking_balance = unknown
        credit_history = poor
        purpose in {business, furniture/appliances}
        percent_of_income > 3
        age <= 29
        ->  class yes  [0.912]

    Rule 17/3: (12.9/1.2, lift 1.7)
        checking_balance = < 0 DM
        purpose = car
        savings_balance = < 100 DM
        percent_of_income > 2
        percent_of_income <= 3
        ->  class yes  [0.853]

    Rule 17/4: (16.1/2.3, lift 1.7)
        checking_balance = unknown
        credit_history = good
        savings_balance = < 100 DM
        age <= 24
        phone = FALSE
        ->  class yes  [0.818]

    Rule 17/5: (10.4/1.5, lift 1.6)
        checking_balance = > 200 DM
        dependents > 1
        ->  class yes  [0.798]

    Rule 17/6: (23/4.1, lift 1.6)
        checking_balance = < 0 DM
        credit_history in {perfect, poor}
        savings_balance = < 100 DM
        ->  class yes  [0.797]

    Rule 17/7: (15.6/2.6, lift 1.6)
        checking_balance = unknown
        savings_balance in {500 - 1000 DM, unknown}
        percent_of_income > 1
        other_credit = bank
        housing = own
        ->  class yes  [0.793]

    Rule 17/8: (7.7/1.1, lift 1.6)
        checking_balance = 1 - 200 DM
        purpose = car
        job = unskilled
        ->  class yes  [0.785]

    Rule 17/9: (11.2/2, lift 1.6)
        checking_balance = 1 - 200 DM
        savings_balance in {100 - 500 DM, 500 - 1000 DM}
        job = unskilled
        ->  class yes  [0.776]

    Rule 17/10: (19.1/4.1, lift 1.5)
        checking_balance = 1 - 200 DM
        amount > 1808
        savings_balance = < 100 DM
        job = unskilled
        ->  class yes  [0.758]

    Rule 17/11: (27/6.7, lift 1.5)
        checking_balance = unknown
        purpose in {business, car, education}
        percent_of_income > 1
        other_credit = bank
        ->  class yes  [0.735]

    Rule 17/12: (37/10.3, lift 1.4)
        checking_balance = < 0 DM
        purpose = car
        savings_balance = < 100 DM
        employment_duration in {< 1 year, > 7 years, 4 - 7 years}
        percent_of_income > 2
        ->  class yes  [0.710]

    Rule 17/13: (26.7/7.6, lift 1.4)
        checking_balance = > 200 DM
        percent_of_income > 3
        age > 24
        ->  class yes  [0.701]

    Rule 17/14: (51.7/15.1, lift 1.4)
        checking_balance = 1 - 200 DM
        months_loan_duration > 8
        credit_history in {critical, good, perfect}
        purpose = furniture/appliances
        years_at_residence <= 3
        job = skilled
        ->  class yes  [0.701]

    Rule 17/15: (50.3/16.7, lift 1.3)
        checking_balance = 1 - 200 DM
        job = management
        ->  class yes  [0.663]

    Rule 17/16: (524.3/244.1, lift 1.1)
        months_loan_duration > 16
        ->  class yes  [0.534]

    Rule 17/17: (310.5/148, lift 1.1)
        checking_balance = < 0 DM
        ->  class yes  [0.523]

    Rule 17/18: (23.8, lift 1.9)
        checking_balance = unknown
        credit_history = critical
        age > 30
        other_credit = none
        ->  class no  [0.961]

    Rule 17/19: (19.8, lift 1.9)
        checking_balance = unknown
        credit_history = good
        savings_balance = < 100 DM
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years, unemployed}
        age > 24
        other_credit = none
        existing_loans_count <= 1
        ->  class no  [0.954]

    Rule 17/20: (16.6, lift 1.9)
        checking_balance = 1 - 200 DM
        purpose in {education, furniture/appliances, renovations}
        amount <= 1808
        savings_balance = < 100 DM
        job = unskilled
        ->  class no  [0.946]

    Rule 17/21: (13.3, lift 1.8)
        months_loan_duration <= 8
        credit_history in {critical, good, perfect}
        years_at_residence <= 3
        job = skilled
        ->  class no  [0.934]

    Rule 17/22: (12.7, lift 1.8)
        checking_balance = < 0 DM
        credit_history = good
        purpose = furniture/appliances
        employment_duration = < 1 year
        years_at_residence <= 2
        ->  class no  [0.932]

    Rule 17/23: (8.7, lift 1.8)
        checking_balance = < 0 DM
        credit_history = critical
        purpose = furniture/appliances
        amount <= 2039
        ->  class no  [0.907]

    Rule 17/24: (7.8, lift 1.8)
        checking_balance = > 200 DM
        age <= 24
        ->  class no  [0.898]

    Rule 17/25: (7.5, lift 1.8)
        checking_balance = unknown
        purpose = furniture/appliances
        savings_balance in {< 100 DM, 100 - 500 DM}
        other_credit = bank
        housing = own
        ->  class no  [0.895]

    Rule 17/26: (7.3, lift 1.8)
        checking_balance = < 0 DM
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        job in {management, unskilled}
        ->  class no  [0.893]

    Rule 17/27: (4.4, lift 1.7)
        checking_balance = 1 - 200 DM
        savings_balance in {> 1000 DM, unknown}
        job = unskilled
        ->  class no  [0.843]

    Rule 17/28: (18.6/2.3, lift 1.6)
        checking_balance = < 0 DM
        purpose = car
        savings_balance = < 100 DM
        percent_of_income <= 2
        housing in {other, rent}
        ->  class no  [0.839]

    Rule 17/29: (3.9, lift 1.6)
        checking_balance = > 200 DM
        other_credit = bank
        dependents <= 1
        ->  class no  [0.832]

    Rule 17/30: (24/5.1, lift 1.5)
        checking_balance = unknown
        percent_of_income <= 1
        ->  class no  [0.767]

    Rule 17/31: (35.9/8.3, lift 1.5)
        checking_balance = 1 - 200 DM
        purpose = furniture/appliances
        years_at_residence > 3
        ->  class no  [0.754]

    Rule 17/32: (41.4/11.6, lift 1.4)
        checking_balance = unknown
        credit_history = good
        savings_balance in {> 1000 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        other_credit = none
        existing_loans_count <= 1
        ->  class no  [0.710]

    Rule 17/33: (43.5/14.4, lift 1.3)
        checking_balance = < 0 DM
        purpose = car
        percent_of_income <= 2
        ->  class no  [0.662]

    Rule 17/34: (65.3/22.4, lift 1.3)
        checking_balance = unknown
        percent_of_income > 2
        existing_loans_count > 1
        ->  class no  [0.653]

    Rule 17/35: (394.3/178.7, lift 1.1)
        housing = own
        job = skilled
        ->  class no  [0.547]

    Default class: yes

    -----  Trial 18:  -----

    Rules:

    Rule 18/1: (8.8, lift 2.0)
        credit_history = good
        purpose = car
        amount <= 1680
        savings_balance = < 100 DM
        employment_duration = 4 - 7 years
        ->  class yes  [0.907]

    Rule 18/2: (8.2, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        purpose = car
        employment_duration = < 1 year
        housing in {other, rent}
        ->  class yes  [0.901]

    Rule 18/3: (7.5, lift 2.0)
        purpose in {education, renovations}
        savings_balance = > 1000 DM
        ->  class yes  [0.895]

    Rule 18/4: (6.4, lift 2.0)
        credit_history = perfect
        savings_balance = < 100 DM
        housing in {other, rent}
        ->  class yes  [0.881]

    Rule 18/5: (6.3, lift 2.0)
        credit_history = very good
        savings_balance = < 100 DM
        percent_of_income > 1
        percent_of_income <= 2
        ->  class yes  [0.879]

    Rule 18/6: (6, lift 2.0)
        purpose = furniture/appliances
        savings_balance = < 100 DM
        age > 36
        other_credit = none
        job = management
        ->  class yes  [0.876]

    Rule 18/7: (5.9, lift 1.9)
        credit_history = very good
        savings_balance = < 100 DM
        other_credit = none
        ->  class yes  [0.874]

    Rule 18/8: (5.6, lift 1.9)
        credit_history = perfect
        savings_balance = < 100 DM
        age > 40
        ->  class yes  [0.869]

    Rule 18/9: (5.4, lift 1.9)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        ->  class yes  [0.865]

    Rule 18/10: (4.5, lift 1.9)
        credit_history = perfect
        savings_balance = < 100 DM
        job = management
        ->  class yes  [0.847]

    Rule 18/11: (9.7/0.8, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration <= 13
        credit_history = critical
        savings_balance = < 100 DM
        percent_of_income <= 3
        other_credit = none
        dependents <= 1
        ->  class yes  [0.846]

    Rule 18/12: (7.3/0.4, lift 1.9)
        credit_history = good
        purpose = education
        savings_balance = < 100 DM
        existing_loans_count <= 1
        phone = FALSE
        ->  class yes  [0.845]

    Rule 18/13: (12/1.2, lift 1.9)
        checking_balance = 1 - 200 DM
        credit_history = critical
        savings_balance = < 100 DM
        percent_of_income <= 2
        years_at_residence > 1
        housing = own
        ->  class yes  [0.843]

    Rule 18/14: (3.9, lift 1.8)
        credit_history = poor
        purpose = car
        employment_duration = unemployed
        ->  class yes  [0.829]

    Rule 18/15: (8.8/0.9, lift 1.8)
        credit_history = critical
        amount <= 1345
        savings_balance = unknown
        ->  class yes  [0.828]

    Rule 18/16: (19.6/2.8, lift 1.8)
        checking_balance = 1 - 200 DM
        credit_history = critical
        savings_balance = < 100 DM
        years_at_residence > 1
        age <= 33
        housing = own
        ->  class yes  [0.825]

    Rule 18/17: (3.5, lift 1.8)
        credit_history = perfect
        savings_balance = < 100 DM
        dependents > 1
        ->  class yes  [0.818]

    Rule 18/18: (25.8/4.8, lift 1.8)
        months_loan_duration > 11
        purpose in {car, furniture/appliances}
        savings_balance = 500 - 1000 DM
        years_at_residence > 1
        other_credit in {bank, none}
        job in {skilled, unskilled}
        ->  class yes  [0.793]

    Rule 18/19: (10.6/1.6, lift 1.8)
        checking_balance = 1 - 200 DM
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        employment_duration in {< 1 year, 1 - 4 years}
        ->  class yes  [0.789]

    Rule 18/20: (13.6/2.3, lift 1.8)
        purpose = education
        savings_balance = 100 - 500 DM
        ->  class yes  [0.788]

    Rule 18/21: (30/6, lift 1.7)
        purpose = furniture/appliances
        amount > 1316
        savings_balance = < 100 DM
        employment_duration = < 1 year
        age > 22
        age <= 43
        other_credit = none
        job in {skilled, unskilled}
        ->  class yes  [0.782]

    Rule 18/22: (8.9/1.7, lift 1.7)
        credit_history = good
        purpose = renovations
        savings_balance = < 100 DM
        ->  class yes  [0.756]

    Rule 18/23: (28.3/7.4, lift 1.6)
        credit_history = poor
        purpose in {business, education, furniture/appliances}
        savings_balance = < 100 DM
        ->  class yes  [0.723]

    Rule 18/24: (760.8/408.4, lift 1.0)
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, unemployed}
        ->  class yes  [0.463]

    Rule 18/25: (12.6, lift 1.7)
        purpose = furniture/appliances
        amount <= 1316
        savings_balance = < 100 DM
        employment_duration = < 1 year
        age <= 43
        ->  class no  [0.931]

    Rule 18/26: (12.1, lift 1.7)
        credit_history = critical
        years_at_residence <= 1
        housing = own
        ->  class no  [0.929]

    Rule 18/27: (9.1, lift 1.7)
        checking_balance = unknown
        savings_balance = unknown
        other_credit = none
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.910]

    Rule 18/28: (9, lift 1.6)
        checking_balance = 1 - 200 DM
        credit_history = critical
        housing in {other, rent}
        dependents <= 1
        ->  class no  [0.909]

    Rule 18/29: (25/1.7, lift 1.6)
        purpose in {business, car, furniture/appliances}
        savings_balance = > 1000 DM
        existing_loans_count <= 2
        ->  class no  [0.899]

    Rule 18/30: (7.3, lift 1.6)
        months_loan_duration > 10
        savings_balance = < 100 DM
        employment_duration = < 1 year
        age <= 22
        other_credit = none
        ->  class no  [0.892]

    Rule 18/31: (7.1, lift 1.6)
        savings_balance = 500 - 1000 DM
        job in {management, unemployed}
        ->  class no  [0.890]

    Rule 18/32: (4.3, lift 1.5)
        credit_history = very good
        percent_of_income <= 1
        other_credit = bank
        ->  class no  [0.841]

    Rule 18/33: (3.6, lift 1.5)
        savings_balance = 500 - 1000 DM
        other_credit = store
        ->  class no  [0.820]

    Rule 18/34: (11/1.5, lift 1.5)
        credit_history = good
        purpose = car
        employment_duration = unemployed
        ->  class no  [0.805]

    Rule 18/35: (28.1/5.3, lift 1.4)
        checking_balance = < 0 DM
        credit_history = critical
        percent_of_income > 3
        other_credit = none
        ->  class no  [0.791]

    Rule 18/36: (15.4/2.8, lift 1.4)
        purpose in {business, car0, renovations}
        savings_balance = 100 - 500 DM
        ->  class no  [0.783]

    Rule 18/37: (25.8/5.1, lift 1.4)
        credit_history in {perfect, poor, very good}
        savings_balance = unknown
        ->  class no  [0.782]

    Rule 18/38: (28.6/6.5, lift 1.4)
        purpose = car
        amount > 1680
        employment_duration = 4 - 7 years
        ->  class no  [0.756]

    Rule 18/39: (141.7/49.1, lift 1.2)
        purpose = furniture/appliances
        employment_duration in {> 7 years, 4 - 7 years}
        ->  class no  [0.651]

    Rule 18/40: (700.9/302.9, lift 1.0)
        age > 25
        ->  class no  [0.568]

    Default class: no

    -----  Trial 19:  -----

    Rules:

    Rule 19/1: (20.5, lift 2.0)
        checking_balance = < 0 DM
        months_loan_duration > 15
        credit_history = good
        amount > 1797
        amount <= 3384
        savings_balance = < 100 DM
        other_credit = none
        job in {skilled, unemployed, unskilled}
        dependents <= 1
        ->  class yes  [0.956]

    Rule 19/2: (12.8, lift 2.0)
        checking_balance = < 0 DM
        months_loan_duration > 15
        credit_history in {poor, very good}
        savings_balance = < 100 DM
        housing in {other, own}
        job in {skilled, unskilled}
        ->  class yes  [0.933]

    Rule 19/3: (10.9, lift 1.9)
        months_loan_duration <= 15
        employment_duration in {< 1 year, > 7 years, unemployed}
        other_credit = store
        ->  class yes  [0.918]

    Rule 19/4: (14.8/0.4, lift 1.9)
        months_loan_duration > 15
        months_loan_duration <= 22
        savings_balance = < 100 DM
        dependents > 1
        ->  class yes  [0.915]

    Rule 19/5: (25.3/2, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration > 15
        credit_history = good
        amount > 1797
        savings_balance = < 100 DM
        years_at_residence <= 3
        other_credit = none
        job in {skilled, unemployed, unskilled}
        dependents <= 1
        ->  class yes  [0.889]

    Rule 19/6: (20.5/1.9, lift 1.8)
        months_loan_duration <= 15
        amount > 3949
        phone = TRUE
        ->  class yes  [0.871]

    Rule 19/7: (4.5, lift 1.8)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        ->  class yes  [0.846]

    Rule 19/8: (13.9/1.6, lift 1.8)
        months_loan_duration <= 15
        credit_history = critical
        employment_duration = > 7 years
        years_at_residence > 2
        years_at_residence <= 3
        ->  class yes  [0.836]

    Rule 19/9: (46.1/8.2, lift 1.7)
        checking_balance = 1 - 200 DM
        months_loan_duration > 22
        savings_balance = < 100 DM
        years_at_residence > 1
        housing in {other, own}
        ->  class yes  [0.808]

    Rule 19/10: (2.6, lift 1.6)
        savings_balance = > 1000 DM
        existing_loans_count > 2
        ->  class yes  [0.784]

    Rule 19/11: (36.7/7.7, lift 1.6)
        months_loan_duration > 15
        savings_balance = < 100 DM
        percent_of_income > 1
        age <= 61
        housing = rent
        ->  class yes  [0.776]

    Rule 19/12: (12.7/2.4, lift 1.6)
        credit_history = good
        purpose = car
        existing_loans_count > 1
        ->  class yes  [0.768]

    Rule 19/13: (21.3/5.2, lift 1.5)
        savings_balance = < 100 DM
        other_credit in {bank, store}
        dependents > 1
        ->  class yes  [0.735]

    Rule 19/14: (39.6/10.6, lift 1.5)
        months_loan_duration > 15
        savings_balance = 100 - 500 DM
        employment_duration in {< 1 year, > 7 years, 1 - 4 years}
        years_at_residence > 1
        ->  class yes  [0.722]

    Rule 19/15: (19/5.2, lift 1.5)
        months_loan_duration <= 15
        other_credit = none
        housing = other
        ->  class yes  [0.706]

    Rule 19/16: (44.8/14.5, lift 1.4)
        months_loan_duration > 15
        credit_history = good
        savings_balance = unknown
        employment_duration in {< 1 year, > 7 years, 1 - 4 years}
        ->  class yes  [0.669]

    Rule 19/17: (84/30.1, lift 1.3)
        credit_history = good
        purpose = car
        percent_of_income > 1
        phone = FALSE
        ->  class yes  [0.638]

    Rule 19/18: (535.2/254.6, lift 1.1)
        months_loan_duration > 15
        ->  class yes  [0.524]

    Rule 19/19: (15.6, lift 1.8)
        months_loan_duration <= 15
        credit_history = critical
        amount <= 3949
        employment_duration = > 7 years
        years_at_residence > 3
        ->  class no  [0.943]

    Rule 19/20: (12.1, lift 1.8)
        months_loan_duration > 15
        months_loan_duration <= 42
        savings_balance = > 1000 DM
        existing_loans_count <= 2
        ->  class no  [0.929]

    Rule 19/21: (11.1, lift 1.8)
        checking_balance in {> 200 DM, unknown}
        months_loan_duration > 15
        savings_balance = 500 - 1000 DM
        other_credit = none
        ->  class no  [0.924]

    Rule 19/22: (10.8, lift 1.8)
        months_loan_duration > 22
        amount <= 1797
        savings_balance = < 100 DM
        other_credit in {bank, none}
        housing = own
        dependents <= 1
        ->  class no  [0.922]

    Rule 19/23: (10.6, lift 1.8)
        checking_balance = > 200 DM
        months_loan_duration > 15
        percent_of_income <= 3
        housing in {other, own}
        dependents <= 1
        ->  class no  [0.921]

    Rule 19/24: (18.2/1.2, lift 1.7)
        checking_balance = 1 - 200 DM
        months_loan_duration > 15
        years_at_residence <= 1
        housing = own
        ->  class no  [0.890]

    Rule 19/25: (6.9, lift 1.7)
        months_loan_duration > 13
        months_loan_duration <= 15
        credit_history = good
        purpose = car
        phone = FALSE
        ->  class no  [0.888]

    Rule 19/26: (10.9/0.5, lift 1.7)
        purpose = car
        percent_of_income <= 1
        other_credit in {bank, none}
        housing in {own, rent}
        existing_loans_count <= 1
        phone = FALSE
        ->  class no  [0.881]

    Rule 19/27: (19.7/1.7, lift 1.7)
        months_loan_duration <= 7
        credit_history = good
        amount <= 3949
        ->  class no  [0.874]

    Rule 19/28: (17.8/1.5, lift 1.7)
        checking_balance = < 0 DM
        months_loan_duration > 15
        savings_balance = < 100 DM
        housing in {other, own}
        job = management
        dependents <= 1
        ->  class no  [0.873]

    Rule 19/29: (44.6/5, lift 1.7)
        months_loan_duration <= 15
        credit_history = critical
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years, unemployed}
        other_credit in {bank, none}
        housing in {own, rent}
        ->  class no  [0.871]

    Rule 19/30: (15/1.2, lift 1.7)
        months_loan_duration > 15
        savings_balance = 100 - 500 DM
        employment_duration in {4 - 7 years, unemployed}
        ->  class no  [0.869]

    Rule 19/31: (32.5/5.5, lift 1.6)
        months_loan_duration > 7
        months_loan_duration <= 15
        purpose in {business, education}
        amount <= 3949
        other_credit in {bank, none}
        housing in {own, rent}
        ->  class no  [0.813]

    Rule 19/32: (24.2/5.5, lift 1.4)
        months_loan_duration > 15
        credit_history in {critical, perfect, poor}
        savings_balance = unknown
        ->  class no  [0.751]

    Rule 19/33: (236.4/96.9, lift 1.1)
        checking_balance = unknown
        ->  class no  [0.589]

    Rule 19/34: (415.1/180.7, lift 1.1)
        purpose = furniture/appliances
        ->  class no  [0.564]

    Default class: yes

    -----  Trial 20:  -----

    Rules:

    Rule 20/1: (17.5/0.2, lift 2.0)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 11
        credit_history = good
        amount > 888
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        percent_of_income > 1
        age > 30
        housing = own
        existing_loans_count <= 1
        job in {skilled, unskilled}
        ->  class yes  [0.940]

    Rule 20/2: (12.2, lift 2.0)
        checking_balance = < 0 DM
        credit_history = good
        purpose = car
        amount <= 7596
        savings_balance = < 100 DM
        years_at_residence <= 2
        job in {skilled, unskilled}
        ->  class yes  [0.929]

    Rule 20/3: (10.1/0.2, lift 1.9)
        months_loan_duration <= 10
        credit_history = good
        purpose = furniture/appliances
        amount > 1316
        savings_balance = < 100 DM
        employment_duration = < 1 year
        job = skilled
        ->  class yes  [0.902]

    Rule 20/4: (14.5/0.8, lift 1.9)
        credit_history = poor
        savings_balance = < 100 DM
        percent_of_income > 2
        phone = TRUE
        ->  class yes  [0.890]

    Rule 20/5: (10.3/0.4, lift 1.9)
        credit_history = perfect
        housing in {other, rent}
        ->  class yes  [0.886]

    Rule 20/6: (16.9/1.3, lift 1.9)
        purpose = car
        amount <= 7596
        savings_balance = < 100 DM
        other_credit = bank
        job in {skilled, unskilled}
        ->  class yes  [0.878]

    Rule 20/7: (11.2/0.7, lift 1.9)
        credit_history = perfect
        savings_balance = < 100 DM
        age > 33
        ->  class yes  [0.871]

    Rule 20/8: (5.1, lift 1.8)
        purpose in {education, renovations}
        savings_balance = > 1000 DM
        ->  class yes  [0.860]

    Rule 20/9: (3, lift 1.7)
        months_loan_duration > 42
        savings_balance = > 1000 DM
        ->  class yes  [0.800]

    Rule 20/10: (17.4/2.9, lift 1.7)
        months_loan_duration <= 42
        purpose = car
        amount > 6887
        employment_duration = > 7 years
        ->  class yes  [0.798]

    Rule 20/11: (12.9/2.1, lift 1.7)
        credit_history = good
        purpose = education
        savings_balance = < 100 DM
        years_at_residence > 3
        job in {skilled, unskilled}
        ->  class yes  [0.789]

    Rule 20/12: (23.1/5.2, lift 1.6)
        months_loan_duration > 10
        amount <= 7596
        employment_duration = < 1 year
        existing_loans_count <= 1
        job = unskilled
        ->  class yes  [0.755]

    Rule 20/13: (16.4/4, lift 1.6)
        credit_history = good
        purpose = furniture/appliances
        employment_duration = unemployed
        ->  class yes  [0.731]

    Rule 20/14: (32.9/10.3, lift 1.4)
        checking_balance = < 0 DM
        savings_balance = < 100 DM
        employment_duration = 4 - 7 years
        job in {skilled, unskilled}
        ->  class yes  [0.677]

    Rule 20/15: (92.5/35.1, lift 1.3)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = furniture/appliances
        amount > 888
        employment_duration = 1 - 4 years
        other_credit = none
        job in {skilled, unskilled}
        ->  class yes  [0.618]

    Rule 20/16: (574.6/290.6, lift 1.1)
        savings_balance = < 100 DM
        ->  class yes  [0.494]

    Rule 20/17: (17.5, lift 1.8)
        months_loan_duration <= 42
        purpose in {business, car, furniture/appliances}
        savings_balance = > 1000 DM
        existing_loans_count <= 2
        ->  class no  [0.949]

    Rule 20/18: (14.6, lift 1.8)
        months_loan_duration > 11
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        years_at_residence > 2
        age <= 30
        other_credit = none
        housing = own
        job in {skilled, unskilled}
        ->  class no  [0.940]

    Rule 20/19: (14.5, lift 1.8)
        credit_history = good
        purpose = car
        amount > 4308
        amount <= 7596
        savings_balance = < 100 DM
        ->  class no  [0.939]

    Rule 20/20: (10.3, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = good
        amount > 1283
        amount <= 7596
        percent_of_income <= 1
        housing = own
        existing_loans_count <= 1
        ->  class no  [0.919]

    Rule 20/21: (9.8, lift 1.7)
        months_loan_duration <= 11
        amount > 1283
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        other_credit = none
        job in {skilled, unskilled}
        ->  class no  [0.915]

    Rule 20/22: (5.3, lift 1.6)
        months_loan_duration > 42
        purpose = car
        savings_balance = unknown
        ->  class no  [0.863]

    Rule 20/23: (15.9/2.6, lift 1.5)
        savings_balance = 500 - 1000 DM
        employment_duration in {> 7 years, unemployed}
        ->  class no  [0.801]

    Rule 20/24: (16.9/3.5, lift 1.4)
        savings_balance = < 100 DM
        employment_duration = unemployed
        percent_of_income <= 2
        ->  class no  [0.761]

    Rule 20/25: (26.1/7, lift 1.3)
        purpose in {business, education, renovations}
        savings_balance = unknown
        ->  class no  [0.717]

    Rule 20/26: (40.1/10.9, lift 1.3)
        savings_balance = unknown
        employment_duration in {< 1 year, 4 - 7 years}
        ->  class no  [0.717]

    Rule 20/27: (100.5/32.1, lift 1.3)
        credit_history = critical
        amount <= 7596
        savings_balance = < 100 DM
        other_credit = none
        ->  class no  [0.677]

    Rule 20/28: (144.9/56.7, lift 1.1)
        checking_balance in {> 200 DM, unknown}
        purpose = furniture/appliances
        ->  class no  [0.607]

    Rule 20/29: (807.8/362.4, lift 1.0)
        amount <= 7596
        ->  class no  [0.551]

    Default class: yes

    -----  Trial 21:  -----

    Rules:

    Rule 21/1: (11.5, lift 1.8)
        checking_balance = < 0 DM
        percent_of_income > 1
        percent_of_income <= 3
        age <= 24
        job = skilled
        ->  class yes  [0.926]

    Rule 21/2: (11.5, lift 1.8)
        amount > 12204
        savings_balance in {< 100 DM, 100 - 500 DM, unknown}
        age <= 57
        existing_loans_count <= 1
        phone = TRUE
        ->  class yes  [0.926]

    Rule 21/3: (15.4/0.7, lift 1.7)
        months_loan_duration > 42
        percent_of_income > 3
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.900]

    Rule 21/4: (24/1.8, lift 1.7)
        amount <= 1967
        savings_balance = unknown
        percent_of_income > 3
        years_at_residence > 1
        age <= 42
        existing_loans_count <= 1
        job in {skilled, unskilled}
        phone = FALSE
        ->  class yes  [0.894]

    Rule 21/5: (22.3/1.7, lift 1.7)
        checking_balance = 1 - 200 DM
        amount <= 1659
        percent_of_income <= 3
        job = skilled
        phone = FALSE
        ->  class yes  [0.888]

    Rule 21/6: (40.6/5.7, lift 1.6)
        amount <= 4351
        percent_of_income > 3
        dependents > 1
        phone = FALSE
        ->  class yes  [0.843]

    Rule 21/7: (29.4/4.6, lift 1.6)
        months_loan_duration > 33
        existing_loans_count > 1
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.822]

    Rule 21/8: (19.3/2.9, lift 1.6)
        credit_history = good
        percent_of_income <= 3
        existing_loans_count > 1
        phone = TRUE
        ->  class yes  [0.817]

    Rule 21/9: (32.2/5.3, lift 1.6)
        savings_balance in {100 - 500 DM, 500 - 1000 DM}
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years, unemployed}
        percent_of_income > 3
        years_at_residence > 1
        phone = FALSE
        ->  class yes  [0.815]

    Rule 21/10: (39.1/11.3, lift 1.4)
        employment_duration = 4 - 7 years
        percent_of_income > 3
        phone = FALSE
        ->  class yes  [0.699]

    Rule 21/11: (81.9/28.5, lift 1.3)
        months_loan_duration <= 33
        credit_history in {perfect, very good}
        ->  class yes  [0.649]

    Rule 21/12: (51.9/20.5, lift 1.2)
        other_credit = store
        ->  class yes  [0.600]

    Rule 21/13: (653.1/299.9, lift 1.0)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        ->  class yes  [0.541]

    Rule 21/14: (24.2, lift 2.0)
        checking_balance = unknown
        credit_history = good
        purpose in {business, car, furniture/appliances}
        other_credit = none
        existing_loans_count <= 1
        phone = TRUE
        ->  class no  [0.962]

    Rule 21/15: (17.8, lift 2.0)
        amount <= 3711
        dependents > 1
        phone = TRUE
        ->  class no  [0.949]

    Rule 21/16: (13.4, lift 1.9)
        months_loan_duration <= 16
        credit_history in {critical, poor}
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years, unemployed}
        percent_of_income > 3
        phone = FALSE
        ->  class no  [0.935]

    Rule 21/17: (13.3, lift 1.9)
        checking_balance in {> 200 DM, unknown}
        employment_duration = > 7 years
        percent_of_income > 3
        dependents <= 1
        phone = FALSE
        ->  class no  [0.935]

    Rule 21/18: (12.3, lift 1.9)
        savings_balance = unknown
        percent_of_income > 3
        years_at_residence > 1
        age > 42
        ->  class no  [0.930]

    Rule 21/19: (11.9, lift 1.9)
        purpose in {business, education, renovations}
        percent_of_income <= 3
        job = unskilled
        phone = FALSE
        ->  class no  [0.928]

    Rule 21/20: (11.8, lift 1.9)
        checking_balance = unknown
        credit_history = critical
        amount <= 5045
        other_credit in {bank, none}
        phone = TRUE
        ->  class no  [0.927]

    Rule 21/21: (10.9, lift 1.9)
        savings_balance in {> 1000 DM, 500 - 1000 DM}
        existing_loans_count <= 1
        phone = TRUE
        ->  class no  [0.923]

    Rule 21/22: (7.7, lift 1.9)
        checking_balance = > 200 DM
        age <= 57
        existing_loans_count <= 1
        phone = TRUE
        ->  class no  [0.897]

    Rule 21/23: (24.9/1.8, lift 1.9)
        savings_balance in {> 1000 DM, unknown}
        percent_of_income <= 3
        age <= 45
        phone = FALSE
        ->  class no  [0.895]

    Rule 21/24: (7.3, lift 1.9)
        savings_balance = unknown
        percent_of_income > 3
        existing_loans_count > 1
        dependents <= 1
        phone = FALSE
        ->  class no  [0.892]

    Rule 21/25: (12.7/0.7, lift 1.8)
        savings_balance = < 100 DM
        employment_duration in {> 7 years, 4 - 7 years}
        percent_of_income <= 3
        job = unskilled
        ->  class no  [0.884]

    Rule 21/26: (6, lift 1.8)
        savings_balance = > 1000 DM
        percent_of_income > 3
        dependents <= 1
        ->  class no  [0.875]

    Rule 21/27: (13.6/1, lift 1.8)
        months_loan_duration <= 33
        credit_history = poor
        existing_loans_count > 1
        phone = TRUE
        ->  class no  [0.869]

    Rule 21/28: (17.5/1.8, lift 1.8)
        amount > 9271
        amount <= 12204
        existing_loans_count <= 1
        phone = TRUE
        ->  class no  [0.857]

    Rule 21/29: (4.4, lift 1.8)
        savings_balance = 500 - 1000 DM
        percent_of_income <= 3
        job = unskilled
        ->  class no  [0.844]

    Rule 21/30: (16.9/2.3, lift 1.7)
        age > 57
        dependents <= 1
        phone = TRUE
        ->  class no  [0.823]

    Rule 21/31: (3.5, lift 1.7)
        percent_of_income > 3
        years_at_residence > 1
        job = unemployed
        ->  class no  [0.819]

    Rule 21/32: (14.4/2, lift 1.7)
        credit_history = good
        amount <= 9271
        savings_balance in {< 100 DM, 100 - 500 DM, unknown}
        other_credit = bank
        phone = TRUE
        ->  class no  [0.815]

    Rule 21/33: (27.5/4.8, lift 1.7)
        amount <= 4686
        employment_duration = > 7 years
        percent_of_income > 3
        years_at_residence > 1
        other_credit = none
        dependents <= 1
        phone = FALSE
        ->  class no  [0.803]

    Rule 21/34: (31.5/6.2, lift 1.6)
        employment_duration in {< 1 year, > 7 years, 1 - 4 years}
        percent_of_income > 3
        years_at_residence <= 1
        job in {skilled, unskilled}
        dependents <= 1
        phone = FALSE
        ->  class no  [0.785]

    Rule 21/35: (23.1/5, lift 1.6)
        months_loan_duration > 18
        dependents > 1
        phone = TRUE
        ->  class no  [0.760]

    Rule 21/36: (36.2/10.7, lift 1.4)
        percent_of_income > 3
        years_at_residence <= 1
        job in {skilled, unskilled}
        dependents <= 1
        phone = FALSE
        ->  class no  [0.694]

    Rule 21/37: (46.1/13.8, lift 1.4)
        checking_balance = 1 - 200 DM
        amount > 1659
        percent_of_income <= 2
        job = skilled
        ->  class no  [0.692]

    Rule 21/38: (178.1/76.4, lift 1.2)
        months_loan_duration <= 16
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years, unemployed}
        job in {management, skilled, unskilled}
        ->  class no  [0.570]

    Rule 21/39: (441.2/210.6, lift 1.1)
        percent_of_income <= 3
        ->  class no  [0.523]

    Default class: yes

    -----  Trial 22:  -----

    Rules:

    Rule 22/1: (13.9, lift 1.9)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 15
        months_loan_duration <= 21
        purpose = furniture/appliances
        amount > 888
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        existing_loans_count <= 1
        ->  class yes  [0.937]

    Rule 22/2: (6.4, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 40
        employment_duration = > 7 years
        ->  class yes  [0.881]

    Rule 22/3: (6.4, lift 1.8)
        credit_history = good
        purpose in {education, renovations}
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        ->  class yes  [0.881]

    Rule 22/4: (18.4/1.5, lift 1.8)
        checking_balance = 1 - 200 DM
        employment_duration = < 1 year
        housing = rent
        ->  class yes  [0.876]

    Rule 22/5: (10.1/0.6, lift 1.8)
        amount > 2647
        savings_balance = 100 - 500 DM
        employment_duration = 1 - 4 years
        other_credit = none
        existing_loans_count <= 1
        ->  class yes  [0.871]

    Rule 22/6: (10.6/0.7, lift 1.8)
        credit_history = critical
        purpose = furniture/appliances
        employment_duration = 1 - 4 years
        years_at_residence > 3
        ->  class yes  [0.863]

    Rule 22/7: (14.7/1.5, lift 1.8)
        credit_history = poor
        employment_duration = < 1 year
        ->  class yes  [0.848]

    Rule 22/8: (23.6/3, lift 1.7)
        employment_duration = 1 - 4 years
        percent_of_income > 1
        years_at_residence > 1
        other_credit = bank
        phone = FALSE
        ->  class yes  [0.844]

    Rule 22/9: (11/1.1, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        savings_balance = < 100 DM
        employment_duration = > 7 years
        other_credit = bank
        job in {management, skilled}
        ->  class yes  [0.841]

    Rule 22/10: (21.1/2.7, lift 1.7)
        months_loan_duration > 24
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        other_credit = none
        existing_loans_count <= 1
        job in {skilled, unskilled}
        ->  class yes  [0.840]

    Rule 22/11: (26.5/3.6, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        savings_balance = < 100 DM
        employment_duration = > 7 years
        years_at_residence <= 3
        age > 31
        other_credit = none
        job in {management, skilled}
        ->  class yes  [0.838]

    Rule 22/12: (24.4/3.7, lift 1.7)
        employment_duration = unemployed
        years_at_residence <= 2
        housing in {own, rent}
        ->  class yes  [0.823]

    Rule 22/13: (19.5/3.2, lift 1.7)
        checking_balance = 1 - 200 DM
        purpose = furniture/appliances
        amount > 888
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        other_credit = none
        existing_loans_count <= 1
        ->  class yes  [0.805]

    Rule 22/14: (12.7/2, lift 1.7)
        checking_balance in {> 200 DM, 1 - 200 DM}
        purpose = car
        employment_duration = 4 - 7 years
        housing in {other, own}
        ->  class yes  [0.798]

    Rule 22/15: (12.6/2.1, lift 1.6)
        amount <= 1291
        employment_duration = unemployed
        ->  class yes  [0.789]

    Rule 22/16: (18.9/3.6, lift 1.6)
        amount <= 1597
        savings_balance = unknown
        employment_duration = 1 - 4 years
        other_credit = none
        ->  class yes  [0.782]

    Rule 22/17: (18.4/4.1, lift 1.6)
        credit_history = very good
        other_credit = none
        ->  class yes  [0.751]

    Rule 22/18: (65.4/26.5, lift 1.2)
        credit_history = good
        existing_loans_count > 1
        ->  class yes  [0.592]

    Rule 22/19: (110.6/46.2, lift 1.2)
        purpose = car
        savings_balance = < 100 DM
        percent_of_income > 2
        ->  class yes  [0.581]

    Rule 22/20: (118.2/49.4, lift 1.2)
        employment_duration = < 1 year
        years_at_residence > 1
        ->  class yes  [0.580]

    Rule 22/21: (550.3/271.5, lift 1.0)
        years_at_residence <= 3
        ->  class yes  [0.507]

    Rule 22/22: (14.8, lift 1.8)
        credit_history in {good, poor, very good}
        employment_duration = 1 - 4 years
        age <= 36
        other_credit = store
        ->  class no  [0.935]

    Rule 22/23: (8.6, lift 1.8)
        employment_duration = > 7 years
        years_at_residence <= 3
        age <= 31
        other_credit = none
        ->  class no  [0.906]

    Rule 22/24: (11.7/0.7, lift 1.7)
        employment_duration = unemployed
        housing = other
        dependents <= 1
        ->  class no  [0.877]

    Rule 22/25: (24/3.3, lift 1.6)
        employment_duration = 4 - 7 years
        housing = rent
        ->  class no  [0.836]

    Rule 22/26: (11.8/1.4, lift 1.6)
        purpose = furniture/appliances
        amount <= 1449
        employment_duration = 4 - 7 years
        ->  class no  [0.822]

    Rule 22/27: (23.6/4.2, lift 1.5)
        purpose in {business, education}
        employment_duration = 4 - 7 years
        housing in {other, own}
        ->  class no  [0.797]

    Rule 22/28: (11.9/2, lift 1.5)
        months_loan_duration > 40
        purpose = car
        housing = own
        ->  class no  [0.787]

    Rule 22/29: (55.1/11.7, lift 1.5)
        checking_balance in {> 200 DM, unknown}
        employment_duration = > 7 years
        age > 33
        ->  class no  [0.777]

    Rule 22/30: (17/4.6, lift 1.4)
        years_at_residence <= 1
        other_credit = bank
        ->  class no  [0.705]

    Rule 22/31: (334.6/146.2, lift 1.1)
        phone = TRUE
        ->  class no  [0.563]

    Rule 22/32: (703.1/330.9, lift 1.0)
        other_credit = none
        ->  class no  [0.529]

    Default class: no

    -----  Trial 23:  -----

    Rules:

    Rule 23/1: (12.1, lift 1.9)
        months_loan_duration <= 26
        credit_history = good
        savings_balance in {< 100 DM, 100 - 500 DM}
        housing = rent
        existing_loans_count > 1
        dependents <= 1
        ->  class yes  [0.929]

    Rule 23/2: (11.4, lift 1.9)
        checking_balance = < 0 DM
        purpose = education
        savings_balance = < 100 DM
        age <= 42
        ->  class yes  [0.926]

    Rule 23/3: (10, lift 1.9)
        checking_balance = < 0 DM
        purpose = car
        savings_balance = < 100 DM
        age > 41
        age <= 46
        dependents <= 1
        ->  class yes  [0.917]

    Rule 23/4: (6.1, lift 1.8)
        checking_balance = < 0 DM
        purpose = car
        savings_balance = < 100 DM
        age <= 28
        phone = TRUE
        ->  class yes  [0.876]

    Rule 23/5: (19/1.8, lift 1.8)
        checking_balance = 1 - 200 DM
        savings_balance in {< 100 DM, > 1000 DM, 100 - 500 DM}
        age > 33
        dependents > 1
        ->  class yes  [0.866]

    Rule 23/6: (4.6, lift 1.8)
        checking_balance = < 0 DM
        purpose = car
        amount > 7308
        dependents > 1
        ->  class yes  [0.848]

    Rule 23/7: (7.2/0.5, lift 1.7)
        checking_balance = > 200 DM
        months_loan_duration > 7
        dependents > 1
        ->  class yes  [0.836]

    Rule 23/8: (12.1/1.4, lift 1.7)
        checking_balance = unknown
        employment_duration in {> 7 years, unemployed}
        age <= 31
        other_credit = none
        ->  class yes  [0.833]

    Rule 23/9: (18.3/2.6, lift 1.7)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        savings_balance = < 100 DM
        other_credit = bank
        housing in {own, rent}
        ->  class yes  [0.823]

    Rule 23/10: (15.7/3, lift 1.6)
        checking_balance = unknown
        credit_history in {critical, very good}
        purpose = car
        other_credit = bank
        ->  class yes  [0.776]

    Rule 23/11: (33.1/8.7, lift 1.5)
        checking_balance = < 0 DM
        purpose = car
        savings_balance = < 100 DM
        age <= 46
        other_credit = none
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.722]

    Rule 23/12: (59.8/19.6, lift 1.4)
        checking_balance = 1 - 200 DM
        months_loan_duration > 26
        savings_balance in {< 100 DM, > 1000 DM, 100 - 500 DM, 500 - 1000 DM}
        dependents <= 1
        ->  class yes  [0.667]

    Rule 23/13: (38.3/13.6, lift 1.3)
        checking_balance = 1 - 200 DM
        credit_history in {perfect, very good}
        ->  class yes  [0.638]

    Rule 23/14: (39.8/14.3, lift 1.3)
        checking_balance = unknown
        years_at_residence <= 3
        other_credit = bank
        ->  class yes  [0.635]

    Rule 23/15: (22.3/8.4, lift 1.3)
        purpose = renovations
        ->  class yes  [0.614]

    Rule 23/16: (849.5/435.1, lift 1.0)
        months_loan_duration > 7
        ->  class yes  [0.488]

    Rule 23/17: (17.7, lift 1.8)
        checking_balance = unknown
        amount <= 1459
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years}
        ->  class no  [0.949]

    Rule 23/18: (10.4, lift 1.8)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 26
        credit_history = critical
        age > 35
        dependents <= 1
        ->  class no  [0.919]

    Rule 23/19: (8.8, lift 1.7)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        amount <= 1108
        savings_balance = < 100 DM
        other_credit = none
        ->  class no  [0.907]

    Rule 23/20: (8.7, lift 1.7)
        checking_balance = > 200 DM
        age <= 24
        ->  class no  [0.906]

    Rule 23/21: (8.1, lift 1.7)
        checking_balance = > 200 DM
        purpose = car
        savings_balance = < 100 DM
        other_credit in {none, store}
        ->  class no  [0.901]

    Rule 23/22: (8, lift 1.7)
        checking_balance = 1 - 200 DM
        months_loan_duration > 15
        months_loan_duration <= 26
        credit_history = good
        savings_balance in {< 100 DM, 100 - 500 DM}
        housing = own
        dependents <= 1
        phone = FALSE
        ->  class no  [0.900]

    Rule 23/23: (12.5/0.5, lift 1.7)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 26
        credit_history = good
        housing = own
        existing_loans_count > 1
        ->  class no  [0.899]

    Rule 23/24: (7.4, lift 1.7)
        checking_balance = < 0 DM
        purpose = car
        savings_balance = < 100 DM
        age > 46
        other_credit = none
        dependents <= 1
        ->  class no  [0.893]

    Rule 23/25: (10.9/0.7, lift 1.7)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 26
        credit_history = critical
        age <= 29
        ->  class no  [0.870]

    Rule 23/26: (5.4, lift 1.7)
        checking_balance = > 200 DM
        other_credit = bank
        dependents <= 1
        ->  class no  [0.865]

    Rule 23/27: (4.5, lift 1.6)
        checking_balance = unknown
        purpose = furniture/appliances
        years_at_residence > 3
        other_credit = bank
        ->  class no  [0.845]

    Rule 23/28: (21/2.7, lift 1.6)
        checking_balance = unknown
        credit_history = good
        savings_balance in {> 1000 DM, 100 - 500 DM, unknown}
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years}
        age <= 31
        ->  class no  [0.840]

    Rule 23/29: (17.4/2.4, lift 1.6)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 26
        other_credit = bank
        housing = own
        dependents <= 1
        ->  class no  [0.825]

    Rule 23/30: (8.6/0.9, lift 1.6)
        purpose = education
        savings_balance = < 100 DM
        age > 42
        ->  class no  [0.817]

    Rule 23/31: (29.1/4.7, lift 1.6)
        purpose = car
        savings_balance = < 100 DM
        age > 28
        age <= 41
        other_credit = none
        phone = TRUE
        ->  class no  [0.816]

    Rule 23/32: (47.1/9.1, lift 1.5)
        checking_balance = unknown
        credit_history in {critical, perfect}
        other_credit = none
        ->  class no  [0.794]

    Rule 23/33: (77.1/15.9, lift 1.5)
        checking_balance = unknown
        age > 31
        other_credit = none
        ->  class no  [0.786]

    Rule 23/34: (37.1/10.1, lift 1.4)
        checking_balance = 1 - 200 DM
        savings_balance = unknown
        ->  class no  [0.717]

    Rule 23/35: (23.9/6.8, lift 1.3)
        checking_balance = < 0 DM
        savings_balance in {> 1000 DM, 100 - 500 DM, 500 - 1000 DM}
        ->  class no  [0.700]

    Rule 23/36: (41.7/12.3, lift 1.3)
        checking_balance = unknown
        credit_history in {good, perfect}
        purpose = car
        ->  class no  [0.696]

    Rule 23/37: (455.4/206.1, lift 1.0)
        percent_of_income <= 3
        ->  class no  [0.547]

    Default class: no

    -----  Trial 24:  -----

    Rules:

    Rule 24/1: (16.8, lift 2.0)
        checking_balance = < 0 DM
        credit_history = good
        percent_of_income > 1
        years_at_residence > 1
        age <= 23
        other_credit = none
        job = skilled
        ->  class yes  [0.947]

    Rule 24/2: (13.7, lift 2.0)
        checking_balance = 1 - 200 DM
        credit_history = good
        amount > 8648
        ->  class yes  [0.936]

    Rule 24/3: (9, lift 1.9)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        savings_balance = < 100 DM
        years_at_residence > 1
        other_credit = bank
        job = skilled
        ->  class yes  [0.909]

    Rule 24/4: (8.5, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration > 27
        credit_history = good
        purpose = furniture/appliances
        amount <= 5743
        percent_of_income > 1
        phone = FALSE
        ->  class yes  [0.905]

    Rule 24/5: (8.5, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration > 27
        savings_balance = unknown
        other_credit = none
        ->  class yes  [0.905]

    Rule 24/6: (23.6/2, lift 1.9)
        checking_balance = 1 - 200 DM
        credit_history = good
        employment_duration in {< 1 year, > 7 years, 1 - 4 years}
        housing = rent
        job in {skilled, unskilled}
        ->  class yes  [0.885]

    Rule 24/7: (6.3, lift 1.9)
        checking_balance = < 0 DM
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        job = skilled
        phone = TRUE
        ->  class yes  [0.880]

    Rule 24/8: (22/2.6, lift 1.8)
        checking_balance = unknown
        purpose = business
        employment_duration in {< 1 year, 1 - 4 years, unemployed}
        age > 26
        ->  class yes  [0.851]

    Rule 24/9: (11.8/1.4, lift 1.8)
        credit_history = poor
        age > 45
        ->  class yes  [0.823]

    Rule 24/10: (3.3, lift 1.7)
        checking_balance = 1 - 200 DM
        existing_loans_count > 3
        ->  class yes  [0.813]

    Rule 24/11: (12.3/1.7, lift 1.7)
        checking_balance = unknown
        purpose = education
        amount > 1670
        employment_duration in {< 1 year, 1 - 4 years}
        ->  class yes  [0.812]

    Rule 24/12: (3.2, lift 1.7)
        checking_balance = 1 - 200 DM
        credit_history = critical
        dependents > 1
        ->  class yes  [0.809]

    Rule 24/13: (13.6/2.3, lift 1.7)
        checking_balance = unknown
        purpose = furniture/appliances
        amount > 4297
        employment_duration in {< 1 year, 1 - 4 years, unemployed}
        ->  class yes  [0.789]

    Rule 24/14: (21.1/4.2, lift 1.7)
        checking_balance = < 0 DM
        months_loan_duration > 30
        existing_loans_count > 1
        ->  class yes  [0.773]

    Rule 24/15: (21.5/4.4, lift 1.7)
        checking_balance = < 0 DM
        purpose in {business, education, renovations}
        savings_balance = < 100 DM
        job in {skilled, unemployed}
        ->  class yes  [0.772]

    Rule 24/16: (30.3/7.7, lift 1.6)
        checking_balance = unknown
        credit_history in {critical, good, very good}
        purpose = car
        amount > 1271
        employment_duration in {1 - 4 years, unemployed}
        dependents <= 1
        ->  class yes  [0.730]

    Rule 24/17: (49.7/16.1, lift 1.4)
        checking_balance = < 0 DM
        months_loan_duration > 9
        purpose = car
        age <= 48
        job in {skilled, unemployed}
        dependents <= 1
        ->  class yes  [0.669]

    Rule 24/18: (62.5/25, lift 1.3)
        credit_history in {perfect, very good}
        savings_balance = < 100 DM
        ->  class yes  [0.596]

    Rule 24/19: (308.6/148.6, lift 1.1)
        checking_balance = < 0 DM
        ->  class yes  [0.518]

    Rule 24/20: (20.3, lift 1.8)
        checking_balance = unknown
        months_loan_duration > 24
        employment_duration in {> 7 years, 4 - 7 years}
        ->  class no  [0.955]

    Rule 24/21: (22.2/1.7, lift 1.7)
        checking_balance = 1 - 200 DM
        credit_history = poor
        age <= 45
        existing_loans_count <= 3
        ->  class no  [0.890]

    Rule 24/22: (9.2/0.7, lift 1.6)
        checking_balance = 1 - 200 DM
        months_loan_duration > 27
        credit_history = very good
        ->  class no  [0.845]

    Rule 24/23: (35.8/8, lift 1.4)
        checking_balance = unknown
        employment_duration = 4 - 7 years
        ->  class no  [0.763]

    Rule 24/24: (65.8/18.3, lift 1.3)
        checking_balance = < 0 DM
        savings_balance = < 100 DM
        percent_of_income > 1
        other_credit in {bank, none}
        job in {management, unskilled}
        ->  class no  [0.715]

    Rule 24/25: (842.6/384.7, lift 1.0)
        amount <= 8648
        ->  class no  [0.543]

    Default class: yes

    -----  Trial 25:  -----

    Rules:

    Rule 25/1: (10.8/0.7, lift 1.7)
        checking_balance = unknown
        employment_duration in {< 1 year, 1 - 4 years}
        percent_of_income > 1
        housing = rent
        existing_loans_count > 1
        dependents <= 1
        ->  class yes  [0.869]

    Rule 25/2: (21.3/3.4, lift 1.6)
        checking_balance = unknown
        savings_balance in {< 100 DM, 500 - 1000 DM}
        employment_duration in {< 1 year, 1 - 4 years, unemployed}
        percent_of_income > 1
        housing = rent
        dependents <= 1
        ->  class yes  [0.811]

    Rule 25/3: (18.8/3.5, lift 1.6)
        months_loan_duration > 22
        credit_history = good
        purpose = furniture/appliances
        housing = own
        existing_loans_count > 1
        ->  class yes  [0.783]

    Rule 25/4: (649.7/299.8, lift 1.1)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        ->  class yes  [0.538]

    Rule 25/5: (22.7, lift 1.9)
        purpose = car
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 1 - 4 years, unemployed}
        percent_of_income <= 2
        age > 29
        ->  class no  [0.960]

    Rule 25/6: (14.2, lift 1.9)
        months_loan_duration <= 8
        savings_balance = < 100 DM
        years_at_residence <= 3
        ->  class no  [0.938]

    Rule 25/7: (12.2, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 47
        purpose = furniture/appliances
        amount > 1851
        savings_balance = < 100 DM
        age <= 37
        job = management
        ->  class no  [0.930]

    Rule 25/8: (11.9, lift 1.8)
        checking_balance = 1 - 200 DM
        months_loan_duration > 8
        months_loan_duration <= 47
        purpose = furniture/appliances
        savings_balance = < 100 DM
        years_at_residence > 3
        job = skilled
        ->  class no  [0.928]

    Rule 25/9: (11.7, lift 1.8)
        checking_balance = < 0 DM
        months_loan_duration <= 16
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        years_at_residence <= 3
        ->  class no  [0.927]

    Rule 25/10: (10.7, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 11
        purpose = furniture/appliances
        job = unskilled
        ->  class no  [0.921]

    Rule 25/11: (10.2, lift 1.8)
        purpose in {business, renovations}
        savings_balance = 100 - 500 DM
        housing = own
        dependents <= 1
        ->  class no  [0.918]

    Rule 25/12: (9.5, lift 1.8)
        savings_balance = unknown
        employment_duration = 4 - 7 years
        job = skilled
        ->  class no  [0.913]

    Rule 25/13: (8.8, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = good
        purpose = furniture/appliances
        amount <= 1901
        savings_balance = < 100 DM
        other_credit = none
        job = unskilled
        ->  class no  [0.907]

    Rule 25/14: (10.3/0.2, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration <= 47
        purpose = furniture/appliances
        amount > 3599
        savings_balance = < 100 DM
        years_at_residence > 3
        job = skilled
        ->  class no  [0.904]

    Rule 25/15: (7.7, lift 1.8)
        purpose = furniture/appliances
        percent_of_income <= 1
        job = unskilled
        ->  class no  [0.897]

    Rule 25/16: (6.7, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = furniture/appliances
        savings_balance = unknown
        job in {unemployed, unskilled}
        ->  class no  [0.885]

    Rule 25/17: (6.4, lift 1.8)
        checking_balance = > 200 DM
        savings_balance = < 100 DM
        years_at_residence > 3
        job = skilled
        ->  class no  [0.881]

    Rule 25/18: (21.1/1.9, lift 1.7)
        purpose = car
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 1 - 4 years, unemployed}
        percent_of_income <= 2
        years_at_residence > 2
        ->  class no  [0.875]

    Rule 25/19: (5.7, lift 1.7)
        months_loan_duration > 11
        credit_history = good
        purpose = furniture/appliances
        other_credit = bank
        job = unskilled
        ->  class no  [0.871]

    Rule 25/20: (5.3, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = business
        savings_balance = < 100 DM
        other_credit = bank
        ->  class no  [0.864]

    Rule 25/21: (9.3/0.5, lift 1.7)
        checking_balance = < 0 DM
        credit_history in {critical, perfect}
        purpose = furniture/appliances
        years_at_residence <= 3
        age <= 50
        ->  class no  [0.863]

    Rule 25/22: (4.9, lift 1.7)
        checking_balance = > 200 DM
        purpose = car
        savings_balance = < 100 DM
        other_credit = none
        ->  class no  [0.855]

    Rule 25/23: (3.7, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        savings_balance = 500 - 1000 DM
        other_credit in {bank, store}
        ->  class no  [0.824]

    Rule 25/24: (21.3/3.7, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {critical, poor}
        savings_balance = 100 - 500 DM
        ->  class no  [0.799]

    Rule 25/25: (32.9/6.4, lift 1.6)
        months_loan_duration <= 33
        purpose = car
        savings_balance = < 100 DM
        other_credit = none
        housing = own
        existing_loans_count > 1
        ->  class no  [0.787]

    Rule 25/26: (14.7/3.5, lift 1.5)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        savings_balance = > 1000 DM
        ->  class no  [0.729]

    Rule 25/27: (18.7/4.9, lift 1.4)
        purpose in {business, education}
        savings_balance = unknown
        ->  class no  [0.714]

    Rule 25/28: (250.3/97.9, lift 1.2)
        checking_balance = unknown
        ->  class no  [0.608]

    Default class: no

    -----  Trial 26:  -----

    Rules:

    Rule 26/1: (10, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM, unknown}
        credit_history = good
        purpose in {education, renovations}
        savings_balance = < 100 DM
        age <= 47
        job = skilled
        dependents <= 1
        ->  class yes  [0.916]

    Rule 26/2: (12.5/0.4, lift 2.0)
        months_loan_duration > 11
        months_loan_duration <= 13
        savings_balance = 500 - 1000 DM
        years_at_residence > 1
        job in {skilled, unskilled}
        ->  class yes  [0.906]

    Rule 26/3: (9.6/0.1, lift 2.0)
        purpose = car
        amount <= 1804
        savings_balance = unknown
        employment_duration = 1 - 4 years
        ->  class yes  [0.902]

    Rule 26/4: (8, lift 2.0)
        credit_history = perfect
        savings_balance = < 100 DM
        housing in {other, rent}
        ->  class yes  [0.900]

    Rule 26/5: (7.5, lift 1.9)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        ->  class yes  [0.894]

    Rule 26/6: (6.5, lift 1.9)
        purpose in {education, renovations}
        savings_balance = > 1000 DM
        ->  class yes  [0.882]

    Rule 26/7: (5.5, lift 1.9)
        credit_history = very good
        savings_balance = < 100 DM
        other_credit = none
        ->  class yes  [0.866]

    Rule 26/8: (15.4/1.8, lift 1.8)
        credit_history = critical
        amount > 1922
        savings_balance = < 100 DM
        housing = rent
        dependents <= 1
        ->  class yes  [0.838]

    Rule 26/9: (25.2/3.7, lift 1.8)
        months_loan_duration <= 9
        credit_history = good
        purpose = furniture/appliances
        amount > 717
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years}
        age <= 47
        job = skilled
        dependents <= 1
        ->  class yes  [0.828]

    Rule 26/10: (7/0.6, lift 1.8)
        months_loan_duration > 11
        savings_balance = 500 - 1000 DM
        other_credit = bank
        job in {skilled, unskilled}
        ->  class yes  [0.822]

    Rule 26/11: (3.3, lift 1.8)
        credit_history = perfect
        other_credit = store
        ->  class yes  [0.810]

    Rule 26/12: (2.9, lift 1.7)
        savings_balance = > 1000 DM
        existing_loans_count > 2
        ->  class yes  [0.795]

    Rule 26/13: (12.2/1.9, lift 1.7)
        purpose = education
        savings_balance = 100 - 500 DM
        ->  class yes  [0.793]

    Rule 26/14: (14/2.7, lift 1.7)
        purpose = car
        savings_balance = 100 - 500 DM
        job in {management, unskilled}
        ->  class yes  [0.770]

    Rule 26/15: (2.2, lift 1.7)
        purpose = car0
        savings_balance = unknown
        ->  class yes  [0.764]

    Rule 26/16: (15.6/3.3, lift 1.6)
        purpose = car
        amount > 6887
        employment_duration = > 7 years
        ->  class yes  [0.755]

    Rule 26/17: (9.8/2, lift 1.6)
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        existing_loans_count > 1
        ->  class yes  [0.750]

    Rule 26/18: (32/7.6, lift 1.6)
        months_loan_duration > 16
        credit_history = poor
        savings_balance = < 100 DM
        ->  class yes  [0.747]

    Rule 26/19: (18.9/4.4, lift 1.6)
        amount > 7596
        savings_balance = < 100 DM
        job = management
        ->  class yes  [0.744]

    Rule 26/20: (27/7.7, lift 1.5)
        credit_history = good
        savings_balance = < 100 DM
        percent_of_income > 1
        age > 35
        dependents > 1
        ->  class yes  [0.700]

    Rule 26/21: (21.9/6.7, lift 1.5)
        credit_history = very good
        savings_balance = < 100 DM
        percent_of_income > 1
        ->  class yes  [0.678]

    Rule 26/22: (122/53.3, lift 1.2)
        months_loan_duration > 8
        existing_loans_count <= 1
        job = unskilled
        ->  class yes  [0.562]

    Rule 26/23: (186.3/86.6, lift 1.2)
        months_loan_duration > 26
        dependents <= 1
        ->  class yes  [0.535]

    Rule 26/24: (26.4, lift 1.8)
        credit_history = critical
        years_at_residence > 3
        age > 32
        housing = own
        dependents <= 1
        ->  class no  [0.965]

    Rule 26/25: (15.1, lift 1.7)
        purpose = car
        savings_balance = 100 - 500 DM
        age > 29
        job = skilled
        ->  class no  [0.941]

    Rule 26/26: (12.4, lift 1.7)
        months_loan_duration <= 39
        credit_history = good
        purpose = business
        age <= 47
        dependents <= 1
        ->  class no  [0.930]

    Rule 26/27: (11.5, lift 1.7)
        checking_balance = unknown
        months_loan_duration > 9
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years}
        job = skilled
        ->  class no  [0.926]

    Rule 26/28: (10.4, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        months_loan_duration > 13
        savings_balance = 500 - 1000 DM
        other_credit = none
        ->  class no  [0.919]

    Rule 26/29: (9.9, lift 1.7)
        checking_balance = < 0 DM
        months_loan_duration <= 39
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 1 - 4 years}
        years_at_residence <= 1
        ->  class no  [0.916]

    Rule 26/30: (9.1, lift 1.7)
        months_loan_duration <= 8
        savings_balance = < 100 DM
        job = unskilled
        ->  class no  [0.910]

    Rule 26/31: (9.1, lift 1.7)
        credit_history = critical
        years_at_residence <= 1
        housing = own
        ->  class no  [0.910]

    Rule 26/32: (8.7, lift 1.7)
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = > 7 years
        job = skilled
        dependents <= 1
        ->  class no  [0.906]

    Rule 26/33: (8, lift 1.7)
        savings_balance = 500 - 1000 DM
        job in {management, unemployed}
        ->  class no  [0.900]

    Rule 26/34: (7.7, lift 1.7)
        credit_history = good
        purpose in {business, car0}
        existing_loans_count <= 1
        job = unskilled
        ->  class no  [0.897]

    Rule 26/35: (7.6, lift 1.7)
        savings_balance = < 100 DM
        percent_of_income <= 1
        dependents > 1
        ->  class no  [0.896]

    Rule 26/36: (6.8, lift 1.6)
        months_loan_duration <= 11
        savings_balance = 500 - 1000 DM
        ->  class no  [0.887]

    Rule 26/37: (14.8/1, lift 1.6)
        purpose in {business, car, furniture/appliances}
        savings_balance = > 1000 DM
        existing_loans_count <= 2
        ->  class no  [0.882]

    Rule 26/38: (6.1, lift 1.6)
        savings_balance = 500 - 1000 DM
        years_at_residence <= 1
        ->  class no  [0.877]

    Rule 26/39: (15.9/1.7, lift 1.6)
        credit_history = good
        age <= 35
        job = skilled
        dependents > 1
        ->  class no  [0.850]

    Rule 26/40: (3.6, lift 1.5)
        credit_history = very good
        savings_balance = < 100 DM
        other_credit = store
        ->  class no  [0.822]

    Rule 26/41: (14.6/2, lift 1.5)
        credit_history = critical
        savings_balance = < 100 DM
        dependents > 1
        ->  class no  [0.821]

    Rule 26/42: (31.2/5.4, lift 1.5)
        credit_history = good
        amount <= 7596
        savings_balance = < 100 DM
        job = management
        ->  class no  [0.807]

    Rule 26/43: (31.1/6.5, lift 1.4)
        credit_history = perfect
        other_credit in {bank, none}
        housing = own
        ->  class no  [0.774]

    Rule 26/44: (48.3/10.5, lift 1.4)
        credit_history = critical
        amount <= 1922
        savings_balance = < 100 DM
        ->  class no  [0.772]

    Rule 26/45: (27.2/6.1, lift 1.4)
        savings_balance = < 100 DM
        existing_loans_count > 1
        job = unskilled
        ->  class no  [0.758]

    Rule 26/46: (24.8/6, lift 1.4)
        checking_balance = > 200 DM
        credit_history = good
        job = skilled
        ->  class no  [0.738]

    Rule 26/47: (141.5/54.7, lift 1.1)
        savings_balance = unknown
        ->  class no  [0.612]

    Default class: no

    -----  Trial 27:  -----

    Rules:

    Rule 27/1: (8.6, lift 1.9)
        employment_duration = unemployed
        years_at_residence <= 1
        ->  class yes  [0.906]

    Rule 27/2: (6.3, lift 1.9)
        purpose = business
        employment_duration = unemployed
        ->  class yes  [0.879]

    Rule 27/3: (5.6, lift 1.8)
        employment_duration = unemployed
        dependents > 1
        ->  class yes  [0.858]

    Rule 27/4: (12.4/1.1, lift 1.8)
        checking_balance = < 0 DM
        amount > 5804
        employment_duration = > 7 years
        ->  class yes  [0.857]

    Rule 27/5: (17.8/2, lift 1.8)
        checking_balance = 1 - 200 DM
        employment_duration = < 1 year
        housing = rent
        ->  class yes  [0.846]

    Rule 27/6: (26.2/6.2, lift 1.6)
        checking_balance = < 0 DM
        credit_history = good
        employment_duration = < 1 year
        percent_of_income > 2
        other_credit = none
        ->  class yes  [0.745]

    Rule 27/7: (24.6/5.8, lift 1.6)
        checking_balance = 1 - 200 DM
        savings_balance in {< 100 DM, > 1000 DM}
        employment_duration = > 7 years
        age > 30
        ->  class yes  [0.744]

    Rule 27/8: (22.2/5.5, lift 1.6)
        amount > 3149
        employment_duration = 1 - 4 years
        job = management
        ->  class yes  [0.732]

    Rule 27/9: (18.2/4.6, lift 1.5)
        credit_history in {perfect, poor, very good}
        employment_duration = < 1 year
        other_credit = none
        ->  class yes  [0.721]

    Rule 27/10: (75.3/21.3, lift 1.5)
        months_loan_duration > 13
        savings_balance in {< 100 DM, 100 - 500 DM, unknown}
        employment_duration = 1 - 4 years
        percent_of_income > 2
        years_at_residence > 1
        job = skilled
        ->  class yes  [0.711]

    Rule 27/11: (80.7/23.1, lift 1.5)
        months_loan_duration > 13
        savings_balance in {< 100 DM, 100 - 500 DM, unknown}
        employment_duration = 1 - 4 years
        percent_of_income > 1
        years_at_residence > 1
        job = skilled
        phone = FALSE
        ->  class yes  [0.709]

    Rule 27/12: (258.1/124, lift 1.1)
        amount <= 1484
        ->  class yes  [0.519]

    Rule 27/13: (11.8, lift 1.7)
        months_loan_duration <= 13
        employment_duration = 1 - 4 years
        age > 31
        other_credit = none
        job = skilled
        ->  class no  [0.927]

    Rule 27/14: (8, lift 1.7)
        employment_duration = 4 - 7 years
        age > 49
        ->  class no  [0.900]

    Rule 27/15: (7.1, lift 1.7)
        savings_balance = unknown
        employment_duration = 4 - 7 years
        job = skilled
        ->  class no  [0.890]

    Rule 27/16: (5.8, lift 1.6)
        credit_history = critical
        amount <= 1659
        savings_balance = < 100 DM
        employment_duration = < 1 year
        ->  class no  [0.872]

    Rule 27/17: (10/0.8, lift 1.6)
        checking_balance = unknown
        credit_history = good
        savings_balance = < 100 DM
        other_credit = none
        phone = TRUE
        ->  class no  [0.847]

    Rule 27/18: (31.9/6.2, lift 1.5)
        checking_balance = < 0 DM
        purpose in {business, car, car0, furniture/appliances, renovations}
        amount <= 5804
        employment_duration = > 7 years
        years_at_residence > 3
        ->  class no  [0.787]

    Rule 27/19: (19/3.5, lift 1.5)
        checking_balance = 1 - 200 DM
        savings_balance in {100 - 500 DM, 500 - 1000 DM, unknown}
        employment_duration = > 7 years
        ->  class no  [0.785]

    Rule 27/20: (31.6/6.8, lift 1.4)
        checking_balance = unknown
        employment_duration = > 7 years
        other_credit = none
        ->  class no  [0.768]

    Rule 27/21: (64.4/19.6, lift 1.3)
        employment_duration = 1 - 4 years
        housing in {other, own}
        job = unskilled
        ->  class no  [0.689]

    Rule 27/22: (42/14, lift 1.2)
        checking_balance = > 200 DM
        credit_history = good
        ->  class no  [0.660]

    Rule 27/23: (641.9/288.5, lift 1.0)
        amount > 1484
        ->  class no  [0.550]

    Default class: no

    -----  Trial 28:  -----

    Rules:

    Rule 28/1: (11.1/1.9, lift 1.7)
        months_loan_duration <= 8
        amount > 3380
        ->  class yes  [0.780]

    Rule 28/2: (837.8/432.3, lift 1.0)
        months_loan_duration > 8
        ->  class yes  [0.484]

    Rule 28/3: (36, lift 1.8)
        checking_balance = unknown
        purpose = car
        amount <= 11816
        age > 27
        age <= 66
        other_credit in {none, store}
        ->  class no  [0.974]

    Rule 28/4: (10.8, lift 1.7)
        checking_balance = < 0 DM
        months_loan_duration > 13
        months_loan_duration <= 16
        savings_balance = < 100 DM
        phone = FALSE
        ->  class no  [0.921]

    Rule 28/5: (9.5, lift 1.7)
        credit_history = critical
        years_at_residence <= 1
        housing = own
        ->  class no  [0.913]

    Rule 28/6: (8, lift 1.7)
        checking_balance = > 200 DM
        employment_duration = 1 - 4 years
        percent_of_income <= 2
        dependents <= 1
        ->  class no  [0.900]

    Rule 28/7: (14.8/0.7, lift 1.7)
        credit_history = critical
        years_at_residence > 2
        other_credit = none
        housing = own
        existing_loans_count <= 1
        ->  class no  [0.899]

    Rule 28/8: (7, lift 1.7)
        checking_balance = unknown
        purpose = furniture/appliances
        age <= 32
        other_credit = bank
        housing = own
        ->  class no  [0.889]

    Rule 28/9: (19.9/1.7, lift 1.6)
        checking_balance = 1 - 200 DM
        credit_history = good
        amount <= 8648
        employment_duration in {> 7 years, 4 - 7 years, unemployed}
        age <= 32
        other_credit = none
        ->  class no  [0.875]

    Rule 28/10: (6, lift 1.6)
        checking_balance = > 200 DM
        other_credit = bank
        dependents <= 1
        ->  class no  [0.874]

    Rule 28/11: (15.4/1.4, lift 1.6)
        checking_balance = < 0 DM
        months_loan_duration > 10
        credit_history = good
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years, unemployed}
        job = management
        ->  class no  [0.863]

    Rule 28/12: (5.2, lift 1.6)
        checking_balance = > 200 DM
        amount > 2687
        employment_duration = < 1 year
        ->  class no  [0.861]

    Rule 28/13: (9.3/0.7, lift 1.6)
        checking_balance = < 0 DM
        credit_history = good
        years_at_residence > 2
        housing in {own, rent}
        dependents > 1
        ->  class no  [0.847]

    Rule 28/14: (51.1/7.2, lift 1.6)
        months_loan_duration <= 8
        amount <= 3380
        ->  class no  [0.845]

    Rule 28/15: (4.4, lift 1.6)
        purpose = furniture/appliances
        other_credit = bank
        housing = other
        ->  class no  [0.844]

    Rule 28/16: (3.9, lift 1.6)
        checking_balance = unknown
        percent_of_income <= 1
        other_credit = bank
        ->  class no  [0.831]

    Rule 28/17: (12.6/1.5, lift 1.6)
        checking_balance = > 200 DM
        employment_duration in {> 7 years, unemployed}
        dependents <= 1
        ->  class no  [0.826]

    Rule 28/18: (22.8/4.6, lift 1.5)
        checking_balance = < 0 DM
        months_loan_duration > 13
        credit_history = critical
        savings_balance = < 100 DM
        other_credit = none
        ->  class no  [0.776]

    Rule 28/19: (66.1/17.8, lift 1.4)
        checking_balance = unknown
        months_loan_duration > 8
        purpose = furniture/appliances
        years_at_residence > 1
        other_credit in {none, store}
        ->  class no  [0.723]

    Rule 28/20: (56.9/15.9, lift 1.3)
        checking_balance = 1 - 200 DM
        credit_history = good
        age <= 32
        other_credit = none
        housing in {other, own}
        job in {skilled, unskilled}
        ->  class no  [0.714]

    Default class: no

    -----  Trial 29:  -----

    Rules:

    Rule 29/1: (9.1, lift 2.2)
        checking_balance = 1 - 200 DM
        months_loan_duration > 36
        savings_balance = < 100 DM
        job = skilled
        ->  class yes  [0.910]

    Rule 29/2: (8.5, lift 2.2)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        savings_balance = < 100 DM
        years_at_residence > 1
        other_credit = bank
        job = skilled
        ->  class yes  [0.905]

    Rule 29/3: (8.3, lift 2.2)
        checking_balance = < 0 DM
        purpose = car
        savings_balance = < 100 DM
        years_at_residence <= 2
        existing_loans_count <= 1
        job = skilled
        ->  class yes  [0.902]

    Rule 29/4: (18.9/1.3, lift 2.1)
        checking_balance = unknown
        savings_balance = < 100 DM
        percent_of_income > 1
        age <= 23
        job = skilled
        ->  class yes  [0.889]

    Rule 29/5: (7.1, lift 2.1)
        credit_history = critical
        savings_balance = < 100 DM
        percent_of_income <= 2
        housing in {own, rent}
        job = management
        ->  class yes  [0.885]

    Rule 29/6: (5.4, lift 2.1)
        purpose in {education, renovations}
        savings_balance = > 1000 DM
        ->  class yes  [0.865]

    Rule 29/7: (4.4, lift 2.0)
        months_loan_duration > 8
        employment_duration = 1 - 4 years
        other_credit = bank
        job = unskilled
        ->  class yes  [0.844]

    Rule 29/8: (3.8, lift 2.0)
        checking_balance = < 0 DM
        purpose = education
        savings_balance = < 100 DM
        job = skilled
        ->  class yes  [0.828]

    Rule 29/9: (11.1/1.4, lift 1.9)
        credit_history in {perfect, poor}
        savings_balance = < 100 DM
        job = management
        ->  class yes  [0.815]

    Rule 29/10: (3.3, lift 1.9)
        savings_balance = < 100 DM
        employment_duration = > 7 years
        other_credit = store
        job = unskilled
        ->  class yes  [0.811]

    Rule 29/11: (21.6/4.1, lift 1.9)
        checking_balance = unknown
        savings_balance = < 100 DM
        age <= 23
        job = skilled
        ->  class yes  [0.785]

    Rule 29/12: (9.2/1.4, lift 1.9)
        savings_balance = 100 - 500 DM
        employment_duration = > 7 years
        percent_of_income <= 2
        ->  class yes  [0.782]

    Rule 29/13: (18.9/3.6, lift 1.9)
        months_loan_duration > 8
        months_loan_duration <= 45
        savings_balance = < 100 DM
        employment_duration = < 1 year
        age > 24
        other_credit = none
        job = unskilled
        ->  class yes  [0.779]

    Rule 29/14: (23.9/5.1, lift 1.8)
        checking_balance = 1 - 200 DM
        months_loan_duration > 7
        months_loan_duration <= 27
        credit_history = good
        purpose = furniture/appliances
        amount > 629
        savings_balance = < 100 DM
        existing_loans_count <= 1
        job = skilled
        ->  class yes  [0.764]

    Rule 29/15: (9.5/1.7, lift 1.8)
        savings_balance = 100 - 500 DM
        employment_duration = 1 - 4 years
        years_at_residence > 1
        housing in {other, rent}
        ->  class yes  [0.764]

    Rule 29/16: (13.9/3.2, lift 1.8)
        months_loan_duration > 33
        credit_history = good
        savings_balance = unknown
        dependents <= 1
        ->  class yes  [0.735]

    Rule 29/17: (9.5/2.2, lift 1.7)
        employment_duration = > 7 years
        job = unskilled
        dependents > 1
        ->  class yes  [0.722]

    Rule 29/18: (15.2/4, lift 1.7)
        savings_balance = 100 - 500 DM
        employment_duration = < 1 year
        ->  class yes  [0.708]

    Rule 29/19: (10.3/2.7, lift 1.7)
        savings_balance = 500 - 1000 DM
        job = unskilled
        ->  class yes  [0.698]

    Rule 29/20: (50.3/17.3, lift 1.5)
        checking_balance = < 0 DM
        savings_balance = < 100 DM
        existing_loans_count > 1
        job = skilled
        ->  class yes  [0.649]

    Rule 29/21: (110/54.6, lift 1.2)
        housing in {own, rent}
        job = management
        ->  class yes  [0.504]

    Rule 29/22: (353.4/189.6, lift 1.1)
        savings_balance = < 100 DM
        job = skilled
        ->  class yes  [0.464]

    Rule 29/23: (7.3, lift 1.5)
        savings_balance = 500 - 1000 DM
        years_at_residence <= 1
        ->  class no  [0.892]

    Rule 29/24: (6.5, lift 1.5)
        savings_balance = 500 - 1000 DM
        job in {management, unemployed}
        ->  class no  [0.882]

    Rule 29/25: (15.4/1.6, lift 1.5)
        savings_balance = 100 - 500 DM
        employment_duration = > 7 years
        percent_of_income > 2
        ->  class no  [0.849]

    Rule 29/26: (21.4/3.5, lift 1.4)
        purpose in {business, car, furniture/appliances}
        savings_balance = > 1000 DM
        ->  class no  [0.809]

    Rule 29/27: (55.3/11.1, lift 1.4)
        credit_history = critical
        percent_of_income > 2
        housing in {own, rent}
        phone = TRUE
        ->  class no  [0.789]

    Rule 29/28: (30.8/6.6, lift 1.3)
        savings_balance = 100 - 500 DM
        employment_duration in {4 - 7 years, unemployed}
        ->  class no  [0.769]

    Rule 29/29: (26.9/6.7, lift 1.3)
        checking_balance = unknown
        percent_of_income <= 1
        ->  class no  [0.734]

    Rule 29/30: (104.8/31.5, lift 1.2)
        percent_of_income > 1
        years_at_residence > 3
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.696]

    Rule 29/31: (169/58.4, lift 1.1)
        checking_balance = unknown
        job = skilled
        ->  class no  [0.653]

    Rule 29/32: (146.3/52.3, lift 1.1)
        savings_balance = unknown
        ->  class no  [0.641]

    Rule 29/33: (574.7/253.5, lift 1.0)
        savings_balance = < 100 DM
        ->  class no  [0.559]

    Default class: no

    -----  Trial 30:  -----

    Rules:

    Rule 30/1: (17.2, lift 2.0)
        checking_balance = 1 - 200 DM
        credit_history = good
        amount > 8487
        ->  class yes  [0.948]

    Rule 30/2: (15.3, lift 2.0)
        checking_balance = < 0 DM
        months_loan_duration > 27
        credit_history = good
        savings_balance in {< 100 DM, unknown}
        job = skilled
        dependents <= 1
        ->  class yes  [0.942]

    Rule 30/3: (8.7, lift 1.9)
        checking_balance = < 0 DM
        credit_history = good
        age <= 21
        job = skilled
        ->  class yes  [0.907]

    Rule 30/4: (11.2/0.5, lift 1.8)
        credit_history = perfect
        housing in {other, rent}
        ->  class yes  [0.884]

    Rule 30/5: (8.5/0.4, lift 1.8)
        months_loan_duration <= 47
        employment_duration = 4 - 7 years
        age <= 22
        ->  class yes  [0.872]

    Rule 30/6: (18.8/2.1, lift 1.8)
        months_loan_duration <= 39
        credit_history = very good
        age > 23
        other_credit in {none, store}
        ->  class yes  [0.849]

    Rule 30/7: (21.1/2.6, lift 1.8)
        checking_balance = 1 - 200 DM
        credit_history = good
        employment_duration in {< 1 year, > 7 years, 1 - 4 years}
        housing = rent
        job in {skilled, unskilled}
        ->  class yes  [0.844]

    Rule 30/8: (11/1.1, lift 1.7)
        months_loan_duration <= 39
        credit_history = very good
        amount > 4530
        ->  class yes  [0.841]

    Rule 30/9: (36.7/5.2, lift 1.7)
        checking_balance = < 0 DM
        months_loan_duration <= 47
        credit_history = good
        savings_balance in {< 100 DM, 100 - 500 DM, unknown}
        percent_of_income > 1
        years_at_residence > 1
        years_at_residence <= 3
        other_credit in {bank, none}
        job = skilled
        ->  class yes  [0.840]

    Rule 30/10: (11.6/1.2, lift 1.7)
        credit_history = poor
        savings_balance = < 100 DM
        percent_of_income > 2
        phone = TRUE
        ->  class yes  [0.836]

    Rule 30/11: (3.4, lift 1.7)
        checking_balance = < 0 DM
        savings_balance = 100 - 500 DM
        job = unskilled
        ->  class yes  [0.813]

    Rule 30/12: (25.7/4.5, lift 1.7)
        months_loan_duration > 11
        credit_history = critical
        age <= 46
        other_credit = bank
        job in {skilled, unskilled}
        ->  class yes  [0.803]

    Rule 30/13: (7.1/0.8, lift 1.7)
        checking_balance = < 0 DM
        savings_balance = unknown
        job = management
        ->  class yes  [0.801]

    Rule 30/14: (14.1/2.3, lift 1.7)
        credit_history = poor
        employment_duration = < 1 year
        ->  class yes  [0.795]

    Rule 30/15: (29.9/6.6, lift 1.6)
        months_loan_duration > 47
        credit_history = good
        ->  class yes  [0.762]

    Rule 30/16: (15.5/3.3, lift 1.6)
        credit_history = perfect
        age > 33
        ->  class yes  [0.752]

    Rule 30/17: (17/4.9, lift 1.4)
        checking_balance = unknown
        employment_duration = unemployed
        ->  class yes  [0.689]

    Rule 30/18: (33/10, lift 1.4)
        checking_balance = unknown
        amount > 1512
        employment_duration = < 1 year
        ->  class yes  [0.686]

    Rule 30/19: (50/15.5, lift 1.4)
        checking_balance = < 0 DM
        months_loan_duration <= 36
        credit_history = good
        purpose in {car, education, renovations}
        savings_balance = < 100 DM
        ->  class yes  [0.682]

    Rule 30/20: (485.7/229.8, lift 1.1)
        months_loan_duration > 11
        savings_balance = < 100 DM
        ->  class yes  [0.527]

    Rule 30/21: (578.6/285.7, lift 1.1)
        savings_balance = < 100 DM
        ->  class yes  [0.506]

    Rule 30/22: (16, lift 1.8)
        checking_balance = 1 - 200 DM
        months_loan_duration > 18
        amount <= 8487
        savings_balance = unknown
        ->  class no  [0.944]

    Rule 30/23: (12.8, lift 1.8)
        credit_history = critical
        purpose = car
        other_credit = none
        dependents > 1
        ->  class no  [0.932]

    Rule 30/24: (11.5, lift 1.8)
        checking_balance = 1 - 200 DM
        employment_duration in {4 - 7 years, unemployed}
        housing = rent
        ->  class no  [0.926]

    Rule 30/25: (10.9, lift 1.8)
        credit_history = critical
        age > 60
        ->  class no  [0.922]

    Rule 30/26: (9.3, lift 1.8)
        checking_balance = 1 - 200 DM
        months_loan_duration > 11
        months_loan_duration <= 16
        credit_history = critical
        ->  class no  [0.911]

    Rule 30/27: (9, lift 1.8)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 47
        credit_history = good
        amount <= 8487
        housing = own
        existing_loans_count > 1
        ->  class no  [0.909]

    Rule 30/28: (16.2/0.9, lift 1.7)
        checking_balance = < 0 DM
        months_loan_duration <= 27
        credit_history = good
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        percent_of_income > 1
        years_at_residence > 3
        age > 21
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.897]

    Rule 30/29: (7.3, lift 1.7)
        checking_balance = > 200 DM
        credit_history = good
        phone = TRUE
        ->  class no  [0.892]

    Rule 30/30: (5.2, lift 1.7)
        credit_history = critical
        age > 46
        other_credit = bank
        ->  class no  [0.862]

    Rule 30/31: (12.9/1.1, lift 1.7)
        checking_balance = < 0 DM
        months_loan_duration > 11
        credit_history = critical
        purpose = furniture/appliances
        savings_balance = < 100 DM
        age <= 59
        ->  class no  [0.858]

    Rule 30/32: (4.9, lift 1.6)
        credit_history = critical
        other_credit = bank
        job = management
        ->  class no  [0.855]

    Rule 30/33: (6.8/0.5, lift 1.6)
        months_loan_duration > 39
        credit_history = very good
        ->  class no  [0.825]

    Rule 30/34: (21.4/3.7, lift 1.5)
        credit_history = perfect
        age <= 33
        housing = own
        ->  class no  [0.800]

    Rule 30/35: (61.4/19.1, lift 1.3)
        credit_history = poor
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years, unemployed}
        ->  class no  [0.683]

    Rule 30/36: (40.6/13, lift 1.3)
        checking_balance = 1 - 200 DM
        years_at_residence <= 1
        housing = own
        ->  class no  [0.673]

    Rule 30/37: (51.4/16.9, lift 1.3)
        checking_balance = unknown
        employment_duration = > 7 years
        ->  class no  [0.666]

    Rule 30/38: (116.2/47.5, lift 1.1)
        percent_of_income <= 1
        ->  class no  [0.590]

    Rule 30/39: (245.5/102, lift 1.1)
        checking_balance = unknown
        ->  class no  [0.584]

    Rule 30/40: (841/398.8, lift 1.0)
        months_loan_duration <= 47
        ->  class no  [0.526]

    Default class: no

    -----  Trial 31:  -----

    Rules:

    Rule 31/1: (10.8, lift 2.0)
        amount > 12204
        other_credit = none
        housing = other
        ->  class yes  [0.922]

    Rule 31/2: (17.6/0.8, lift 2.0)
        checking_balance = < 0 DM
        age <= 24
        other_credit = none
        housing = own
        job = skilled
        ->  class yes  [0.909]

    Rule 31/3: (8.9, lift 2.0)
        checking_balance = 1 - 200 DM
        months_loan_duration > 33
        purpose = furniture/appliances
        employment_duration in {1 - 4 years, 4 - 7 years}
        housing = own
        job = skilled
        dependents <= 1
        ->  class yes  [0.909]

    Rule 31/4: (9.5, lift 2.0)
        savings_balance in {100 - 500 DM, 500 - 1000 DM, unknown}
        percent_of_income <= 1
        housing = other
        ->  class yes  [0.905]

    Rule 31/5: (8, lift 1.9)
        checking_balance = < 0 DM
        amount > 5595
        savings_balance = < 100 DM
        housing = own
        job = skilled
        ->  class yes  [0.900]

    Rule 31/6: (7.4, lift 1.9)
        credit_history = very good
        housing = own
        job = management
        ->  class yes  [0.893]

    Rule 31/7: (6.3, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration > 11
        purpose = car
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        percent_of_income <= 3
        housing = own
        ->  class yes  [0.879]

    Rule 31/8: (12.4/1.3, lift 1.8)
        percent_of_income > 2
        percent_of_income <= 3
        other_credit = none
        housing = other
        ->  class yes  [0.837]

    Rule 31/9: (3.6, lift 1.8)
        other_credit = store
        housing = other
        ->  class yes  [0.821]

    Rule 31/10: (3.4, lift 1.8)
        purpose = education
        housing = rent
        dependents > 1
        ->  class yes  [0.816]

    Rule 31/11: (12.6/1.8, lift 1.8)
        savings_balance = unknown
        percent_of_income > 3
        housing = own
        job = unskilled
        ->  class yes  [0.812]

    Rule 31/12: (4.5/0.3, lift 1.7)
        purpose = renovations
        housing = other
        ->  class yes  [0.805]

    Rule 31/13: (8/1, lift 1.7)
        purpose in {education, furniture/appliances}
        savings_balance = unknown
        job = management
        ->  class yes  [0.796]

    Rule 31/14: (14/2.4, lift 1.7)
        percent_of_income > 2
        other_credit = none
        housing = other
        dependents > 1
        ->  class yes  [0.786]

    Rule 31/15: (15.2/3.4, lift 1.6)
        savings_balance in {> 1000 DM, 100 - 500 DM}
        housing = own
        job = management
        ->  class yes  [0.743]

    Rule 31/16: (42.9/13.2, lift 1.5)
        checking_balance = 1 - 200 DM
        purpose = furniture/appliances
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years}
        housing = own
        existing_loans_count <= 1
        job = skilled
        dependents <= 1
        ->  class yes  [0.683]

    Rule 31/17: (37.9/12.3, lift 1.4)
        checking_balance = 1 - 200 DM
        purpose = car
        amount <= 2859
        ->  class yes  [0.666]

    Rule 31/18: (126.1/53.5, lift 1.2)
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, unemployed}
        housing = rent
        dependents <= 1
        ->  class yes  [0.574]

    Rule 31/19: (279.1/139.4, lift 1.1)
        checking_balance = 1 - 200 DM
        ->  class yes  [0.501]

    Rule 31/20: (8.3, lift 1.7)
        savings_balance = < 100 DM
        years_at_residence <= 1
        job = management
        ->  class no  [0.903]

    Rule 31/21: (11.1/0.3, lift 1.7)
        credit_history in {critical, poor}
        amount <= 2039
        housing = rent
        ->  class no  [0.898]

    Rule 31/22: (7.4, lift 1.7)
        credit_history = good
        age > 39
        housing = rent
        existing_loans_count <= 1
        dependents <= 1
        ->  class no  [0.894]

    Rule 31/23: (16.7/1.3, lift 1.6)
        credit_history = good
        purpose in {business, car, renovations}
        percent_of_income <= 2
        housing = rent
        existing_loans_count <= 1
        ->  class no  [0.879]

    Rule 31/24: (4.8, lift 1.6)
        savings_balance = > 1000 DM
        housing = rent
        dependents <= 1
        ->  class no  [0.853]

    Rule 31/25: (875.7/398.4, lift 1.0)
        amount <= 12204
        ->  class no  [0.545]

    Default class: no

    -----  Trial 32:  -----

    Rules:

    Rule 32/1: (14.3, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration > 28
        purpose = furniture/appliances
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, 4 - 7 years}
        existing_loans_count <= 1
        dependents <= 1
        ->  class yes  [0.939]

    Rule 32/2: (11.3, lift 1.9)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = perfect
        percent_of_income > 3
        ->  class yes  [0.925]

    Rule 32/3: (11, lift 1.9)
        credit_history = critical
        purpose = car
        age <= 35
        other_credit in {bank, store}
        ->  class yes  [0.923]

    Rule 32/4: (8.1, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = perfect
        savings_balance = < 100 DM
        age > 34
        ->  class yes  [0.901]

    Rule 32/5: (15/0.7, lift 1.8)
        months_loan_duration > 16
        credit_history = poor
        savings_balance = < 100 DM
        percent_of_income > 3
        ->  class yes  [0.899]

    Rule 32/6: (7.5, lift 1.8)
        months_loan_duration > 16
        credit_history = poor
        savings_balance = < 100 DM
        years_at_residence <= 1
        ->  class yes  [0.895]

    Rule 32/7: (6.7, lift 1.8)
        credit_history = good
        purpose = furniture/appliances
        amount <= 1549
        percent_of_income <= 2
        years_at_residence <= 1
        ->  class yes  [0.885]

    Rule 32/8: (6.3, lift 1.8)
        credit_history = perfect
        savings_balance in {< 100 DM, 100 - 500 DM}
        dependents > 1
        ->  class yes  [0.880]

    Rule 32/9: (13.6/1.4, lift 1.7)
        credit_history = critical
        purpose = furniture/appliances
        age <= 35
        job = unskilled
        ->  class yes  [0.844]

    Rule 32/10: (13.6/1.5, lift 1.7)
        checking_balance = 1 - 200 DM
        purpose = furniture/appliances
        savings_balance in {100 - 500 DM, 500 - 1000 DM}
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, 4 - 7 years}
        years_at_residence > 1
        ->  class yes  [0.838]

    Rule 32/11: (41.6/7.3, lift 1.6)
        credit_history = good
        purpose = car
        amount <= 1393
        years_at_residence > 1
        phone = FALSE
        ->  class yes  [0.811]

    Rule 32/12: (15.1/2.3, lift 1.6)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        savings_balance in {100 - 500 DM, unknown}
        existing_loans_count <= 1
        ->  class yes  [0.808]

    Rule 32/13: (44.2/7.9, lift 1.6)
        credit_history = good
        amount > 1209
        existing_loans_count > 1
        dependents <= 1
        ->  class yes  [0.807]

    Rule 32/14: (16.4/2.8, lift 1.6)
        credit_history = critical
        purpose = car
        age <= 29
        housing = own
        ->  class yes  [0.793]

    Rule 32/15: (17.3/3.2, lift 1.6)
        credit_history = good
        purpose = furniture/appliances
        employment_duration = unemployed
        ->  class yes  [0.783]

    Rule 32/16: (17.9/3.9, lift 1.5)
        months_loan_duration > 18
        credit_history = good
        purpose = business
        ->  class yes  [0.755]

    Rule 32/17: (42.6/10.7, lift 1.5)
        months_loan_duration <= 39
        credit_history = very good
        amount > 409
        age > 23
        ->  class yes  [0.738]

    Rule 32/18: (46.5/14.2, lift 1.4)
        credit_history = good
        purpose = furniture/appliances
        amount > 4057
        ->  class yes  [0.687]

    Rule 32/19: (768.5/379.5, lift 1.0)
        years_at_residence > 1
        ->  class yes  [0.506]

    Rule 32/20: (15, lift 1.9)
        checking_balance = unknown
        purpose = furniture/appliances
        existing_loans_count <= 1
        phone = TRUE
        ->  class no  [0.941]

    Rule 32/21: (4.5, lift 1.7)
        credit_history = perfect
        savings_balance in {500 - 1000 DM, unknown}
        ->  class no  [0.847]

    Rule 32/22: (3.9, lift 1.6)
        credit_history = good
        purpose = car0
        existing_loans_count <= 1
        ->  class no  [0.829]

    Rule 32/23: (24.7/4.9, lift 1.5)
        credit_history = poor
        savings_balance in {100 - 500 DM, unknown}
        ->  class no  [0.777]

    Rule 32/24: (42.8/15.2, lift 1.3)
        checking_balance = > 200 DM
        purpose = furniture/appliances
        ->  class no  [0.639]

    Rule 32/25: (870.4/424.1, lift 1.0)
        amount <= 11328
        ->  class no  [0.513]

    Default class: yes

    -----  Trial 33:  -----

    Rules:

    Rule 33/1: (13, lift 1.8)
        checking_balance = < 0 DM
        employment_duration = > 7 years
        existing_loans_count <= 1
        job = management
        ->  class yes  [0.933]

    Rule 33/2: (11.8, lift 1.8)
        checking_balance = 1 - 200 DM
        credit_history = good
        amount > 10366
        ->  class yes  [0.927]

    Rule 33/3: (11.9/1.3, lift 1.6)
        checking_balance = < 0 DM
        savings_balance = unknown
        job = skilled
        phone = FALSE
        ->  class yes  [0.834]

    Rule 33/4: (36.1/6.5, lift 1.5)
        checking_balance = < 0 DM
        savings_balance = < 100 DM
        job = skilled
        phone = TRUE
        ->  class yes  [0.803]

    Rule 33/5: (2.9, lift 1.5)
        checking_balance = 1 - 200 DM
        existing_loans_count > 3
        ->  class yes  [0.798]

    Rule 33/6: (10.4/1.5, lift 1.5)
        checking_balance = > 200 DM
        dependents > 1
        ->  class yes  [0.797]

    Rule 33/7: (25.9/8.2, lift 1.3)
        purpose = renovations
        ->  class yes  [0.672]

    Rule 33/8: (592.4/267.8, lift 1.1)
        savings_balance = < 100 DM
        ->  class yes  [0.548]

    Rule 33/9: (512.8/233, lift 1.1)
        employment_duration in {< 1 year, 1 - 4 years}
        ->  class yes  [0.546]

    Rule 33/10: (16.6, lift 2.0)
        checking_balance = unknown
        employment_duration = 4 - 7 years
        years_at_residence > 2
        ->  class no  [0.946]

    Rule 33/11: (15.2, lift 2.0)
        checking_balance = 1 - 200 DM
        months_loan_duration > 18
        amount <= 10366
        savings_balance = unknown
        ->  class no  [0.942]

    Rule 33/12: (14.1, lift 2.0)
        checking_balance = < 0 DM
        months_loan_duration <= 16
        credit_history = good
        amount > 1386
        job = skilled
        phone = FALSE
        ->  class no  [0.938]

    Rule 33/13: (13.1, lift 1.9)
        checking_balance = 1 - 200 DM
        credit_history in {critical, perfect, poor, very good}
        savings_balance = unknown
        ->  class no  [0.934]

    Rule 33/14: (13, lift 1.9)
        checking_balance = unknown
        savings_balance in {< 100 DM, unknown}
        employment_duration = > 7 years
        job = skilled
        ->  class no  [0.933]

    Rule 33/15: (9.4, lift 1.9)
        checking_balance = unknown
        months_loan_duration > 24
        employment_duration = > 7 years
        ->  class no  [0.913]

    Rule 33/16: (9.4, lift 1.9)
        checking_balance = unknown
        months_loan_duration <= 9
        employment_duration = 1 - 4 years
        ->  class no  [0.912]

    Rule 33/17: (9.2, lift 1.9)
        checking_balance = 1 - 200 DM
        purpose = furniture/appliances
        amount <= 1829
        savings_balance = < 100 DM
        job = unskilled
        ->  class no  [0.911]

    Rule 33/18: (8.6, lift 1.9)
        checking_balance = unknown
        purpose in {car, education}
        employment_duration = < 1 year
        ->  class no  [0.905]

    Rule 33/19: (8, lift 1.9)
        checking_balance = > 200 DM
        age <= 24
        ->  class no  [0.900]

    Rule 33/20: (8, lift 1.9)
        checking_balance = unknown
        employment_duration = 1 - 4 years
        percent_of_income <= 1
        ->  class no  [0.900]

    Rule 33/21: (7, lift 1.8)
        checking_balance = > 200 DM
        other_credit = bank
        dependents <= 1
        ->  class no  [0.889]

    Rule 33/22: (6.4, lift 1.8)
        checking_balance = < 0 DM
        percent_of_income > 1
        percent_of_income <= 2
        job = management
        ->  class no  [0.881]

    Rule 33/23: (16.2/1.2, lift 1.8)
        checking_balance = unknown
        purpose = furniture/appliances
        amount <= 4594
        employment_duration = < 1 year
        age <= 41
        ->  class no  [0.879]

    Rule 33/24: (5.6, lift 1.8)
        checking_balance = unknown
        employment_duration = 1 - 4 years
        other_credit = none
        dependents > 1
        ->  class no  [0.869]

    Rule 33/25: (17.1/1.5, lift 1.8)
        checking_balance = > 200 DM
        employment_duration in {> 7 years, 4 - 7 years}
        existing_loans_count <= 2
        dependents <= 1
        ->  class no  [0.867]

    Rule 33/26: (5.1, lift 1.8)
        checking_balance = < 0 DM
        amount <= 7596
        existing_loans_count > 1
        job = management
        ->  class no  [0.859]

    Rule 33/27: (12/1.2, lift 1.8)
        checking_balance = < 0 DM
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years, unemployed}
        percent_of_income > 3
        other_credit = none
        job = management
        ->  class no  [0.846]

    Rule 33/28: (25.8/3.5, lift 1.7)
        checking_balance = unknown
        employment_duration = 4 - 7 years
        age > 23
        ->  class no  [0.840]

    Rule 33/29: (8.4/0.7, lift 1.7)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        savings_balance = < 100 DM
        years_at_residence <= 3
        job = unskilled
        ->  class no  [0.834]

    Rule 33/30: (13.6/1.7, lift 1.7)
        checking_balance = 1 - 200 DM
        credit_history in {critical, poor}
        savings_balance = 100 - 500 DM
        existing_loans_count <= 3
        ->  class no  [0.823]

    Rule 33/31: (3.6, lift 1.7)
        checking_balance = > 200 DM
        savings_balance in {> 1000 DM, 500 - 1000 DM}
        existing_loans_count <= 2
        ->  class no  [0.821]

    Rule 33/32: (51.1/19.1, lift 1.3)
        purpose = car
        years_at_residence <= 3
        phone = TRUE
        ->  class no  [0.621]

    Rule 33/33: (808.7/408.6, lift 1.0)
        amount <= 7596
        ->  class no  [0.495]

    Default class: yes

    -----  Trial 34:  -----

    Rules:

    Rule 34/1: (12.8, lift 2.1)
        months_loan_duration > 14
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        percent_of_income > 1
        years_at_residence > 1
        years_at_residence <= 3
        age > 32
        other_credit = none
        existing_loans_count <= 1
        ->  class yes  [0.933]

    Rule 34/2: (11.9, lift 2.1)
        checking_balance = 1 - 200 DM
        credit_history = good
        amount > 8858
        ->  class yes  [0.928]

    Rule 34/3: (11.9, lift 2.1)
        months_loan_duration > 14
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        percent_of_income > 1
        years_at_residence > 2
        years_at_residence <= 3
        other_credit = none
        job in {skilled, unskilled}
        ->  class yes  [0.928]

    Rule 34/4: (13.5/0.3, lift 2.0)
        months_loan_duration > 10
        purpose = renovations
        amount <= 1995
        ->  class yes  [0.913]

    Rule 34/5: (9.2, lift 2.0)
        credit_history in {poor, very good}
        purpose = business
        employment_duration = < 1 year
        ->  class yes  [0.910]

    Rule 34/6: (7.2, lift 2.0)
        months_loan_duration <= 14
        credit_history = good
        purpose = furniture/appliances
        amount > 4057
        ->  class yes  [0.892]

    Rule 34/7: (18.9/1.4, lift 2.0)
        checking_balance = < 0 DM
        months_loan_duration <= 40
        credit_history = good
        purpose = car
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years}
        years_at_residence <= 2
        ->  class yes  [0.887]

    Rule 34/8: (18.3/1.6, lift 1.9)
        purpose = car
        savings_balance = < 100 DM
        percent_of_income > 1
        other_credit = bank
        ->  class yes  [0.872]

    Rule 34/9: (12.4/0.9, lift 1.9)
        checking_balance = < 0 DM
        credit_history = good
        purpose = car
        age > 43
        other_credit = none
        ->  class yes  [0.866]

    Rule 34/10: (5.5, lift 1.9)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        ->  class yes  [0.866]

    Rule 34/11: (4.7, lift 1.9)
        purpose = business
        employment_duration = unemployed
        ->  class yes  [0.851]

    Rule 34/12: (4.5, lift 1.9)
        purpose = furniture/appliances
        amount <= 1138
        savings_balance = 500 - 1000 DM
        existing_loans_count <= 1
        ->  class yes  [0.847]

    Rule 34/13: (4.4, lift 1.9)
        months_loan_duration > 36
        credit_history = critical
        purpose = furniture/appliances
        savings_balance = < 100 DM
        ->  class yes  [0.843]

    Rule 34/14: (7.9/0.6, lift 1.9)
        credit_history = critical
        purpose = car
        savings_balance = < 100 DM
        age <= 29
        ->  class yes  [0.834]

    Rule 34/15: (5.2/0.2, lift 1.8)
        purpose = furniture/appliances
        savings_balance = < 100 DM
        years_at_residence <= 1
        dependents > 1
        ->  class yes  [0.829]

    Rule 34/16: (3.5, lift 1.8)
        purpose = business
        employment_duration = 1 - 4 years
        existing_loans_count > 2
        ->  class yes  [0.817]

    Rule 34/17: (16/3.3, lift 1.7)
        months_loan_duration > 14
        savings_balance = < 100 DM
        age > 22
        age <= 23
        ->  class yes  [0.761]

    Rule 34/18: (10.8/2.1, lift 1.7)
        purpose = business
        employment_duration = > 7 years
        other_credit = none
        ->  class yes  [0.760]

    Rule 34/19: (45.4/10.8, lift 1.7)
        purpose = education
        savings_balance in {< 100 DM, > 1000 DM, 100 - 500 DM}
        age <= 43
        ->  class yes  [0.751]

    Rule 34/20: (20.6/4.8, lift 1.6)
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        years_at_residence > 1
        age <= 22
        ->  class yes  [0.742]

    Rule 34/21: (17/5.6, lift 1.4)
        savings_balance = 500 - 1000 DM
        existing_loans_count > 1
        ->  class yes  [0.653]

    Rule 34/22: (41.3/15.6, lift 1.4)
        credit_history in {perfect, very good}
        other_credit = none
        ->  class yes  [0.616]

    Rule 34/23: (132.1/63.1, lift 1.2)
        job = management
        ->  class yes  [0.522]

    Rule 34/24: (198.4/95.5, lift 1.2)
        months_loan_duration > 27
        ->  class yes  [0.518]

    Rule 34/25: (10.3/1.1, lift 1.5)
        purpose = furniture/appliances
        savings_balance = > 1000 DM
        ->  class no  [0.826]

    Rule 34/26: (11.4/1.5, lift 1.5)
        purpose = education
        age > 43
        ->  class no  [0.811]

    Rule 34/27: (52.5/11.7, lift 1.4)
        checking_balance = unknown
        purpose = car
        other_credit = none
        ->  class no  [0.767]

    Rule 34/28: (58.5/16.5, lift 1.3)
        months_loan_duration <= 36
        credit_history = critical
        purpose = furniture/appliances
        savings_balance = < 100 DM
        ->  class no  [0.712]

    Rule 34/29: (852.7/376.1, lift 1.0)
        amount <= 8858
        ->  class no  [0.559]

    Default class: no

    -----  Trial 35:  -----

    Rules:

    Rule 35/1: (9.8, lift 1.9)
        checking_balance = 1 - 200 DM
        credit_history = good
        amount > 8648
        ->  class yes  [0.916]

    Rule 35/2: (14.8/0.7, lift 1.9)
        checking_balance = unknown
        credit_history = good
        percent_of_income <= 2
        other_credit = none
        existing_loans_count > 1
        ->  class yes  [0.897]

    Rule 35/3: (6.3, lift 1.8)
        months_loan_duration <= 36
        credit_history = poor
        years_at_residence <= 1
        other_credit = none
        ->  class yes  [0.879]

    Rule 35/4: (5.2, lift 1.8)
        amount <= 1007
        years_at_residence > 3
        other_credit = bank
        ->  class yes  [0.862]

    Rule 35/5: (22.2/2.4, lift 1.8)
        checking_balance in {> 200 DM, unknown}
        purpose in {business, car}
        percent_of_income > 1
        years_at_residence <= 3
        other_credit = bank
        ->  class yes  [0.861]

    Rule 35/6: (12.6/1.1, lift 1.8)
        checking_balance in {> 200 DM, unknown}
        amount <= 8648
        years_at_residence > 1
        other_credit = store
        job in {management, skilled}
        ->  class yes  [0.854]

    Rule 35/7: (8.3/0.8, lift 1.7)
        months_loan_duration <= 36
        credit_history = very good
        other_credit = none
        ->  class yes  [0.830]

    Rule 35/8: (15.9/2.5, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        amount <= 1372
        savings_balance = < 100 DM
        percent_of_income <= 2
        ->  class yes  [0.806]

    Rule 35/9: (16.3/3.7, lift 1.5)
        months_loan_duration <= 16
        percent_of_income > 2
        housing = other
        ->  class yes  [0.742]

    Rule 35/10: (20.7/5.1, lift 1.5)
        purpose = furniture/appliances
        employment_duration in {> 7 years, 1 - 4 years}
        years_at_residence <= 3
        other_credit = bank
        ->  class yes  [0.729]

    Rule 35/11: (25.9/6.6, lift 1.5)
        checking_balance = 1 - 200 DM
        credit_history = good
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        percent_of_income > 2
        ->  class yes  [0.726]

    Rule 35/12: (588.4/280.9, lift 1.1)
        checking_balance in {< 0 DM, 1 - 200 DM}
        ->  class yes  [0.523]

    Rule 35/13: (12.4, lift 1.8)
        checking_balance = < 0 DM
        months_loan_duration > 16
        employment_duration in {< 1 year, 1 - 4 years, unemployed}
        other_credit in {bank, none}
        existing_loans_count <= 1
        job = management
        ->  class no  [0.931]

    Rule 35/14: (9.9, lift 1.8)
        credit_history = good
        savings_balance = < 100 DM
        percent_of_income <= 2
        housing = other
        ->  class no  [0.916]

    Rule 35/15: (8, lift 1.7)
        checking_balance = < 0 DM
        credit_history = good
        amount > 1372
        percent_of_income <= 2
        other_credit = none
        housing = rent
        ->  class no  [0.900]

    Rule 35/16: (7.7, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        savings_balance = unknown
        employment_duration in {< 1 year, > 7 years, 4 - 7 years}
        percent_of_income <= 2
        housing = own
        ->  class no  [0.897]

    Rule 35/17: (7.6, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        purpose = furniture/appliances
        percent_of_income > 2
        phone = TRUE
        ->  class no  [0.896]

    Rule 35/18: (7.4, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 16
        credit_history = critical
        purpose = furniture/appliances
        percent_of_income > 2
        age <= 54
        ->  class no  [0.893]

    Rule 35/19: (6.9, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        purpose = furniture/appliances
        employment_duration in {< 1 year, 4 - 7 years}
        other_credit = bank
        ->  class no  [0.888]

    Rule 35/20: (21.8/1.7, lift 1.7)
        checking_balance = 1 - 200 DM
        credit_history = good
        amount > 1372
        amount <= 8648
        savings_balance = < 100 DM
        percent_of_income <= 2
        ->  class no  [0.885]

    Rule 35/21: (9.1/0.3, lift 1.7)
        credit_history = good
        savings_balance in {< 100 DM, unknown}
        percent_of_income > 2
        age > 54
        other_credit in {bank, none}
        existing_loans_count <= 1
        ->  class no  [0.881]

    Rule 35/22: (46.2/5.1, lift 1.7)
        checking_balance = unknown
        credit_history = critical
        amount <= 11816
        other_credit = none
        ->  class no  [0.873]

    Rule 35/23: (5.6, lift 1.7)
        credit_history = good
        amount > 1372
        percent_of_income <= 2
        other_credit = store
        ->  class no  [0.868]

    Rule 35/24: (5.4, lift 1.7)
        credit_history = good
        savings_balance = > 1000 DM
        percent_of_income > 2
        ->  class no  [0.865]

    Rule 35/25: (5.3, lift 1.7)
        credit_history = critical
        purpose in {car0, renovations}
        percent_of_income > 2
        ->  class no  [0.864]

    Rule 35/26: (15/1.3, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        amount > 1007
        years_at_residence > 3
        other_credit = bank
        ->  class no  [0.862]

    Rule 35/27: (23.5/3.6, lift 1.6)
        checking_balance = < 0 DM
        months_loan_duration <= 16
        amount > 674
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        other_credit = none
        housing = own
        ->  class no  [0.821]

    Rule 35/28: (21.1/3.3, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        purpose = car
        age > 29
        housing = own
        ->  class no  [0.815]

    Rule 35/29: (15.9/2.9, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = poor
        savings_balance in {100 - 500 DM, unknown}
        ->  class no  [0.782]

    Rule 35/30: (198.1/64.7, lift 1.3)
        checking_balance in {> 200 DM, unknown}
        years_at_residence > 1
        other_credit = none
        ->  class no  [0.672]

    Rule 35/31: (71.6/26.3, lift 1.2)
        percent_of_income > 2
        years_at_residence <= 1
        ->  class no  [0.629]

    Rule 35/32: (168.6/65.7, lift 1.2)
        purpose in {business, car, education}
        percent_of_income <= 2
        ->  class no  [0.609]

    Default class: no

    -----  Trial 36:  -----

    Rules:

    Rule 36/1: (10, lift 2.0)
        checking_balance = 1 - 200 DM
        amount > 888
        amount <= 1316
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        other_credit = none
        phone = FALSE
        ->  class yes  [0.917]

    Rule 36/2: (9.2, lift 1.9)
        checking_balance = unknown
        amount > 4594
        employment_duration = < 1 year
        housing = own
        ->  class yes  [0.911]

    Rule 36/3: (6.8, lift 1.9)
        checking_balance = < 0 DM
        percent_of_income <= 1
        job = management
        ->  class yes  [0.886]

    Rule 36/4: (6.3, lift 1.9)
        checking_balance = 1 - 200 DM
        employment_duration = < 1 year
        other_credit = store
        ->  class yes  [0.879]

    Rule 36/5: (12.3/1.4, lift 1.8)
        checking_balance = 1 - 200 DM
        employment_duration = > 7 years
        job = management
        ->  class yes  [0.832]

    Rule 36/6: (11.1/2.2, lift 1.6)
        checking_balance = > 200 DM
        dependents > 1
        ->  class yes  [0.755]

    Rule 36/7: (758.7/390.9, lift 1.0)
        months_loan_duration > 11
        ->  class yes  [0.485]

    Rule 36/8: (22.5, lift 1.8)
        checking_balance = unknown
        purpose in {furniture/appliances, renovations}
        employment_duration = 1 - 4 years
        job = skilled
        ->  class no  [0.959]

    Rule 36/9: (14.4, lift 1.8)
        checking_balance = 1 - 200 DM
        employment_duration = < 1 year
        years_at_residence <= 1
        other_credit in {bank, none}
        housing = own
        ->  class no  [0.939]

    Rule 36/10: (13.5, lift 1.8)
        checking_balance = unknown
        employment_duration = 4 - 7 years
        years_at_residence > 2
        ->  class no  [0.935]

    Rule 36/11: (12.2, lift 1.8)
        checking_balance = unknown
        employment_duration = 4 - 7 years
        phone = TRUE
        ->  class no  [0.930]

    Rule 36/12: (12.1, lift 1.7)
        checking_balance = unknown
        months_loan_duration > 24
        employment_duration = > 7 years
        ->  class no  [0.929]

    Rule 36/13: (10, lift 1.7)
        savings_balance in {> 1000 DM, unknown}
        employment_duration = < 1 year
        years_at_residence > 1
        housing = own
        ->  class no  [0.917]

    Rule 36/14: (9.5, lift 1.7)
        checking_balance = 1 - 200 DM
        employment_duration = 4 - 7 years
        housing = rent
        ->  class no  [0.913]

    Rule 36/15: (9.3, lift 1.7)
        purpose = furniture/appliances
        amount <= 888
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        other_credit = none
        ->  class no  [0.912]

    Rule 36/16: (9.3, lift 1.7)
        checking_balance = 1 - 200 DM
        purpose in {business, education}
        employment_duration = 4 - 7 years
        housing = own
        ->  class no  [0.911]

    Rule 36/17: (8.4, lift 1.7)
        checking_balance = unknown
        purpose in {business, car, education}
        amount <= 1271
        employment_duration = 1 - 4 years
        ->  class no  [0.904]

    Rule 36/18: (21.6/1.4, lift 1.7)
        checking_balance = unknown
        employment_duration = > 7 years
        age > 41
        ->  class no  [0.897]

    Rule 36/19: (7.6, lift 1.7)
        checking_balance = < 0 DM
        percent_of_income > 1
        percent_of_income <= 2
        job = management
        ->  class no  [0.895]

    Rule 36/20: (6.5, lift 1.7)
        checking_balance = > 200 DM
        age <= 24
        ->  class no  [0.883]

    Rule 36/21: (6.4, lift 1.7)
        checking_balance = unknown
        employment_duration = 1 - 4 years
        percent_of_income <= 1
        ->  class no  [0.880]

    Rule 36/22: (6.2, lift 1.7)
        checking_balance = > 200 DM
        purpose = car
        existing_loans_count <= 2
        dependents <= 1
        ->  class no  [0.878]

    Rule 36/23: (29.7/2.9, lift 1.7)
        checking_balance = 1 - 200 DM
        purpose in {car, furniture/appliances}
        employment_duration = > 7 years
        job in {skilled, unskilled}
        ->  class no  [0.877]

    Rule 36/24: (21.5/2, lift 1.6)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 22
        employment_duration = 4 - 7 years
        ->  class no  [0.874]

    Rule 36/25: (5.9, lift 1.6)
        purpose = furniture/appliances
        savings_balance = > 1000 DM
        other_credit = none
        ->  class no  [0.873]

    Rule 36/26: (11.6/0.9, lift 1.6)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        savings_balance = < 100 DM
        percent_of_income <= 1
        other_credit = none
        phone = FALSE
        ->  class no  [0.861]

    Rule 36/27: (4.6, lift 1.6)
        checking_balance = > 200 DM
        other_credit = bank
        dependents <= 1
        ->  class no  [0.848]

    Rule 36/28: (18.6/2.6, lift 1.5)
        checking_balance = < 0 DM
        amount <= 5511
        percent_of_income > 1
        years_at_residence > 2
        job = management
        ->  class no  [0.823]

    Rule 36/29: (22.1/3.4, lift 1.5)
        checking_balance = < 0 DM
        months_loan_duration <= 16
        purpose = furniture/appliances
        savings_balance = < 100 DM
        age <= 57
        other_credit = none
        phone = FALSE
        ->  class no  [0.816]

    Rule 36/30: (51.9/16, lift 1.3)
        checking_balance = 1 - 200 DM
        savings_balance in {> 1000 DM, unknown}
        ->  class no  [0.684]

    Rule 36/31: (141.3/54.4, lift 1.2)
        months_loan_duration <= 11
        ->  class no  [0.613]

    Default class: no

    -----  Trial 37:  -----

    Rules:

    Rule 37/1: (9.6, lift 2.2)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        savings_balance = < 100 DM
        years_at_residence > 1
        other_credit = bank
        job = skilled
        ->  class yes  [0.914]

    Rule 37/2: (9.3, lift 2.2)
        checking_balance = 1 - 200 DM
        purpose = car
        amount <= 1493
        years_at_residence > 1
        housing = own
        ->  class yes  [0.912]

    Rule 37/3: (6.4, lift 2.1)
        checking_balance = < 0 DM
        months_loan_duration <= 10
        credit_history = good
        purpose = car
        savings_balance = < 100 DM
        dependents <= 1
        ->  class yes  [0.881]

    Rule 37/4: (5.4, lift 2.1)
        checking_balance = < 0 DM
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        job = skilled
        phone = TRUE
        ->  class yes  [0.864]

    Rule 37/5: (27.4/3.1, lift 2.1)
        checking_balance = 1 - 200 DM
        credit_history in {good, perfect, very good}
        employment_duration in {< 1 year, > 7 years, 1 - 4 years}
        housing = rent
        job in {skilled, unemployed, unskilled}
        ->  class yes  [0.861]

    Rule 37/6: (8.3/0.5, lift 2.1)
        checking_balance = unknown
        credit_history = poor
        percent_of_income > 3
        age <= 30
        ->  class yes  [0.860]

    Rule 37/7: (19.7/2.2, lift 2.0)
        checking_balance = < 0 DM
        months_loan_duration <= 40
        credit_history = good
        purpose = car
        savings_balance = < 100 DM
        years_at_residence <= 2
        ->  class yes  [0.855]

    Rule 37/8: (4.4, lift 2.0)
        checking_balance = < 0 DM
        credit_history = good
        savings_balance = < 100 DM
        job = unemployed
        ->  class yes  [0.844]

    Rule 37/9: (11.3/1.2, lift 2.0)
        checking_balance = < 0 DM
        credit_history = good
        purpose = car
        savings_balance = < 100 DM
        employment_duration in {< 1 year, > 7 years, 4 - 7 years}
        years_at_residence > 2
        dependents <= 1
        ->  class yes  [0.838]

    Rule 37/10: (8/0.6, lift 2.0)
        checking_balance = 1 - 200 DM
        months_loan_duration > 36
        purpose = furniture/appliances
        ->  class yes  [0.835]

    Rule 37/11: (5.3/0.5, lift 1.9)
        checking_balance = 1 - 200 DM
        purpose = car
        housing = own
        job = unskilled
        ->  class yes  [0.796]

    Rule 37/12: (14.5/2.6, lift 1.9)
        checking_balance = < 0 DM
        credit_history in {poor, very good}
        purpose = furniture/appliances
        savings_balance = < 100 DM
        ->  class yes  [0.784]

    Rule 37/13: (17/3.3, lift 1.9)
        checking_balance = unknown
        credit_history = good
        savings_balance in {< 100 DM, 500 - 1000 DM, unknown}
        existing_loans_count <= 1
        job = unskilled
        phone = FALSE
        ->  class yes  [0.774]

    Rule 37/14: (11.2/2.3, lift 1.8)
        credit_history = critical
        amount > 6967
        other_credit = none
        ->  class yes  [0.748]

    Rule 37/15: (25.8/6.4, lift 1.8)
        checking_balance = unknown
        months_loan_duration <= 30
        credit_history = good
        savings_balance in {< 100 DM, 500 - 1000 DM}
        years_at_residence > 1
        age <= 32
        other_credit = none
        existing_loans_count <= 1
        phone = FALSE
        ->  class yes  [0.735]

    Rule 37/16: (15.6/3.7, lift 1.7)
        checking_balance = < 0 DM
        savings_balance = unknown
        employment_duration in {> 7 years, 1 - 4 years}
        other_credit = none
        ->  class yes  [0.731]

    Rule 37/17: (16.8/4.2, lift 1.7)
        checking_balance = 1 - 200 DM
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        housing = other
        ->  class yes  [0.723]

    Rule 37/18: (10.9/3.4, lift 1.6)
        checking_balance = > 200 DM
        dependents > 1
        ->  class yes  [0.658]

    Rule 37/19: (27.8/10.3, lift 1.5)
        checking_balance = 1 - 200 DM
        housing = own
        job = management
        ->  class yes  [0.621]

    Rule 37/20: (298.2/157, lift 1.1)
        checking_balance = < 0 DM
        ->  class yes  [0.474]

    Rule 37/21: (25.3, lift 1.7)
        checking_balance = unknown
        months_loan_duration <= 30
        age > 32
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.963]

    Rule 37/22: (11.9, lift 1.6)
        checking_balance = unknown
        savings_balance in {> 1000 DM, 100 - 500 DM}
        existing_loans_count <= 1
        phone = FALSE
        ->  class no  [0.928]

    Rule 37/23: (10.6, lift 1.6)
        checking_balance = 1 - 200 DM
        employment_duration in {4 - 7 years, unemployed}
        housing = rent
        ->  class no  [0.921]

    Rule 37/24: (9.4, lift 1.6)
        credit_history = critical
        purpose = car
        savings_balance = < 100 DM
        dependents > 1
        ->  class no  [0.913]

    Rule 37/25: (9, lift 1.6)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 7
        purpose = furniture/appliances
        ->  class no  [0.909]

    Rule 37/26: (7.7, lift 1.5)
        checking_balance = 1 - 200 DM
        credit_history in {critical, poor}
        housing = rent
        ->  class no  [0.897]

    Rule 37/27: (7.7, lift 1.5)
        checking_balance = < 0 DM
        credit_history = good
        percent_of_income <= 1
        other_credit = none
        job = skilled
        phone = FALSE
        ->  class no  [0.897]

    Rule 37/28: (11.1/0.5, lift 1.5)
        checking_balance = 1 - 200 DM
        employment_duration in {< 1 year, unemployed}
        housing = other
        ->  class no  [0.885]

    Rule 37/29: (38.4/4.2, lift 1.5)
        checking_balance = unknown
        credit_history = good
        existing_loans_count <= 1
        phone = TRUE
        ->  class no  [0.871]

    Rule 37/30: (76.6/19.4, lift 1.3)
        checking_balance = unknown
        credit_history = critical
        ->  class no  [0.740]

    Rule 37/31: (639.2/252.3, lift 1.0)
        housing = own
        ->  class no  [0.605]

    Default class: no

    -----  Trial 38:  -----

    Rules:

    Rule 38/1: (11.7/0.2, lift 2.0)
        checking_balance = < 0 DM
        amount > 5804
        employment_duration = > 7 years
        ->  class yes  [0.910]

    Rule 38/2: (7.4, lift 2.0)
        months_loan_duration <= 39
        credit_history = perfect
        employment_duration = 1 - 4 years
        age > 35
        ->  class yes  [0.894]

    Rule 38/3: (7.4, lift 2.0)
        employment_duration = unemployed
        years_at_residence <= 1
        ->  class yes  [0.894]

    Rule 38/4: (6, lift 2.0)
        checking_balance = < 0 DM
        months_loan_duration > 33
        employment_duration = > 7 years
        housing in {own, rent}
        ->  class yes  [0.874]

    Rule 38/5: (10.2/0.6, lift 2.0)
        credit_history = poor
        age > 51
        ->  class yes  [0.869]

    Rule 38/6: (5.2, lift 1.9)
        purpose = business
        employment_duration = unemployed
        ->  class yes  [0.862]

    Rule 38/7: (7.6/0.6, lift 1.9)
        checking_balance = 1 - 200 DM
        employment_duration = > 7 years
        age > 28
        dependents > 1
        ->  class yes  [0.838]

    Rule 38/8: (18.8/2.6, lift 1.9)
        months_loan_duration > 39
        employment_duration = 1 - 4 years
        percent_of_income > 1
        dependents <= 1
        ->  class yes  [0.826]

    Rule 38/9: (12.9/1.7, lift 1.8)
        credit_history = poor
        employment_duration = < 1 year
        ->  class yes  [0.821]

    Rule 38/10: (13.6/1.9, lift 1.8)
        months_loan_duration > 13
        credit_history = good
        employment_duration = 1 - 4 years
        existing_loans_count > 1
        ->  class yes  [0.816]

    Rule 38/11: (25/4.1, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        employment_duration = < 1 year
        age <= 34
        ->  class yes  [0.812]

    Rule 38/12: (12.3/1.7, lift 1.8)
        credit_history in {critical, perfect}
        purpose = furniture/appliances
        amount > 1313
        employment_duration = < 1 year
        ->  class yes  [0.812]

    Rule 38/13: (11.1/1.9, lift 1.7)
        months_loan_duration > 15
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        dependents > 1
        ->  class yes  [0.777]

    Rule 38/14: (13.3/3, lift 1.7)
        credit_history = critical
        employment_duration = 1 - 4 years
        housing in {other, rent}
        dependents <= 1
        ->  class yes  [0.736]

    Rule 38/15: (31.8/8, lift 1.6)
        checking_balance = < 0 DM
        months_loan_duration > 9
        employment_duration = > 7 years
        age > 28
        housing in {own, rent}
        job in {management, skilled}
        dependents <= 1
        ->  class yes  [0.733]

    Rule 38/16: (560.7/301.6, lift 1.0)
        phone = FALSE
        ->  class yes  [0.462]

    Rule 38/17: (21.6, lift 1.7)
        checking_balance = unknown
        employment_duration = > 7 years
        age > 35
        dependents <= 1
        ->  class no  [0.958]

    Rule 38/18: (18.9, lift 1.7)
        checking_balance = unknown
        purpose = furniture/appliances
        employment_duration = 1 - 4 years
        job = skilled
        ->  class no  [0.952]

    Rule 38/19: (18.2, lift 1.7)
        months_loan_duration <= 39
        credit_history = good
        purpose in {business, car0}
        existing_loans_count <= 1
        dependents <= 1
        ->  class no  [0.951]

    Rule 38/20: (13.8, lift 1.7)
        months_loan_duration <= 15
        credit_history = good
        amount > 1289
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        job = skilled
        phone = FALSE
        ->  class no  [0.937]

    Rule 38/21: (12.5, lift 1.7)
        checking_balance = unknown
        months_loan_duration > 21
        dependents > 1
        ->  class no  [0.931]

    Rule 38/22: (11.4, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        purpose = car
        employment_duration = < 1 year
        ->  class no  [0.925]

    Rule 38/23: (10.8, lift 1.7)
        months_loan_duration <= 7
        employment_duration = 1 - 4 years
        phone = FALSE
        ->  class no  [0.922]

    Rule 38/24: (10.6, lift 1.7)
        checking_balance = < 0 DM
        employment_duration = > 7 years
        other_credit in {bank, none}
        job = unskilled
        ->  class no  [0.921]

    Rule 38/25: (17.5/0.7, lift 1.6)
        employment_duration = 1 - 4 years
        percent_of_income <= 3
        years_at_residence <= 3
        housing = own
        job = unskilled
        dependents <= 1
        ->  class no  [0.911]

    Rule 38/26: (12/0.5, lift 1.6)
        months_loan_duration <= 39
        credit_history = perfect
        employment_duration = 1 - 4 years
        age <= 35
        ->  class no  [0.893]

    Rule 38/27: (13.4/0.7, lift 1.6)
        months_loan_duration <= 15
        employment_duration = 1 - 4 years
        dependents > 1
        ->  class no  [0.888]

    Rule 38/28: (6.9, lift 1.6)
        months_loan_duration <= 39
        existing_loans_count <= 1
        job = management
        dependents <= 1
        phone = FALSE
        ->  class no  [0.887]

    Rule 38/29: (14.3/0.8, lift 1.6)
        employment_duration = 4 - 7 years
        percent_of_income <= 3
        job = unskilled
        ->  class no  [0.887]

    Rule 38/30: (16.5/2.3, lift 1.5)
        purpose = car
        employment_duration = < 1 year
        age > 34
        ->  class no  [0.819]

    Rule 38/31: (21/3.4, lift 1.5)
        months_loan_duration <= 13
        employment_duration = 1 - 4 years
        existing_loans_count > 1
        ->  class no  [0.808]

    Rule 38/32: (22.3/4.1, lift 1.4)
        credit_history = poor
        employment_duration = 1 - 4 years
        age <= 51
        ->  class no  [0.791]

    Rule 38/33: (64.7/17.8, lift 1.3)
        checking_balance in {1 - 200 DM, unknown}
        employment_duration = 4 - 7 years
        ->  class no  [0.718]

    Rule 38/34: (46.6/12.8, lift 1.3)
        purpose in {car, car0, furniture/appliances, renovations}
        employment_duration = unemployed
        years_at_residence > 1
        dependents <= 1
        ->  class no  [0.715]

    Rule 38/35: (121.3/39, lift 1.2)
        amount <= 5804
        employment_duration = > 7 years
        other_credit = none
        ->  class no  [0.675]

    Rule 38/36: (339.3/140.6, lift 1.1)
        phone = TRUE
        ->  class no  [0.585]

    Default class: no

    -----  Trial 39:  -----

    Rules:

    Rule 39/1: (23/1, lift 2.1)
        credit_history in {perfect, poor, very good}
        savings_balance = < 100 DM
        percent_of_income > 3
        phone = TRUE
        ->  class yes  [0.919]

    Rule 39/2: (10.1, lift 2.1)
        checking_balance = < 0 DM
        months_loan_duration > 15
        months_loan_duration <= 21
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        ->  class yes  [0.918]

    Rule 39/3: (8.7, lift 2.0)
        purpose = furniture/appliances
        employment_duration = < 1 year
        years_at_residence > 1
        age > 30
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.907]

    Rule 39/4: (13.8/0.5, lift 2.0)
        credit_history in {critical, good}
        purpose = car
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        percent_of_income > 1
        other_credit = bank
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.904]

    Rule 39/5: (7.7, lift 2.0)
        amount > 12204
        savings_balance = unknown
        other_credit = none
        ->  class yes  [0.896]

    Rule 39/6: (7.1, lift 2.0)
        savings_balance = < 100 DM
        employment_duration = 4 - 7 years
        age <= 23
        phone = FALSE
        ->  class yes  [0.890]

    Rule 39/7: (15.6/1, lift 2.0)
        amount <= 1223
        percent_of_income > 1
        dependents > 1
        phone = FALSE
        ->  class yes  [0.889]

    Rule 39/8: (14/1.1, lift 2.0)
        credit_history in {perfect, poor, very good}
        amount <= 7824
        age > 43
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.870]

    Rule 39/9: (4.6, lift 1.9)
        credit_history in {perfect, very good}
        purpose = car
        percent_of_income > 1
        other_credit = none
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.849]

    Rule 39/10: (12.1/1.1, lift 1.9)
        savings_balance = unknown
        years_at_residence <= 1
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.849]

    Rule 39/11: (32.8/4.9, lift 1.9)
        amount > 7824
        savings_balance = < 100 DM
        phone = TRUE
        ->  class yes  [0.831]

    Rule 39/12: (12.4/1.5, lift 1.9)
        amount > 3711
        savings_balance in {< 100 DM, 100 - 500 DM}
        dependents > 1
        phone = TRUE
        ->  class yes  [0.824]

    Rule 39/13: (17.3/2.7, lift 1.8)
        credit_history = critical
        purpose = car
        savings_balance in {< 100 DM, 100 - 500 DM}
        percent_of_income > 1
        age <= 35
        existing_loans_count > 1
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.808]

    Rule 39/14: (9.2/1.3, lift 1.8)
        months_loan_duration <= 16
        savings_balance = 100 - 500 DM
        phone = TRUE
        ->  class yes  [0.790]

    Rule 39/15: (8.6/1.3, lift 1.8)
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = unemployed
        phone = FALSE
        ->  class yes  [0.784]

    Rule 39/16: (18.1/3.6, lift 1.7)
        checking_balance = < 0 DM
        credit_history = good
        purpose = car
        percent_of_income > 2
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.769]

    Rule 39/17: (29.2/6.5, lift 1.7)
        checking_balance = < 0 DM
        credit_history in {critical, good}
        savings_balance = < 100 DM
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, 4 - 7 years}
        years_at_residence > 1
        age <= 47
        existing_loans_count <= 2
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.760]

    Rule 39/18: (25.2/6.1, lift 1.7)
        checking_balance = 1 - 200 DM
        purpose = furniture/appliances
        amount > 888
        employment_duration = 1 - 4 years
        percent_of_income > 1
        other_credit = none
        existing_loans_count <= 1
        dependents <= 1
        ->  class yes  [0.740]

    Rule 39/19: (14.9/3.4, lift 1.7)
        existing_loans_count > 2
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.738]

    Rule 39/20: (9.5/2.2, lift 1.6)
        purpose = business
        housing = rent
        ->  class yes  [0.723]

    Rule 39/21: (62.4/17.7, lift 1.6)
        amount > 1316
        savings_balance = < 100 DM
        percent_of_income > 1
        years_at_residence > 1
        age <= 25
        ->  class yes  [0.710]

    Rule 39/22: (26.8/8.2, lift 1.5)
        purpose = education
        savings_balance in {< 100 DM, > 1000 DM, 100 - 500 DM}
        percent_of_income > 1
        phone = FALSE
        ->  class yes  [0.681]

    Rule 39/23: (32/12.5, lift 1.4)
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        ->  class yes  [0.603]

    Rule 39/24: (20.9/8.2, lift 1.4)
        purpose = renovations
        ->  class yes  [0.598]

    Rule 39/25: (17.6, lift 1.7)
        amount <= 3711
        dependents > 1
        phone = TRUE
        ->  class no  [0.949]

    Rule 39/26: (6.6, lift 1.6)
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        employment_duration in {< 1 year, > 7 years}
        existing_loans_count <= 1
        phone = FALSE
        ->  class no  [0.884]

    Rule 39/27: (25.4/3.1, lift 1.5)
        purpose = business
        percent_of_income > 1
        housing = own
        dependents <= 1
        phone = FALSE
        ->  class no  [0.852]

    Rule 39/28: (877.4/384.5, lift 1.0)
        existing_loans_count <= 2
        ->  class no  [0.562]

    Default class: no

    -----  Trial 40:  -----

    Rules:

    Rule 40/1: (14.5, lift 2.1)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 20
        credit_history = poor
        savings_balance = < 100 DM
        percent_of_income > 1
        ->  class yes  [0.939]

    Rule 40/2: (13.7, lift 2.1)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration <= 16
        credit_history = very good
        amount > 409
        age > 23
        ->  class yes  [0.936]

    Rule 40/3: (11.1, lift 2.1)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = perfect
        housing in {other, rent}
        ->  class yes  [0.924]

    Rule 40/4: (10.6, lift 2.1)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = perfect
        percent_of_income > 3
        ->  class yes  [0.921]

    Rule 40/5: (9.1, lift 2.0)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 7
        purpose = furniture/appliances
        employment_duration = < 1 year
        years_at_residence > 2
        years_at_residence <= 3
        ->  class yes  [0.910]

    Rule 40/6: (15.7/0.6, lift 2.0)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 7
        credit_history = good
        purpose = car
        age > 43
        other_credit = none
        ->  class yes  [0.909]

    Rule 40/7: (7.3, lift 2.0)
        checking_balance = 1 - 200 DM
        credit_history = critical
        purpose in {business, education}
        employment_duration = > 7 years
        dependents <= 1
        ->  class yes  [0.893]

    Rule 40/8: (15.5/1.1, lift 2.0)
        checking_balance = < 0 DM
        months_loan_duration > 27
        credit_history = good
        purpose = furniture/appliances
        savings_balance in {< 100 DM, unknown}
        dependents <= 1
        ->  class yes  [0.882]

    Rule 40/9: (6.2, lift 2.0)
        months_loan_duration > 7
        months_loan_duration <= 9
        purpose = furniture/appliances
        employment_duration = < 1 year
        years_at_residence <= 3
        ->  class yes  [0.878]

    Rule 40/10: (30/3.2, lift 1.9)
        checking_balance = unknown
        purpose in {business, car, education}
        percent_of_income > 1
        years_at_residence > 1
        age <= 44
        other_credit in {bank, store}
        ->  class yes  [0.870]

    Rule 40/11: (5, lift 1.9)
        checking_balance = unknown
        credit_history in {good, poor}
        percent_of_income <= 1
        existing_loans_count > 1
        job = skilled
        ->  class yes  [0.858]

    Rule 40/12: (4.4, lift 1.9)
        checking_balance = < 0 DM
        credit_history = good
        years_at_residence > 3
        age <= 21
        ->  class yes  [0.844]

    Rule 40/13: (25.6/3.9, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = very good
        amount > 409
        amount <= 8072
        savings_balance in {< 100 DM, unknown}
        age > 23
        ->  class yes  [0.823]

    Rule 40/14: (16.9/2.7, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 7
        credit_history = good
        purpose = car
        employment_duration = < 1 year
        percent_of_income > 2
        ->  class yes  [0.803]

    Rule 40/15: (36.8/6.9, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 7
        credit_history = good
        amount > 629
        amount <= 1047
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        ->  class yes  [0.797]

    Rule 40/16: (31.3/8.1, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = good
        purpose = furniture/appliances
        savings_balance in {< 100 DM, 100 - 500 DM, unknown}
        employment_duration in {> 7 years, 4 - 7 years, unemployed}
        years_at_residence <= 3
        other_credit = none
        ->  class yes  [0.726]

    Rule 40/17: (95.3/38.2, lift 1.3)
        credit_history in {good, poor}
        existing_loans_count > 1
        job in {management, skilled}
        ->  class yes  [0.597]

    Rule 40/18: (638.2/328.9, lift 1.1)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        ->  class yes  [0.485]

    Rule 40/19: (34.9, lift 1.8)
        checking_balance = unknown
        savings_balance in {< 100 DM, 500 - 1000 DM, unknown}
        percent_of_income > 1
        other_credit = none
        housing in {other, own}
        existing_loans_count > 1
        job = skilled
        ->  class no  [0.973]

    Rule 40/20: (6.1, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = poor
        percent_of_income <= 1
        ->  class no  [0.876]

    Rule 40/21: (10.4/1.2, lift 1.5)
        checking_balance = unknown
        age > 44
        other_credit in {bank, store}
        ->  class no  [0.821]

    Rule 40/22: (22.1/4, lift 1.4)
        credit_history = poor
        savings_balance in {100 - 500 DM, unknown}
        ->  class no  [0.792]

    Rule 40/23: (25.9/6.6, lift 1.3)
        amount <= 629
        ->  class no  [0.728]

    Rule 40/24: (57.3/15.2, lift 1.3)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration <= 42
        credit_history = good
        purpose = furniture/appliances
        years_at_residence > 3
        other_credit = none
        ->  class no  [0.726]

    Rule 40/25: (24.9/6.7, lift 1.3)
        checking_balance = unknown
        percent_of_income <= 1
        ->  class no  [0.715]

    Rule 40/26: (106.5/30.2, lift 1.3)
        checking_balance = unknown
        other_credit = none
        existing_loans_count <= 1
        ->  class no  [0.713]

    Rule 40/27: (830.9/360.5, lift 1.0)
        amount <= 8086
        ->  class no  [0.566]

    Default class: no

    -----  Trial 41:  -----

    Rules:

    Rule 41/1: (8.8, lift 1.8)
        checking_balance = unknown
        purpose in {business, renovations}
        employment_duration = < 1 year
        ->  class yes  [0.908]

    Rule 41/2: (6.8, lift 1.8)
        checking_balance = unknown
        purpose = furniture/appliances
        amount > 4594
        employment_duration = < 1 year
        ->  class yes  [0.886]

    Rule 41/3: (6.4, lift 1.8)
        checking_balance = unknown
        employment_duration = 4 - 7 years
        years_at_residence <= 2
        age <= 23
        ->  class yes  [0.881]

    Rule 41/4: (5.7, lift 1.8)
        checking_balance = unknown
        purpose = furniture/appliances
        employment_duration = < 1 year
        age > 41
        ->  class yes  [0.870]

    Rule 41/5: (18/1.7, lift 1.8)
        purpose = education
        amount > 1670
        employment_duration = 1 - 4 years
        percent_of_income > 1
        ->  class yes  [0.864]

    Rule 41/6: (38.2/9.9, lift 1.5)
        credit_history in {critical, good}
        purpose = car
        amount > 1271
        employment_duration = 1 - 4 years
        percent_of_income > 1
        phone = FALSE
        ->  class yes  [0.729]

    Rule 41/7: (36.7/13.1, lift 1.3)
        employment_duration = unemployed
        percent_of_income > 2
        ->  class yes  [0.635]

    Rule 41/8: (652.1/309.6, lift 1.1)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        ->  class yes  [0.525]

    Rule 41/9: (10.8, lift 1.8)
        months_loan_duration <= 11
        purpose = furniture/appliances
        housing = own
        job = unskilled
        ->  class no  [0.922]

    Rule 41/10: (10.7, lift 1.8)
        checking_balance = unknown
        purpose in {car, education}
        employment_duration = < 1 year
        ->  class no  [0.921]

    Rule 41/11: (9.9, lift 1.8)
        credit_history = critical
        age > 61
        ->  class no  [0.916]

    Rule 41/12: (9.8, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration <= 7
        purpose = furniture/appliances
        housing = own
        ->  class no  [0.915]

    Rule 41/13: (8.1, lift 1.8)
        credit_history = good
        percent_of_income > 1
        percent_of_income <= 2
        housing = other
        ->  class no  [0.901]

    Rule 41/14: (6.9, lift 1.7)
        purpose = furniture/appliances
        age <= 24
        other_credit = none
        housing = own
        existing_loans_count <= 1
        job = unskilled
        ->  class no  [0.888]

    Rule 41/15: (25.4/2.3, lift 1.7)
        checking_balance = unknown
        employment_duration = 4 - 7 years
        age > 23
        ->  class no  [0.881]

    Rule 41/16: (6.4, lift 1.7)
        credit_history = critical
        amount <= 7238
        employment_duration = > 7 years
        other_credit = none
        existing_loans_count <= 1
        ->  class no  [0.881]

    Rule 41/17: (15.1/1.2, lift 1.7)
        checking_balance = unknown
        purpose = furniture/appliances
        amount <= 4594
        employment_duration = < 1 year
        age <= 41
        ->  class no  [0.872]

    Rule 41/18: (5.4, lift 1.7)
        checking_balance = < 0 DM
        months_loan_duration > 40
        purpose = car
        housing = own
        ->  class no  [0.865]

    Rule 41/19: (5.3, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = good
        savings_balance = > 1000 DM
        employment_duration in {< 1 year, > 7 years, 1 - 4 years}
        ->  class no  [0.864]

    Rule 41/20: (5.3, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        savings_balance = 100 - 500 DM
        other_credit = none
        ->  class no  [0.863]

    Rule 41/21: (5, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = poor
        percent_of_income <= 1
        ->  class no  [0.857]

    Rule 41/22: (14.6/1.5, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        amount > 2171
        age <= 49
        other_credit = store
        ->  class no  [0.851]

    Rule 41/23: (16.7/1.8, lift 1.7)
        checking_balance = 1 - 200 DM
        credit_history = good
        other_credit = none
        housing = own
        existing_loans_count > 1
        ->  class no  [0.850]

    Rule 41/24: (10.5/1, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        years_at_residence <= 1
        other_credit = bank
        ->  class no  [0.841]

    Rule 41/25: (9.1/0.9, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = good
        employment_duration in {4 - 7 years, unemployed}
        housing = rent
        ->  class no  [0.828]

    Rule 41/26: (27/4.2, lift 1.6)
        amount <= 1271
        age > 24
        other_credit = none
        housing = own
        job = unskilled
        ->  class no  [0.821]

    Rule 41/27: (22.5/3.7, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        savings_balance in {100 - 500 DM, 500 - 1000 DM, unknown}
        other_credit = bank
        dependents <= 1
        ->  class no  [0.808]

    Rule 41/28: (24/5.5, lift 1.5)
        checking_balance in {> 200 DM, 1 - 200 DM}
        credit_history = good
        purpose = car
        other_credit = none
        housing = own
        ->  class no  [0.749]

    Rule 41/29: (29.2/7.1, lift 1.5)
        checking_balance = unknown
        purpose in {furniture/appliances, renovations}
        employment_duration = 1 - 4 years
        ->  class no  [0.742]

    Rule 41/30: (51.1/16.6, lift 1.3)
        checking_balance = unknown
        employment_duration = > 7 years
        ->  class no  [0.669]

    Rule 41/31: (641.9/306, lift 1.0)
        housing = own
        ->  class no  [0.523]

    Default class: no

    -----  Trial 42:  -----

    Rules:

    Rule 42/1: (9.2, lift 2.1)
        checking_balance = < 0 DM
        purpose = education
        savings_balance = < 100 DM
        age <= 43
        ->  class yes  [0.911]

    Rule 42/2: (7.3, lift 2.0)
        months_loan_duration <= 36
        savings_balance in {< 100 DM, 100 - 500 DM}
        employment_duration = 4 - 7 years
        age <= 22
        ->  class yes  [0.893]

    Rule 42/3: (6.7, lift 2.0)
        purpose = furniture/appliances
        employment_duration = unemployed
        years_at_residence <= 2
        ->  class yes  [0.885]

    Rule 42/4: (10.3/0.5, lift 2.0)
        months_loan_duration <= 36
        credit_history = critical
        purpose = furniture/appliances
        savings_balance in {< 100 DM, 500 - 1000 DM, unknown}
        employment_duration = 1 - 4 years
        years_at_residence > 3
        ->  class yes  [0.878]

    Rule 42/5: (9.7/0.5, lift 2.0)
        purpose = furniture/appliances
        amount <= 1597
        savings_balance = unknown
        employment_duration = 1 - 4 years
        ->  class yes  [0.876]

    Rule 42/6: (16/1.4, lift 2.0)
        purpose = car
        savings_balance = < 100 DM
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, 4 - 7 years}
        age > 29
        other_credit = bank
        ->  class yes  [0.868]

    Rule 42/7: (12.7/1.2, lift 1.9)
        purpose = education
        savings_balance = < 100 DM
        age > 32
        age <= 43
        ->  class yes  [0.851]

    Rule 42/8: (22.7/3.1, lift 1.9)
        months_loan_duration > 16
        purpose = business
        savings_balance = < 100 DM
        years_at_residence > 1
        other_credit in {bank, none}
        ->  class yes  [0.834]

    Rule 42/9: (22.9/3.2, lift 1.9)
        purpose = furniture/appliances
        employment_duration = < 1 year
        age > 30
        other_credit in {bank, none}
        housing = own
        ->  class yes  [0.832]

    Rule 42/10: (17.5/2.3, lift 1.9)
        months_loan_duration > 15
        months_loan_duration <= 21
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        housing = own
        phone = FALSE
        ->  class yes  [0.832]

    Rule 42/11: (6.2/0.4, lift 1.9)
        percent_of_income <= 1
        age > 29
        other_credit = none
        housing = other
        ->  class yes  [0.824]

    Rule 42/12: (18/2.5, lift 1.9)
        checking_balance in {< 0 DM, unknown}
        months_loan_duration <= 36
        purpose = furniture/appliances
        employment_duration = > 7 years
        percent_of_income > 1
        years_at_residence <= 3
        age > 32
        other_credit in {bank, none}
        ->  class yes  [0.824]

    Rule 42/13: (19.5/2.8, lift 1.9)
        purpose = car
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM}
        age <= 29
        other_credit in {bank, none}
        existing_loans_count > 1
        ->  class yes  [0.823]

    Rule 42/14: (8.9/1.1, lift 1.8)
        purpose = car
        age > 42
        housing = rent
        ->  class yes  [0.811]

    Rule 42/15: (22.4/3.7, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 36
        purpose = furniture/appliances
        ->  class yes  [0.806]

    Rule 42/16: (19.7/3.6, lift 1.8)
        checking_balance in {> 200 DM, 1 - 200 DM}
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        employment_duration = < 1 year
        housing = rent
        ->  class yes  [0.787]

    Rule 42/17: (14.2/2.7, lift 1.8)
        purpose = furniture/appliances
        savings_balance = 500 - 1000 DM
        employment_duration = 1 - 4 years
        phone = FALSE
        ->  class yes  [0.770]

    Rule 42/18: (15.8/3.1, lift 1.7)
        checking_balance = < 0 DM
        purpose = car
        years_at_residence <= 2
        age <= 29
        ->  class yes  [0.768]

    Rule 42/19: (29.3/6.4, lift 1.7)
        checking_balance = 1 - 200 DM
        purpose = car
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        age <= 29
        ->  class yes  [0.762]

    Rule 42/20: (20/4.6, lift 1.7)
        credit_history = good
        purpose = furniture/appliances
        employment_duration = 1 - 4 years
        housing in {other, rent}
        phone = FALSE
        ->  class yes  [0.744]

    Rule 42/21: (24.4/6.1, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 22
        purpose = furniture/appliances
        employment_duration = 4 - 7 years
        ->  class yes  [0.731]

    Rule 42/22: (12.1/3, lift 1.6)
        purpose = education
        savings_balance in {> 1000 DM, 100 - 500 DM}
        ->  class yes  [0.715]

    Rule 42/23: (49.8/17.7, lift 1.5)
        purpose = car
        amount <= 2622
        age <= 29
        ->  class yes  [0.639]

    Rule 42/24: (49.1/18.9, lift 1.4)
        other_credit = store
        ->  class yes  [0.611]

    Rule 42/25: (44.2/18.8, lift 1.3)
        purpose = car
        other_credit = bank
        ->  class yes  [0.570]

    Rule 42/26: (9.2, lift 1.6)
        checking_balance = unknown
        amount > 2622
        age <= 29
        existing_loans_count <= 1
        ->  class no  [0.911]

    Rule 42/27: (6.5, lift 1.6)
        purpose = car
        savings_balance = > 1000 DM
        ->  class no  [0.882]

    Rule 42/28: (5.1, lift 1.5)
        savings_balance = 500 - 1000 DM
        employment_duration = 1 - 4 years
        phone = TRUE
        ->  class no  [0.859]

    Rule 42/29: (69.7/19.3, lift 1.3)
        months_loan_duration <= 15
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        housing = own
        ->  class no  [0.717]

    Rule 42/30: (37.2/10.3, lift 1.3)
        employment_duration = unemployed
        years_at_residence > 2
        ->  class no  [0.712]

    Rule 42/31: (850.9/365.2, lift 1.0)
        other_credit in {bank, none}
        ->  class no  [0.571]

    Default class: no

    -----  Trial 43:  -----

    Rules:

    Rule 43/1: (4.1, lift 1.9)
        months_loan_duration <= 7
        credit_history = good
        amount > 4057
        ->  class yes  [0.837]

    Rule 43/2: (18.1/3, lift 1.8)
        credit_history = poor
        employment_duration in {< 1 year, unemployed}
        ->  class yes  [0.799]

    Rule 43/3: (850/455.3, lift 1.0)
        months_loan_duration > 7
        ->  class yes  [0.464]

    Rule 43/4: (20.2, lift 1.7)
        checking_balance = unknown
        credit_history = good
        purpose in {car, furniture/appliances}
        existing_loans_count <= 1
        phone = TRUE
        ->  class no  [0.955]

    Rule 43/5: (18.9, lift 1.7)
        checking_balance = unknown
        months_loan_duration <= 24
        credit_history = good
        purpose in {car, furniture/appliances}
        employment_duration in {< 1 year, > 7 years, 1 - 4 years}
        percent_of_income > 2
        years_at_residence > 1
        years_at_residence <= 3
        existing_loans_count <= 1
        ->  class no  [0.952]

    Rule 43/6: (19.9/0.3, lift 1.7)
        checking_balance = > 200 DM
        months_loan_duration <= 24
        purpose in {car, education, furniture/appliances}
        percent_of_income > 2
        other_credit in {bank, none}
        existing_loans_count <= 1
        ->  class no  [0.941]

    Rule 43/7: (25.5/1.4, lift 1.7)
        months_loan_duration <= 7
        credit_history = good
        amount <= 4057
        ->  class no  [0.912]

    Rule 43/8: (9.3, lift 1.7)
        months_loan_duration <= 24
        credit_history = good
        purpose in {business, car0}
        percent_of_income > 2
        existing_loans_count <= 1
        ->  class no  [0.911]

    Rule 43/9: (9.1, lift 1.7)
        credit_history = poor
        employment_duration = 1 - 4 years
        percent_of_income <= 2
        ->  class no  [0.910]

    Rule 43/10: (6.8, lift 1.6)
        amount <= 1549
        percent_of_income <= 2
        age > 44
        housing = own
        ->  class no  [0.886]

    Rule 43/11: (11.5/0.7, lift 1.6)
        months_loan_duration <= 24
        purpose in {car, furniture/appliances}
        employment_duration = < 1 year
        existing_loans_count <= 1
        job in {management, unemployed}
        ->  class no  [0.872]

    Rule 43/12: (5.7, lift 1.6)
        credit_history = poor
        employment_duration = 1 - 4 years
        job = unskilled
        ->  class no  [0.870]

    Rule 43/13: (6.8/0.2, lift 1.6)
        months_loan_duration > 39
        credit_history = very good
        ->  class no  [0.868]

    Rule 43/14: (41.9/5, lift 1.6)
        months_loan_duration > 10
        credit_history = good
        percent_of_income > 2
        years_at_residence <= 1
        job in {management, skilled}
        ->  class no  [0.864]

    Rule 43/15: (5.3, lift 1.6)
        credit_history = perfect
        savings_balance in {500 - 1000 DM, unknown}
        ->  class no  [0.863]

    Rule 43/16: (11.7/0.9, lift 1.6)
        credit_history = poor
        employment_duration = 4 - 7 years
        years_at_residence > 1
        ->  class no  [0.863]

    Rule 43/17: (4.4, lift 1.5)
        credit_history = critical
        other_credit = bank
        job in {management, unemployed}
        ->  class no  [0.843]

    Rule 43/18: (37.4/7, lift 1.5)
        credit_history = critical
        amount <= 7763
        savings_balance in {> 1000 DM, 100 - 500 DM, unknown}
        other_credit in {none, store}
        ->  class no  [0.797]

    Rule 43/19: (36.2/7.2, lift 1.4)
        credit_history = good
        percent_of_income <= 2
        housing = rent
        existing_loans_count <= 1
        ->  class no  [0.785]

    Rule 43/20: (50.5/11.5, lift 1.4)
        months_loan_duration <= 24
        credit_history = good
        employment_duration in {> 7 years, 4 - 7 years}
        percent_of_income > 2
        years_at_residence > 1
        other_credit = none
        ->  class no  [0.762]

    Rule 43/21: (108.1/28, lift 1.3)
        months_loan_duration > 7
        credit_history = good
        amount > 1549
        percent_of_income <= 2
        existing_loans_count <= 1
        ->  class no  [0.736]

    Rule 43/22: (111/32.1, lift 1.3)
        months_loan_duration <= 42
        credit_history = critical
        amount <= 7763
        savings_balance = < 100 DM
        other_credit in {none, store}
        ->  class no  [0.707]

    Default class: no

    -----  Trial 44:  -----

    Rules:

    Rule 44/1: (13.4, lift 2.1)
        amount > 7596
        savings_balance = < 100 DM
        percent_of_income > 2
        ->  class yes  [0.932]

    Rule 44/2: (12.5, lift 2.1)
        credit_history = good
        purpose = furniture/appliances
        percent_of_income <= 3
        existing_loans_count > 1
        job = skilled
        ->  class yes  [0.931]

    Rule 44/3: (11.3, lift 2.1)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 42
        amount > 7511
        other_credit = none
        job = management
        ->  class yes  [0.925]

    Rule 44/4: (13.3/0.5, lift 2.1)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        savings_balance in {< 100 DM, unknown}
        employment_duration = 1 - 4 years
        age <= 24
        job = skilled
        ->  class yes  [0.901]

    Rule 44/5: (8.1, lift 2.1)
        checking_balance = < 0 DM
        credit_history = poor
        percent_of_income > 1
        job = skilled
        ->  class yes  [0.901]

    Rule 44/6: (17.7/1, lift 2.1)
        checking_balance = 1 - 200 DM
        months_loan_duration > 42
        savings_balance in {< 100 DM, > 1000 DM, 100 - 500 DM}
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, 4 - 7 years}
        ->  class yes  [0.900]

    Rule 44/7: (20.4/2.3, lift 1.9)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 42
        savings_balance in {< 100 DM, 100 - 500 DM}
        employment_duration in {> 7 years, unemployed}
        other_credit = none
        job = management
        ->  class yes  [0.855]

    Rule 44/8: (4.6, lift 1.9)
        checking_balance = < 0 DM
        percent_of_income <= 1
        job = management
        ->  class yes  [0.849]

    Rule 44/9: (4.5, lift 1.9)
        checking_balance = < 0 DM
        employment_duration = 1 - 4 years
        housing = other
        ->  class yes  [0.847]

    Rule 44/10: (4.5, lift 1.9)
        checking_balance = < 0 DM
        employment_duration = < 1 year
        percent_of_income > 1
        job = skilled
        phone = TRUE
        ->  class yes  [0.847]

    Rule 44/11: (4.2, lift 1.9)
        checking_balance = > 200 DM
        credit_history in {critical, very good}
        employment_duration = < 1 year
        ->  class yes  [0.838]

    Rule 44/12: (12.6/1.7, lift 1.9)
        checking_balance = 1 - 200 DM
        months_loan_duration > 18
        savings_balance in {< 100 DM, 100 - 500 DM}
        job = unskilled
        dependents <= 1
        ->  class yes  [0.815]

    Rule 44/13: (3.1, lift 1.8)
        checking_balance = < 0 DM
        savings_balance = 100 - 500 DM
        job = unskilled
        ->  class yes  [0.804]

    Rule 44/14: (5.8/0.7, lift 1.8)
        checking_balance = < 0 DM
        savings_balance = unknown
        job = management
        ->  class yes  [0.780]

    Rule 44/15: (18.2/3.5, lift 1.8)
        checking_balance = < 0 DM
        employment_duration = 4 - 7 years
        percent_of_income > 1
        other_credit = none
        job = skilled
        dependents <= 1
        ->  class yes  [0.776]

    Rule 44/16: (18.7/4.3, lift 1.7)
        checking_balance = 1 - 200 DM
        savings_balance in {< 100 DM, > 1000 DM, 100 - 500 DM}
        other_credit in {bank, none}
        dependents > 1
        ->  class yes  [0.746]

    Rule 44/17: (8.2/1.8, lift 1.6)
        housing = other
        job = unskilled
        ->  class yes  [0.723]

    Rule 44/18: (45/12.1, lift 1.6)
        checking_balance = < 0 DM
        savings_balance in {< 100 DM, 100 - 500 DM, unknown}
        employment_duration in {> 7 years, unemployed}
        percent_of_income > 1
        job = skilled
        ->  class yes  [0.721]

    Rule 44/19: (20.9/5.6, lift 1.6)
        checking_balance = unknown
        percent_of_income > 1
        age <= 23
        job = skilled
        ->  class yes  [0.712]

    Rule 44/20: (47.7/14.1, lift 1.6)
        checking_balance = < 0 DM
        purpose = car
        percent_of_income > 2
        job = skilled
        ->  class yes  [0.697]

    Rule 44/21: (23.2/7.5, lift 1.5)
        checking_balance = unknown
        purpose = business
        employment_duration in {< 1 year, 1 - 4 years}
        ->  class yes  [0.661]

    Rule 44/22: (300.8/145.2, lift 1.2)
        checking_balance = < 0 DM
        ->  class yes  [0.517]

    Rule 44/23: (9.5, lift 1.6)
        checking_balance = unknown
        percent_of_income <= 3
        other_credit = none
        job = unskilled
        ->  class no  [0.913]

    Rule 44/24: (20.5/3.5, lift 1.4)
        checking_balance = unknown
        credit_history = critical
        housing = own
        phone = TRUE
        ->  class no  [0.798]

    Rule 44/25: (60.1/15, lift 1.3)
        checking_balance = 1 - 200 DM
        savings_balance in {500 - 1000 DM, unknown}
        ->  class no  [0.742]

    Rule 44/26: (129/47.9, lift 1.1)
        employment_duration = 4 - 7 years
        ->  class no  [0.627]

    Rule 44/27: (775.2/321.4, lift 1.0)
        months_loan_duration <= 42
        amount <= 7511
        ->  class no  [0.585]

    Default class: no

    -----  Trial 45:  -----

    Rules:

    Rule 45/1: (11, lift 1.9)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = perfect
        percent_of_income > 3
        ->  class yes  [0.923]

    Rule 45/2: (15.5/2, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = poor
        savings_balance = < 100 DM
        percent_of_income > 2
        ->  class yes  [0.826]

    Rule 45/3: (3.4, lift 1.7)
        savings_balance = 100 - 500 DM
        existing_loans_count > 3
        ->  class yes  [0.814]

    Rule 45/4: (24/3.9, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history in {critical, good, very good}
        purpose in {car, furniture/appliances}
        savings_balance = unknown
        employment_duration = 1 - 4 years
        ->  class yes  [0.810]

    Rule 45/5: (32.6/5.8, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose in {business, car, education, furniture/appliances}
        savings_balance = 100 - 500 DM
        employment_duration in {< 1 year, 1 - 4 years, unemployed}
        ->  class yes  [0.803]

    Rule 45/6: (19.3/3.2, lift 1.7)
        checking_balance = unknown
        amount > 4530
        employment_duration in {> 7 years, 1 - 4 years, unemployed}
        job = management
        ->  class yes  [0.801]

    Rule 45/7: (12.7/2.1, lift 1.6)
        checking_balance = unknown
        employment_duration in {> 7 years, 1 - 4 years, unemployed}
        existing_loans_count > 1
        job = management
        ->  class yes  [0.791]

    Rule 45/8: (36.6/8.1, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        purpose = car
        savings_balance = < 100 DM
        years_at_residence <= 2
        ->  class yes  [0.765]

    Rule 45/9: (41.2/9.4, lift 1.6)
        amount > 6615
        employment_duration in {< 1 year, > 7 years, 1 - 4 years}
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.759]

    Rule 45/10: (19.3/4.5, lift 1.5)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = very good
        savings_balance = < 100 DM
        percent_of_income > 1
        ->  class yes  [0.743]

    Rule 45/11: (55.5/16.3, lift 1.4)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = critical
        amount > 1922
        savings_balance = < 100 DM
        job in {management, skilled, unskilled}
        dependents <= 1
        ->  class yes  [0.700]

    Rule 45/12: (189.3/83.5, lift 1.2)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        employment_duration in {< 1 year, unemployed}
        ->  class yes  [0.558]

    Rule 45/13: (569.6/280, lift 1.0)
        phone = FALSE
        ->  class yes  [0.508]

    Rule 45/14: (19.1, lift 1.9)
        checking_balance = unknown
        purpose = furniture/appliances
        employment_duration = 1 - 4 years
        job = skilled
        ->  class no  [0.953]

    Rule 45/15: (14.7, lift 1.8)
        credit_history = critical
        amount <= 1922
        savings_balance = < 100 DM
        age > 36
        job in {management, skilled, unskilled}
        ->  class no  [0.940]

    Rule 45/16: (12.7, lift 1.8)
        checking_balance = unknown
        purpose = furniture/appliances
        age > 32
        job = skilled
        phone = FALSE
        ->  class no  [0.932]

    Rule 45/17: (11.3, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {perfect, poor}
        savings_balance = unknown
        ->  class no  [0.925]

    Rule 45/18: (9.9, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose in {business, education}
        savings_balance = unknown
        employment_duration = 1 - 4 years
        ->  class no  [0.916]

    Rule 45/19: (9.7, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = good
        purpose = furniture/appliances
        amount <= 1901
        savings_balance = < 100 DM
        other_credit = none
        job = unskilled
        ->  class no  [0.914]

    Rule 45/20: (9.6, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        amount <= 1922
        savings_balance = < 100 DM
        age <= 32
        ->  class no  [0.914]

    Rule 45/21: (8.8, lift 1.8)
        checking_balance = unknown
        purpose = car
        percent_of_income > 3
        job = skilled
        phone = FALSE
        ->  class no  [0.907]

    Rule 45/22: (11.2/0.7, lift 1.7)
        credit_history = critical
        savings_balance = < 100 DM
        job in {management, skilled, unskilled}
        dependents > 1
        ->  class no  [0.873]

    Rule 45/23: (5.6, lift 1.7)
        checking_balance = unknown
        amount <= 4530
        existing_loans_count <= 1
        job = management
        ->  class no  [0.868]

    Rule 45/24: (13.9/1.1, lift 1.7)
        checking_balance = unknown
        existing_loans_count > 1
        job = unskilled
        ->  class no  [0.866]

    Rule 45/25: (5.1, lift 1.7)
        credit_history = very good
        savings_balance = < 100 DM
        percent_of_income <= 1
        job in {management, skilled}
        ->  class no  [0.859]

    Rule 45/26: (24.4/2.7, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        savings_balance = 100 - 500 DM
        employment_duration in {> 7 years, 4 - 7 years}
        existing_loans_count <= 3
        job in {management, skilled}
        ->  class no  [0.858]

    Rule 45/27: (4.7, lift 1.7)
        purpose in {car0, renovations}
        savings_balance = 100 - 500 DM
        ->  class no  [0.851]

    Rule 45/28: (10.4/0.9, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose in {business, car, furniture/appliances}
        savings_balance = > 1000 DM
        ->  class no  [0.849]

    Rule 45/29: (14.4/1.5, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        savings_balance = 500 - 1000 DM
        age <= 61
        existing_loans_count <= 1
        ->  class no  [0.844]

    Rule 45/30: (16.8/2, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration <= 33
        amount <= 5711
        savings_balance = unknown
        employment_duration = > 7 years
        ->  class no  [0.842]

    Rule 45/31: (44.4/7.4, lift 1.6)
        checking_balance = unknown
        amount <= 6615
        job = skilled
        phone = TRUE
        ->  class no  [0.818]

    Rule 45/32: (14.4/2.8, lift 1.5)
        checking_balance = unknown
        savings_balance in {> 1000 DM, 100 - 500 DM}
        existing_loans_count <= 1
        ->  class no  [0.771]

    Rule 45/33: (29.1/6.4, lift 1.5)
        credit_history = poor
        percent_of_income <= 2
        ->  class no  [0.762]

    Rule 45/34: (43.1/10.2, lift 1.5)
        purpose = furniture/appliances
        employment_duration = > 7 years
        other_credit = none
        job = skilled
        ->  class no  [0.752]

    Rule 45/35: (17.8/4.7, lift 1.4)
        savings_balance = unknown
        employment_duration = < 1 year
        ->  class no  [0.713]

    Rule 45/36: (128.3/52, lift 1.2)
        employment_duration = 4 - 7 years
        ->  class no  [0.594]

    Default class: no

    -----  Trial 46:  -----

    Rules:

    Rule 46/1: (11.7, lift 2.1)
        checking_balance = < 0 DM
        months_loan_duration > 22
        amount <= 7418
        savings_balance = unknown
        housing = own
        ->  class yes  [0.927]

    Rule 46/2: (12.1, lift 2.1)
        checking_balance = 1 - 200 DM
        months_loan_duration > 8
        credit_history = good
        years_at_residence <= 3
        age > 35
        existing_loans_count <= 1
        dependents <= 1
        ->  class yes  [0.923]

    Rule 46/3: (8.4, lift 2.0)
        checking_balance = unknown
        months_loan_duration > 8
        amount > 1300
        years_at_residence <= 2
        age <= 23
        ->  class yes  [0.904]

    Rule 46/4: (8, lift 2.0)
        checking_balance = 1 - 200 DM
        months_loan_duration > 42
        savings_balance = < 100 DM
        ->  class yes  [0.900]

    Rule 46/5: (10.5/0.3, lift 2.0)
        months_loan_duration > 8
        age > 50
        dependents > 1
        ->  class yes  [0.893]

    Rule 46/6: (7.2, lift 2.0)
        checking_balance in {< 0 DM, > 200 DM}
        employment_duration = 1 - 4 years
        housing = other
        ->  class yes  [0.892]

    Rule 46/7: (7.2, lift 2.0)
        checking_balance = < 0 DM
        months_loan_duration <= 40
        purpose = car
        years_at_residence <= 1
        housing = own
        ->  class yes  [0.891]

    Rule 46/8: (6.8, lift 2.0)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        years_at_residence > 1
        years_at_residence <= 3
        age > 32
        job = skilled
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.886]

    Rule 46/9: (5.4, lift 2.0)
        checking_balance = < 0 DM
        credit_history = very good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        housing = own
        ->  class yes  [0.865]

    Rule 46/10: (14.4/1.3, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration > 8
        months_loan_duration <= 40
        purpose = car
        savings_balance = < 100 DM
        percent_of_income <= 3
        housing = own
        dependents <= 1
        ->  class yes  [0.858]

    Rule 46/11: (10/0.8, lift 1.9)
        amount > 12204
        housing = other
        ->  class yes  [0.853]

    Rule 46/12: (10.7/0.9, lift 1.9)
        checking_balance = > 200 DM
        months_loan_duration > 8
        dependents > 1
        ->  class yes  [0.851]

    Rule 46/13: (4.7, lift 1.9)
        other_credit = store
        housing = other
        ->  class yes  [0.850]

    Rule 46/14: (38.4/5.2, lift 1.9)
        months_loan_duration > 8
        amount <= 4351
        percent_of_income > 3
        dependents > 1
        phone = FALSE
        ->  class yes  [0.846]

    Rule 46/15: (4.2, lift 1.9)
        checking_balance = 1 - 200 DM
        savings_balance = 100 - 500 DM
        housing = own
        job = management
        ->  class yes  [0.839]

    Rule 46/16: (9.1/0.9, lift 1.9)
        credit_history = good
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years}
        housing = rent
        existing_loans_count > 1
        dependents <= 1
        ->  class yes  [0.830]

    Rule 46/17: (14.2/1.9, lift 1.9)
        checking_balance = 1 - 200 DM
        credit_history = critical
        savings_balance = < 100 DM
        percent_of_income <= 2
        years_at_residence > 1
        housing = own
        ->  class yes  [0.823]

    Rule 46/18: (13.8/2, lift 1.8)
        amount > 5848
        employment_duration = > 7 years
        housing = other
        dependents <= 1
        ->  class yes  [0.810]

    Rule 46/19: (19.6/3.4, lift 1.8)
        checking_balance = 1 - 200 DM
        credit_history = critical
        savings_balance = < 100 DM
        years_at_residence > 1
        age <= 33
        housing = own
        ->  class yes  [0.794]

    Rule 46/20: (11.6/1.9, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 8
        credit_history in {perfect, very good}
        housing = rent
        dependents <= 1
        ->  class yes  [0.789]

    Rule 46/21: (20.1/3.7, lift 1.8)
        months_loan_duration > 8
        purpose = car
        years_at_residence <= 2
        other_credit = bank
        ->  class yes  [0.786]

    Rule 46/22: (14.3/3.2, lift 1.7)
        checking_balance = < 0 DM
        purpose in {education, renovations}
        housing = own
        dependents <= 1
        ->  class yes  [0.742]

    Rule 46/23: (30.4/7.7, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM, unknown}
        months_loan_duration > 8
        credit_history = good
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years, unemployed}
        percent_of_income > 2
        years_at_residence > 2
        housing = rent
        ->  class yes  [0.731]

    Rule 46/24: (7.8/1.8, lift 1.6)
        checking_balance = < 0 DM
        job = unemployed
        ->  class yes  [0.715]

    Rule 46/25: (35/10.8, lift 1.5)
        checking_balance = < 0 DM
        savings_balance = < 100 DM
        job = skilled
        phone = TRUE
        ->  class yes  [0.681]

    Rule 46/26: (40.5/12.9, lift 1.5)
        months_loan_duration > 8
        years_at_residence <= 2
        housing = rent
        dependents <= 1
        ->  class yes  [0.672]

    Rule 46/27: (58.2/19.6, lift 1.5)
        checking_balance = 1 - 200 DM
        months_loan_duration > 8
        savings_balance = < 100 DM
        years_at_residence <= 3
        existing_loans_count <= 1
        job in {management, skilled}
        ->  class yes  [0.657]

    Rule 46/28: (17.2, lift 1.7)
        amount <= 3711
        dependents > 1
        phone = TRUE
        ->  class no  [0.948]

    Rule 46/29: (13.4, lift 1.7)
        checking_balance = 1 - 200 DM
        months_loan_duration > 8
        months_loan_duration <= 42
        amount > 2528
        years_at_residence <= 3
        age <= 35
        housing = own
        existing_loans_count <= 1
        dependents <= 1
        ->  class no  [0.935]

    Rule 46/30: (9.3, lift 1.6)
        checking_balance = < 0 DM
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        years_at_residence <= 1
        job = skilled
        phone = FALSE
        ->  class no  [0.912]

    Rule 46/31: (60.8/16.1, lift 1.3)
        months_loan_duration <= 8
        ->  class no  [0.728]

    Rule 46/32: (839.2/382.5, lift 1.0)
        months_loan_duration > 8
        ->  class no  [0.544]

    Default class: no

    -----  Trial 47:  -----

    Rules:

    Rule 47/1: (16, lift 2.0)
        credit_history = good
        purpose = furniture/appliances
        percent_of_income <= 3
        existing_loans_count > 1
        job = skilled
        ->  class yes  [0.944]

    Rule 47/2: (15.3, lift 2.0)
        checking_balance = < 0 DM
        credit_history = good
        purpose = car
        employment_duration in {< 1 year, 1 - 4 years}
        percent_of_income > 1
        years_at_residence <= 2
        ->  class yes  [0.942]

    Rule 47/3: (27/1.5, lift 1.9)
        credit_history = poor
        savings_balance in {< 100 DM, > 1000 DM}
        percent_of_income > 2
        housing in {other, own}
        job in {management, skilled}
        ->  class yes  [0.915]

    Rule 47/4: (10.8/0.2, lift 1.9)
        credit_history = perfect
        housing in {other, rent}
        ->  class yes  [0.904]

    Rule 47/5: (8.3, lift 1.9)
        checking_balance = < 0 DM
        credit_history = good
        purpose = car
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        age > 44
        other_credit = none
        ->  class yes  [0.903]

    Rule 47/6: (14.2/0.7, lift 1.9)
        credit_history = good
        purpose = furniture/appliances
        amount <= 2325
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years, unemployed}
        years_at_residence > 1
        age <= 22
        job = skilled
        ->  class yes  [0.895]

    Rule 47/7: (20.8/2, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        ->  class yes  [0.871]

    Rule 47/8: (7.2/0.2, lift 1.8)
        checking_balance = 1 - 200 DM
        credit_history = good
        purpose = car
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years}
        years_at_residence > 3
        other_credit = none
        ->  class yes  [0.865]

    Rule 47/9: (12.8/1, lift 1.8)
        checking_balance = 1 - 200 DM
        credit_history = critical
        years_at_residence > 1
        years_at_residence <= 2
        phone = FALSE
        ->  class yes  [0.863]

    Rule 47/10: (20.6/3.2, lift 1.7)
        credit_history = critical
        years_at_residence > 1
        age <= 46
        other_credit = bank
        job in {skilled, unskilled}
        ->  class yes  [0.814]

    Rule 47/11: (13.7/2.3, lift 1.7)
        credit_history = poor
        employment_duration = < 1 year
        ->  class yes  [0.791]

    Rule 47/12: (10.9/1.8, lift 1.6)
        credit_history = good
        housing in {other, rent}
        existing_loans_count > 1
        job = skilled
        ->  class yes  [0.779]

    Rule 47/13: (20.7/4.6, lift 1.6)
        credit_history = perfect
        percent_of_income > 2
        ->  class yes  [0.754]

    Rule 47/14: (43.4/14.3, lift 1.4)
        months_loan_duration <= 33
        credit_history = very good
        age > 23
        ->  class yes  [0.663]

    Rule 47/15: (769.1/393.4, lift 1.0)
        years_at_residence > 1
        ->  class yes  [0.489]

    Rule 47/16: (25.7, lift 1.8)
        checking_balance = unknown
        credit_history = critical
        age > 30
        other_credit = none
        ->  class no  [0.964]

    Rule 47/17: (12.5, lift 1.8)
        credit_history = critical
        age > 60
        ->  class no  [0.931]

    Rule 47/18: (12.1, lift 1.8)
        months_loan_duration <= 18
        credit_history = good
        purpose = business
        ->  class no  [0.929]

    Rule 47/19: (10.6, lift 1.8)
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = > 7 years
        other_credit = none
        existing_loans_count <= 1
        job in {skilled, unskilled}
        dependents <= 1
        ->  class no  [0.921]

    Rule 47/20: (9.8, lift 1.7)
        credit_history = good
        purpose in {business, car, education, renovations}
        housing = own
        existing_loans_count > 1
        job = skilled
        ->  class no  [0.915]

    Rule 47/21: (22.8/1.5, lift 1.7)
        credit_history = poor
        savings_balance in {100 - 500 DM, 500 - 1000 DM, unknown}
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years, unemployed}
        ->  class no  [0.901]

    Rule 47/22: (8, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        existing_loans_count <= 1
        ->  class no  [0.900]

    Rule 47/23: (7.4, lift 1.7)
        credit_history = poor
        savings_balance = < 100 DM
        percent_of_income <= 2
        age > 28
        ->  class no  [0.894]

    Rule 47/24: (6.2, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        credit_history = good
        purpose = furniture/appliances
        savings_balance = unknown
        existing_loans_count <= 1
        ->  class no  [0.878]

    Rule 47/25: (5.6, lift 1.7)
        credit_history = critical
        age > 46
        other_credit = bank
        ->  class no  [0.869]

    Rule 47/26: (5.6, lift 1.7)
        credit_history = very good
        age <= 23
        ->  class no  [0.869]

    Rule 47/27: (4.6, lift 1.6)
        credit_history = critical
        other_credit = bank
        job = management
        ->  class no  [0.848]

    Rule 47/28: (14.1/1.5, lift 1.6)
        credit_history = critical
        years_at_residence <= 1
        ->  class no  [0.847]

    Rule 47/29: (9.1/1.3, lift 1.5)
        months_loan_duration > 33
        credit_history = very good
        ->  class no  [0.795]

    Rule 47/30: (31.1/6.4, lift 1.5)
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        years_at_residence <= 1
        existing_loans_count <= 1
        dependents <= 1
        ->  class no  [0.775]

    Rule 47/31: (18.4/3.7, lift 1.5)
        purpose = furniture/appliances
        savings_balance in {> 1000 DM, 500 - 1000 DM}
        existing_loans_count <= 1
        ->  class no  [0.768]

    Rule 47/32: (143.1/52.4, lift 1.2)
        purpose = furniture/appliances
        amount <= 2325
        years_at_residence > 1
        other_credit = none
        ->  class no  [0.632]

    Rule 47/33: (239.3/91.9, lift 1.2)
        purpose = car
        amount > 1103
        amount <= 11328
        ->  class no  [0.615]

    Default class: yes

    -----  Trial 48:  -----

    Rules:

    Rule 48/1: (14.8, lift 2.2)
        checking_balance = < 0 DM
        credit_history = good
        percent_of_income > 1
        years_at_residence > 1
        age <= 23
        other_credit = none
        job = skilled
        ->  class yes  [0.941]

    Rule 48/2: (13.5, lift 2.2)
        checking_balance = < 0 DM
        months_loan_duration > 16
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        percent_of_income > 1
        years_at_residence <= 3
        other_credit = none
        job = skilled
        ->  class yes  [0.936]

    Rule 48/3: (12.3, lift 2.1)
        checking_balance = < 0 DM
        savings_balance = < 100 DM
        percent_of_income > 1
        years_at_residence > 1
        other_credit = bank
        job = skilled
        ->  class yes  [0.930]

    Rule 48/4: (11.3, lift 2.1)
        checking_balance = < 0 DM
        credit_history in {poor, very good}
        savings_balance = < 100 DM
        percent_of_income > 1
        job = skilled
        ->  class yes  [0.925]

    Rule 48/5: (7.2, lift 2.1)
        checking_balance = 1 - 200 DM
        months_loan_duration > 27
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        ->  class yes  [0.892]

    Rule 48/6: (6.8, lift 2.0)
        checking_balance = 1 - 200 DM
        purpose = car
        savings_balance = < 100 DM
        age <= 25
        phone = FALSE
        ->  class yes  [0.886]

    Rule 48/7: (6.6, lift 2.0)
        checking_balance = 1 - 200 DM
        purpose = business
        savings_balance = < 100 DM
        years_at_residence > 3
        ->  class yes  [0.884]

    Rule 48/8: (7, lift 2.0)
        purpose = car
        savings_balance = 100 - 500 DM
        job = unskilled
        ->  class yes  [0.880]

    Rule 48/9: (5.8, lift 2.0)
        checking_balance = > 200 DM
        percent_of_income > 1
        dependents > 1
        ->  class yes  [0.872]

    Rule 48/10: (5.7, lift 2.0)
        purpose in {education, renovations}
        savings_balance = > 1000 DM
        ->  class yes  [0.870]

    Rule 48/11: (4, lift 1.9)
        checking_balance = < 0 DM
        percent_of_income <= 1
        job = management
        ->  class yes  [0.832]

    Rule 48/12: (9.3/1, lift 1.9)
        months_loan_duration > 11
        savings_balance = 500 - 1000 DM
        job = unskilled
        ->  class yes  [0.823]

    Rule 48/13: (3.5, lift 1.9)
        savings_balance = 500 - 1000 DM
        existing_loans_count > 2
        job = skilled
        ->  class yes  [0.820]

    Rule 48/14: (19/2.8, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration > 11
        purpose = car
        amount > 2212
        savings_balance = < 100 DM
        percent_of_income > 1
        age <= 48
        job = skilled
        ->  class yes  [0.817]

    Rule 48/15: (14.9/2.4, lift 1.8)
        checking_balance = unknown
        savings_balance = < 100 DM
        percent_of_income > 1
        age <= 23
        ->  class yes  [0.800]

    Rule 48/16: (2.8, lift 1.8)
        savings_balance = 100 - 500 DM
        existing_loans_count > 3
        ->  class yes  [0.793]

    Rule 48/17: (12/2.2, lift 1.8)
        checking_balance = 1 - 200 DM
        purpose = car
        savings_balance = < 100 DM
        years_at_residence > 3
        ->  class yes  [0.770]

    Rule 48/18: (10.7/1.9, lift 1.8)
        checking_balance = unknown
        purpose in {business, car, education}
        savings_balance = < 100 DM
        other_credit = store
        ->  class yes  [0.768]

    Rule 48/19: (24.4/5.3, lift 1.7)
        amount <= 1804
        savings_balance = unknown
        employment_duration = 1 - 4 years
        other_credit = none
        ->  class yes  [0.760]

    Rule 48/20: (2, lift 1.7)
        savings_balance = > 1000 DM
        existing_loans_count > 2
        ->  class yes  [0.753]

    Rule 48/21: (26.6/6.1, lift 1.7)
        checking_balance = 1 - 200 DM
        months_loan_duration > 7
        purpose = furniture/appliances
        amount > 685
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years}
        housing = own
        existing_loans_count <= 1
        job = skilled
        ->  class yes  [0.752]

    Rule 48/22: (8.4/1.8, lift 1.7)
        purpose = education
        savings_balance = 100 - 500 DM
        ->  class yes  [0.735]

    Rule 48/23: (302.2/156.5, lift 1.1)
        checking_balance = < 0 DM
        ->  class yes  [0.482]

    Rule 48/24: (18.2, lift 1.7)
        months_loan_duration <= 33
        purpose in {business, car, furniture/appliances}
        savings_balance = > 1000 DM
        existing_loans_count <= 2
        ->  class no  [0.951]

    Rule 48/25: (13.9, lift 1.7)
        checking_balance = < 0 DM
        months_loan_duration <= 16
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        years_at_residence <= 3
        ->  class no  [0.937]

    Rule 48/26: (11, lift 1.6)
        months_loan_duration <= 11
        credit_history in {critical, good, perfect}
        purpose = car
        job = skilled
        ->  class no  [0.923]

    Rule 48/27: (8, lift 1.6)
        savings_balance = 500 - 1000 DM
        percent_of_income > 3
        existing_loans_count <= 2
        job = skilled
        ->  class no  [0.900]

    Rule 48/28: (6.9, lift 1.6)
        months_loan_duration <= 11
        savings_balance = 500 - 1000 DM
        ->  class no  [0.888]

    Rule 48/29: (6.4, lift 1.6)
        checking_balance = 1 - 200 DM
        purpose = furniture/appliances
        amount <= 685
        ->  class no  [0.880]

    Rule 48/30: (54.9/6.4, lift 1.5)
        checking_balance = unknown
        months_loan_duration <= 33
        savings_balance = < 100 DM
        age > 23
        other_credit = none
        ->  class no  [0.870]

    Rule 48/31: (9.8/0.6, lift 1.5)
        checking_balance = < 0 DM
        purpose in {business, car0, education}
        job = unskilled
        ->  class no  [0.861]

    Rule 48/32: (17.3/2.1, lift 1.5)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 30
        purpose = furniture/appliances
        savings_balance = < 100 DM
        existing_loans_count > 1
        ->  class no  [0.840]

    Rule 48/33: (25.8/4, lift 1.5)
        savings_balance = < 100 DM
        existing_loans_count > 1
        job = unskilled
        phone = FALSE
        ->  class no  [0.820]

    Rule 48/34: (15.2/2.5, lift 1.4)
        purpose in {business, car0, renovations}
        savings_balance = 100 - 500 DM
        ->  class no  [0.795]

    Rule 48/35: (25.6/6.6, lift 1.3)
        checking_balance = unknown
        percent_of_income <= 1
        ->  class no  [0.724]

    Rule 48/36: (89.3/26.1, lift 1.2)
        savings_balance = unknown
        employment_duration in {< 1 year, > 7 years, 4 - 7 years, unemployed}
        ->  class no  [0.703]

    Rule 48/37: (659.8/270.6, lift 1.0)
        amount > 1386
        ->  class no  [0.590]

    Default class: no

    -----  Trial 49:  -----

    Rules:

    Rule 49/1: (17.1, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration > 27
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, 4 - 7 years}
        existing_loans_count <= 1
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.948]

    Rule 49/2: (12.8, lift 1.9)
        months_loan_duration <= 42
        purpose = car
        amount > 11590
        ->  class yes  [0.932]

    Rule 49/3: (11.5, lift 1.9)
        credit_history = perfect
        savings_balance = < 100 DM
        housing in {other, rent}
        ->  class yes  [0.926]

    Rule 49/4: (18.4/1.2, lift 1.8)
        credit_history = poor
        savings_balance = < 100 DM
        percent_of_income > 2
        phone = TRUE
        ->  class yes  [0.892]

    Rule 49/5: (6.1, lift 1.8)
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = unemployed
        phone = FALSE
        ->  class yes  [0.876]

    Rule 49/6: (5.3, lift 1.7)
        credit_history = very good
        savings_balance = < 100 DM
        existing_loans_count > 1
        ->  class yes  [0.864]

    Rule 49/7: (19.2/2, lift 1.7)
        months_loan_duration > 11
        savings_balance = 500 - 1000 DM
        years_at_residence > 1
        age <= 30
        other_credit = none
        job in {skilled, unskilled}
        ->  class yes  [0.860]

    Rule 49/8: (28.5/3.5, lift 1.7)
        months_loan_duration <= 42
        amount > 8133
        amount <= 14896
        savings_balance = < 100 DM
        ->  class yes  [0.852]

    Rule 49/9: (4.7, lift 1.7)
        purpose in {education, renovations}
        savings_balance = > 1000 DM
        ->  class yes  [0.850]

    Rule 49/10: (13/1.5, lift 1.7)
        credit_history = perfect
        savings_balance = < 100 DM
        age > 33
        ->  class yes  [0.834]

    Rule 49/11: (22.3/4.2, lift 1.6)
        credit_history = good
        amount <= 2603
        savings_balance = < 100 DM
        percent_of_income <= 3
        housing = rent
        phone = FALSE
        ->  class yes  [0.787]

    Rule 49/12: (15.1/2.7, lift 1.6)
        checking_balance = < 0 DM
        savings_balance = unknown
        employment_duration = > 7 years
        ->  class yes  [0.783]

    Rule 49/13: (17.6/3.3, lift 1.6)
        credit_history = critical
        amount > 2012
        savings_balance = < 100 DM
        housing = rent
        dependents <= 1
        ->  class yes  [0.780]

    Rule 49/14: (23/5.3, lift 1.5)
        credit_history = good
        savings_balance = < 100 DM
        percent_of_income > 2
        housing = other
        ->  class yes  [0.750]

    Rule 49/15: (116.5/37.3, lift 1.4)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 13
        credit_history = good
        percent_of_income > 2
        years_at_residence > 1
        ->  class yes  [0.677]

    Rule 49/16: (69.1/22.8, lift 1.3)
        months_loan_duration > 42
        ->  class yes  [0.665]

    Rule 49/17: (59.9/22.4, lift 1.3)
        checking_balance in {< 0 DM, 1 - 200 DM}
        savings_balance = 100 - 500 DM
        ->  class yes  [0.622]

    Rule 49/18: (12.3, lift 1.8)
        purpose = education
        savings_balance = unknown
        job in {skilled, unskilled}
        ->  class no  [0.930]

    Rule 49/19: (9.3/0.5, lift 1.7)
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        percent_of_income > 3
        housing = rent
        existing_loans_count <= 1
        phone = FALSE
        ->  class no  [0.864]

    Rule 49/20: (16.9/1.7, lift 1.7)
        months_loan_duration <= 42
        purpose in {business, car, furniture/appliances}
        savings_balance = > 1000 DM
        ->  class no  [0.858]

    Rule 49/21: (3.9, lift 1.6)
        months_loan_duration > 42
        purpose = car
        savings_balance = unknown
        ->  class no  [0.829]

    Rule 49/22: (2.8, lift 1.6)
        savings_balance = 500 - 1000 DM
        years_at_residence <= 1
        ->  class no  [0.790]

    Rule 49/23: (26.5/5.4, lift 1.5)
        purpose = furniture/appliances
        savings_balance = unknown
        age <= 43
        housing = own
        job = skilled
        ->  class no  [0.776]

    Rule 49/24: (830.9/400.6, lift 1.0)
        months_loan_duration <= 42
        ->  class no  [0.518]

    Default class: no

    -----  Trial 50:  -----

    Rules:

    Rule 50/1: (19.8, lift 1.9)
        months_loan_duration > 7
        amount <= 1264
        dependents > 1
        phone = FALSE
        ->  class yes  [0.952]

    Rule 50/2: (18.6, lift 1.9)
        months_loan_duration > 15
        months_loan_duration <= 21
        amount > 888
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        existing_loans_count <= 1
        phone = FALSE
        ->  class yes  [0.951]

    Rule 50/3: (16.9/1.3, lift 1.7)
        months_loan_duration <= 16
        amount > 4042
        phone = TRUE
        ->  class yes  [0.880]

    Rule 50/4: (14.2/1.3, lift 1.7)
        purpose = business
        amount > 1721
        amount <= 2483
        savings_balance = < 100 DM
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.861]

    Rule 50/5: (13.5/1.4, lift 1.7)
        months_loan_duration > 16
        purpose = education
        savings_balance in {< 100 DM, 100 - 500 DM}
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.848]

    Rule 50/6: (14.8/1.8, lift 1.6)
        months_loan_duration > 7
        credit_history in {perfect, poor, very good}
        purpose = furniture/appliances
        savings_balance = < 100 DM
        existing_loans_count <= 1
        phone = FALSE
        ->  class yes  [0.833]

    Rule 50/7: (12.1/1.5, lift 1.6)
        purpose = business
        housing = rent
        job in {skilled, unskilled}
        ->  class yes  [0.825]

    Rule 50/8: (23.4/3.5, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        amount > 1808
        savings_balance = < 100 DM
        percent_of_income > 1
        existing_loans_count > 1
        phone = FALSE
        ->  class yes  [0.822]

    Rule 50/9: (24.9/4.8, lift 1.5)
        purpose in {business, car0, renovations}
        amount > 4042
        savings_balance in {< 100 DM, > 1000 DM, unknown}
        phone = TRUE
        ->  class yes  [0.785]

    Rule 50/10: (34.6/8.1, lift 1.5)
        purpose = furniture/appliances
        amount > 4042
        amount <= 12976
        savings_balance in {< 100 DM, > 1000 DM, unknown}
        phone = TRUE
        ->  class yes  [0.751]

    Rule 50/11: (29.2/7.1, lift 1.5)
        percent_of_income > 1
        years_at_residence <= 2
        dependents > 1
        phone = FALSE
        ->  class yes  [0.740]

    Rule 50/12: (107.5/36.4, lift 1.3)
        months_loan_duration > 7
        credit_history in {critical, good, very good}
        purpose = car
        percent_of_income > 1
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.658]

    Rule 50/13: (593.8/278.4, lift 1.0)
        savings_balance = < 100 DM
        ->  class yes  [0.531]

    Rule 50/14: (13.2, lift 1.9)
        amount > 1264
        years_at_residence > 2
        housing = own
        dependents > 1
        phone = FALSE
        ->  class no  [0.934]

    Rule 50/15: (14.2/0.2, lift 1.9)
        purpose = car
        amount > 1867
        percent_of_income <= 1
        phone = FALSE
        ->  class no  [0.924]

    Rule 50/16: (11.2, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration <= 15
        purpose = furniture/appliances
        amount > 888
        housing = own
        existing_loans_count <= 1
        ->  class no  [0.924]

    Rule 50/17: (26/1.8, lift 1.8)
        months_loan_duration <= 7
        purpose in {business, car, furniture/appliances, renovations}
        amount <= 4139
        phone = FALSE
        ->  class no  [0.901]

    Rule 50/18: (6, lift 1.8)
        credit_history = good
        purpose = furniture/appliances
        amount <= 888
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        ->  class no  [0.875]

    Rule 50/19: (24.8/2.8, lift 1.8)
        purpose = business
        amount <= 8648
        housing = own
        job in {skilled, unskilled}
        dependents <= 1
        phone = FALSE
        ->  class no  [0.859]

    Rule 50/20: (5, lift 1.7)
        checking_balance = unknown
        credit_history = critical
        purpose = car
        other_credit = none
        phone = FALSE
        ->  class no  [0.858]

    Rule 50/21: (4.8, lift 1.7)
        purpose = education
        savings_balance = unknown
        phone = FALSE
        ->  class no  [0.852]

    Rule 50/22: (3.5, lift 1.7)
        purpose = furniture/appliances
        savings_balance = > 1000 DM
        phone = FALSE
        ->  class no  [0.817]

    Rule 50/23: (18.8/3.5, lift 1.6)
        checking_balance = < 0 DM
        months_loan_duration > 21
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        ->  class no  [0.782]

    Rule 50/24: (16.7/3.3, lift 1.6)
        purpose = furniture/appliances
        savings_balance = unknown
        housing in {other, own}
        dependents <= 1
        phone = FALSE
        ->  class no  [0.771]

    Rule 50/25: (182.8/69, lift 1.3)
        amount > 2483
        amount <= 4042
        ->  class no  [0.621]

    Rule 50/26: (135.6/57.7, lift 1.2)
        purpose = furniture/appliances
        employment_duration in {> 7 years, 4 - 7 years}
        ->  class no  [0.574]

    Rule 50/27: (202.8/93.7, lift 1.1)
        credit_history = critical
        ->  class no  [0.538]

    Rule 50/28: (328/151.8, lift 1.1)
        phone = TRUE
        ->  class no  [0.537]

    Default class: yes

    -----  Trial 51:  -----

    Rules:

    Rule 51/1: (11.4, lift 1.9)
        checking_balance = 1 - 200 DM
        credit_history in {good, perfect, very good}
        savings_balance = 100 - 500 DM
        housing = rent
        ->  class yes  [0.926]

    Rule 51/2: (10.9, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = perfect
        housing in {other, rent}
        ->  class yes  [0.923]

    Rule 51/3: (13.5/0.4, lift 1.8)
        checking_balance = unknown
        credit_history = good
        percent_of_income <= 2
        other_credit = none
        existing_loans_count > 1
        ->  class yes  [0.911]

    Rule 51/4: (7.1, lift 1.8)
        checking_balance = 1 - 200 DM
        months_loan_duration > 30
        credit_history = good
        savings_balance = 100 - 500 DM
        ->  class yes  [0.890]

    Rule 51/5: (6.3, lift 1.8)
        checking_balance = < 0 DM
        amount > 5711
        savings_balance = unknown
        employment_duration = > 7 years
        ->  class yes  [0.880]

    Rule 51/6: (6.2, lift 1.8)
        credit_history = critical
        amount > 7308
        savings_balance = < 100 DM
        other_credit = none
        ->  class yes  [0.878]

    Rule 51/7: (13.1/1.3, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = poor
        savings_balance = < 100 DM
        phone = TRUE
        ->  class yes  [0.846]

    Rule 51/8: (27.1/3.8, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 47
        savings_balance = < 100 DM
        ->  class yes  [0.836]

    Rule 51/9: (10/1.1, lift 1.6)
        credit_history in {perfect, very good}
        savings_balance = 100 - 500 DM
        other_credit = none
        ->  class yes  [0.824]

    Rule 51/10: (18/2.6, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        amount > 7582
        job = management
        ->  class yes  [0.822]

    Rule 51/11: (2.9, lift 1.6)
        months_loan_duration > 39
        savings_balance = > 1000 DM
        ->  class yes  [0.794]

    Rule 51/12: (31.5/5.9, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 21
        credit_history = good
        amount <= 1347
        savings_balance = < 100 DM
        job = skilled
        phone = FALSE
        ->  class yes  [0.793]

    Rule 51/13: (9.9/1.6, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        savings_balance = < 100 DM
        job = unemployed
        ->  class yes  [0.781]

    Rule 51/14: (2.5, lift 1.6)
        savings_balance = 100 - 500 DM
        existing_loans_count > 3
        ->  class yes  [0.776]

    Rule 51/15: (39.9/10.9, lift 1.4)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        savings_balance = < 100 DM
        other_credit in {bank, store}
        job = skilled
        ->  class yes  [0.717]

    Rule 51/16: (41.7/11.7, lift 1.4)
        checking_balance = unknown
        percent_of_income > 1
        age <= 44
        other_credit = bank
        ->  class yes  [0.708]

    Rule 51/17: (66.4/22.7, lift 1.3)
        months_loan_duration > 11
        purpose = furniture/appliances
        percent_of_income > 1
        job = unskilled
        ->  class yes  [0.653]

    Rule 51/18: (743.6/358.2, lift 1.0)
        age <= 44
        ->  class yes  [0.518]

    Rule 51/19: (17, lift 1.9)
        credit_history = critical
        purpose = car
        age > 35
        other_credit = none
        phone = FALSE
        ->  class no  [0.947]

    Rule 51/20: (15.9, lift 1.9)
        checking_balance = unknown
        months_loan_duration > 18
        other_credit = none
        housing = own
        existing_loans_count <= 1
        ->  class no  [0.944]

    Rule 51/21: (14.8, lift 1.9)
        checking_balance = unknown
        credit_history = good
        years_at_residence <= 3
        age > 27
        other_credit = none
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.940]

    Rule 51/22: (11.7, lift 1.9)
        checking_balance = unknown
        years_at_residence > 1
        age > 44
        other_credit = bank
        ->  class no  [0.927]

    Rule 51/23: (10.1, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose in {business, education}
        savings_balance = unknown
        job = skilled
        ->  class no  [0.917]

    Rule 51/24: (8.5, lift 1.8)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 30
        credit_history = good
        savings_balance = 100 - 500 DM
        other_credit = none
        housing in {other, own}
        ->  class no  [0.905]

    Rule 51/25: (19.8/1.3, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {critical, poor}
        savings_balance = 100 - 500 DM
        existing_loans_count <= 3
        ->  class no  [0.894]

    Rule 51/26: (14.9/1.2, lift 1.7)
        amount <= 5711
        savings_balance = unknown
        employment_duration = > 7 years
        job = skilled
        ->  class no  [0.869]

    Rule 51/27: (32.4/4.3, lift 1.7)
        checking_balance = unknown
        credit_history = critical
        amount <= 11816
        other_credit = none
        ->  class no  [0.847]

    Rule 51/28: (24.3/3.1, lift 1.7)
        checking_balance = unknown
        credit_history = good
        other_credit = none
        housing in {other, rent}
        existing_loans_count <= 1
        ->  class no  [0.846]

    Rule 51/29: (4.4, lift 1.7)
        checking_balance = unknown
        credit_history = perfect
        other_credit = none
        ->  class no  [0.844]

    Rule 51/30: (27.2/5.8, lift 1.5)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration <= 15
        credit_history = good
        amount > 1347
        job = skilled
        phone = FALSE
        ->  class no  [0.767]

    Rule 51/31: (13.8/2.9, lift 1.5)
        amount <= 458
        ->  class no  [0.755]

    Rule 51/32: (31/7.1, lift 1.5)
        months_loan_duration <= 47
        credit_history = good
        amount <= 7582
        savings_balance = < 100 DM
        job = management
        ->  class no  [0.753]

    Rule 51/33: (19/5.1, lift 1.4)
        months_loan_duration <= 39
        savings_balance = > 1000 DM
        ->  class no  [0.707]

    Rule 51/34: (43.6/13.4, lift 1.4)
        amount <= 1901
        savings_balance = < 100 DM
        other_credit = none
        job = unskilled
        ->  class no  [0.684]

    Rule 51/35: (452.8/212.1, lift 1.1)
        percent_of_income <= 3
        ->  class no  [0.531]

    Default class: no

    -----  Trial 52:  -----

    Rules:

    Rule 52/1: (15.7, lift 1.9)
        checking_balance in {< 0 DM, > 200 DM}
        months_loan_duration > 15
        months_loan_duration <= 22
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        job = skilled
        ->  class yes  [0.943]

    Rule 52/2: (13.7, lift 1.9)
        checking_balance in {< 0 DM, > 200 DM}
        credit_history = critical
        purpose = furniture/appliances
        employment_duration = 1 - 4 years
        years_at_residence > 3
        ->  class yes  [0.933]

    Rule 52/3: (8.4, lift 1.9)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        housing = other
        job = unskilled
        ->  class yes  [0.904]

    Rule 52/4: (8.3, lift 1.9)
        checking_balance = unknown
        credit_history = good
        savings_balance = 500 - 1000 DM
        years_at_residence > 1
        years_at_residence <= 2
        job = skilled
        ->  class yes  [0.900]

    Rule 52/5: (16.1/0.9, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        savings_balance = < 100 DM
        age > 41
        age <= 48
        dependents <= 1
        ->  class yes  [0.894]

    Rule 52/6: (10.1/0.3, lift 1.8)
        credit_history = critical
        purpose = car
        savings_balance = < 100 DM
        age <= 29
        ->  class yes  [0.892]

    Rule 52/7: (10.4/0.7, lift 1.8)
        checking_balance = < 0 DM
        credit_history = good
        purpose = car
        savings_balance = < 100 DM
        employment_duration in {> 7 years, 4 - 7 years, unemployed}
        housing in {own, rent}
        dependents <= 1
        ->  class yes  [0.862]

    Rule 52/8: (18.5/2.3, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 36
        purpose = furniture/appliances
        housing in {own, rent}
        ->  class yes  [0.841]

    Rule 52/9: (8.7/0.8, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        other_credit = bank
        housing in {own, rent}
        dependents > 1
        ->  class yes  [0.830]

    Rule 52/10: (21.4/4.1, lift 1.6)
        purpose = car
        savings_balance = < 100 DM
        other_credit in {bank, store}
        housing in {own, rent}
        ->  class yes  [0.782]

    Rule 52/11: (16.6/3.1, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        housing in {own, rent}
        ->  class yes  [0.781]

    Rule 52/12: (45/9.4, lift 1.6)
        checking_balance = unknown
        credit_history in {good, poor, very good}
        amount > 3612
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, unemployed}
        dependents <= 1
        ->  class yes  [0.779]

    Rule 52/13: (26.7/5.8, lift 1.6)
        checking_balance = < 0 DM
        credit_history = good
        purpose = car
        employment_duration in {< 1 year, 1 - 4 years}
        percent_of_income > 2
        dependents <= 1
        ->  class yes  [0.764]

    Rule 52/14: (28.5/6.7, lift 1.5)
        checking_balance = unknown
        amount > 6850
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, unemployed}
        dependents <= 1
        ->  class yes  [0.748]

    Rule 52/15: (36.6/10, lift 1.5)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = education
        age <= 42
        ->  class yes  [0.716]

    Rule 52/16: (45/13.7, lift 1.4)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        employment_duration = > 7 years
        years_at_residence <= 3
        job in {management, skilled}
        ->  class yes  [0.688]

    Rule 52/17: (42.8/13.7, lift 1.4)
        credit_history in {perfect, very good}
        other_credit = none
        ->  class yes  [0.671]

    Rule 52/18: (80.8/29.5, lift 1.3)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        housing = other
        ->  class yes  [0.631]

    Rule 52/19: (76.9/30.4, lift 1.2)
        savings_balance = < 100 DM
        employment_duration = < 1 year
        percent_of_income <= 3
        ->  class yes  [0.602]

    Rule 52/20: (202.8/89.3, lift 1.2)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        other_credit in {none, store}
        phone = TRUE
        ->  class yes  [0.559]

    Rule 52/21: (16.7, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration <= 22
        purpose = furniture/appliances
        savings_balance in {< 100 DM, unknown}
        employment_duration = 4 - 7 years
        ->  class no  [0.946]

    Rule 52/22: (7.3, lift 1.7)
        credit_history = poor
        purpose = car
        age <= 41
        ->  class no  [0.892]

    Rule 52/23: (7.2, lift 1.7)
        purpose = furniture/appliances
        savings_balance in {< 100 DM, unknown}
        employment_duration = < 1 year
        existing_loans_count <= 1
        job in {management, unemployed}
        ->  class no  [0.892]

    Rule 52/24: (30.5/3.4, lift 1.7)
        checking_balance = unknown
        credit_history = good
        amount <= 3612
        savings_balance in {< 100 DM, > 1000 DM, 100 - 500 DM, unknown}
        years_at_residence <= 2
        job = skilled
        ->  class no  [0.863]

    Rule 52/25: (43.2/7.3, lift 1.6)
        checking_balance = unknown
        amount <= 3612
        years_at_residence > 2
        job = skilled
        ->  class no  [0.816]

    Rule 52/26: (19.2/3.1, lift 1.6)
        purpose = car
        amount <= 5804
        other_credit = none
        housing in {own, rent}
        dependents > 1
        ->  class no  [0.807]

    Rule 52/27: (30.6/5.5, lift 1.5)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = critical
        purpose = car
        age > 29
        other_credit = none
        ->  class no  [0.800]

    Rule 52/28: (31.3/5.7, lift 1.5)
        purpose = furniture/appliances
        savings_balance in {< 100 DM, 500 - 1000 DM, unknown}
        employment_duration = > 7 years
        years_at_residence > 3
        housing in {own, rent}
        ->  class no  [0.800]

    Rule 52/29: (17.5/3.3, lift 1.5)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = business
        housing = own
        phone = FALSE
        ->  class no  [0.780]

    Rule 52/30: (20.7/5.5, lift 1.4)
        other_credit = bank
        housing = other
        job in {management, skilled, unemployed}
        ->  class no  [0.715]

    Rule 52/31: (37.3/12.1, lift 1.3)
        employment_duration = > 7 years
        job = unskilled
        ->  class no  [0.666]

    Rule 52/32: (61.6/22, lift 1.2)
        credit_history in {perfect, poor, very good}
        employment_duration = 1 - 4 years
        ->  class no  [0.639]

    Rule 52/33: (798.8/377.2, lift 1.0)
        housing in {own, rent}
        ->  class no  [0.528]

    Default class: no

    -----  Trial 53:  -----

    Rules:

    Rule 53/1: (17.6, lift 2.0)
        checking_balance = 1 - 200 DM
        credit_history = good
        amount > 8648
        ->  class yes  [0.949]

    Rule 53/2: (8.8/0.2, lift 1.9)
        checking_balance = unknown
        purpose = education
        amount > 1680
        percent_of_income > 2
        ->  class yes  [0.893]

    Rule 53/3: (6.7, lift 1.9)
        checking_balance = unknown
        purpose = furniture/appliances
        amount > 4594
        employment_duration = < 1 year
        ->  class yes  [0.885]

    Rule 53/4: (6.5, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration > 27
        credit_history = good
        savings_balance = unknown
        ->  class yes  [0.882]

    Rule 53/5: (9.1/0.3, lift 1.9)
        checking_balance = unknown
        purpose = car
        percent_of_income > 1
        other_credit = bank
        housing = own
        ->  class yes  [0.881]

    Rule 53/6: (4.5, lift 1.8)
        checking_balance = unknown
        purpose = furniture/appliances
        employment_duration = < 1 year
        age > 41
        ->  class yes  [0.846]

    Rule 53/7: (22.9/3.3, lift 1.8)
        checking_balance = 1 - 200 DM
        credit_history = good
        amount <= 5179
        employment_duration in {< 1 year, > 7 years, 1 - 4 years}
        housing = rent
        ->  class yes  [0.828]

    Rule 53/8: (13.2/1.7, lift 1.7)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 27
        credit_history = very good
        ->  class yes  [0.823]

    Rule 53/9: (3.1, lift 1.7)
        checking_balance = 1 - 200 DM
        credit_history = critical
        dependents > 1
        ->  class yes  [0.804]

    Rule 53/10: (7.9/1.2, lift 1.6)
        checking_balance = > 200 DM
        dependents > 1
        ->  class yes  [0.774]

    Rule 53/11: (10/2.2, lift 1.6)
        purpose = car
        other_credit = store
        ->  class yes  [0.731]

    Rule 53/12: (26.6/7.1, lift 1.5)
        checking_balance = unknown
        purpose = business
        employment_duration in {< 1 year, 1 - 4 years, unemployed}
        ->  class yes  [0.717]

    Rule 53/13: (37.6/12.2, lift 1.4)
        purpose = car
        percent_of_income > 1
        other_credit = bank
        ->  class yes  [0.667]

    Rule 53/14: (58.9/19.9, lift 1.4)
        checking_balance = < 0 DM
        credit_history = good
        purpose = furniture/appliances
        percent_of_income > 1
        years_at_residence > 1
        other_credit = none
        existing_loans_count <= 1
        dependents <= 1
        ->  class yes  [0.657]

    Rule 53/15: (23.8/8.1, lift 1.4)
        checking_balance = < 0 DM
        credit_history = very good
        ->  class yes  [0.648]

    Rule 53/16: (301.5/141.7, lift 1.1)
        checking_balance = < 0 DM
        ->  class yes  [0.530]

    Rule 53/17: (11.7, lift 1.8)
        checking_balance = 1 - 200 DM
        employment_duration in {4 - 7 years, unemployed}
        housing = rent
        ->  class no  [0.927]

    Rule 53/18: (11.4, lift 1.8)
        checking_balance = < 0 DM
        months_loan_duration > 10
        credit_history = good
        purpose = car
        savings_balance = < 100 DM
        employment_duration in {1 - 4 years, unemployed}
        years_at_residence > 2
        ->  class no  [0.925]

    Rule 53/19: (20.6/0.8, lift 1.7)
        checking_balance = 1 - 200 DM
        credit_history = poor
        age <= 39
        ->  class no  [0.919]

    Rule 53/20: (8.7, lift 1.7)
        credit_history = critical
        years_at_residence <= 1
        housing = own
        ->  class no  [0.907]

    Rule 53/21: (7.9, lift 1.7)
        checking_balance = 1 - 200 DM
        credit_history = critical
        savings_balance in {> 1000 DM, 100 - 500 DM, unknown}
        dependents <= 1
        ->  class no  [0.899]

    Rule 53/22: (6.8, lift 1.7)
        checking_balance = 1 - 200 DM
        credit_history = critical
        housing in {other, rent}
        dependents <= 1
        ->  class no  [0.887]

    Rule 53/23: (9.7/1.2, lift 1.5)
        checking_balance = < 0 DM
        credit_history = critical
        dependents > 1
        ->  class no  [0.812]

    Rule 53/24: (48.7/12.6, lift 1.4)
        credit_history = good
        amount <= 1766
        age > 30
        other_credit = none
        dependents <= 1
        ->  class no  [0.731]

    Rule 53/25: (66.1/23.5, lift 1.2)
        checking_balance = > 200 DM
        dependents <= 1
        ->  class no  [0.640]

    Rule 53/26: (234.5/96.7, lift 1.1)
        checking_balance = unknown
        ->  class no  [0.587]

    Rule 53/27: (145.3/60.3, lift 1.1)
        savings_balance = unknown
        ->  class no  [0.584]

    Default class: no

    -----  Trial 54:  -----

    Rules:

    Rule 54/1: (15.7, lift 2.0)
        credit_history = good
        amount > 10722
        phone = TRUE
        ->  class yes  [0.944]

    Rule 54/2: (11.2, lift 2.0)
        months_loan_duration > 16
        credit_history = poor
        savings_balance = < 100 DM
        percent_of_income > 3
        age <= 46
        ->  class yes  [0.924]

    Rule 54/3: (11.2/0.5, lift 1.9)
        credit_history = good
        purpose = furniture/appliances
        age <= 45
        existing_loans_count > 1
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.884]

    Rule 54/4: (12.3/1, lift 1.8)
        months_loan_duration > 16
        credit_history = poor
        employment_duration = < 1 year
        ->  class yes  [0.860]

    Rule 54/5: (16.8/2, lift 1.8)
        credit_history = poor
        age > 46
        ->  class yes  [0.841]

    Rule 54/6: (18.1/3.4, lift 1.7)
        months_loan_duration > 8
        purpose = car
        age <= 28
        phone = TRUE
        ->  class yes  [0.780]

    Rule 54/7: (47.1/14.2, lift 1.5)
        credit_history = very good
        amount <= 7629
        age > 23
        ->  class yes  [0.690]

    Rule 54/8: (41.6/16.8, lift 1.3)
        credit_history = perfect
        ->  class yes  [0.592]

    Rule 54/9: (569.9/288.2, lift 1.0)
        phone = FALSE
        ->  class yes  [0.494]

    Rule 54/10: (13.5, lift 1.8)
        credit_history = critical
        age > 60
        ->  class no  [0.935]

    Rule 54/11: (11.6, lift 1.8)
        months_loan_duration <= 16
        credit_history = poor
        age <= 46
        ->  class no  [0.926]

    Rule 54/12: (10, lift 1.7)
        checking_balance = 1 - 200 DM
        credit_history = good
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 4 - 7 years}
        other_credit = none
        housing = own
        existing_loans_count <= 1
        phone = FALSE
        ->  class no  [0.917]

    Rule 54/13: (8.2, lift 1.7)
        checking_balance = > 200 DM
        age <= 24
        ->  class no  [0.902]

    Rule 54/14: (7.1, lift 1.7)
        months_loan_duration > 27
        credit_history = critical
        savings_balance = unknown
        ->  class no  [0.891]

    Rule 54/15: (24/1.8, lift 1.7)
        credit_history = poor
        savings_balance in {> 1000 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years, unemployed}
        age <= 46
        ->  class no  [0.891]

    Rule 54/16: (6.6, lift 1.7)
        credit_history = very good
        age <= 23
        ->  class no  [0.884]

    Rule 54/17: (8.7/0.3, lift 1.7)
        checking_balance = < 0 DM
        credit_history = good
        purpose = furniture/appliances
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        other_credit = none
        dependents > 1
        ->  class no  [0.879]

    Rule 54/18: (12.3/0.8, lift 1.7)
        purpose = car
        percent_of_income <= 1
        job = skilled
        phone = FALSE
        ->  class no  [0.875]

    Rule 54/19: (18.1/1.6, lift 1.7)
        months_loan_duration <= 36
        credit_history = critical
        purpose = furniture/appliances
        amount <= 6742
        savings_balance = < 100 DM
        age <= 30
        other_credit = none
        ->  class no  [0.873]

    Rule 54/20: (5.6, lift 1.6)
        credit_history = critical
        purpose = furniture/appliances
        other_credit = store
        ->  class no  [0.869]

    Rule 54/21: (4.8, lift 1.6)
        purpose = education
        savings_balance = unknown
        phone = FALSE
        ->  class no  [0.853]

    Rule 54/22: (25.5/3, lift 1.6)
        credit_history = poor
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years, unemployed}
        percent_of_income <= 3
        age <= 46
        ->  class no  [0.853]

    Rule 54/23: (33.7/4.8, lift 1.6)
        credit_history = critical
        purpose = car
        savings_balance = < 100 DM
        age > 29
        other_credit = none
        ->  class no  [0.837]

    Rule 54/24: (21.5/3.1, lift 1.6)
        months_loan_duration > 11
        credit_history = good
        purpose = furniture/appliances
        other_credit = bank
        phone = FALSE
        ->  class no  [0.827]

    Rule 54/25: (3.1, lift 1.5)
        purpose = car0
        phone = FALSE
        ->  class no  [0.803]

    Rule 54/26: (53.1/13.2, lift 1.4)
        checking_balance = < 0 DM
        months_loan_duration <= 27
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, 4 - 7 years}
        existing_loans_count <= 1
        phone = FALSE
        ->  class no  [0.743]

    Rule 54/27: (330.1/143, lift 1.1)
        phone = TRUE
        ->  class no  [0.567]

    Default class: no

    -----  Trial 55:  -----

    Rules:

    Rule 55/1: (15.4, lift 2.2)
        checking_balance = < 0 DM
        amount > 3123
        savings_balance = < 100 DM
        job = skilled
        phone = TRUE
        ->  class yes  [0.943]

    Rule 55/2: (8.7, lift 2.1)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        savings_balance = < 100 DM
        years_at_residence > 1
        other_credit = bank
        job = skilled
        ->  class yes  [0.907]

    Rule 55/3: (8.9, lift 2.1)
        checking_balance = unknown
        months_loan_duration > 7
        savings_balance in {< 100 DM, 500 - 1000 DM, unknown}
        other_credit = bank
        existing_loans_count <= 1
        job = unskilled
        ->  class yes  [0.906]

    Rule 55/4: (8.2, lift 2.1)
        checking_balance = 1 - 200 DM
        credit_history in {critical, good}
        purpose = furniture/appliances
        amount > 4006
        employment_duration = 4 - 7 years
        ->  class yes  [0.902]

    Rule 55/5: (6.6, lift 2.0)
        checking_balance = 1 - 200 DM
        purpose = furniture/appliances
        employment_duration = < 1 year
        housing = rent
        job = skilled
        ->  class yes  [0.884]

    Rule 55/6: (6.3, lift 2.0)
        checking_balance = < 0 DM
        credit_history = poor
        savings_balance = < 100 DM
        job = skilled
        ->  class yes  [0.879]

    Rule 55/7: (5.8, lift 2.0)
        checking_balance = < 0 DM
        months_loan_duration > 7
        months_loan_duration <= 21
        employment_duration = < 1 year
        existing_loans_count <= 1
        job = unskilled
        ->  class yes  [0.871]

    Rule 55/8: (5.3, lift 2.0)
        months_loan_duration > 26
        months_loan_duration <= 36
        savings_balance = < 100 DM
        job = unskilled
        ->  class yes  [0.863]

    Rule 55/9: (4.7, lift 1.9)
        checking_balance = < 0 DM
        savings_balance = < 100 DM
        employment_duration = < 1 year
        job = skilled
        phone = TRUE
        ->  class yes  [0.850]

    Rule 55/10: (28.2/5, lift 1.8)
        checking_balance = < 0 DM
        months_loan_duration > 7
        purpose = furniture/appliances
        amount > 1092
        amount <= 3416
        savings_balance = < 100 DM
        years_at_residence > 1
        job = skilled
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.800]

    Rule 55/11: (9.7/1.6, lift 1.8)
        checking_balance = 1 - 200 DM
        months_loan_duration > 21
        savings_balance = < 100 DM
        job = unskilled
        ->  class yes  [0.778]

    Rule 55/12: (15.6/3.3, lift 1.7)
        checking_balance = 1 - 200 DM
        savings_balance in {100 - 500 DM, 500 - 1000 DM}
        job = unskilled
        ->  class yes  [0.752]

    Rule 55/13: (11.4/2.4, lift 1.7)
        checking_balance = < 0 DM
        savings_balance = unknown
        job = skilled
        phone = FALSE
        ->  class yes  [0.747]

    Rule 55/14: (47.4/15.2, lift 1.5)
        checking_balance = 1 - 200 DM
        months_loan_duration > 7
        credit_history in {critical, good, perfect}
        purpose = furniture/appliances
        employment_duration in {1 - 4 years, unemployed}
        other_credit = none
        ->  class yes  [0.672]

    Rule 55/15: (50.1/19.2, lift 1.4)
        checking_balance = 1 - 200 DM
        job = management
        ->  class yes  [0.612]

    Rule 55/16: (139.1/66.9, lift 1.2)
        job = management
        ->  class yes  [0.519]

    Rule 55/17: (299.8/159.1, lift 1.1)
        checking_balance = < 0 DM
        ->  class yes  [0.470]

    Rule 55/18: (7.5, lift 1.6)
        savings_balance = < 100 DM
        years_at_residence <= 1
        job = management
        ->  class no  [0.895]

    Rule 55/19: (18.6/2.6, lift 1.5)
        checking_balance = < 0 DM
        months_loan_duration > 7
        savings_balance = < 100 DM
        years_at_residence > 2
        job = management
        ->  class no  [0.826]

    Rule 55/20: (57.9/13, lift 1.4)
        months_loan_duration <= 7
        ->  class no  [0.766]

    Rule 55/21: (24.4/5.8, lift 1.3)
        checking_balance = < 0 DM
        savings_balance in {> 1000 DM, 100 - 500 DM, 500 - 1000 DM}
        ->  class no  [0.742]

    Rule 55/22: (842.1/380.2, lift 1.0)
        months_loan_duration > 7
        ->  class no  [0.548]

    Default class: no

    -----  Trial 56:  -----

    Rules:

    Rule 56/1: (8.5, lift 1.9)
        employment_duration = unemployed
        years_at_residence <= 1
        ->  class yes  [0.905]

    Rule 56/2: (19.2/1.1, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        savings_balance in {< 100 DM, 100 - 500 DM, unknown}
        employment_duration = > 7 years
        age > 33
        age <= 62
        job = management
        dependents <= 1
        ->  class yes  [0.902]

    Rule 56/3: (12/0.6, lift 1.8)
        checking_balance = unknown
        months_loan_duration > 18
        employment_duration = > 7 years
        age > 28
        age <= 35
        dependents <= 1
        ->  class yes  [0.884]

    Rule 56/4: (5.2, lift 1.8)
        months_loan_duration <= 8
        credit_history = good
        amount > 4057
        ->  class yes  [0.860]

    Rule 56/5: (5, lift 1.8)
        purpose = business
        employment_duration = unemployed
        ->  class yes  [0.858]

    Rule 56/6: (14.8/1.4, lift 1.8)
        months_loan_duration > 8
        credit_history = good
        amount <= 1512
        savings_balance = < 100 DM
        employment_duration = 4 - 7 years
        age <= 49
        ->  class yes  [0.857]

    Rule 56/7: (4.5, lift 1.7)
        savings_balance = < 100 DM
        employment_duration = < 1 year
        job = unemployed
        ->  class yes  [0.847]

    Rule 56/8: (20/2.5, lift 1.7)
        months_loan_duration > 9
        savings_balance = < 100 DM
        employment_duration = < 1 year
        age > 22
        housing = rent
        job in {skilled, unskilled}
        ->  class yes  [0.841]

    Rule 56/9: (4.3, lift 1.7)
        months_loan_duration > 8
        employment_duration = unemployed
        dependents > 1
        ->  class yes  [0.841]

    Rule 56/10: (4.1, lift 1.7)
        purpose = business
        employment_duration = 1 - 4 years
        existing_loans_count > 2
        ->  class yes  [0.836]

    Rule 56/11: (22.6/3.5, lift 1.7)
        months_loan_duration > 16
        months_loan_duration <= 22
        purpose = furniture/appliances
        amount > 1055
        employment_duration = 1 - 4 years
        housing in {own, rent}
        ->  class yes  [0.819]

    Rule 56/12: (15.1/2.3, lift 1.7)
        purpose = business
        employment_duration = 1 - 4 years
        years_at_residence > 1
        years_at_residence <= 3
        housing in {own, rent}
        job = skilled
        ->  class yes  [0.805]

    Rule 56/13: (19.4/3.6, lift 1.6)
        months_loan_duration > 8
        credit_history in {critical, poor, very good}
        savings_balance = < 100 DM
        employment_duration = < 1 year
        housing = own
        ->  class yes  [0.783]

    Rule 56/14: (28.9/5.7, lift 1.6)
        months_loan_duration > 8
        months_loan_duration <= 22
        employment_duration = > 7 years
        dependents > 1
        ->  class yes  [0.782]

    Rule 56/15: (33.3/7.4, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 8
        savings_balance in {< 100 DM, unknown}
        employment_duration = > 7 years
        age > 28
        age <= 62
        job = skilled
        dependents <= 1
        ->  class yes  [0.763]

    Rule 56/16: (13.2/2.8, lift 1.5)
        savings_balance = 100 - 500 DM
        employment_duration = < 1 year
        percent_of_income > 2
        ->  class yes  [0.750]

    Rule 56/17: (20.3/4.6, lift 1.5)
        purpose = education
        employment_duration = 1 - 4 years
        percent_of_income > 2
        ->  class yes  [0.746]

    Rule 56/18: (27.8/7.2, lift 1.5)
        employment_duration = 1 - 4 years
        housing = other
        ->  class yes  [0.726]

    Rule 56/19: (27.3/7.7, lift 1.4)
        months_loan_duration > 30
        purpose = furniture/appliances
        employment_duration = 1 - 4 years
        ->  class yes  [0.703]

    Rule 56/20: (61.9/20.3, lift 1.4)
        months_loan_duration > 8
        credit_history in {critical, good, perfect, very good}
        purpose = car
        employment_duration = 1 - 4 years
        percent_of_income > 1
        job in {skilled, unskilled}
        ->  class yes  [0.667]

    Rule 56/21: (59.3/25.3, lift 1.2)
        credit_history = very good
        ->  class yes  [0.572]

    Rule 56/22: (837.3/417.9, lift 1.0)
        months_loan_duration > 8
        ->  class yes  [0.501]

    Rule 56/23: (9.5, lift 1.8)
        employment_duration = 4 - 7 years
        age > 49
        ->  class no  [0.913]

    Rule 56/24: (30.4/4.1, lift 1.6)
        months_loan_duration <= 8
        credit_history = good
        amount <= 4057
        ->  class no  [0.844]

    Rule 56/25: (9/0.8, lift 1.6)
        purpose = car
        employment_duration = 1 - 4 years
        percent_of_income <= 1
        housing in {own, rent}
        ->  class no  [0.837]

    Rule 56/26: (19.9/3.3, lift 1.6)
        months_loan_duration <= 8
        credit_history in {critical, perfect, poor}
        ->  class no  [0.805]

    Rule 56/27: (40.4/8.7, lift 1.5)
        savings_balance in {> 1000 DM, 100 - 500 DM, unknown}
        employment_duration = 4 - 7 years
        ->  class no  [0.772]

    Rule 56/28: (24.1/5.9, lift 1.4)
        employment_duration = > 7 years
        job = unskilled
        dependents <= 1
        ->  class no  [0.736]

    Rule 56/29: (76/26, lift 1.3)
        months_loan_duration <= 16
        purpose = furniture/appliances
        employment_duration = 1 - 4 years
        housing in {own, rent}
        ->  class no  [0.654]

    Rule 56/30: (170.7/73, lift 1.1)
        savings_balance in {> 1000 DM, unknown}
        ->  class no  [0.572]

    Rule 56/31: (787.5/391.1, lift 1.0)
        months_loan_duration > 9
        ->  class no  [0.503]

    Default class: no

    -----  Trial 57:  -----

    Rules:

    Rule 57/1: (11.4, lift 1.9)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = perfect
        percent_of_income > 3
        ->  class yes  [0.925]

    Rule 57/2: (10.6, lift 1.9)
        checking_balance = unknown
        purpose in {business, renovations}
        employment_duration = < 1 year
        ->  class yes  [0.921]

    Rule 57/3: (8.5, lift 1.9)
        purpose in {education, renovations}
        savings_balance = > 1000 DM
        ->  class yes  [0.905]

    Rule 57/4: (6.7, lift 1.8)
        checking_balance = unknown
        purpose = furniture/appliances
        amount > 4594
        employment_duration = < 1 year
        ->  class yes  [0.885]

    Rule 57/5: (6, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        amount > 7678
        savings_balance = < 100 DM
        ->  class yes  [0.874]

    Rule 57/6: (29.2/4.1, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 47
        savings_balance = < 100 DM
        ->  class yes  [0.836]

    Rule 57/7: (14.6/1.9, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = poor
        savings_balance = < 100 DM
        percent_of_income > 2
        ->  class yes  [0.829]

    Rule 57/8: (10.3/2, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        savings_balance = < 100 DM
        job = unemployed
        ->  class yes  [0.758]

    Rule 57/9: (14.8/3.3, lift 1.6)
        checking_balance = unknown
        employment_duration = unemployed
        ->  class yes  [0.745]

    Rule 57/10: (20.8/5.4, lift 1.5)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = very good
        savings_balance = < 100 DM
        percent_of_income > 1
        ->  class yes  [0.722]

    Rule 57/11: (31.2/8.4, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        amount > 7485
        ->  class yes  [0.717]

    Rule 57/12: (747.6/377.6, lift 1.0)
        age <= 44
        ->  class yes  [0.495]

    Rule 57/13: (24.6, lift 1.9)
        checking_balance = unknown
        purpose = furniture/appliances
        employment_duration = 1 - 4 years
        job = skilled
        ->  class no  [0.962]

    Rule 57/14: (23.3, lift 1.8)
        checking_balance = unknown
        months_loan_duration <= 28
        credit_history in {critical, good}
        percent_of_income > 1
        other_credit = none
        job = skilled
        phone = TRUE
        ->  class no  [0.960]

    Rule 57/15: (13.8, lift 1.8)
        checking_balance = unknown
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        percent_of_income <= 1
        ->  class no  [0.937]

    Rule 57/16: (13, lift 1.8)
        checking_balance = unknown
        months_loan_duration > 28
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        other_credit = none
        job = skilled
        ->  class no  [0.933]

    Rule 57/17: (12.2, lift 1.8)
        years_at_residence <= 1
        age <= 44
        other_credit = bank
        ->  class no  [0.929]

    Rule 57/18: (11.5, lift 1.8)
        checking_balance = unknown
        savings_balance in {> 1000 DM, unknown}
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        age <= 31
        job = skilled
        ->  class no  [0.926]

    Rule 57/19: (9.5, lift 1.8)
        checking_balance = unknown
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        age > 44
        other_credit = bank
        ->  class no  [0.913]

    Rule 57/20: (8.3, lift 1.7)
        checking_balance = unknown
        purpose in {car, education}
        employment_duration = < 1 year
        ->  class no  [0.903]

    Rule 57/21: (7.9, lift 1.7)
        savings_balance in {> 1000 DM, 100 - 500 DM}
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        percent_of_income > 1
        other_credit = bank
        housing = own
        ->  class no  [0.899]

    Rule 57/22: (9.8/0.2, lift 1.7)
        checking_balance = < 0 DM
        months_loan_duration <= 47
        credit_history = good
        age > 45
        other_credit in {bank, none}
        housing in {own, rent}
        ->  class no  [0.898]

    Rule 57/23: (7.3, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = furniture/appliances
        savings_balance = unknown
        job in {unemployed, unskilled}
        ->  class no  [0.892]

    Rule 57/24: (40.5/3.9, lift 1.7)
        months_loan_duration <= 47
        credit_history = critical
        purpose = car
        amount <= 7678
        age > 29
        other_credit = none
        ->  class no  [0.884]

    Rule 57/25: (16.2/1.3, lift 1.7)
        checking_balance = < 0 DM
        months_loan_duration <= 16
        credit_history = good
        amount > 1386
        job = skilled
        ->  class no  [0.875]

    Rule 57/26: (5.5, lift 1.7)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 7
        credit_history = good
        savings_balance = < 100 DM
        ->  class no  [0.867]

    Rule 57/27: (15.9/1.4, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose in {business, education}
        savings_balance = unknown
        ->  class no  [0.865]

    Rule 57/28: (15.5/1.4, lift 1.7)
        checking_balance = unknown
        purpose = furniture/appliances
        amount <= 4594
        employment_duration = < 1 year
        age <= 41
        ->  class no  [0.861]

    Rule 57/29: (14.4/1.3, lift 1.7)
        purpose = furniture/appliances
        savings_balance = unknown
        percent_of_income > 3
        housing = own
        job = skilled
        ->  class no  [0.860]

    Rule 57/30: (67.1/8.7, lift 1.7)
        checking_balance = unknown
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        age > 31
        other_credit = none
        ->  class no  [0.860]

    Rule 57/31: (4.9, lift 1.6)
        savings_balance = 500 - 1000 DM
        job in {management, unemployed}
        ->  class no  [0.856]

    Rule 57/32: (16.4/1.8, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = car
        amount > 1372
        amount <= 7485
        savings_balance = unknown
        ->  class no  [0.849]

    Rule 57/33: (4.2, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        savings_balance = 500 - 1000 DM
        other_credit in {bank, store}
        ->  class no  [0.838]

    Rule 57/34: (3.2, lift 1.6)
        savings_balance = 500 - 1000 DM
        years_at_residence <= 1
        ->  class no  [0.809]

    Rule 57/35: (12.3/1.8, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose in {business, car, furniture/appliances}
        savings_balance = > 1000 DM
        ->  class no  [0.806]

    Rule 57/36: (27.1/4.9, lift 1.5)
        months_loan_duration <= 47
        credit_history = good
        amount <= 7582
        savings_balance = < 100 DM
        job = management
        ->  class no  [0.798]

    Rule 57/37: (40.6/8.3, lift 1.5)
        months_loan_duration <= 36
        credit_history = critical
        purpose = furniture/appliances
        amount <= 7678
        savings_balance = < 100 DM
        other_credit = none
        job in {skilled, unemployed, unskilled}
        ->  class no  [0.781]

    Rule 57/38: (18.3/3.7, lift 1.5)
        checking_balance = > 200 DM
        credit_history = good
        other_credit = none
        job = skilled
        ->  class no  [0.771]

    Rule 57/39: (23.9/6.7, lift 1.3)
        credit_history = poor
        percent_of_income <= 2
        ->  class no  [0.701]

    Rule 57/40: (43.7/12.7, lift 1.3)
        months_loan_duration <= 14
        savings_balance = < 100 DM
        other_credit = none
        job = unskilled
        ->  class no  [0.701]

    Default class: yes

    -----  Trial 58:  -----

    Rules:

    Rule 58/1: (11.4, lift 2.2)
        checking_balance = 1 - 200 DM
        months_loan_duration > 36
        savings_balance = < 100 DM
        other_credit = none
        ->  class yes  [0.926]

    Rule 58/2: (9.7, lift 2.1)
        checking_balance = < 0 DM
        purpose = education
        savings_balance = < 100 DM
        age <= 42
        ->  class yes  [0.915]

    Rule 58/3: (9.2, lift 2.1)
        checking_balance = 1 - 200 DM
        amount > 888
        amount <= 1316
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        other_credit = none
        phone = FALSE
        ->  class yes  [0.910]

    Rule 58/4: (6.9, lift 2.1)
        checking_balance = unknown
        percent_of_income > 3
        age <= 33
        existing_loans_count <= 1
        job = unskilled
        ->  class yes  [0.888]

    Rule 58/5: (6.3, lift 2.1)
        checking_balance = < 0 DM
        months_loan_duration <= 10
        credit_history = good
        purpose = car
        savings_balance = < 100 DM
        dependents <= 1
        ->  class yes  [0.880]

    Rule 58/6: (17.2/1.5, lift 2.0)
        checking_balance = unknown
        savings_balance = < 100 DM
        percent_of_income > 1
        age <= 23
        job = skilled
        ->  class yes  [0.870]

    Rule 58/7: (5.6, lift 2.0)
        checking_balance = 1 - 200 DM
        savings_balance = 100 - 500 DM
        housing = rent
        phone = FALSE
        ->  class yes  [0.868]

    Rule 58/8: (5, lift 2.0)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        ->  class yes  [0.858]

    Rule 58/9: (12.3/1.1, lift 2.0)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        savings_balance = < 100 DM
        percent_of_income > 3
        years_at_residence > 1
        other_credit in {bank, store}
        ->  class yes  [0.857]

    Rule 58/10: (15.2/1.5, lift 2.0)
        checking_balance = < 0 DM
        purpose = car
        savings_balance = < 100 DM
        percent_of_income > 2
        percent_of_income <= 3
        ->  class yes  [0.856]

    Rule 58/11: (11.5/1.1, lift 2.0)
        years_at_residence <= 2
        other_credit = bank
        existing_loans_count > 1
        job = skilled
        ->  class yes  [0.845]

    Rule 58/12: (5.8/0.3, lift 1.9)
        checking_balance = unknown
        credit_history in {poor, very good}
        job = management
        ->  class yes  [0.829]

    Rule 58/13: (17.5/2.7, lift 1.9)
        checking_balance = > 200 DM
        amount <= 4308
        percent_of_income > 3
        other_credit in {none, store}
        housing = own
        ->  class yes  [0.809]

    Rule 58/14: (11.5/1.6, lift 1.9)
        checking_balance = 1 - 200 DM
        employment_duration = < 1 year
        housing = rent
        phone = FALSE
        ->  class yes  [0.808]

    Rule 58/15: (10.6/1.6, lift 1.9)
        checking_balance = unknown
        housing = rent
        existing_loans_count > 1
        job = skilled
        ->  class yes  [0.792]

    Rule 58/16: (25.5/4.7, lift 1.9)
        checking_balance = < 0 DM
        credit_history = good
        purpose = car
        savings_balance = < 100 DM
        years_at_residence <= 2
        ->  class yes  [0.791]

    Rule 58/17: (23/4.3, lift 1.8)
        checking_balance = < 0 DM
        credit_history = good
        purpose = car
        savings_balance = < 100 DM
        employment_duration in {< 1 year, > 7 years, 4 - 7 years}
        dependents <= 1
        ->  class yes  [0.789]

    Rule 58/18: (26.7/6, lift 1.8)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        amount > 1620
        savings_balance = < 100 DM
        percent_of_income > 3
        years_at_residence > 1
        other_credit = none
        ->  class yes  [0.757]

    Rule 58/19: (21.5/5, lift 1.7)
        checking_balance = 1 - 200 DM
        amount > 10366
        ->  class yes  [0.747]

    Rule 58/20: (7/1.8, lift 1.6)
        checking_balance = > 200 DM
        dependents > 1
        ->  class yes  [0.695]

    Rule 58/21: (27.9/9, lift 1.6)
        purpose = car
        savings_balance = < 100 DM
        other_credit in {bank, store}
        ->  class yes  [0.667]

    Rule 58/22: (29.2/11.5, lift 1.4)
        checking_balance = < 0 DM
        savings_balance = unknown
        ->  class yes  [0.598]

    Rule 58/23: (581.2/317.7, lift 1.1)
        savings_balance = < 100 DM
        ->  class yes  [0.454]

    Rule 58/24: (5.6, lift 1.5)
        checking_balance = < 0 DM
        purpose = education
        age > 42
        ->  class no  [0.869]

    Rule 58/25: (62.4/10.4, lift 1.4)
        checking_balance = unknown
        age > 23
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.823]

    Rule 58/26: (41.6/7.8, lift 1.4)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 36
        employment_duration in {> 7 years, 4 - 7 years}
        other_credit = none
        phone = FALSE
        ->  class no  [0.798]

    Rule 58/27: (95.1/28.2, lift 1.2)
        checking_balance = 1 - 200 DM
        amount <= 10366
        phone = TRUE
        ->  class no  [0.699]

    Rule 58/28: (84/25.1, lift 1.2)
        credit_history = critical
        percent_of_income > 3
        other_credit = none
        ->  class no  [0.696]

    Rule 58/29: (456.8/180.1, lift 1.1)
        percent_of_income <= 3
        ->  class no  [0.605]

    Rule 58/30: (641/265, lift 1.0)
        housing = own
        ->  class no  [0.586]

    Default class: no

    -----  Trial 59:  -----

    Rules:

    Rule 59/1: (19.3, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 8
        credit_history = good
        savings_balance = < 100 DM
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, 4 - 7 years}
        percent_of_income > 2
        years_at_residence > 1
        years_at_residence <= 3
        phone = TRUE
        ->  class yes  [0.953]

    Rule 59/2: (14.6, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = perfect
        housing in {other, rent}
        ->  class yes  [0.940]

    Rule 59/3: (11.2, lift 1.9)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = perfect
        percent_of_income > 3
        ->  class yes  [0.924]

    Rule 59/4: (9.4, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        savings_balance = 100 - 500 DM
        age <= 24
        ->  class yes  [0.913]

    Rule 59/5: (8.1, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 9
        credit_history = very good
        ->  class yes  [0.901]

    Rule 59/6: (10.9/0.3, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        employment_duration = < 1 year
        percent_of_income > 2
        years_at_residence > 2
        phone = FALSE
        ->  class yes  [0.899]

    Rule 59/7: (7.6, lift 1.9)
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = unemployed
        phone = FALSE
        ->  class yes  [0.896]

    Rule 59/8: (12.3/0.5, lift 1.9)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = good
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        percent_of_income > 2
        age > 35
        phone = FALSE
        ->  class yes  [0.893]

    Rule 59/9: (6.7, lift 1.8)
        credit_history = critical
        purpose = car
        savings_balance = < 100 DM
        other_credit in {bank, store}
        ->  class yes  [0.886]

    Rule 59/10: (10.1/0.4, lift 1.8)
        checking_balance = unknown
        purpose = education
        amount > 1680
        percent_of_income > 2
        ->  class yes  [0.881]

    Rule 59/11: (11.1/0.6, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = poor
        age > 45
        ->  class yes  [0.876]

    Rule 59/12: (10.4/0.6, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration <= 33
        credit_history = good
        percent_of_income > 2
        dependents > 1
        phone = FALSE
        ->  class yes  [0.874]

    Rule 59/13: (10.1/1.1, lift 1.7)
        checking_balance = < 0 DM
        credit_history = good
        savings_balance = unknown
        job = skilled
        phone = FALSE
        ->  class yes  [0.823]

    Rule 59/14: (16.3/2.4, lift 1.7)
        checking_balance = unknown
        credit_history = good
        purpose = furniture/appliances
        employment_duration in {< 1 year, > 7 years, unemployed}
        existing_loans_count > 1
        ->  class yes  [0.815]

    Rule 59/15: (19.7/3.2, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        purpose = furniture/appliances
        percent_of_income <= 2
        ->  class yes  [0.808]

    Rule 59/16: (10.1/2.4, lift 1.5)
        checking_balance = < 0 DM
        credit_history = poor
        ->  class yes  [0.722]

    Rule 59/17: (31/8.6, lift 1.5)
        checking_balance = unknown
        purpose = business
        amount > 2150
        ->  class yes  [0.708]

    Rule 59/18: (61.2/17.8, lift 1.5)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        amount <= 3384
        savings_balance = < 100 DM
        percent_of_income <= 2
        phone = FALSE
        ->  class yes  [0.702]

    Rule 59/19: (63.6/21.4, lift 1.4)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = car
        savings_balance = < 100 DM
        percent_of_income > 2
        dependents <= 1
        ->  class yes  [0.659]

    Rule 59/20: (663.2/326.6, lift 1.1)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        ->  class yes  [0.508]

    Rule 59/21: (11.8, lift 1.8)
        months_loan_duration <= 33
        credit_history = good
        purpose = furniture/appliances
        amount <= 3384
        percent_of_income <= 2
        age > 35
        ->  class no  [0.928]

    Rule 59/22: (7.9, lift 1.7)
        months_loan_duration > 8
        amount <= 8086
        savings_balance = < 100 DM
        employment_duration = unemployed
        other_credit = none
        phone = TRUE
        ->  class no  [0.899]

    Rule 59/23: (6.6, lift 1.7)
        credit_history = critical
        purpose = car
        age > 46
        other_credit = none
        ->  class no  [0.884]

    Rule 59/24: (37.4/6.7, lift 1.5)
        checking_balance = unknown
        credit_history in {critical, perfect, poor, very good}
        purpose = furniture/appliances
        ->  class no  [0.806]

    Rule 59/25: (47.3/9, lift 1.5)
        checking_balance = unknown
        purpose = car
        amount <= 11816
        other_credit = none
        ->  class no  [0.798]

    Rule 59/26: (23.7/4.6, lift 1.5)
        checking_balance in {> 200 DM, 1 - 200 DM}
        credit_history = poor
        age <= 45
        ->  class no  [0.781]

    Rule 59/27: (828.9/388.3, lift 1.0)
        amount <= 8086
        ->  class no  [0.531]

    Default class: yes

    -----  Trial 60:  -----

    Rules:

    Rule 60/1: (11.9/0.2, lift 1.8)
        checking_balance = unknown
        amount > 4594
        employment_duration = < 1 year
        dependents <= 1
        ->  class yes  [0.915]

    Rule 60/2: (5.8/0.4, lift 1.7)
        checking_balance = unknown
        employment_duration = < 1 year
        housing = own
        dependents > 1
        ->  class yes  [0.826]

    Rule 60/3: (10.5/1.8, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        savings_balance = < 100 DM
        job = unemployed
        ->  class yes  [0.775]

    Rule 60/4: (28/8.5, lift 1.4)
        purpose = renovations
        ->  class yes  [0.684]

    Rule 60/5: (662.4/311.7, lift 1.1)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        ->  class yes  [0.529]

    Rule 60/6: (13.2, lift 1.9)
        credit_history = critical
        years_at_residence <= 3
        other_credit = none
        existing_loans_count <= 2
        job = skilled
        phone = TRUE
        ->  class no  [0.934]

    Rule 60/7: (13, lift 1.9)
        months_loan_duration <= 18
        credit_history = good
        purpose = business
        ->  class no  [0.933]

    Rule 60/8: (11.3, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = furniture/appliances
        amount <= 5711
        savings_balance = unknown
        employment_duration in {> 7 years, 4 - 7 years}
        ->  class no  [0.925]

    Rule 60/9: (10.5, lift 1.8)
        checking_balance = < 0 DM
        months_loan_duration <= 27
        credit_history = good
        purpose = furniture/appliances
        age > 24
        age <= 31
        other_credit = none
        job = skilled
        phone = FALSE
        ->  class no  [0.920]

    Rule 60/10: (9, lift 1.8)
        checking_balance = unknown
        months_loan_duration > 24
        employment_duration = > 7 years
        ->  class no  [0.909]

    Rule 60/11: (8.5, lift 1.8)
        credit_history = good
        purpose = car
        amount <= 11054
        employment_duration = unemployed
        ->  class no  [0.905]

    Rule 60/12: (12.3/0.4, lift 1.8)
        months_loan_duration <= 42
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        age > 44
        job = skilled
        ->  class no  [0.905]

    Rule 60/13: (8.5, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        purpose = furniture/appliances
        amount <= 701
        savings_balance = < 100 DM
        ->  class no  [0.904]

    Rule 60/14: (8.4, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        savings_balance = 100 - 500 DM
        other_credit = none
        ->  class no  [0.904]

    Rule 60/15: (8.3, lift 1.8)
        checking_balance = unknown
        employment_duration = 1 - 4 years
        percent_of_income <= 1
        ->  class no  [0.903]

    Rule 60/16: (7.4, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = good
        purpose = furniture/appliances
        amount <= 1901
        savings_balance = < 100 DM
        other_credit = none
        job = unskilled
        ->  class no  [0.894]

    Rule 60/17: (7.3, lift 1.8)
        amount <= 409
        ->  class no  [0.892]

    Rule 60/18: (7.2, lift 1.8)
        credit_history = very good
        amount > 8072
        ->  class no  [0.891]

    Rule 60/19: (6.6, lift 1.8)
        credit_history = critical
        savings_balance = < 100 DM
        percent_of_income <= 1
        other_credit = none
        job = skilled
        ->  class no  [0.884]

    Rule 60/20: (5.9, lift 1.7)
        credit_history = poor
        savings_balance = unknown
        housing in {other, own}
        ->  class no  [0.874]

    Rule 60/21: (5.8, lift 1.7)
        credit_history = perfect
        savings_balance in {500 - 1000 DM, unknown}
        ->  class no  [0.872]

    Rule 60/22: (5.3, lift 1.7)
        months_loan_duration <= 42
        purpose = furniture/appliances
        savings_balance = > 1000 DM
        ->  class no  [0.864]

    Rule 60/23: (5.4, lift 1.7)
        credit_history = very good
        age <= 23
        ->  class no  [0.864]

    Rule 60/24: (5.1, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = poor
        housing = own
        job = unskilled
        ->  class no  [0.860]

    Rule 60/25: (24/3.2, lift 1.7)
        credit_history = good
        purpose = car
        amount <= 11054
        employment_duration in {> 7 years, 4 - 7 years}
        age <= 44
        other_credit = none
        ->  class no  [0.838]

    Rule 60/26: (8.5/0.7, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = poor
        housing = rent
        ->  class no  [0.837]

    Rule 60/27: (4.1, lift 1.7)
        credit_history = good
        purpose = education
        age > 48
        ->  class no  [0.837]

    Rule 60/28: (10.9/1.3, lift 1.6)
        credit_history = perfect
        age > 29
        age <= 33
        ->  class no  [0.821]

    Rule 60/29: (3.6, lift 1.6)
        credit_history = good
        purpose = education
        existing_loans_count > 1
        ->  class no  [0.821]

    Rule 60/30: (32.8/7.1, lift 1.5)
        checking_balance = unknown
        employment_duration = 4 - 7 years
        ->  class no  [0.768]

    Rule 60/31: (12.6/2.4, lift 1.5)
        credit_history = critical
        savings_balance = < 100 DM
        dependents > 1
        ->  class no  [0.766]

    Rule 60/32: (34.4/8, lift 1.5)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = good
        purpose = car
        amount <= 11054
        percent_of_income <= 2
        age <= 44
        other_credit = none
        ->  class no  [0.754]

    Rule 60/33: (75/28.3, lift 1.2)
        checking_balance = unknown
        employment_duration = 1 - 4 years
        other_credit = none
        ->  class no  [0.620]

    Rule 60/34: (679.1/328.9, lift 1.0)
        months_loan_duration <= 27
        ->  class no  [0.516]

    Default class: yes

    -----  Trial 61:  -----

    Rules:

    Rule 61/1: (16.4/0.6, lift 2.0)
        checking_balance = 1 - 200 DM
        months_loan_duration > 8
        credit_history = good
        amount <= 1928
        employment_duration = 1 - 4 years
        age > 28
        housing = own
        existing_loans_count <= 1
        ->  class yes  [0.913]

    Rule 61/2: (24.3/1.7, lift 2.0)
        checking_balance = unknown
        months_loan_duration > 8
        savings_balance in {< 100 DM, 500 - 1000 DM, unknown}
        employment_duration in {< 1 year, 1 - 4 years, unemployed}
        percent_of_income > 1
        housing = rent
        dependents <= 1
        ->  class yes  [0.897]

    Rule 61/3: (9.9/0.2, lift 2.0)
        months_loan_duration > 8
        age > 50
        dependents > 1
        ->  class yes  [0.896]

    Rule 61/4: (7.2, lift 2.0)
        employment_duration = unemployed
        years_at_residence <= 1
        ->  class yes  [0.891]

    Rule 61/5: (13.1/0.8, lift 1.9)
        checking_balance = < 0 DM
        purpose in {business, education, renovations}
        savings_balance = < 100 DM
        job = skilled
        ->  class yes  [0.881]

    Rule 61/6: (17.3/1.4, lift 1.9)
        months_loan_duration > 8
        credit_history in {perfect, very good}
        amount <= 7308
        dependents > 1
        ->  class yes  [0.879]

    Rule 61/7: (21.3/2, lift 1.9)
        months_loan_duration > 8
        amount <= 1264
        dependents > 1
        ->  class yes  [0.870]

    Rule 61/8: (5.6, lift 1.9)
        checking_balance = 1 - 200 DM
        employment_duration = < 1 year
        other_credit = store
        ->  class yes  [0.868]

    Rule 61/9: (24.3/2.7, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration > 30
        job = skilled
        dependents <= 1
        ->  class yes  [0.858]

    Rule 61/10: (5, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration > 8
        savings_balance = unknown
        job = management
        ->  class yes  [0.857]

    Rule 61/11: (6.5/0.3, lift 1.9)
        checking_balance = 1 - 200 DM
        employment_duration = > 7 years
        age > 49
        ->  class yes  [0.849]

    Rule 61/12: (11.8/1.5, lift 1.8)
        checking_balance = 1 - 200 DM
        employment_duration = > 7 years
        job = management
        ->  class yes  [0.821]

    Rule 61/13: (20.1/3.2, lift 1.8)
        credit_history = good
        purpose = furniture/appliances
        amount > 1300
        housing = own
        existing_loans_count > 1
        dependents <= 1
        ->  class yes  [0.809]

    Rule 61/14: (42.9/11.6, lift 1.6)
        checking_balance = < 0 DM
        months_loan_duration > 8
        purpose = furniture/appliances
        amount > 1092
        amount <= 4153
        savings_balance = < 100 DM
        years_at_residence > 1
        job = skilled
        dependents <= 1
        ->  class yes  [0.721]

    Rule 61/15: (48.7/14.6, lift 1.5)
        checking_balance = unknown
        months_loan_duration > 8
        months_loan_duration <= 21
        amount > 1300
        years_at_residence <= 2
        housing = own
        ->  class yes  [0.692]

    Rule 61/16: (62.2/22.3, lift 1.4)
        months_loan_duration > 8
        purpose = car
        amount <= 1386
        ->  class yes  [0.637]

    Rule 61/17: (239.6/114.5, lift 1.2)
        amount > 4057
        ->  class yes  [0.522]

    Rule 61/18: (761/407.1, lift 1.0)
        years_at_residence > 1
        ->  class yes  [0.465]

    Rule 61/19: (17.4, lift 1.7)
        checking_balance = 1 - 200 DM
        employment_duration = < 1 year
        years_at_residence <= 1
        other_credit in {bank, none}
        housing = own
        ->  class no  [0.949]

    Rule 61/20: (15, lift 1.7)
        checking_balance = unknown
        amount <= 1300
        housing = own
        dependents <= 1
        ->  class no  [0.941]

    Rule 61/21: (10.5, lift 1.7)
        checking_balance = 1 - 200 DM
        employment_duration = 4 - 7 years
        housing = rent
        ->  class no  [0.920]

    Rule 61/22: (29.8/1.6, lift 1.7)
        credit_history in {critical, good, poor}
        amount > 1264
        amount <= 9857
        years_at_residence > 2
        age <= 50
        dependents > 1
        ->  class no  [0.918]

    Rule 61/23: (21.4/1.2, lift 1.7)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 22
        employment_duration = 4 - 7 years
        other_credit = none
        ->  class no  [0.905]

    Rule 61/24: (8.3, lift 1.7)
        checking_balance = unknown
        employment_duration in {> 7 years, 4 - 7 years}
        housing = rent
        ->  class no  [0.903]

    Rule 61/25: (8.2, lift 1.6)
        checking_balance = unknown
        credit_history = good
        purpose = car
        amount > 1388
        housing = own
        ->  class no  [0.902]

    Rule 61/26: (6.9, lift 1.6)
        checking_balance = < 0 DM
        savings_balance = unknown
        job = skilled
        dependents <= 1
        phone = TRUE
        ->  class no  [0.887]

    Rule 61/27: (12.7/0.8, lift 1.6)
        checking_balance = unknown
        months_loan_duration > 21
        credit_history = critical
        housing = own
        ->  class no  [0.876]

    Rule 61/28: (5.8, lift 1.6)
        checking_balance = unknown
        credit_history = poor
        years_at_residence > 2
        housing = own
        ->  class no  [0.872]

    Rule 61/29: (19.6/2, lift 1.6)
        checking_balance = 1 - 200 DM
        months_loan_duration > 8
        employment_duration = unemployed
        years_at_residence > 1
        dependents <= 1
        ->  class no  [0.862]

    Rule 61/30: (25.5/3.4, lift 1.5)
        checking_balance = < 0 DM
        months_loan_duration > 16
        savings_balance = < 100 DM
        job = management
        dependents <= 1
        ->  class no  [0.840]

    Rule 61/31: (53.5/10.1, lift 1.5)
        months_loan_duration <= 8
        amount <= 4057
        ->  class no  [0.800]

    Rule 61/32: (41.3/8.2, lift 1.4)
        checking_balance = > 200 DM
        credit_history in {good, perfect}
        dependents <= 1
        ->  class no  [0.787]

    Rule 61/33: (36.4/8, lift 1.4)
        checking_balance = 1 - 200 DM
        employment_duration = > 7 years
        job in {skilled, unskilled}
        dependents <= 1
        ->  class no  [0.765]

    Rule 61/34: (840.7/393.3, lift 1.0)
        months_loan_duration > 8
        ->  class no  [0.532]

    Default class: no

    -----  Trial 62:  -----

    Rules:

    Rule 62/1: (11.2, lift 2.0)
        credit_history = good
        purpose = car
        housing in {other, rent}
        existing_loans_count > 1
        ->  class yes  [0.924]

    Rule 62/2: (9, lift 2.0)
        credit_history = very good
        amount > 4530
        amount <= 7629
        ->  class yes  [0.909]

    Rule 62/3: (8.1, lift 2.0)
        credit_history = critical
        purpose = car
        savings_balance = < 100 DM
        years_at_residence > 1
        age <= 29
        ->  class yes  [0.901]

    Rule 62/4: (8.5/0.2, lift 1.9)
        months_loan_duration <= 33
        credit_history = good
        employment_duration = 4 - 7 years
        age <= 24
        job = skilled
        phone = FALSE
        ->  class yes  [0.886]

    Rule 62/5: (7.9/0.1, lift 1.9)
        credit_history = good
        purpose = renovations
        percent_of_income > 3
        ->  class yes  [0.886]

    Rule 62/6: (5.2, lift 1.9)
        credit_history = poor
        purpose = renovations
        ->  class yes  [0.862]

    Rule 62/7: (8.7/0.7, lift 1.8)
        credit_history = good
        purpose = furniture/appliances
        years_at_residence <= 3
        existing_loans_count > 1
        job = skilled
        phone = TRUE
        ->  class yes  [0.845]

    Rule 62/8: (4, lift 1.8)
        credit_history = poor
        purpose = business
        years_at_residence <= 1
        ->  class yes  [0.835]

    Rule 62/9: (17.2/2.3, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        purpose = furniture/appliances
        employment_duration = 1 - 4 years
        age <= 24
        job = skilled
        phone = FALSE
        ->  class yes  [0.830]

    Rule 62/10: (6.2/0.4, lift 1.8)
        months_loan_duration > 33
        credit_history = good
        purpose = furniture/appliances
        existing_loans_count > 1
        ->  class yes  [0.827]

    Rule 62/11: (28.2/4.3, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = very good
        amount <= 3556
        phone = FALSE
        ->  class yes  [0.826]

    Rule 62/12: (14.7/2, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 33
        credit_history = good
        purpose = furniture/appliances
        percent_of_income > 1
        years_at_residence <= 3
        job = skilled
        phone = TRUE
        ->  class yes  [0.819]

    Rule 62/13: (21.8/3.5, lift 1.8)
        checking_balance = < 0 DM
        months_loan_duration <= 40
        credit_history = good
        purpose = car
        years_at_residence <= 2
        ->  class yes  [0.811]

    Rule 62/14: (13.8/2.1, lift 1.8)
        purpose = furniture/appliances
        employment_duration = < 1 year
        years_at_residence > 2
        age > 24
        job = skilled
        ->  class yes  [0.804]

    Rule 62/15: (11/1.6, lift 1.8)
        months_loan_duration <= 33
        credit_history = good
        purpose = furniture/appliances
        employment_duration = unemployed
        ->  class yes  [0.801]

    Rule 62/16: (14.5/2.3, lift 1.7)
        months_loan_duration > 18
        credit_history = good
        purpose = business
        amount <= 5293
        ->  class yes  [0.798]

    Rule 62/17: (12.3/2, lift 1.7)
        checking_balance = 1 - 200 DM
        credit_history = good
        amount > 8335
        ->  class yes  [0.792]

    Rule 62/18: (24.6/4.5, lift 1.7)
        months_loan_duration > 33
        months_loan_duration <= 39
        credit_history = good
        percent_of_income > 2
        age > 25
        existing_loans_count <= 1
        ->  class yes  [0.792]

    Rule 62/19: (13.9/2.8, lift 1.7)
        credit_history = poor
        purpose = furniture/appliances
        percent_of_income > 3
        ->  class yes  [0.762]

    Rule 62/20: (34.3/10, lift 1.5)
        credit_history = perfect
        savings_balance in {< 100 DM, 100 - 500 DM}
        ->  class yes  [0.698]

    Rule 62/21: (15.2/4.6, lift 1.5)
        years_at_residence <= 1
        housing = rent
        ->  class yes  [0.674]

    Rule 62/22: (62.6/21, lift 1.4)
        months_loan_duration <= 33
        purpose = furniture/appliances
        percent_of_income > 1
        existing_loans_count <= 1
        job = unskilled
        ->  class yes  [0.659]

    Rule 62/23: (40.7/16.4, lift 1.3)
        credit_history = perfect
        ->  class yes  [0.592]

    Rule 62/24: (301.2/152.5, lift 1.1)
        checking_balance = < 0 DM
        ->  class yes  [0.494]

    Rule 62/25: (15.8, lift 1.7)
        credit_history = critical
        years_at_residence <= 1
        housing in {other, own}
        ->  class no  [0.944]

    Rule 62/26: (13.4, lift 1.7)
        credit_history = critical
        age > 60
        ->  class no  [0.935]

    Rule 62/27: (12, lift 1.7)
        months_loan_duration <= 18
        credit_history = good
        purpose = business
        ->  class no  [0.928]

    Rule 62/28: (11, lift 1.7)
        months_loan_duration <= 33
        credit_history = good
        purpose = furniture/appliances
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, 4 - 7 years}
        age <= 38
        job = management
        ->  class no  [0.923]

    Rule 62/29: (11, lift 1.7)
        purpose = furniture/appliances
        percent_of_income <= 1
        job = unskilled
        ->  class no  [0.923]

    Rule 62/30: (11.1/0.1, lift 1.7)
        credit_history = poor
        purpose = car
        years_at_residence <= 3
        ->  class no  [0.914]

    Rule 62/31: (47.2/4, lift 1.7)
        checking_balance = unknown
        credit_history = critical
        amount <= 11816
        other_credit = none
        ->  class no  [0.899]

    Rule 62/32: (7.2, lift 1.6)
        credit_history = very good
        amount > 7629
        ->  class no  [0.891]

    Rule 62/33: (6.4, lift 1.6)
        credit_history = perfect
        savings_balance in {500 - 1000 DM, unknown}
        ->  class no  [0.882]

    Rule 62/34: (35.8/4.3, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = critical
        purpose = car
        age > 29
        other_credit = none
        ->  class no  [0.860]

    Rule 62/35: (3.9, lift 1.5)
        credit_history = critical
        other_credit = bank
        job = management
        ->  class no  [0.830]

    Rule 62/36: (38/7, lift 1.5)
        credit_history = critical
        purpose = furniture/appliances
        savings_balance = < 100 DM
        age <= 48
        job = skilled
        ->  class no  [0.799]

    Rule 62/37: (30.4/6.6, lift 1.4)
        checking_balance = < 0 DM
        purpose = car
        years_at_residence > 2
        age <= 43
        existing_loans_count <= 1
        ->  class no  [0.766]

    Rule 62/38: (45.6/12.6, lift 1.3)
        months_loan_duration > 33
        percent_of_income <= 2
        existing_loans_count <= 1
        ->  class no  [0.714]

    Rule 62/39: (51.2/17, lift 1.2)
        existing_loans_count > 1
        job = unskilled
        ->  class no  [0.662]

    Rule 62/40: (86/29.9, lift 1.2)
        checking_balance in {> 200 DM, unknown}
        purpose = car
        ->  class no  [0.649]

    Rule 62/41: (447.7/185.7, lift 1.1)
        months_loan_duration <= 33
        job = skilled
        ->  class no  [0.585]

    Default class: no

    -----  Trial 63:  -----

    Rules:

    Rule 63/1: (19.1/0.5, lift 2.0)
        checking_balance = 1 - 200 DM
        months_loan_duration > 8
        savings_balance = < 100 DM
        years_at_residence > 1
        years_at_residence <= 2
        job = skilled
        ->  class yes  [0.931]

    Rule 63/2: (16.2/0.6, lift 1.9)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 14
        months_loan_duration <= 36
        savings_balance = < 100 DM
        percent_of_income <= 3
        other_credit in {bank, none}
        housing = own
        job = unskilled
        ->  class yes  [0.910]

    Rule 63/3: (6.6, lift 1.9)
        years_at_residence <= 3
        other_credit = store
        job = management
        ->  class yes  [0.883]

    Rule 63/4: (14.4/1.5, lift 1.8)
        credit_history in {perfect, poor, very good}
        savings_balance = < 100 DM
        housing = own
        job = management
        ->  class yes  [0.847]

    Rule 63/5: (8.8/0.7, lift 1.8)
        employment_duration = < 1 year
        age <= 33
        housing = rent
        existing_loans_count > 1
        job = skilled
        ->  class yes  [0.844]

    Rule 63/6: (18.8/2.4, lift 1.8)
        employment_duration = < 1 year
        age > 22
        age <= 33
        housing = rent
        job = skilled
        ->  class yes  [0.834]

    Rule 63/7: (25.4/4, lift 1.7)
        checking_balance = < 0 DM
        months_loan_duration > 16
        credit_history = good
        savings_balance = < 100 DM
        years_at_residence <= 3
        other_credit in {bank, none}
        housing = own
        job = skilled
        ->  class yes  [0.819]

    Rule 63/8: (12.9/1.9, lift 1.7)
        checking_balance = 1 - 200 DM
        purpose = business
        savings_balance = < 100 DM
        years_at_residence > 1
        ->  class yes  [0.809]

    Rule 63/9: (3.2, lift 1.7)
        savings_balance = unknown
        years_at_residence <= 1
        job = management
        ->  class yes  [0.806]

    Rule 63/10: (12.3/1.8, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        savings_balance = < 100 DM
        years_at_residence <= 3
        job = management
        ->  class yes  [0.804]

    Rule 63/11: (33.7/6.6, lift 1.7)
        savings_balance in {100 - 500 DM, 500 - 1000 DM, unknown}
        percent_of_income > 3
        job = unskilled
        ->  class yes  [0.788]

    Rule 63/12: (11.2/1.9, lift 1.6)
        checking_balance = < 0 DM
        credit_history in {poor, very good}
        housing = own
        job = skilled
        ->  class yes  [0.777]

    Rule 63/13: (20/4.2, lift 1.6)
        amount <= 1449
        housing = other
        ->  class yes  [0.763]

    Rule 63/14: (24.3/5.4, lift 1.6)
        percent_of_income > 3
        job = unskilled
        dependents > 1
        ->  class yes  [0.758]

    Rule 63/15: (16.7/3.6, lift 1.6)
        savings_balance in {> 1000 DM, 100 - 500 DM}
        housing = own
        job = management
        ->  class yes  [0.755]

    Rule 63/16: (30/7.9, lift 1.5)
        checking_balance in {> 200 DM, 1 - 200 DM}
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        housing = other
        ->  class yes  [0.723]

    Rule 63/17: (49.1/13.5, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM, unknown}
        credit_history in {critical, good, perfect, very good}
        employment_duration = 1 - 4 years
        housing = rent
        dependents <= 1
        ->  class yes  [0.717]

    Rule 63/18: (59.4/18, lift 1.5)
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        percent_of_income > 2
        housing = other
        ->  class yes  [0.691]

    Rule 63/19: (16.9/5, lift 1.4)
        credit_history = critical
        savings_balance = 500 - 1000 DM
        ->  class yes  [0.681]

    Rule 63/20: (433.2/205.7, lift 1.1)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 14
        ->  class yes  [0.525]

    Rule 63/21: (10.7, lift 1.7)
        checking_balance in {> 200 DM, 1 - 200 DM, unknown}
        employment_duration = 4 - 7 years
        housing = rent
        ->  class no  [0.921]

    Rule 63/22: (16.1/0.9, lift 1.7)
        amount > 1449
        employment_duration in {< 1 year, unemployed}
        housing = other
        ->  class no  [0.894]

    Rule 63/23: (5.4, lift 1.6)
        months_loan_duration <= 22
        employment_duration = < 1 year
        housing = rent
        job in {management, unskilled}
        ->  class no  [0.866]

    Rule 63/24: (5.6/0.2, lift 1.6)
        employment_duration = < 1 year
        age > 33
        housing = rent
        ->  class no  [0.838]

    Rule 63/25: (3.8, lift 1.6)
        credit_history = poor
        employment_duration = 1 - 4 years
        housing = rent
        ->  class no  [0.827]

    Rule 63/26: (70.4/22.7, lift 1.3)
        checking_balance in {> 200 DM, unknown}
        employment_duration = > 7 years
        ->  class no  [0.672]

    Rule 63/27: (328.2/141.4, lift 1.1)
        percent_of_income <= 2
        ->  class no  [0.569]

    Rule 63/28: (636.8/282.8, lift 1.1)
        housing = own
        ->  class no  [0.556]

    Default class: no

    -----  Trial 64:  -----

    Rules:

    Rule 64/1: (6.9, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        purpose = car
        savings_balance = < 100 DM
        age <= 29
        ->  class yes  [0.888]

    Rule 64/2: (6.1, lift 1.8)
        checking_balance = unknown
        purpose = furniture/appliances
        amount > 4594
        employment_duration = < 1 year
        ->  class yes  [0.877]

    Rule 64/3: (14.6/1.3, lift 1.8)
        checking_balance in {> 200 DM, unknown}
        purpose in {business, renovations}
        employment_duration = < 1 year
        ->  class yes  [0.859]

    Rule 64/4: (16.7/2.9, lift 1.6)
        checking_balance = unknown
        purpose = car
        employment_duration = 1 - 4 years
        age <= 27
        ->  class yes  [0.793]

    Rule 64/5: (44.2/12.8, lift 1.4)
        checking_balance in {> 200 DM, unknown}
        years_at_residence > 1
        age <= 44
        other_credit = bank
        ->  class yes  [0.701]

    Rule 64/6: (33.6/10.6, lift 1.4)
        purpose = business
        other_credit in {none, store}
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.675]

    Rule 64/7: (63.5/26.5, lift 1.2)
        purpose = education
        ->  class yes  [0.581]

    Rule 64/8: (582.5/276.8, lift 1.1)
        checking_balance in {< 0 DM, 1 - 200 DM}
        ->  class yes  [0.525]

    Rule 64/9: (25.8, lift 1.9)
        checking_balance in {> 200 DM, unknown}
        purpose in {education, furniture/appliances}
        age > 44
        ->  class no  [0.964]

    Rule 64/10: (15.1, lift 1.8)
        months_loan_duration <= 15
        credit_history = good
        amount > 1347
        savings_balance in {< 100 DM, unknown}
        employment_duration = 1 - 4 years
        job = skilled
        phone = FALSE
        ->  class no  [0.942]

    Rule 64/11: (13.1, lift 1.8)
        credit_history = good
        purpose = car
        amount > 1386
        amount <= 8613
        savings_balance = < 100 DM
        percent_of_income <= 2
        dependents <= 1
        ->  class no  [0.934]

    Rule 64/12: (11.9, lift 1.8)
        checking_balance = < 0 DM
        months_loan_duration > 21
        months_loan_duration <= 36
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        phone = FALSE
        ->  class no  [0.928]

    Rule 64/13: (11, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 11
        purpose = furniture/appliances
        job = unskilled
        ->  class no  [0.923]

    Rule 64/14: (10.4, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 36
        credit_history = critical
        purpose = furniture/appliances
        percent_of_income > 1
        phone = TRUE
        ->  class no  [0.919]

    Rule 64/15: (8.2, lift 1.8)
        checking_balance = 1 - 200 DM
        months_loan_duration > 42
        savings_balance = unknown
        ->  class no  [0.902]

    Rule 64/16: (6.9, lift 1.7)
        months_loan_duration <= 36
        purpose = furniture/appliances
        savings_balance = > 1000 DM
        ->  class no  [0.887]

    Rule 64/17: (12.7/0.8, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        savings_balance = unknown
        other_credit = bank
        ->  class no  [0.875]

    Rule 64/18: (5.7, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        savings_balance = 500 - 1000 DM
        percent_of_income > 1
        job = skilled
        ->  class no  [0.871]

    Rule 64/19: (15.5/1.3, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 36
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years, unemployed}
        percent_of_income > 1
        job = management
        ->  class no  [0.868]

    Rule 64/20: (5.3, lift 1.7)
        checking_balance = < 0 DM
        purpose = car
        savings_balance = 100 - 500 DM
        ->  class no  [0.863]

    Rule 64/21: (4.5, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        savings_balance = unknown
        job = unskilled
        ->  class no  [0.847]

    Rule 64/22: (17.4/2.1, lift 1.6)
        months_loan_duration <= 36
        credit_history = critical
        purpose = furniture/appliances
        savings_balance in {< 100 DM, unknown}
        age <= 30
        job = skilled
        phone = FALSE
        ->  class no  [0.838]

    Rule 64/23: (13.5/1.7, lift 1.6)
        checking_balance in {> 200 DM, unknown}
        purpose in {car, education}
        employment_duration = < 1 year
        ->  class no  [0.829]

    Rule 64/24: (30.9/6, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        other_credit in {bank, none}
        job = unskilled
        ->  class no  [0.786]

    Rule 64/25: (36.7/10.4, lift 1.4)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = business
        housing = own
        ->  class no  [0.705]

    Rule 64/26: (60/20.2, lift 1.3)
        credit_history = critical
        purpose = car
        other_credit = none
        ->  class no  [0.658]

    Rule 64/27: (317.5/135.4, lift 1.1)
        checking_balance in {> 200 DM, unknown}
        ->  class no  [0.573]

    Default class: no

    -----  Trial 65:  -----

    Rules:

    Rule 65/1: (13.9, lift 2.1)
        months_loan_duration > 8
        credit_history = good
        amount > 1766
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 1 - 4 years}
        percent_of_income > 1
        years_at_residence > 1
        age > 30
        housing = own
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.937]

    Rule 65/2: (12.7, lift 2.1)
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        percent_of_income > 1
        years_at_residence > 1
        age <= 22
        phone = FALSE
        ->  class yes  [0.932]

    Rule 65/3: (12.7, lift 2.1)
        months_loan_duration > 8
        amount > 1766
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 1 - 4 years}
        percent_of_income > 1
        years_at_residence > 1
        housing = rent
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.932]

    Rule 65/4: (19.2/1.4, lift 2.0)
        months_loan_duration > 42
        percent_of_income > 1
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.886]

    Rule 65/5: (6.1, lift 2.0)
        employment_duration = unemployed
        years_at_residence <= 1
        ->  class yes  [0.877]

    Rule 65/6: (14.4/1.5, lift 1.9)
        months_loan_duration > 13
        purpose = furniture/appliances
        savings_balance = < 100 DM
        percent_of_income > 1
        dependents > 1
        phone = FALSE
        ->  class yes  [0.850]

    Rule 65/7: (4.7, lift 1.9)
        purpose = business
        employment_duration = unemployed
        ->  class yes  [0.850]

    Rule 65/8: (9.3/0.7, lift 1.9)
        credit_history in {perfect, very good}
        purpose = car
        percent_of_income > 1
        other_credit = none
        phone = FALSE
        ->  class yes  [0.846]

    Rule 65/9: (16.3/1.8, lift 1.9)
        months_loan_duration > 28
        credit_history = good
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years}
        percent_of_income > 1
        years_at_residence > 1
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.846]

    Rule 65/10: (21.4/2.7, lift 1.9)
        months_loan_duration > 8
        purpose = car
        percent_of_income > 1
        other_credit = bank
        phone = FALSE
        ->  class yes  [0.840]

    Rule 65/11: (4.2, lift 1.9)
        months_loan_duration <= 8
        amount > 4057
        existing_loans_count <= 1
        ->  class yes  [0.838]

    Rule 65/12: (6.8/0.4, lift 1.9)
        months_loan_duration > 8
        purpose = business
        dependents > 1
        phone = FALSE
        ->  class yes  [0.836]

    Rule 65/13: (13.2/1.5, lift 1.9)
        checking_balance = < 0 DM
        credit_history = good
        purpose = car
        years_at_residence <= 2
        phone = FALSE
        ->  class yes  [0.834]

    Rule 65/14: (13.4/1.6, lift 1.9)
        months_loan_duration > 8
        purpose = renovations
        amount <= 2483
        phone = FALSE
        ->  class yes  [0.834]

    Rule 65/15: (5.2/0.2, lift 1.9)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        savings_balance = unknown
        phone = FALSE
        ->  class yes  [0.831]

    Rule 65/16: (19.8/2.8, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        purpose = furniture/appliances
        amount > 1835
        percent_of_income > 1
        phone = FALSE
        ->  class yes  [0.827]

    Rule 65/17: (15.5/2.2, lift 1.8)
        credit_history in {critical, poor}
        purpose = car
        employment_duration = > 7 years
        percent_of_income > 1
        age <= 58
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.819]

    Rule 65/18: (10.3/1.3, lift 1.8)
        purpose = furniture/appliances
        savings_balance = unknown
        employment_duration = 1 - 4 years
        percent_of_income > 1
        phone = TRUE
        ->  class yes  [0.816]

    Rule 65/19: (4/0.1, lift 1.8)
        months_loan_duration > 8
        employment_duration = 4 - 7 years
        existing_loans_count > 2
        ->  class yes  [0.815]

    Rule 65/20: (12.6/2.3, lift 1.7)
        purpose in {business, education}
        employment_duration = > 7 years
        percent_of_income > 1
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.777]

    Rule 65/21: (2.3, lift 1.7)
        purpose = business
        job = unemployed
        ->  class yes  [0.770]

    Rule 65/22: (20.6/4.2, lift 1.7)
        months_loan_duration > 8
        purpose = education
        amount > 709
        savings_balance in {< 100 DM, > 1000 DM, 100 - 500 DM}
        phone = FALSE
        ->  class yes  [0.768]

    Rule 65/23: (22.3/4.7, lift 1.7)
        months_loan_duration > 13
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        percent_of_income > 1
        age <= 31
        phone = TRUE
        ->  class yes  [0.765]

    Rule 65/24: (12.2/2.4, lift 1.7)
        amount > 9283
        percent_of_income <= 1
        dependents <= 1
        ->  class yes  [0.757]

    Rule 65/25: (9.8/2, lift 1.7)
        amount > 3949
        existing_loans_count > 2
        ->  class yes  [0.746]

    Rule 65/26: (36.5/10.5, lift 1.6)
        months_loan_duration > 8
        purpose = car
        percent_of_income > 1
        existing_loans_count > 1
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.700]

    Rule 65/27: (19.6/5.9, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        ->  class yes  [0.683]

    Rule 65/28: (11.8, lift 1.7)
        months_loan_duration <= 8
        existing_loans_count > 1
        ->  class no  [0.928]

    Rule 65/29: (12/1.4, lift 1.5)
        credit_history = poor
        purpose = car
        phone = FALSE
        ->  class no  [0.826]

    Rule 65/30: (54.1/10.5, lift 1.4)
        months_loan_duration <= 8
        amount <= 4057
        ->  class no  [0.795]

    Rule 65/31: (840.4/385.6, lift 1.0)
        months_loan_duration > 8
        ->  class no  [0.541]

    Default class: no

    -----  Trial 66:  -----

    Rules:

    Rule 66/1: (8.8, lift 1.9)
        credit_history = perfect
        other_credit = none
        housing in {other, rent}
        ->  class yes  [0.907]

    Rule 66/2: (12.6/0.6, lift 1.8)
        credit_history = good
        amount > 7418
        employment_duration = > 7 years
        ->  class yes  [0.888]

    Rule 66/3: (7.6/0.2, lift 1.8)
        credit_history = good
        purpose = car
        amount <= 1480
        employment_duration = 4 - 7 years
        ->  class yes  [0.870]

    Rule 66/4: (14.9/1.2, lift 1.8)
        months_loan_duration <= 22
        credit_history = good
        housing = rent
        existing_loans_count > 1
        ->  class yes  [0.867]

    Rule 66/5: (5, lift 1.8)
        employment_duration = unemployed
        years_at_residence <= 1
        ->  class yes  [0.857]

    Rule 66/6: (16.2/1.8, lift 1.8)
        credit_history = good
        amount <= 1249
        employment_duration = > 7 years
        age > 28
        existing_loans_count <= 1
        phone = FALSE
        ->  class yes  [0.845]

    Rule 66/7: (5.9/0.2, lift 1.8)
        credit_history = good
        amount <= 1175
        employment_duration = unemployed
        ->  class yes  [0.845]

    Rule 66/8: (16.9/2.3, lift 1.7)
        credit_history = good
        amount > 10222
        years_at_residence > 1
        ->  class yes  [0.825]

    Rule 66/9: (9.9/1.3, lift 1.7)
        credit_history = poor
        years_at_residence <= 1
        ->  class yes  [0.811]

    Rule 66/10: (15.6/2.6, lift 1.7)
        months_loan_duration <= 24
        credit_history = good
        employment_duration = > 7 years
        age > 28
        existing_loans_count > 1
        ->  class yes  [0.794]

    Rule 66/11: (19.8/3.5, lift 1.6)
        credit_history = poor
        employment_duration in {< 1 year, unemployed}
        ->  class yes  [0.793]

    Rule 66/12: (19.5/3.5, lift 1.6)
        credit_history = critical
        years_at_residence > 1
        age <= 46
        other_credit = bank
        job in {skilled, unskilled}
        ->  class yes  [0.792]

    Rule 66/13: (14.2/2.8, lift 1.6)
        credit_history = critical
        amount > 6967
        other_credit = none
        ->  class yes  [0.766]

    Rule 66/14: (9.5/1.8, lift 1.6)
        credit_history = poor
        age > 52
        ->  class yes  [0.760]

    Rule 66/15: (21.8/4.9, lift 1.6)
        purpose = furniture/appliances
        employment_duration = < 1 year
        other_credit in {none, store}
        job = unskilled
        ->  class yes  [0.751]

    Rule 66/16: (50.6/13.9, lift 1.5)
        months_loan_duration <= 39
        credit_history = very good
        age > 23
        ->  class yes  [0.716]

    Rule 66/17: (47.4/18.2, lift 1.3)
        other_credit = store
        ->  class yes  [0.612]

    Rule 66/18: (183.6/82.8, lift 1.1)
        credit_history = good
        employment_duration = 1 - 4 years
        ->  class yes  [0.548]

    Rule 66/19: (548.3/266.1, lift 1.1)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        age <= 46
        ->  class yes  [0.515]

    Rule 66/20: (24.1, lift 1.9)
        checking_balance in {> 200 DM, 1 - 200 DM, unknown}
        credit_history = good
        amount > 1249
        amount <= 7418
        employment_duration = > 7 years
        existing_loans_count <= 1
        ->  class no  [0.962]

    Rule 66/21: (13.3, lift 1.8)
        credit_history = good
        amount > 1175
        amount <= 10222
        employment_duration = unemployed
        years_at_residence > 1
        other_credit = none
        ->  class no  [0.935]

    Rule 66/22: (48.4/2.6, lift 1.8)
        checking_balance = unknown
        credit_history = critical
        amount <= 6967
        other_credit = none
        ->  class no  [0.928]

    Rule 66/23: (10.1, lift 1.8)
        credit_history = good
        employment_duration = > 7 years
        age <= 28
        ->  class no  [0.918]

    Rule 66/24: (25.1/1.7, lift 1.7)
        months_loan_duration <= 22
        credit_history = good
        amount > 1680
        other_credit = none
        housing = rent
        existing_loans_count <= 1
        ->  class no  [0.900]

    Rule 66/25: (7.9, lift 1.7)
        purpose = education
        savings_balance = unknown
        employment_duration = 1 - 4 years
        ->  class no  [0.899]

    Rule 66/26: (32.7/3, lift 1.7)
        credit_history = good
        amount > 1249
        amount <= 7418
        employment_duration = > 7 years
        years_at_residence > 2
        existing_loans_count <= 1
        ->  class no  [0.885]

    Rule 66/27: (6.2, lift 1.7)
        months_loan_duration > 27
        credit_history = critical
        purpose = furniture/appliances
        percent_of_income > 2
        job = skilled
        ->  class no  [0.878]

    Rule 66/28: (5, lift 1.6)
        credit_history = critical
        other_credit = bank
        job = management
        ->  class no  [0.857]

    Rule 66/29: (39.2/5.3, lift 1.6)
        credit_history = critical
        age > 46
        ->  class no  [0.847]

    Rule 66/30: (4.5, lift 1.6)
        credit_history = very good
        age <= 23
        ->  class no  [0.845]

    Rule 66/31: (10.4/0.9, lift 1.6)
        credit_history = perfect
        percent_of_income <= 2
        other_credit = none
        housing = own
        ->  class no  [0.844]

    Rule 66/32: (18.4/2.3, lift 1.6)
        credit_history = good
        amount <= 1249
        phone = TRUE
        ->  class no  [0.838]

    Rule 66/33: (13.7/1.9, lift 1.6)
        credit_history = critical
        years_at_residence <= 1
        ->  class no  [0.817]

    Rule 66/34: (9.5/1.2, lift 1.6)
        credit_history = perfect
        other_credit = bank
        dependents <= 1
        ->  class no  [0.813]

    Rule 66/35: (15.4/2.3, lift 1.6)
        purpose = furniture/appliances
        employment_duration = < 1 year
        other_credit = none
        housing = own
        job = skilled
        phone = FALSE
        ->  class no  [0.810]

    Rule 66/36: (5.9/0.5, lift 1.6)
        months_loan_duration > 39
        credit_history = very good
        ->  class no  [0.807]

    Rule 66/37: (43.7/9.1, lift 1.5)
        credit_history = poor
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        years_at_residence > 1
        age <= 52
        ->  class no  [0.778]

    Rule 66/38: (29.8/7, lift 1.4)
        credit_history = good
        purpose in {car, furniture/appliances}
        employment_duration = 1 - 4 years
        existing_loans_count <= 1
        phone = TRUE
        ->  class no  [0.750]

    Rule 66/39: (93.2/31.6, lift 1.3)
        amount > 1546
        savings_balance = unknown
        ->  class no  [0.658]

    Rule 66/40: (73.2/26.1, lift 1.2)
        credit_history = good
        employment_duration = 4 - 7 years
        percent_of_income > 1
        ->  class no  [0.640]

    Default class: no

    -----  Trial 67:  -----

    Rules:

    Rule 67/1: (16.6/0.4, lift 2.2)
        credit_history = poor
        savings_balance = < 100 DM
        percent_of_income > 3
        existing_loans_count <= 1
        ->  class yes  [0.926]

    Rule 67/2: (17.9/0.7, lift 2.1)
        checking_balance = 1 - 200 DM
        credit_history = good
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM}
        percent_of_income > 2
        age > 35
        existing_loans_count <= 1
        ->  class yes  [0.914]

    Rule 67/3: (16.5/0.7, lift 2.1)
        credit_history = good
        purpose = furniture/appliances
        amount > 2319
        housing in {other, own}
        existing_loans_count > 1
        job = skilled
        dependents <= 1
        ->  class yes  [0.908]

    Rule 67/4: (8.5, lift 2.1)
        checking_balance = < 0 DM
        employment_duration = 1 - 4 years
        percent_of_income > 2
        age > 23
        other_credit = bank
        ->  class yes  [0.905]

    Rule 67/5: (28.4/2, lift 2.1)
        checking_balance in {< 0 DM, 1 - 200 DM, unknown}
        credit_history = good
        savings_balance in {< 100 DM, 100 - 500 DM}
        percent_of_income > 2
        years_at_residence > 1
        age <= 23
        ->  class yes  [0.900]

    Rule 67/6: (7.5, lift 2.1)
        credit_history = very good
        savings_balance = < 100 DM
        other_credit = none
        ->  class yes  [0.895]

    Rule 67/7: (11/0.5, lift 2.1)
        credit_history = good
        housing = rent
        existing_loans_count > 1
        job = skilled
        ->  class yes  [0.885]

    Rule 67/8: (7.6/0.1, lift 2.1)
        credit_history = good
        purpose = furniture/appliances
        amount <= 1525
        percent_of_income <= 2
        age <= 35
        housing = own
        ->  class yes  [0.881]

    Rule 67/9: (5.9, lift 2.0)
        credit_history = critical
        amount > 7308
        savings_balance = < 100 DM
        other_credit = none
        ->  class yes  [0.873]

    Rule 67/10: (5.1, lift 2.0)
        credit_history = perfect
        age > 38
        job = skilled
        ->  class yes  [0.858]

    Rule 67/11: (5, lift 2.0)
        checking_balance = < 0 DM
        amount <= 674
        employment_duration = 1 - 4 years
        ->  class yes  [0.857]

    Rule 67/12: (20.8/2.6, lift 2.0)
        checking_balance = < 0 DM
        months_loan_duration > 7
        months_loan_duration <= 40
        credit_history = good
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 4 - 7 years}
        percent_of_income > 2
        years_at_residence > 1
        ->  class yes  [0.844]

    Rule 67/13: (20.2/2.7, lift 1.9)
        months_loan_duration > 40
        credit_history = good
        percent_of_income > 2
        years_at_residence > 1
        ->  class yes  [0.835]

    Rule 67/14: (4, lift 1.9)
        credit_history = critical
        other_credit = bank
        job in {skilled, unskilled}
        dependents > 1
        ->  class yes  [0.832]

    Rule 67/15: (7.7/0.7, lift 1.9)
        months_loan_duration <= 18
        percent_of_income <= 2
        housing = other
        existing_loans_count <= 1
        ->  class yes  [0.824]

    Rule 67/16: (3.5, lift 1.9)
        credit_history = perfect
        savings_balance in {< 100 DM, 100 - 500 DM}
        dependents > 1
        ->  class yes  [0.818]

    Rule 67/17: (10/1.3, lift 1.9)
        months_loan_duration <= 40
        credit_history = good
        percent_of_income > 2
        years_at_residence > 1
        other_credit = store
        ->  class yes  [0.811]

    Rule 67/18: (16.2/2.5, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration > 7
        savings_balance in {100 - 500 DM, unknown}
        percent_of_income > 2
        years_at_residence > 1
        age > 23
        existing_loans_count <= 1
        ->  class yes  [0.808]

    Rule 67/19: (17.3/2.8, lift 1.9)
        credit_history = very good
        amount > 458
        savings_balance = < 100 DM
        percent_of_income > 1
        other_credit = bank
        ->  class yes  [0.802]

    Rule 67/20: (5.9/0.6, lift 1.9)
        credit_history = good
        existing_loans_count > 1
        job = management
        dependents <= 1
        ->  class yes  [0.793]

    Rule 67/21: (10.3/1.8, lift 1.8)
        credit_history = perfect
        savings_balance in {< 100 DM, 100 - 500 DM}
        age <= 30
        ->  class yes  [0.774]

    Rule 67/22: (8.9/1.6, lift 1.8)
        credit_history = perfect
        savings_balance in {< 100 DM, 100 - 500 DM}
        job in {management, unemployed, unskilled}
        ->  class yes  [0.765]

    Rule 67/23: (25/7.2, lift 1.6)
        months_loan_duration > 20
        credit_history = poor
        savings_balance = < 100 DM
        ->  class yes  [0.698]

    Rule 67/24: (29.8/9.2, lift 1.6)
        months_loan_duration <= 18
        credit_history = good
        amount > 3868
        ->  class yes  [0.678]

    Rule 67/25: (653/359.1, lift 1.1)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        ->  class yes  [0.450]

    Rule 67/26: (24.2, lift 1.7)
        checking_balance = unknown
        months_loan_duration <= 40
        credit_history = good
        percent_of_income > 2
        age > 23
        other_credit in {bank, none}
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.962]

    Rule 67/27: (14.3/0.2, lift 1.6)
        credit_history = good
        purpose in {business, car, education, renovations}
        housing in {other, own}
        existing_loans_count > 1
        job = skilled
        ->  class no  [0.926]

    Rule 67/28: (10, lift 1.6)
        months_loan_duration <= 20
        credit_history = poor
        savings_balance = < 100 DM
        existing_loans_count > 1
        ->  class no  [0.917]

    Rule 67/29: (9.2, lift 1.6)
        months_loan_duration <= 18
        purpose in {business, education, renovations}
        percent_of_income <= 2
        existing_loans_count <= 1
        ->  class no  [0.911]

    Rule 67/30: (14.1/1, lift 1.5)
        credit_history = critical
        amount <= 7308
        savings_balance = < 100 DM
        dependents > 1
        ->  class no  [0.873]

    Rule 67/31: (4.8, lift 1.5)
        checking_balance = > 200 DM
        age <= 23
        ->  class no  [0.853]

    Rule 67/32: (49.3/7.4, lift 1.5)
        checking_balance = unknown
        credit_history = critical
        other_credit = none
        ->  class no  [0.836]

    Rule 67/33: (29/5.7, lift 1.4)
        credit_history = good
        percent_of_income <= 2
        housing = rent
        existing_loans_count <= 1
        ->  class no  [0.783]

    Rule 67/34: (25.2/5.2, lift 1.4)
        credit_history = poor
        savings_balance in {100 - 500 DM, 500 - 1000 DM, unknown}
        ->  class no  [0.773]

    Rule 67/35: (45.3/10.1, lift 1.3)
        credit_history = critical
        savings_balance in {> 1000 DM, 100 - 500 DM, unknown}
        other_credit = none
        ->  class no  [0.764]

    Rule 67/36: (55/12.8, lift 1.3)
        credit_history = good
        percent_of_income > 2
        years_at_residence <= 1
        ->  class no  [0.759]

    Rule 67/37: (741/306.9, lift 1.0)
        housing in {other, own}
        ->  class no  [0.586]

    Default class: no

    -----  Trial 68:  -----

    Rules:

    Rule 68/1: (15.6/0.4, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 22
        purpose = furniture/appliances
        employment_duration = 4 - 7 years
        housing = own
        ->  class yes  [0.918]

    Rule 68/2: (10.1, lift 2.0)
        checking_balance = 1 - 200 DM
        employment_duration = < 1 year
        housing = rent
        job = skilled
        ->  class yes  [0.917]

    Rule 68/3: (21.1/2.3, lift 1.8)
        purpose in {car, furniture/appliances}
        amount <= 1680
        savings_balance = unknown
        employment_duration = 1 - 4 years
        ->  class yes  [0.856]

    Rule 68/4: (4.4, lift 1.8)
        purpose = car
        employment_duration = 4 - 7 years
        existing_loans_count > 2
        ->  class yes  [0.843]

    Rule 68/5: (4, lift 1.8)
        savings_balance = > 1000 DM
        employment_duration = 1 - 4 years
        age > 52
        ->  class yes  [0.833]

    Rule 68/6: (22.9/3.7, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 33
        employment_duration = > 7 years
        ->  class yes  [0.809]

    Rule 68/7: (11.3/1.7, lift 1.7)
        employment_duration = 4 - 7 years
        age <= 22
        ->  class yes  [0.797]

    Rule 68/8: (18.9/3.4, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 16
        months_loan_duration <= 18
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        ->  class yes  [0.790]

    Rule 68/9: (12.1/2, lift 1.7)
        months_loan_duration > 16
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        dependents > 1
        ->  class yes  [0.784]

    Rule 68/10: (22.5/4.9, lift 1.6)
        months_loan_duration > 11
        savings_balance = 500 - 1000 DM
        employment_duration = 1 - 4 years
        phone = FALSE
        ->  class yes  [0.758]

    Rule 68/11: (43.1/10.9, lift 1.6)
        months_loan_duration > 16
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        percent_of_income > 2
        years_at_residence <= 3
        ->  class yes  [0.735]

    Rule 68/12: (43.6/11.3, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 24
        savings_balance = < 100 DM
        employment_duration = > 7 years
        age > 28
        job in {management, skilled}
        ->  class yes  [0.729]

    Rule 68/13: (18.4/4.7, lift 1.5)
        savings_balance in {100 - 500 DM, 500 - 1000 DM}
        employment_duration = < 1 year
        ->  class yes  [0.719]

    Rule 68/14: (49.7/19.4, lift 1.3)
        employment_duration = unemployed
        age <= 56
        ->  class yes  [0.605]

    Rule 68/15: (112.4/48.2, lift 1.2)
        savings_balance = < 100 DM
        employment_duration = < 1 year
        other_credit = none
        ->  class yes  [0.570]

    Rule 68/16: (588.3/294.5, lift 1.1)
        checking_balance in {< 0 DM, 1 - 200 DM}
        ->  class yes  [0.499]

    Rule 68/17: (25.7, lift 1.8)
        months_loan_duration <= 22
        purpose = furniture/appliances
        employment_duration = 4 - 7 years
        age > 22
        ->  class no  [0.964]

    Rule 68/18: (10.8, lift 1.7)
        purpose = car
        amount > 2775
        amount <= 8858
        employment_duration = 4 - 7 years
        housing = own
        ->  class no  [0.922]

    Rule 68/19: (10.7, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        purpose = furniture/appliances
        employment_duration = 4 - 7 years
        age > 22
        ->  class no  [0.921]

    Rule 68/20: (9, lift 1.7)
        purpose = car
        amount <= 8858
        employment_duration = 4 - 7 years
        existing_loans_count <= 2
        phone = TRUE
        ->  class no  [0.909]

    Rule 68/21: (7.7, lift 1.7)
        months_loan_duration <= 11
        savings_balance = 500 - 1000 DM
        ->  class no  [0.897]

    Rule 68/22: (7.3, lift 1.7)
        employment_duration = unemployed
        age > 56
        ->  class no  [0.892]

    Rule 68/23: (10.7/0.6, lift 1.6)
        months_loan_duration > 16
        employment_duration = 1 - 4 years
        percent_of_income > 2
        years_at_residence > 3
        housing = own
        ->  class no  [0.877]

    Rule 68/24: (24.9/2.8, lift 1.6)
        purpose in {business, education, renovations}
        employment_duration = 4 - 7 years
        age > 22
        ->  class no  [0.857]

    Rule 68/25: (4.6, lift 1.6)
        savings_balance = 500 - 1000 DM
        employment_duration = 1 - 4 years
        phone = TRUE
        ->  class no  [0.849]

    Rule 68/26: (21.6/2.8, lift 1.6)
        employment_duration = 4 - 7 years
        age > 22
        housing = rent
        ->  class no  [0.840]

    Rule 68/27: (15/2.3, lift 1.5)
        purpose = furniture/appliances
        employment_duration = 4 - 7 years
        housing in {other, rent}
        ->  class no  [0.808]

    Rule 68/28: (56.4/15, lift 1.4)
        months_loan_duration <= 42
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        percent_of_income <= 2
        dependents <= 1
        ->  class no  [0.726]

    Rule 68/29: (32.7/10.1, lift 1.3)
        savings_balance = > 1000 DM
        ->  class no  [0.679]

    Rule 68/30: (95.8/30.6, lift 1.3)
        months_loan_duration <= 16
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        ->  class no  [0.677]

    Rule 68/31: (129.2/53.3, lift 1.1)
        savings_balance = unknown
        ->  class no  [0.586]

    Rule 68/32: (199.6/84.8, lift 1.1)
        employment_duration = > 7 years
        ->  class no  [0.574]

    Default class: no

    -----  Trial 69:  -----

    Rules:

    Rule 69/1: (20.8/0.3, lift 2.1)
        months_loan_duration > 7
        amount <= 1264
        dependents > 1
        phone = FALSE
        ->  class yes  [0.944]

    Rule 69/2: (13.5, lift 2.1)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 36
        savings_balance = < 100 DM
        job = skilled
        dependents <= 1
        ->  class yes  [0.935]

    Rule 69/3: (12.8, lift 2.1)
        checking_balance = < 0 DM
        purpose = car
        amount > 3031
        savings_balance = < 100 DM
        age > 22
        job = skilled
        dependents <= 1
        ->  class yes  [0.933]

    Rule 69/4: (8.5, lift 2.0)
        checking_balance = unknown
        amount > 6681
        employment_duration = < 1 year
        dependents <= 1
        ->  class yes  [0.905]

    Rule 69/5: (7.4, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        savings_balance = < 100 DM
        age > 22
        other_credit in {bank, store}
        job = skilled
        ->  class yes  [0.894]

    Rule 69/6: (6.3, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        employment_duration = > 7 years
        years_at_residence > 1
        job = management
        ->  class yes  [0.879]

    Rule 69/7: (8.3/0.4, lift 1.9)
        checking_balance = unknown
        credit_history = good
        savings_balance = unknown
        employment_duration in {< 1 year, > 7 years, 1 - 4 years}
        age > 37
        job = skilled
        dependents <= 1
        ->  class yes  [0.868]

    Rule 69/8: (5.3, lift 1.9)
        credit_history = very good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        job = skilled
        ->  class yes  [0.862]

    Rule 69/9: (11/0.9, lift 1.9)
        months_loan_duration > 20
        months_loan_duration <= 45
        savings_balance = unknown
        job = management
        dependents <= 1
        ->  class yes  [0.856]

    Rule 69/10: (15.8/1.9, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 15
        months_loan_duration <= 21
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        job = skilled
        ->  class yes  [0.840]

    Rule 69/11: (8.8/0.7, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 7
        months_loan_duration <= 36
        purpose = renovations
        savings_balance = < 100 DM
        ->  class yes  [0.839]

    Rule 69/12: (19.5/2.5, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 21
        savings_balance = < 100 DM
        employment_duration = 4 - 7 years
        job = skilled
        dependents <= 1
        ->  class yes  [0.835]

    Rule 69/13: (3.9, lift 1.9)
        checking_balance = 1 - 200 DM
        years_at_residence > 1
        job = management
        phone = FALSE
        ->  class yes  [0.830]

    Rule 69/14: (14.2/2, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        savings_balance = < 100 DM
        age > 22
        housing in {other, rent}
        job = skilled
        dependents <= 1
        ->  class yes  [0.816]

    Rule 69/15: (24/4.2, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 7
        amount <= 1316
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        job = skilled
        ->  class yes  [0.800]

    Rule 69/16: (10.7/1.6, lift 1.8)
        months_loan_duration > 7
        credit_history = good
        age > 45
        existing_loans_count <= 1
        dependents > 1
        ->  class yes  [0.799]

    Rule 69/17: (25/5, lift 1.7)
        months_loan_duration > 7
        savings_balance = 100 - 500 DM
        employment_duration in {< 1 year, > 7 years, 1 - 4 years}
        existing_loans_count > 1
        dependents <= 1
        ->  class yes  [0.778]

    Rule 69/18: (16.1/3.1, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        amount <= 6199
        savings_balance = < 100 DM
        job = unskilled
        dependents <= 1
        ->  class yes  [0.776]

    Rule 69/19: (28.7/7.2, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 7
        credit_history = good
        purpose in {car, furniture/appliances}
        savings_balance = unknown
        employment_duration in {< 1 year, > 7 years, 1 - 4 years}
        job = skilled
        ->  class yes  [0.732]

    Rule 69/20: (20.3/5.8, lift 1.6)
        checking_balance = 1 - 200 DM
        savings_balance = < 100 DM
        years_at_residence > 2
        job = management
        ->  class yes  [0.695]

    Rule 69/21: (227.5/109.2, lift 1.2)
        amount > 4057
        ->  class yes  [0.520]

    Rule 69/22: (50.2/9, lift 1.5)
        months_loan_duration <= 7
        amount <= 4057
        ->  class no  [0.808]

    Rule 69/23: (846.7/391, lift 1.0)
        months_loan_duration > 7
        ->  class no  [0.538]

    Default class: no

    -----  Trial 70:  -----

    Rules:

    Rule 70/1: (8.8, lift 1.8)
        checking_balance = 1 - 200 DM
        months_loan_duration > 27
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        ->  class yes  [0.908]

    Rule 70/2: (10.8/0.2, lift 1.8)
        checking_balance = 1 - 200 DM
        months_loan_duration > 36
        savings_balance = < 100 DM
        housing = own
        ->  class yes  [0.906]

    Rule 70/3: (8.4, lift 1.8)
        checking_balance = < 0 DM
        age <= 20
        dependents <= 1
        ->  class yes  [0.903]

    Rule 70/4: (6.7, lift 1.8)
        months_loan_duration > 8
        purpose = car
        amount <= 1455
        savings_balance = 100 - 500 DM
        ->  class yes  [0.884]

    Rule 70/5: (6.6, lift 1.8)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        ->  class yes  [0.884]

    Rule 70/6: (21.9/1.9, lift 1.8)
        months_loan_duration > 8
        savings_balance = 500 - 1000 DM
        years_at_residence > 1
        age <= 30
        other_credit = none
        job in {skilled, unskilled}
        ->  class yes  [0.879]

    Rule 70/7: (30.7/3.1, lift 1.7)
        checking_balance = < 0 DM
        credit_history in {perfect, poor, very good}
        savings_balance = < 100 DM
        job = skilled
        ->  class yes  [0.873]

    Rule 70/8: (5.1, lift 1.7)
        checking_balance = 1 - 200 DM
        savings_balance = < 100 DM
        percent_of_income > 2
        housing = other
        ->  class yes  [0.859]

    Rule 70/9: (16.5/1.7, lift 1.7)
        checking_balance = < 0 DM
        savings_balance = < 100 DM
        housing = rent
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.852]

    Rule 70/10: (14.8/1.6, lift 1.7)
        checking_balance = unknown
        months_loan_duration > 8
        credit_history = critical
        years_at_residence <= 2
        other_credit in {bank, store}
        dependents <= 1
        ->  class yes  [0.843]

    Rule 70/11: (22.2/3.1, lift 1.7)
        checking_balance = < 0 DM
        months_loan_duration > 16
        credit_history = good
        savings_balance = < 100 DM
        other_credit = none
        housing = own
        job = skilled
        dependents <= 1
        ->  class yes  [0.833]

    Rule 70/12: (7.8/0.7, lift 1.7)
        checking_balance = < 0 DM
        months_loan_duration > 8
        job = unemployed
        ->  class yes  [0.831]

    Rule 70/13: (10.8/1.4, lift 1.6)
        checking_balance = 1 - 200 DM
        years_at_residence <= 3
        other_credit = store
        dependents <= 1
        ->  class yes  [0.811]

    Rule 70/14: (25.8/4.3, lift 1.6)
        checking_balance = 1 - 200 DM
        months_loan_duration > 8
        purpose = furniture/appliances
        savings_balance = < 100 DM
        years_at_residence <= 3
        age > 28
        other_credit = none
        housing = own
        dependents <= 1
        ->  class yes  [0.809]

    Rule 70/15: (29.3/6.6, lift 1.5)
        months_loan_duration > 8
        savings_balance = < 100 DM
        years_at_residence <= 2
        housing = rent
        dependents <= 1
        ->  class yes  [0.759]

    Rule 70/16: (92.5/35.3, lift 1.2)
        months_loan_duration > 21
        savings_balance = < 100 DM
        years_at_residence > 3
        ->  class yes  [0.616]

    Rule 70/17: (92.1/38.4, lift 1.2)
        purpose in {education, renovations}
        ->  class yes  [0.582]

    Rule 70/18: (130.1/58, lift 1.1)
        dependents > 1
        ->  class yes  [0.554]

    Rule 70/19: (23.1, lift 1.9)
        checking_balance = 1 - 200 DM
        months_loan_duration > 8
        months_loan_duration <= 21
        years_at_residence > 3
        housing = own
        dependents <= 1
        ->  class no  [0.960]

    Rule 70/20: (16, lift 1.9)
        checking_balance = unknown
        credit_history = critical
        savings_balance = < 100 DM
        other_credit = none
        housing = own
        ->  class no  [0.944]

    Rule 70/21: (15, lift 1.9)
        amount <= 3711
        dependents > 1
        phone = TRUE
        ->  class no  [0.941]

    Rule 70/22: (9.2, lift 1.8)
        checking_balance in {> 200 DM, unknown}
        savings_balance = < 100 DM
        other_credit in {bank, none}
        housing = other
        dependents <= 1
        ->  class no  [0.910]

    Rule 70/23: (12.4/0.6, lift 1.8)
        savings_balance = < 100 DM
        percent_of_income <= 2
        housing = other
        dependents <= 1
        ->  class no  [0.889]

    Rule 70/24: (6.8, lift 1.8)
        savings_balance = unknown
        employment_duration = 4 - 7 years
        job = skilled
        ->  class no  [0.886]

    Rule 70/25: (5.4, lift 1.7)
        purpose = education
        savings_balance = unknown
        job = skilled
        ->  class no  [0.864]

    Rule 70/26: (5.3, lift 1.7)
        checking_balance = > 200 DM
        years_at_residence > 2
        housing = rent
        ->  class no  [0.863]

    Rule 70/27: (20.3/2.1, lift 1.7)
        months_loan_duration <= 16
        amount > 1223
        percent_of_income > 1
        age <= 52
        dependents > 1
        ->  class no  [0.859]

    Rule 70/28: (4.8, lift 1.7)
        age <= 22
        housing = own
        job = unskilled
        ->  class no  [0.853]

    Rule 70/29: (8/0.6, lift 1.7)
        checking_balance = < 0 DM
        savings_balance = < 100 DM
        years_at_residence <= 3
        housing = rent
        phone = FALSE
        ->  class no  [0.841]

    Rule 70/30: (13.8/1.9, lift 1.6)
        purpose in {business, car0, renovations}
        savings_balance = 100 - 500 DM
        dependents <= 1
        ->  class no  [0.816]

    Rule 70/31: (14.9/2.2, lift 1.6)
        percent_of_income <= 1
        dependents > 1
        phone = FALSE
        ->  class no  [0.810]

    Rule 70/32: (34.4/9.2, lift 1.4)
        checking_balance = unknown
        credit_history in {good, perfect, very good}
        savings_balance = < 100 DM
        housing = own
        dependents <= 1
        ->  class no  [0.720]

    Rule 70/33: (28.1/7.4, lift 1.4)
        checking_balance = < 0 DM
        months_loan_duration > 8
        savings_balance = < 100 DM
        job = management
        ->  class no  [0.719]

    Rule 70/34: (769.9/379.6, lift 1.0)
        dependents <= 1
        ->  class no  [0.507]

    Default class: yes

    -----  Trial 71:  -----

    Rules:

    Rule 71/1: (14.6, lift 1.8)
        checking_balance = 1 - 200 DM
        credit_history = good
        amount > 8858
        ->  class yes  [0.940]

    Rule 71/2: (28.8/1.6, lift 1.8)
        checking_balance = < 0 DM
        months_loan_duration <= 40
        credit_history = good
        purpose = car
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, 4 - 7 years}
        years_at_residence <= 2
        ->  class yes  [0.916]

    Rule 71/3: (9.3, lift 1.8)
        credit_history = very good
        amount > 4530
        amount <= 7629
        ->  class yes  [0.912]

    Rule 71/4: (13.4/0.5, lift 1.7)
        credit_history = perfect
        housing in {other, rent}
        ->  class yes  [0.904]

    Rule 71/5: (6.8, lift 1.7)
        credit_history = critical
        employment_duration = 1 - 4 years
        age > 35
        housing = other
        ->  class yes  [0.886]

    Rule 71/6: (27/4.1, lift 1.6)
        amount <= 2241
        other_credit = store
        ->  class yes  [0.825]

    Rule 71/7: (35.1/6.3, lift 1.6)
        credit_history = good
        years_at_residence > 1
        years_at_residence <= 3
        other_credit = bank
        ->  class yes  [0.804]

    Rule 71/8: (34.1/6.5, lift 1.5)
        checking_balance = < 0 DM
        credit_history = good
        purpose = car
        amount <= 1867
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, 4 - 7 years}
        ->  class yes  [0.793]

    Rule 71/9: (2.7, lift 1.5)
        credit_history = poor
        existing_loans_count > 3
        ->  class yes  [0.789]

    Rule 71/10: (17.3/3.4, lift 1.5)
        credit_history = perfect
        age > 33
        ->  class yes  [0.774]

    Rule 71/11: (30.8/8.9, lift 1.3)
        checking_balance = unknown
        credit_history = good
        existing_loans_count > 1
        ->  class yes  [0.697]

    Rule 71/12: (44.1/13.8, lift 1.3)
        credit_history = poor
        percent_of_income > 3
        ->  class yes  [0.679]

    Rule 71/13: (576.8/262.1, lift 1.1)
        age <= 35
        ->  class yes  [0.545]

    Rule 71/14: (17.8, lift 2.0)
        checking_balance = unknown
        credit_history = good
        purpose = car
        age > 27
        existing_loans_count <= 1
        ->  class no  [0.949]

    Rule 71/15: (13.9, lift 1.9)
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years}
        age > 23
        other_credit = none
        housing = rent
        ->  class no  [0.937]

    Rule 71/16: (8.1, lift 1.9)
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = > 7 years
        job = skilled
        dependents <= 1
        ->  class no  [0.901]

    Rule 71/17: (7.9, lift 1.9)
        months_loan_duration <= 18
        credit_history = good
        purpose = business
        ->  class no  [0.899]

    Rule 71/18: (7.7, lift 1.9)
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = < 1 year
        years_at_residence <= 1
        other_credit = none
        housing = own
        dependents <= 1
        ->  class no  [0.896]

    Rule 71/19: (6.3, lift 1.8)
        credit_history = good
        years_at_residence <= 1
        other_credit = bank
        ->  class no  [0.880]

    Rule 71/20: (5.2, lift 1.8)
        checking_balance in {> 200 DM, unknown}
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        existing_loans_count <= 1
        ->  class no  [0.862]

    Rule 71/21: (5.2, lift 1.8)
        checking_balance = < 0 DM
        credit_history = good
        purpose = car
        employment_duration = unemployed
        ->  class no  [0.861]

    Rule 71/22: (12.5/1.2, lift 1.8)
        checking_balance = 1 - 200 DM
        credit_history = good
        purpose = car
        amount > 4933
        amount <= 8858
        ->  class no  [0.849]

    Rule 71/23: (4.4, lift 1.8)
        credit_history = very good
        age <= 23
        ->  class no  [0.843]

    Rule 71/24: (27.3/3.9, lift 1.7)
        credit_history = poor
        percent_of_income <= 3
        years_at_residence > 1
        age <= 49
        existing_loans_count <= 3
        ->  class no  [0.831]

    Rule 71/25: (17.1/2.6, lift 1.7)
        months_loan_duration <= 14
        purpose = furniture/appliances
        savings_balance = < 100 DM
        other_credit = none
        job = unskilled
        ->  class no  [0.810]

    Rule 71/26: (3.2, lift 1.7)
        credit_history = critical
        purpose = car
        years_at_residence <= 1
        ->  class no  [0.809]

    Rule 71/27: (8.1/1, lift 1.7)
        credit_history = good
        years_at_residence > 3
        other_credit = bank
        phone = TRUE
        ->  class no  [0.798]

    Rule 71/28: (16.8/4.1, lift 1.5)
        credit_history = perfect
        age <= 33
        housing = own
        ->  class no  [0.731]

    Rule 71/29: (23.5/6.5, lift 1.5)
        credit_history = critical
        purpose in {business, car0, renovations}
        ->  class no  [0.704]

    Rule 71/30: (91.7/38.1, lift 1.2)
        other_credit = none
        dependents > 1
        ->  class no  [0.583]

    Rule 71/31: (546.8/269.1, lift 1.1)
        age > 28
        ->  class no  [0.508]

    Default class: yes

    -----  Trial 72:  -----

    Rules:

    Rule 72/1: (12.8, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        savings_balance = < 100 DM
        age > 41
        age <= 46
        dependents <= 1
        ->  class yes  [0.933]

    Rule 72/2: (15.3/0.5, lift 1.9)
        checking_balance = unknown
        credit_history = poor
        purpose in {business, furniture/appliances, renovations}
        savings_balance in {< 100 DM, > 1000 DM, unknown}
        percent_of_income > 3
        ->  class yes  [0.911]

    Rule 72/3: (8.8, lift 1.9)
        checking_balance in {< 0 DM, > 200 DM}
        months_loan_duration > 16
        months_loan_duration <= 21
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        years_at_residence > 1
        housing = own
        job = skilled
        ->  class yes  [0.908]

    Rule 72/4: (18.1/0.9, lift 1.9)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = car
        amount <= 960
        percent_of_income > 3
        age <= 46
        ->  class yes  [0.906]

    Rule 72/5: (7.4, lift 1.9)
        employment_duration = unemployed
        years_at_residence <= 1
        ->  class yes  [0.893]

    Rule 72/6: (9.5/0.4, lift 1.8)
        checking_balance = 1 - 200 DM
        purpose in {business, car, education}
        employment_duration = > 7 years
        age > 41
        dependents <= 1
        ->  class yes  [0.874]

    Rule 72/7: (13.1/0.9, lift 1.8)
        checking_balance = unknown
        credit_history = good
        percent_of_income <= 2
        other_credit = none
        existing_loans_count > 1
        ->  class yes  [0.872]

    Rule 72/8: (12.2/1.1, lift 1.8)
        checking_balance = 1 - 200 DM
        employment_duration = 1 - 4 years
        years_at_residence <= 3
        age > 35
        other_credit = none
        ->  class yes  [0.852]

    Rule 72/9: (4.5, lift 1.8)
        checking_balance = 1 - 200 DM
        employment_duration = < 1 year
        other_credit = store
        ->  class yes  [0.847]

    Rule 72/10: (4.5, lift 1.8)
        credit_history = critical
        amount > 11816
        other_credit = none
        ->  class yes  [0.845]

    Rule 72/11: (9.5/0.9, lift 1.8)
        checking_balance = unknown
        credit_history = good
        employment_duration = 1 - 4 years
        job = unskilled
        dependents <= 1
        ->  class yes  [0.834]

    Rule 72/12: (17.9/2.3, lift 1.7)
        checking_balance = < 0 DM
        months_loan_duration <= 40
        purpose = car
        savings_balance = < 100 DM
        percent_of_income <= 3
        years_at_residence <= 3
        dependents <= 1
        ->  class yes  [0.834]

    Rule 72/13: (15.4/2, lift 1.7)
        checking_balance = 1 - 200 DM
        credit_history = good
        employment_duration = 1 - 4 years
        years_at_residence > 3
        ->  class yes  [0.828]

    Rule 72/14: (17.4/2.8, lift 1.7)
        checking_balance = < 0 DM
        savings_balance = unknown
        percent_of_income > 2
        dependents <= 1
        ->  class yes  [0.805]

    Rule 72/15: (20.5/3.8, lift 1.6)
        checking_balance = < 0 DM
        credit_history in {perfect, poor, very good}
        purpose = furniture/appliances
        savings_balance = < 100 DM
        ->  class yes  [0.786]

    Rule 72/16: (17.7/3.2, lift 1.6)
        checking_balance = 1 - 200 DM
        employment_duration = < 1 year
        housing = rent
        ->  class yes  [0.786]

    Rule 72/17: (31/6.4, lift 1.6)
        purpose = car
        savings_balance = < 100 DM
        other_credit in {bank, store}
        ->  class yes  [0.775]

    Rule 72/18: (16.8/3.8, lift 1.6)
        checking_balance = < 0 DM
        purpose in {education, renovations}
        savings_balance = < 100 DM
        ->  class yes  [0.745]

    Rule 72/19: (30.7/8.1, lift 1.5)
        checking_balance = unknown
        purpose in {business, car, education}
        other_credit = bank
        ->  class yes  [0.721]

    Rule 72/20: (39.2/13.1, lift 1.4)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        age > 38
        ->  class yes  [0.656]

    Rule 72/21: (632.7/306, lift 1.1)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        ->  class yes  [0.516]

    Rule 72/22: (51.5/4, lift 1.7)
        checking_balance = unknown
        credit_history = critical
        amount <= 11816
        other_credit = none
        ->  class no  [0.906]

    Rule 72/23: (8.5, lift 1.7)
        checking_balance = unknown
        credit_history = poor
        purpose = car
        ->  class no  [0.905]

    Rule 72/24: (7.4, lift 1.7)
        checking_balance = unknown
        employment_duration = 1 - 4 years
        existing_loans_count <= 1
        dependents > 1
        ->  class no  [0.894]

    Rule 72/25: (12/1.3, lift 1.6)
        credit_history = good
        age <= 32
        other_credit = none
        dependents > 1
        ->  class no  [0.839]

    Rule 72/26: (9.7/1.5, lift 1.5)
        checking_balance = < 0 DM
        credit_history = critical
        dependents > 1
        ->  class no  [0.786]

    Rule 72/27: (773.1/361.3, lift 1.0)
        dependents <= 1
        ->  class no  [0.533]

    Default class: no

    -----  Trial 73:  -----

    Rules:

    Rule 73/1: (11, lift 1.9)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = perfect
        percent_of_income > 3
        ->  class yes  [0.923]

    Rule 73/2: (5.7, lift 1.7)
        credit_history = critical
        purpose = car
        savings_balance = < 100 DM
        other_credit = bank
        ->  class yes  [0.871]

    Rule 73/3: (14.8/1.6, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 16
        credit_history = poor
        savings_balance = < 100 DM
        ->  class yes  [0.845]

    Rule 73/4: (44.4/6.4, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 7
        credit_history = good
        amount > 629
        amount <= 1047
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        ->  class yes  [0.841]

    Rule 73/5: (18.5/2.4, lift 1.7)
        credit_history = very good
        other_credit = none
        ->  class yes  [0.837]

    Rule 73/6: (8.1/0.6, lift 1.7)
        months_loan_duration <= 7
        amount > 3380
        existing_loans_count <= 1
        ->  class yes  [0.836]

    Rule 73/7: (16.7/3.1, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        purpose = furniture/appliances
        years_at_residence <= 2
        phone = FALSE
        ->  class yes  [0.780]

    Rule 73/8: (24.9/4.9, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 47
        credit_history = good
        ->  class yes  [0.780]

    Rule 73/9: (27.5/7.8, lift 1.4)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = very good
        percent_of_income > 3
        ->  class yes  [0.700]

    Rule 73/10: (844.2/411.4, lift 1.0)
        months_loan_duration > 7
        ->  class yes  [0.513]

    Rule 73/11: (10.9, lift 1.8)
        checking_balance = unknown
        credit_history = good
        purpose = car
        employment_duration in {< 1 year, > 7 years, 4 - 7 years}
        ->  class no  [0.923]

    Rule 73/12: (24.3/1.1, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = good
        amount > 1249
        amount <= 5866
        employment_duration = > 7 years
        years_at_residence > 2
        ->  class no  [0.921]

    Rule 73/13: (9.7, lift 1.8)
        checking_balance = unknown
        purpose = furniture/appliances
        percent_of_income > 3
        existing_loans_count > 1
        job = skilled
        ->  class no  [0.914]

    Rule 73/14: (9.5, lift 1.8)
        credit_history = good
        amount > 1047
        amount <= 7582
        employment_duration = 1 - 4 years
        percent_of_income <= 1
        other_credit = none
        ->  class no  [0.913]

    Rule 73/15: (11.5/0.3, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 42
        credit_history = critical
        amount > 5117
        amount <= 7678
        ->  class no  [0.907]

    Rule 73/16: (8.6, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = good
        employment_duration = < 1 year
        percent_of_income <= 2
        years_at_residence > 1
        job in {skilled, unemployed}
        ->  class no  [0.906]

    Rule 73/17: (8, lift 1.8)
        checking_balance = unknown
        credit_history = good
        purpose = car
        phone = TRUE
        ->  class no  [0.900]

    Rule 73/18: (7.8, lift 1.8)
        months_loan_duration > 7
        months_loan_duration <= 16
        credit_history = poor
        savings_balance = < 100 DM
        ->  class no  [0.898]

    Rule 73/19: (7.1, lift 1.8)
        checking_balance = unknown
        credit_history = poor
        purpose = car
        ->  class no  [0.890]

    Rule 73/20: (6.7, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration <= 47
        credit_history = good
        savings_balance = > 1000 DM
        ->  class no  [0.885]

    Rule 73/21: (44.7/5.4, lift 1.7)
        checking_balance = unknown
        credit_history = critical
        amount <= 11816
        other_credit = none
        ->  class no  [0.864]

    Rule 73/22: (20.9/2.2, lift 1.7)
        months_loan_duration > 7
        months_loan_duration <= 42
        credit_history = critical
        purpose = furniture/appliances
        amount <= 5117
        savings_balance = < 100 DM
        years_at_residence > 2
        age <= 54
        other_credit = none
        ->  class no  [0.862]

    Rule 73/23: (36.1/4.3, lift 1.7)
        months_loan_duration <= 42
        credit_history = critical
        purpose = car
        amount <= 7678
        age > 29
        other_credit = none
        ->  class no  [0.861]

    Rule 73/24: (4.8, lift 1.7)
        credit_history = perfect
        purpose in {education, furniture/appliances}
        percent_of_income <= 3
        housing = own
        ->  class no  [0.854]

    Rule 73/25: (4.5, lift 1.7)
        checking_balance = unknown
        credit_history = critical
        other_credit = bank
        phone = TRUE
        ->  class no  [0.847]

    Rule 73/26: (19.2/2.5, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = poor
        savings_balance in {100 - 500 DM, unknown}
        ->  class no  [0.837]

    Rule 73/27: (11.9/1.3, lift 1.7)
        credit_history = good
        amount > 1047
        employment_duration = unemployed
        years_at_residence > 2
        ->  class no  [0.831]

    Rule 73/28: (26.5/3.8, lift 1.7)
        checking_balance = unknown
        months_loan_duration > 7
        credit_history = good
        purpose = furniture/appliances
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, 4 - 7 years}
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.831]

    Rule 73/29: (18.8/4.8, lift 1.4)
        credit_history = very good
        percent_of_income <= 3
        other_credit in {bank, store}
        ->  class no  [0.720]

    Rule 73/30: (36.6/10.6, lift 1.4)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = good
        employment_duration = < 1 year
        years_at_residence <= 1
        ->  class no  [0.699]

    Rule 73/31: (135.8/58.4, lift 1.1)
        employment_duration = 4 - 7 years
        ->  class no  [0.569]

    Rule 73/32: (840.3/410.9, lift 1.0)
        months_loan_duration <= 47
        ->  class no  [0.511]

    Default class: no

    -----  Trial 74:  -----

    Rules:

    Rule 74/1: (12.9, lift 2.1)
        months_loan_duration <= 33
        amount > 7980
        savings_balance = < 100 DM
        ->  class yes  [0.933]

    Rule 74/2: (8.3, lift 2.0)
        credit_history = critical
        purpose = car
        savings_balance in {< 100 DM, 100 - 500 DM}
        other_credit in {bank, store}
        ->  class yes  [0.903]

    Rule 74/3: (24.5/1.6, lift 2.0)
        amount > 7980
        savings_balance in {< 100 DM, 100 - 500 DM}
        age <= 29
        ->  class yes  [0.900]

    Rule 74/4: (9.6/0.2, lift 2.0)
        amount > 7980
        savings_balance in {< 100 DM, 100 - 500 DM}
        years_at_residence > 3
        age > 29
        ->  class yes  [0.899]

    Rule 74/5: (17.4/1.1, lift 2.0)
        months_loan_duration > 28
        credit_history = good
        purpose = furniture/appliances
        amount > 2360
        amount <= 5511
        savings_balance in {< 100 DM, 100 - 500 DM}
        percent_of_income > 1
        age > 22
        phone = FALSE
        ->  class yes  [0.892]

    Rule 74/6: (14/0.8, lift 2.0)
        months_loan_duration > 24
        savings_balance = < 100 DM
        percent_of_income > 3
        job = skilled
        phone = TRUE
        ->  class yes  [0.885]

    Rule 74/7: (6.6, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 27
        credit_history = critical
        purpose = furniture/appliances
        employment_duration in {< 1 year, 1 - 4 years}
        age > 29
        ->  class yes  [0.883]

    Rule 74/8: (6.1, lift 2.0)
        checking_balance = < 0 DM
        credit_history = very good
        other_credit = none
        ->  class yes  [0.877]

    Rule 74/9: (5.9, lift 1.9)
        credit_history = good
        purpose = car
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        age <= 25
        phone = FALSE
        ->  class yes  [0.873]

    Rule 74/10: (7.2/0.2, lift 1.9)
        credit_history = good
        purpose = furniture/appliances
        savings_balance in {< 100 DM, 100 - 500 DM}
        employment_duration = unemployed
        phone = FALSE
        ->  class yes  [0.872]

    Rule 74/11: (19.5/2.3, lift 1.9)
        purpose = car
        amount <= 1433
        savings_balance in {500 - 1000 DM, unknown}
        age <= 40
        ->  class yes  [0.846]

    Rule 74/12: (16.5/1.9, lift 1.9)
        months_loan_duration > 8
        credit_history = good
        savings_balance in {< 100 DM, 100 - 500 DM}
        percent_of_income > 2
        percent_of_income <= 3
        phone = TRUE
        ->  class yes  [0.841]

    Rule 74/13: (25.3/3.6, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM, unknown}
        credit_history = good
        purpose = car
        amount <= 4370
        savings_balance = < 100 DM
        employment_duration in {< 1 year, > 7 years, 4 - 7 years}
        phone = FALSE
        ->  class yes  [0.832]

    Rule 74/14: (26.7/5.8, lift 1.7)
        credit_history = good
        purpose = furniture/appliances
        savings_balance in {< 100 DM, 100 - 500 DM}
        age <= 22
        phone = FALSE
        ->  class yes  [0.763]

    Rule 74/15: (2.2, lift 1.7)
        purpose = education
        savings_balance = > 1000 DM
        ->  class yes  [0.762]

    Rule 74/16: (26.6/6.9, lift 1.6)
        credit_history = poor
        savings_balance = < 100 DM
        percent_of_income > 2
        ->  class yes  [0.725]

    Rule 74/17: (28.2/7.9, lift 1.6)
        credit_history = perfect
        savings_balance in {< 100 DM, 100 - 500 DM}
        other_credit in {none, store}
        ->  class yes  [0.706]

    Rule 74/18: (73/27, lift 1.4)
        purpose = furniture/appliances
        amount > 4153
        ->  class yes  [0.627]

    Rule 74/19: (43.9/17.2, lift 1.3)
        credit_history = very good
        savings_balance in {< 100 DM, 100 - 500 DM}
        ->  class yes  [0.603]

    Rule 74/20: (585/311.1, lift 1.0)
        percent_of_income > 2
        ->  class yes  [0.468]

    Rule 74/21: (19.5, lift 1.7)
        months_loan_duration <= 36
        credit_history = critical
        purpose = furniture/appliances
        amount <= 7980
        savings_balance in {< 100 DM, 100 - 500 DM}
        employment_duration in {> 7 years, 4 - 7 years, unemployed}
        age <= 59
        ->  class no  [0.954]

    Rule 74/22: (11.5, lift 1.7)
        credit_history in {poor, very good}
        purpose = furniture/appliances
        savings_balance in {> 1000 DM, 500 - 1000 DM, unknown}
        ->  class no  [0.926]

    Rule 74/23: (7, lift 1.6)
        checking_balance = < 0 DM
        credit_history = very good
        purpose in {car, education}
        other_credit = bank
        existing_loans_count <= 1
        ->  class no  [0.889]

    Rule 74/24: (17.4/1.8, lift 1.6)
        purpose = education
        savings_balance in {500 - 1000 DM, unknown}
        ->  class no  [0.857]

    Rule 74/25: (20.8/3, lift 1.5)
        months_loan_duration > 33
        amount > 7980
        years_at_residence <= 3
        age > 29
        ->  class no  [0.824]

    Rule 74/26: (25.8/5.9, lift 1.4)
        purpose = business
        savings_balance in {> 1000 DM, 500 - 1000 DM, unknown}
        ->  class no  [0.751]

    Rule 74/27: (826.1/356.1, lift 1.0)
        amount <= 7980
        ->  class no  [0.569]

    Default class: no

    -----  Trial 75:  -----

    Rules:

    Rule 75/1: (21.4, lift 2.0)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 15
        months_loan_duration <= 21
        amount > 999
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        phone = FALSE
        ->  class yes  [0.957]

    Rule 75/2: (10.8, lift 1.9)
        checking_balance = unknown
        purpose in {business, renovations}
        employment_duration = < 1 year
        ->  class yes  [0.922]

    Rule 75/3: (9.2, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 4 - 7 years}
        job = skilled
        phone = TRUE
        ->  class yes  [0.911]

    Rule 75/4: (8.5, lift 1.9)
        checking_balance = 1 - 200 DM
        credit_history = good
        purpose = furniture/appliances
        amount > 781
        amount <= 999
        job = skilled
        ->  class yes  [0.905]

    Rule 75/5: (7.7, lift 1.9)
        purpose in {education, renovations}
        savings_balance = > 1000 DM
        ->  class yes  [0.897]

    Rule 75/6: (7.6, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 45
        purpose = car
        amount > 7485
        savings_balance = unknown
        ->  class yes  [0.896]

    Rule 75/7: (6.5, lift 1.9)
        checking_balance = unknown
        employment_duration = 4 - 7 years
        years_at_residence <= 2
        age <= 23
        ->  class yes  [0.882]

    Rule 75/8: (5.3, lift 1.8)
        checking_balance = unknown
        purpose = furniture/appliances
        amount > 4594
        employment_duration = < 1 year
        ->  class yes  [0.862]

    Rule 75/9: (12.3/1.1, lift 1.8)
        credit_history = good
        purpose = furniture/appliances
        amount > 2320
        savings_balance = < 100 DM
        other_credit = none
        job = unskilled
        ->  class yes  [0.855]

    Rule 75/10: (20.3/2.4, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        amount <= 2241
        other_credit = store
        ->  class yes  [0.849]

    Rule 75/11: (8.6/0.6, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        savings_balance = 500 - 1000 DM
        percent_of_income > 3
        other_credit = none
        ->  class yes  [0.848]

    Rule 75/12: (4, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        other_credit = bank
        dependents > 1
        ->  class yes  [0.833]

    Rule 75/13: (8.9/0.9, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = furniture/appliances
        savings_balance = unknown
        housing in {other, rent}
        job = skilled
        ->  class yes  [0.825]

    Rule 75/14: (11.8/1.5, lift 1.7)
        checking_balance = unknown
        employment_duration = unemployed
        percent_of_income > 2
        ->  class yes  [0.815]

    Rule 75/15: (23.5/3.8, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 47
        savings_balance = < 100 DM
        ->  class yes  [0.812]

    Rule 75/16: (28.6/6.2, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        savings_balance = 100 - 500 DM
        job in {management, unskilled}
        ->  class yes  [0.763]

    Rule 75/17: (24.7/5.7, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = education
        savings_balance = < 100 DM
        age <= 42
        ->  class yes  [0.748]

    Rule 75/18: (28/7.9, lift 1.5)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        savings_balance = 100 - 500 DM
        age <= 29
        ->  class yes  [0.705]

    Rule 75/19: (85.5/25.9, lift 1.5)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = car
        savings_balance = < 100 DM
        percent_of_income > 2
        ->  class yes  [0.692]

    Rule 75/20: (771.4/394.5, lift 1.0)
        months_loan_duration > 11
        ->  class yes  [0.489]

    Rule 75/21: (13.1, lift 1.8)
        checking_balance = unknown
        months_loan_duration > 24
        employment_duration = > 7 years
        ->  class no  [0.934]

    Rule 75/22: (12.8, lift 1.8)
        purpose = furniture/appliances
        amount > 2241
        other_credit = store
        ->  class no  [0.933]

    Rule 75/23: (24.4/1.5, lift 1.7)
        checking_balance = unknown
        employment_duration = 4 - 7 years
        age > 23
        ->  class no  [0.907]

    Rule 75/24: (27.8/1.8, lift 1.7)
        checking_balance = unknown
        savings_balance in {< 100 DM, 500 - 1000 DM, unknown}
        employment_duration = > 7 years
        other_credit = none
        ->  class no  [0.906]

    Rule 75/25: (8.1, lift 1.7)
        checking_balance = unknown
        purpose in {car, education}
        employment_duration = < 1 year
        ->  class no  [0.901]

    Rule 75/26: (7.7, lift 1.7)
        months_loan_duration <= 47
        purpose = education
        savings_balance = < 100 DM
        age > 42
        ->  class no  [0.896]

    Rule 75/27: (7.2, lift 1.7)
        checking_balance = 1 - 200 DM
        months_loan_duration > 45
        savings_balance = unknown
        ->  class no  [0.892]

    Rule 75/28: (12.7/0.7, lift 1.7)
        checking_balance = unknown
        purpose = furniture/appliances
        amount <= 4594
        employment_duration = < 1 year
        age <= 41
        ->  class no  [0.881]

    Rule 75/29: (6.4, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = furniture/appliances
        years_at_residence <= 2
        other_credit = bank
        dependents <= 1
        phone = FALSE
        ->  class no  [0.881]

    Rule 75/30: (19.1/2.1, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        savings_balance = 100 - 500 DM
        age > 29
        job = skilled
        ->  class no  [0.852]

    Rule 75/31: (15.9/2.1, lift 1.6)
        checking_balance = unknown
        purpose = car
        percent_of_income > 3
        phone = FALSE
        ->  class no  [0.830]

    Rule 75/32: (14.6/2.5, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose in {business, education}
        savings_balance = unknown
        ->  class no  [0.787]

    Rule 75/33: (30.5/6.3, lift 1.5)
        checking_balance = unknown
        purpose in {furniture/appliances, renovations}
        employment_duration = 1 - 4 years
        ->  class no  [0.776]

    Rule 75/34: (76.3/24.3, lift 1.3)
        amount > 1372
        amount <= 7485
        savings_balance = unknown
        ->  class no  [0.677]

    Rule 75/35: (251.1/104.8, lift 1.1)
        purpose = furniture/appliances
        employment_duration in {> 7 years, 1 - 4 years}
        ->  class no  [0.582]

    Rule 75/36: (441.3/191.9, lift 1.1)
        percent_of_income <= 3
        ->  class no  [0.565]

    Default class: no

    -----  Trial 76:  -----

    Rules:

    Rule 76/1: (17.6, lift 2.0)
        checking_balance = < 0 DM
        months_loan_duration > 16
        months_loan_duration <= 45
        credit_history = good
        savings_balance = < 100 DM
        percent_of_income > 1
        years_at_residence <= 3
        other_credit = none
        job = skilled
        ->  class yes  [0.949]

    Rule 76/2: (9.5, lift 1.9)
        checking_balance = unknown
        credit_history in {good, poor}
        purpose in {car, furniture/appliances, renovations}
        years_at_residence > 3
        other_credit = none
        existing_loans_count > 1
        phone = FALSE
        ->  class yes  [0.913]

    Rule 76/3: (14.7/0.5, lift 1.9)
        checking_balance = < 0 DM
        purpose in {business, education, renovations}
        savings_balance = < 100 DM
        job = skilled
        ->  class yes  [0.910]

    Rule 76/4: (8.4, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration > 27
        savings_balance = unknown
        other_credit = none
        ->  class yes  [0.904]

    Rule 76/5: (7.4, lift 1.8)
        checking_balance = 1 - 200 DM
        purpose = car
        savings_balance = 100 - 500 DM
        job in {management, unskilled}
        ->  class yes  [0.894]

    Rule 76/6: (20/1.4, lift 1.8)
        checking_balance = < 0 DM
        savings_balance = < 100 DM
        housing in {other, rent}
        existing_loans_count > 1
        job = skilled
        ->  class yes  [0.893]

    Rule 76/7: (18/1.6, lift 1.8)
        checking_balance = < 0 DM
        months_loan_duration > 11
        months_loan_duration <= 13
        existing_loans_count > 1
        job = skilled
        ->  class yes  [0.870]

    Rule 76/8: (18.3/1.7, lift 1.8)
        checking_balance = < 0 DM
        savings_balance = unknown
        employment_duration in {> 7 years, 1 - 4 years}
        other_credit = none
        ->  class yes  [0.867]

    Rule 76/9: (32.8/3.6, lift 1.8)
        credit_history in {critical, good, perfect}
        amount > 10875
        ->  class yes  [0.867]

    Rule 76/10: (5.3, lift 1.8)
        checking_balance = < 0 DM
        percent_of_income <= 1
        job = management
        ->  class yes  [0.864]

    Rule 76/11: (5.2, lift 1.8)
        checking_balance = 1 - 200 DM
        months_loan_duration > 36
        purpose = furniture/appliances
        savings_balance = < 100 DM
        ->  class yes  [0.861]

    Rule 76/12: (13.4/1.3, lift 1.8)
        checking_balance = 1 - 200 DM
        purpose = business
        savings_balance = < 100 DM
        years_at_residence > 1
        ->  class yes  [0.851]

    Rule 76/13: (7.9/0.7, lift 1.7)
        checking_balance = < 0 DM
        amount > 7596
        job = management
        ->  class yes  [0.824]

    Rule 76/14: (13.7/2.3, lift 1.6)
        checking_balance = < 0 DM
        credit_history = good
        savings_balance = < 100 DM
        job = skilled
        phone = TRUE
        ->  class yes  [0.791]

    Rule 76/15: (19.4/3.7, lift 1.6)
        checking_balance = < 0 DM
        purpose in {car, furniture/appliances}
        savings_balance = < 100 DM
        other_credit = bank
        job = skilled
        ->  class yes  [0.782]

    Rule 76/16: (35.6/7.6, lift 1.6)
        checking_balance = unknown
        savings_balance in {< 100 DM, 500 - 1000 DM, unknown}
        percent_of_income > 1
        age <= 44
        other_credit = bank
        ->  class yes  [0.772]

    Rule 76/17: (35.5/8.7, lift 1.5)
        checking_balance = > 200 DM
        amount <= 4308
        percent_of_income > 1
        age > 24
        age <= 39
        ->  class yes  [0.740]

    Rule 76/18: (259.5/122.4, lift 1.1)
        age <= 26
        ->  class yes  [0.528]

    Rule 76/19: (11.6, lift 1.8)
        checking_balance = > 200 DM
        age <= 24
        ->  class no  [0.926]

    Rule 76/20: (7.4, lift 1.7)
        checking_balance = > 200 DM
        amount > 4308
        ->  class no  [0.894]

    Rule 76/21: (17.3/1.3, lift 1.7)
        checking_balance = unknown
        percent_of_income <= 2
        years_at_residence > 3
        existing_loans_count <= 1
        ->  class no  [0.881]

    Rule 76/22: (6.1, lift 1.7)
        checking_balance = 1 - 200 DM
        purpose = furniture/appliances
        amount <= 776
        savings_balance = < 100 DM
        ->  class no  [0.876]

    Rule 76/23: (30.2/3.5, lift 1.7)
        checking_balance = unknown
        credit_history in {good, poor}
        years_at_residence > 1
        other_credit = none
        job in {skilled, unemployed, unskilled}
        phone = TRUE
        ->  class no  [0.860]

    Rule 76/24: (8.9/0.6, lift 1.7)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        savings_balance = < 100 DM
        years_at_residence <= 3
        job = unskilled
        ->  class no  [0.856]

    Rule 76/25: (4.9, lift 1.7)
        checking_balance = 1 - 200 DM
        credit_history = very good
        purpose = furniture/appliances
        ->  class no  [0.856]

    Rule 76/26: (4.3, lift 1.6)
        credit_history in {poor, very good}
        amount > 10875
        ->  class no  [0.842]

    Rule 76/27: (19.5/3.6, lift 1.5)
        checking_balance = unknown
        percent_of_income <= 1
        ->  class no  [0.787]

    Rule 76/28: (18.4/3.6, lift 1.5)
        checking_balance = 1 - 200 DM
        savings_balance = unknown
        phone = FALSE
        ->  class no  [0.776]

    Rule 76/29: (25.8/5.4, lift 1.5)
        credit_history = critical
        other_credit = none
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.769]

    Rule 76/30: (27.9/7.3, lift 1.4)
        savings_balance = < 100 DM
        existing_loans_count > 1
        job = unskilled
        ->  class no  [0.724]

    Rule 76/31: (44.1/15, lift 1.3)
        purpose in {business, car0, education, renovations}
        job = unskilled
        ->  class no  [0.653]

    Rule 76/32: (862.9/406.9, lift 1.0)
        amount <= 10875
        ->  class no  [0.528]

    Default class: no

    -----  Trial 77:  -----

    Rules:

    Rule 77/1: (8.5, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = perfect
        percent_of_income > 3
        ->  class yes  [0.905]

    Rule 77/2: (10.8/0.2, lift 1.7)
        checking_balance = 1 - 200 DM
        credit_history = good
        savings_balance = < 100 DM
        percent_of_income <= 1
        phone = FALSE
        ->  class yes  [0.904]

    Rule 77/3: (7.9, lift 1.7)
        checking_balance = unknown
        purpose in {business, furniture/appliances}
        amount > 6681
        employment_duration = < 1 year
        ->  class yes  [0.899]

    Rule 77/4: (6.5, lift 1.7)
        checking_balance = unknown
        employment_duration = 4 - 7 years
        years_at_residence <= 2
        age <= 23
        ->  class yes  [0.882]

    Rule 77/5: (15.7/1.1, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        credit_history = good
        purpose = furniture/appliances
        percent_of_income > 1
        existing_loans_count <= 1
        job = unskilled
        phone = FALSE
        ->  class yes  [0.880]

    Rule 77/6: (6, lift 1.6)
        checking_balance = unknown
        credit_history = good
        employment_duration = 1 - 4 years
        other_credit = none
        existing_loans_count > 1
        ->  class yes  [0.876]

    Rule 77/7: (27.3/3.1, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        amount > 8086
        ->  class yes  [0.860]

    Rule 77/8: (19.5/2.1, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = poor
        savings_balance in {< 100 DM, > 1000 DM, 100 - 500 DM}
        percent_of_income > 1
        housing = own
        job in {management, skilled}
        ->  class yes  [0.857]

    Rule 77/9: (19.3/2.2, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = perfect
        years_at_residence > 2
        ->  class yes  [0.850]

    Rule 77/10: (32.9/4.9, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 27
        credit_history = very good
        age > 23
        ->  class yes  [0.832]

    Rule 77/11: (36.5/8.4, lift 1.4)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        age > 30
        job = unskilled
        phone = FALSE
        ->  class yes  [0.756]

    Rule 77/12: (51.6/12.9, lift 1.4)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        amount > 1922
        savings_balance = < 100 DM
        years_at_residence > 1
        age <= 61
        dependents <= 1
        ->  class yes  [0.741]

    Rule 77/13: (30.1/7.4, lift 1.4)
        credit_history = good
        purpose in {business, car0, education, renovations}
        savings_balance = < 100 DM
        phone = FALSE
        ->  class yes  [0.739]

    Rule 77/14: (26.2/7.4, lift 1.3)
        employment_duration = 1 - 4 years
        housing = other
        ->  class yes  [0.702]

    Rule 77/15: (63.8/20.6, lift 1.3)
        checking_balance = 1 - 200 DM
        credit_history = good
        purpose = furniture/appliances
        job = skilled
        ->  class yes  [0.671]

    Rule 77/16: (87.2/28.4, lift 1.3)
        purpose = car
        percent_of_income > 1
        age <= 28
        ->  class yes  [0.671]

    Rule 77/17: (305.7/131.5, lift 1.1)
        checking_balance = < 0 DM
        ->  class yes  [0.570]

    Rule 77/18: (17.3, lift 2.0)
        checking_balance in {> 200 DM, unknown}
        months_loan_duration <= 18
        employment_duration = > 7 years
        dependents <= 1
        ->  class no  [0.948]

    Rule 77/19: (10.1, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        purpose = furniture/appliances
        age <= 30
        job = unskilled
        phone = FALSE
        ->  class no  [0.918]

    Rule 77/20: (9.8, lift 2.0)
        credit_history = good
        purpose = car
        amount > 4370
        amount <= 8086
        savings_balance = < 100 DM
        other_credit = none
        ->  class no  [0.915]

    Rule 77/21: (9.6, lift 2.0)
        credit_history = critical
        age > 61
        ->  class no  [0.914]

    Rule 77/22: (9.3, lift 1.9)
        checking_balance in {> 200 DM, unknown}
        employment_duration = 1 - 4 years
        percent_of_income <= 1
        ->  class no  [0.912]

    Rule 77/23: (9.2, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration <= 16
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        age > 23
        job = skilled
        ->  class no  [0.911]

    Rule 77/24: (9, lift 1.9)
        checking_balance = < 0 DM
        credit_history = good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        years_at_residence <= 1
        job = skilled
        phone = FALSE
        ->  class no  [0.909]

    Rule 77/25: (9, lift 1.9)
        checking_balance in {> 200 DM, unknown}
        purpose = car
        employment_duration = < 1 year
        ->  class no  [0.909]

    Rule 77/26: (7.9, lift 1.9)
        credit_history = poor
        savings_balance in {< 100 DM, 100 - 500 DM}
        housing = own
        job = unskilled
        ->  class no  [0.899]

    Rule 77/27: (17.3/1.1, lift 1.9)
        checking_balance in {> 200 DM, unknown}
        employment_duration = 4 - 7 years
        years_at_residence > 2
        ->  class no  [0.892]

    Rule 77/28: (6.1, lift 1.9)
        credit_history = good
        savings_balance = > 1000 DM
        phone = FALSE
        ->  class no  [0.877]

    Rule 77/29: (5.9, lift 1.9)
        credit_history = very good
        age <= 23
        ->  class no  [0.874]

    Rule 77/30: (5.2, lift 1.8)
        purpose = education
        savings_balance = unknown
        job = skilled
        ->  class no  [0.862]

    Rule 77/31: (4.8, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = poor
        savings_balance = unknown
        ->  class no  [0.854]

    Rule 77/32: (24.7/3.3, lift 1.8)
        credit_history = good
        purpose = car
        amount <= 8086
        age > 28
        phone = TRUE
        ->  class no  [0.838]

    Rule 77/33: (26.6/3.9, lift 1.8)
        checking_balance in {> 200 DM, unknown}
        employment_duration = > 7 years
        percent_of_income > 3
        dependents <= 1
        ->  class no  [0.829]

    Rule 77/34: (31.9/5.2, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        purpose in {business, car, furniture/appliances}
        amount <= 1922
        dependents <= 1
        ->  class no  [0.818]

    Rule 77/35: (18.6/3.3, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        purpose = furniture/appliances
        employment_duration = 1 - 4 years
        other_credit = none
        existing_loans_count <= 1
        job in {management, skilled}
        ->  class no  [0.793]

    Rule 77/36: (14.6/2.5, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        savings_balance in {> 1000 DM, 100 - 500 DM, unknown}
        ->  class no  [0.787]

    Rule 77/37: (17/3.6, lift 1.6)
        checking_balance in {> 200 DM, unknown}
        amount <= 1740
        employment_duration = < 1 year
        dependents <= 1
        ->  class no  [0.759]

    Rule 77/38: (628.2/319.4, lift 1.0)
        housing = own
        ->  class no  [0.492]

    Default class: no

    -----  Trial 78:  -----

    Rules:

    Rule 78/1: (9.2/0.4, lift 1.8)
        amount <= 1264
        employment_duration = unemployed
        existing_loans_count <= 1
        dependents <= 1
        ->  class yes  [0.880]

    Rule 78/2: (4.8, lift 1.7)
        months_loan_duration <= 8
        amount > 4057
        existing_loans_count <= 1
        ->  class yes  [0.854]

    Rule 78/3: (12.4/2, lift 1.6)
        months_loan_duration > 16
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        dependents > 1
        ->  class yes  [0.789]

    Rule 78/4: (19.1/4.8, lift 1.5)
        savings_balance = 100 - 500 DM
        employment_duration = < 1 year
        ->  class yes  [0.725]

    Rule 78/5: (842.7/410.9, lift 1.0)
        months_loan_duration > 8
        ->  class yes  [0.512]

    Rule 78/6: (10.9, lift 1.8)
        credit_history = good
        employment_duration = 4 - 7 years
        age > 23
        housing in {other, rent}
        dependents <= 1
        ->  class no  [0.923]

    Rule 78/7: (9.1, lift 1.8)
        savings_balance = unknown
        employment_duration = > 7 years
        existing_loans_count > 1
        ->  class no  [0.910]

    Rule 78/8: (8.2, lift 1.8)
        checking_balance in {> 200 DM, unknown}
        credit_history = good
        employment_duration = 4 - 7 years
        age > 23
        dependents <= 1
        ->  class no  [0.902]

    Rule 78/9: (7.8, lift 1.8)
        employment_duration = 4 - 7 years
        age > 49
        ->  class no  [0.898]

    Rule 78/10: (16.7/1, lift 1.8)
        months_loan_duration > 8
        purpose in {car, car0, furniture/appliances, renovations}
        amount > 1264
        employment_duration = unemployed
        years_at_residence > 1
        existing_loans_count <= 1
        dependents <= 1
        ->  class no  [0.894]

    Rule 78/11: (17/1, lift 1.8)
        employment_duration = > 7 years
        age <= 28
        ->  class no  [0.893]

    Rule 78/12: (20.3/1.8, lift 1.7)
        employment_duration = > 7 years
        other_credit in {bank, none}
        job = unskilled
        dependents <= 1
        ->  class no  [0.875]

    Rule 78/13: (5.3, lift 1.7)
        purpose = education
        savings_balance = unknown
        employment_duration = 1 - 4 years
        ->  class no  [0.862]

    Rule 78/14: (22/2.3, lift 1.7)
        months_loan_duration > 8
        months_loan_duration <= 16
        credit_history in {critical, perfect, poor}
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        other_credit = none
        ->  class no  [0.861]

    Rule 78/15: (4.2, lift 1.7)
        savings_balance = 500 - 1000 DM
        employment_duration = 1 - 4 years
        phone = TRUE
        ->  class no  [0.839]

    Rule 78/16: (3.5, lift 1.6)
        savings_balance = unknown
        employment_duration = 1 - 4 years
        job = management
        ->  class no  [0.818]

    Rule 78/17: (11.3/1.5, lift 1.6)
        months_loan_duration > 9
        savings_balance = < 100 DM
        employment_duration = < 1 year
        job = management
        ->  class no  [0.816]

    Rule 78/18: (28.1/5.1, lift 1.6)
        purpose in {car, furniture/appliances}
        employment_duration = > 7 years
        other_credit = none
        existing_loans_count <= 1
        job = skilled
        phone = FALSE
        ->  class no  [0.798]

    Rule 78/19: (22.9/4.1, lift 1.6)
        employment_duration = > 7 years
        percent_of_income > 1
        other_credit = none
        job = skilled
        phone = TRUE
        ->  class no  [0.793]

    Rule 78/20: (34/6.5, lift 1.6)
        months_loan_duration > 9
        months_loan_duration <= 16
        credit_history = good
        purpose in {business, car, furniture/appliances}
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        age > 20
        other_credit = none
        ->  class no  [0.792]

    Rule 78/21: (34.8/8.1, lift 1.5)
        savings_balance in {100 - 500 DM, unknown}
        employment_duration = 4 - 7 years
        ->  class no  [0.754]

    Rule 78/22: (14.8/3.2, lift 1.5)
        credit_history = good
        employment_duration = 4 - 7 years
        dependents > 1
        ->  class no  [0.749]

    Rule 78/23: (57.3/16.5, lift 1.4)
        months_loan_duration <= 8
        ->  class no  [0.704]

    Rule 78/24: (103.3/36.9, lift 1.3)
        amount <= 10366
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        percent_of_income <= 3
        ->  class no  [0.640]

    Default class: no

    -----  Trial 79:  -----

    Rules:

    Rule 79/1: (10/0.3, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        amount > 1386
        age <= 28
        phone = TRUE
        ->  class yes  [0.888]

    Rule 79/2: (17.3/1.2, lift 2.0)
        checking_balance = 1 - 200 DM
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        employment_duration = < 1 year
        housing = rent
        ->  class yes  [0.885]

    Rule 79/3: (15.1/1, lift 2.0)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = car
        amount <= 1103
        savings_balance in {< 100 DM, 100 - 500 DM, unknown}
        existing_loans_count <= 1
        ->  class yes  [0.884]

    Rule 79/4: (5.3, lift 1.9)
        months_loan_duration > 40
        credit_history = good
        purpose = furniture/appliances
        employment_duration = 1 - 4 years
        ->  class yes  [0.863]

    Rule 79/5: (4.4, lift 1.9)
        checking_balance = unknown
        credit_history = critical
        purpose = education
        age <= 30
        ->  class yes  [0.843]

    Rule 79/6: (19.5/2.6, lift 1.9)
        credit_history = critical
        purpose in {car, furniture/appliances}
        other_credit = bank
        job in {skilled, unskilled}
        ->  class yes  [0.832]

    Rule 79/7: (11.2/1.3, lift 1.8)
        credit_history = perfect
        other_credit = none
        phone = TRUE
        ->  class yes  [0.827]

    Rule 79/8: (3.8, lift 1.8)
        credit_history = perfect
        other_credit = store
        ->  class yes  [0.826]

    Rule 79/9: (13.5/1.7, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        purpose = car
        amount > 1204
        amount <= 1386
        ->  class yes  [0.822]

    Rule 79/10: (16/2.3, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = good
        purpose = car
        age > 43
        ->  class yes  [0.817]

    Rule 79/11: (12.8/1.8, lift 1.8)
        credit_history = good
        purpose = education
        housing = own
        existing_loans_count <= 1
        ->  class yes  [0.812]

    Rule 79/12: (9.3/1.2, lift 1.8)
        credit_history = poor
        age > 52
        ->  class yes  [0.804]

    Rule 79/13: (20.5/3.7, lift 1.8)
        checking_balance = 1 - 200 DM
        credit_history = good
        amount > 888
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.790]

    Rule 79/14: (16.3/3.1, lift 1.7)
        checking_balance = 1 - 200 DM
        credit_history = critical
        years_at_residence > 1
        other_credit = none
        job in {management, unskilled}
        ->  class yes  [0.775]

    Rule 79/15: (13/2.5, lift 1.7)
        credit_history = good
        purpose = furniture/appliances
        employment_duration = unemployed
        ->  class yes  [0.765]

    Rule 79/16: (16.1/3.3, lift 1.7)
        checking_balance = 1 - 200 DM
        credit_history = critical
        years_at_residence > 1
        years_at_residence <= 2
        ->  class yes  [0.763]

    Rule 79/17: (41.1/10.5, lift 1.6)
        credit_history = good
        amount > 1209
        existing_loans_count > 1
        dependents <= 1
        ->  class yes  [0.733]

    Rule 79/18: (25/8.1, lift 1.5)
        employment_duration = > 7 years
        existing_loans_count <= 1
        dependents > 1
        ->  class yes  [0.665]

    Rule 79/19: (32.3/10.8, lift 1.5)
        purpose = furniture/appliances
        employment_duration = 1 - 4 years
        housing in {other, rent}
        job in {skilled, unskilled}
        ->  class yes  [0.656]

    Rule 79/20: (39.8/14.5, lift 1.4)
        credit_history = poor
        percent_of_income > 3
        ->  class yes  [0.630]

    Rule 79/21: (43.5/16.9, lift 1.3)
        credit_history = very good
        other_credit in {bank, none}
        ->  class yes  [0.606]

    Rule 79/22: (645.6/340.2, lift 1.1)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        ->  class yes  [0.473]

    Rule 79/23: (19.5, lift 1.7)
        checking_balance = unknown
        credit_history = good
        purpose = car
        age > 27
        existing_loans_count <= 1
        ->  class no  [0.954]

    Rule 79/24: (13.1, lift 1.7)
        months_loan_duration <= 18
        credit_history = good
        purpose = business
        ->  class no  [0.934]

    Rule 79/25: (7, lift 1.6)
        credit_history = critical
        other_credit = bank
        job in {management, unemployed}
        ->  class no  [0.889]

    Rule 79/26: (24/2.3, lift 1.6)
        credit_history = poor
        percent_of_income <= 3
        years_at_residence > 1
        age <= 52
        ->  class no  [0.872]

    Rule 79/27: (16.2/1.9, lift 1.5)
        purpose = furniture/appliances
        employment_duration = 4 - 7 years
        years_at_residence > 3
        other_credit = none
        ->  class no  [0.842]

    Rule 79/28: (27.3/4.2, lift 1.5)
        credit_history = good
        purpose = furniture/appliances
        amount <= 5711
        employment_duration = > 7 years
        existing_loans_count <= 1
        dependents <= 1
        ->  class no  [0.822]

    Rule 79/29: (14.9/2.2, lift 1.5)
        purpose = education
        housing in {other, rent}
        phone = TRUE
        ->  class no  [0.810]

    Rule 79/30: (36.4/6.6, lift 1.5)
        checking_balance in {< 0 DM, > 200 DM, unknown}
        credit_history = good
        purpose = furniture/appliances
        employment_duration = < 1 year
        existing_loans_count <= 1
        job in {management, skilled}
        ->  class no  [0.801]

    Rule 79/31: (42.4/8.6, lift 1.4)
        credit_history = good
        purpose = car
        amount > 1386
        age > 28
        age <= 43
        ->  class no  [0.783]

    Rule 79/32: (54.4/13.4, lift 1.4)
        purpose = car
        amount > 1386
        age <= 43
        existing_loans_count <= 1
        phone = FALSE
        ->  class no  [0.744]

    Rule 79/33: (85/27.7, lift 1.2)
        credit_history = critical
        years_at_residence > 2
        job = skilled
        ->  class no  [0.671]

    Rule 79/34: (639.8/271.8, lift 1.0)
        housing = own
        ->  class no  [0.575]

    Default class: no

    -----  Trial 80:  -----

    Rules:

    Rule 80/1: (10.3, lift 2.0)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 30
        credit_history = good
        purpose = furniture/appliances
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years}
        years_at_residence > 1
        housing = own
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.919]

    Rule 80/2: (20.7/0.9, lift 2.0)
        months_loan_duration <= 30
        credit_history = good
        amount > 1231
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years}
        percent_of_income > 1
        years_at_residence > 1
        age <= 24
        housing = own
        job = skilled
        ->  class yes  [0.915]

    Rule 80/3: (9, lift 2.0)
        checking_balance = < 0 DM
        purpose = education
        savings_balance = < 100 DM
        age <= 43
        ->  class yes  [0.909]

    Rule 80/4: (7.3, lift 1.9)
        purpose in {education, renovations}
        savings_balance = > 1000 DM
        ->  class yes  [0.892]

    Rule 80/5: (6.3, lift 1.9)
        purpose = furniture/appliances
        savings_balance = < 100 DM
        age > 36
        other_credit = none
        job = management
        ->  class yes  [0.879]

    Rule 80/6: (10.9/0.6, lift 1.9)
        credit_history = good
        purpose in {car, furniture/appliances}
        amount <= 1597
        savings_balance = unknown
        employment_duration = 1 - 4 years
        ->  class yes  [0.874]

    Rule 80/7: (5.8, lift 1.9)
        credit_history in {perfect, very good}
        purpose = car
        savings_balance = 100 - 500 DM
        other_credit = none
        ->  class yes  [0.872]

    Rule 80/8: (13/1.1, lift 1.9)
        purpose = education
        savings_balance = < 100 DM
        age > 32
        age <= 43
        ->  class yes  [0.857]

    Rule 80/9: (12.8/1.4, lift 1.8)
        months_loan_duration > 14
        credit_history = good
        purpose = furniture/appliances
        amount > 1526
        savings_balance = < 100 DM
        other_credit = none
        job = unskilled
        ->  class yes  [0.837]

    Rule 80/10: (4.5, lift 1.8)
        purpose = car
        savings_balance = 100 - 500 DM
        job = unskilled
        ->  class yes  [0.833]

    Rule 80/11: (10.9/1.3, lift 1.8)
        purpose = renovations
        amount <= 2483
        savings_balance = < 100 DM
        ->  class yes  [0.819]

    Rule 80/12: (11/1.4, lift 1.8)
        months_loan_duration > 22
        credit_history = critical
        purpose = furniture/appliances
        percent_of_income <= 2
        ->  class yes  [0.813]

    Rule 80/13: (17.5/2.8, lift 1.8)
        purpose = business
        savings_balance = < 100 DM
        employment_duration in {> 7 years, unemployed}
        ->  class yes  [0.806]

    Rule 80/14: (20/3.6, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        ->  class yes  [0.789]

    Rule 80/15: (10/1.6, lift 1.7)
        purpose = education
        savings_balance = 100 - 500 DM
        ->  class yes  [0.781]

    Rule 80/16: (24.9/4.9, lift 1.7)
        purpose = car
        amount <= 7980
        savings_balance = < 100 DM
        other_credit in {bank, store}
        ->  class yes  [0.780]

    Rule 80/17: (37.8/8, lift 1.7)
        amount > 7980
        savings_balance = < 100 DM
        ->  class yes  [0.773]

    Rule 80/18: (22.9/5.3, lift 1.6)
        checking_balance = unknown
        savings_balance = < 100 DM
        age <= 23
        ->  class yes  [0.746]

    Rule 80/19: (60.8/17.7, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        savings_balance = < 100 DM
        percent_of_income > 2
        age <= 48
        dependents <= 1
        ->  class yes  [0.702]

    Rule 80/20: (16.7/4.7, lift 1.5)
        credit_history = good
        savings_balance = < 100 DM
        other_credit = store
        ->  class yes  [0.697]

    Rule 80/21: (102.8/43.4, lift 1.3)
        credit_history in {perfect, poor, very good}
        savings_balance = < 100 DM
        ->  class yes  [0.576]

    Rule 80/22: (301.9/149.3, lift 1.1)
        checking_balance = < 0 DM
        ->  class yes  [0.505]

    Rule 80/23: (18.7, lift 1.8)
        checking_balance = unknown
        purpose = car
        savings_balance = < 100 DM
        age > 23
        other_credit = none
        ->  class no  [0.952]

    Rule 80/24: (10.6, lift 1.7)
        purpose = car
        amount > 5302
        amount <= 7980
        savings_balance = < 100 DM
        percent_of_income <= 3
        other_credit = none
        ->  class no  [0.921]

    Rule 80/25: (28/3.1, lift 1.6)
        purpose in {business, car, furniture/appliances}
        savings_balance = > 1000 DM
        ->  class no  [0.865]

    Rule 80/26: (4.4, lift 1.6)
        savings_balance = 500 - 1000 DM
        years_at_residence <= 1
        ->  class no  [0.844]

    Rule 80/27: (24.7/4.3, lift 1.5)
        credit_history = good
        amount <= 7980
        savings_balance = < 100 DM
        other_credit = none
        job = management
        ->  class no  [0.801]

    Rule 80/28: (31.2/6.5, lift 1.4)
        checking_balance in {> 200 DM, 1 - 200 DM, unknown}
        amount > 1597
        savings_balance = unknown
        employment_duration = 1 - 4 years
        ->  class no  [0.774]

    Rule 80/29: (44.8/10.9, lift 1.4)
        months_loan_duration <= 14
        savings_balance = < 100 DM
        other_credit = none
        job = unskilled
        ->  class no  [0.745]

    Rule 80/30: (49.8/14.7, lift 1.3)
        savings_balance = unknown
        employment_duration in {< 1 year, 4 - 7 years, unemployed}
        ->  class no  [0.697]

    Rule 80/31: (832.5/369.2, lift 1.0)
        amount <= 7980
        ->  class no  [0.556]

    Default class: yes

    -----  Trial 81:  -----

    Rules:

    Rule 81/1: (11.2, lift 1.9)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 47
        credit_history in {good, perfect}
        amount > 8133
        ->  class yes  [0.924]

    Rule 81/2: (10.9, lift 1.9)
        checking_balance = < 0 DM
        credit_history in {good, perfect}
        savings_balance = < 100 DM
        percent_of_income > 2
        existing_loans_count > 1
        ->  class yes  [0.923]

    Rule 81/3: (7.4, lift 1.9)
        credit_history in {critical, poor, very good}
        purpose = car
        savings_balance = < 100 DM
        percent_of_income > 1
        other_credit = bank
        ->  class yes  [0.894]

    Rule 81/4: (19.5/1.4, lift 1.9)
        checking_balance = < 0 DM
        credit_history = good
        employment_duration = < 1 year
        percent_of_income > 2
        years_at_residence > 2
        age <= 39
        ->  class yes  [0.889]

    Rule 81/5: (24.4/2.2, lift 1.9)
        checking_balance = 1 - 200 DM
        credit_history in {good, perfect}
        employment_duration in {< 1 year, > 7 years, 1 - 4 years}
        housing = rent
        job in {skilled, unemployed, unskilled}
        ->  class yes  [0.879]

    Rule 81/6: (9.1/0.8, lift 1.8)
        checking_balance = < 0 DM
        months_loan_duration <= 47
        credit_history in {good, perfect}
        purpose in {education, renovations}
        ->  class yes  [0.841]

    Rule 81/7: (34.3/6.2, lift 1.7)
        months_loan_duration > 47
        credit_history in {good, perfect}
        years_at_residence > 1
        ->  class yes  [0.801]

    Rule 81/8: (16/2.6, lift 1.7)
        credit_history in {critical, very good}
        savings_balance = unknown
        percent_of_income > 3
        years_at_residence <= 3
        ->  class yes  [0.798]

    Rule 81/9: (8.9/1.2, lift 1.7)
        checking_balance = > 200 DM
        dependents > 1
        ->  class yes  [0.798]

    Rule 81/10: (24.4/5.7, lift 1.6)
        checking_balance = unknown
        credit_history in {good, perfect}
        percent_of_income <= 3
        existing_loans_count > 1
        ->  class yes  [0.747]

    Rule 81/11: (175.8/75.6, lift 1.2)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        years_at_residence <= 3
        ->  class yes  [0.569]

    Rule 81/12: (337/155.9, lift 1.1)
        savings_balance in {< 100 DM, 500 - 1000 DM, unknown}
        existing_loans_count <= 1
        phone = FALSE
        ->  class yes  [0.537]

    Rule 81/13: (18, lift 1.8)
        checking_balance = 1 - 200 DM
        months_loan_duration > 21
        months_loan_duration <= 47
        credit_history in {good, perfect}
        age <= 35
        other_credit = none
        housing = own
        ->  class no  [0.950]

    Rule 81/14: (15.1, lift 1.8)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 47
        credit_history in {good, perfect}
        employment_duration in {> 7 years, 4 - 7 years}
        age <= 35
        job = skilled
        ->  class no  [0.941]

    Rule 81/15: (13.5, lift 1.8)
        months_loan_duration <= 10
        credit_history in {good, perfect}
        amount <= 8133
        age > 35
        housing = own
        ->  class no  [0.936]

    Rule 81/16: (11.1, lift 1.8)
        checking_balance = < 0 DM
        credit_history = good
        purpose = furniture/appliances
        employment_duration = < 1 year
        years_at_residence <= 2
        ->  class no  [0.924]

    Rule 81/17: (11.2, lift 1.8)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 47
        credit_history in {good, perfect}
        amount <= 8133
        age <= 35
        other_credit = none
        housing = own
        job in {management, unemployed, unskilled}
        ->  class no  [0.924]

    Rule 81/18: (10.2, lift 1.7)
        checking_balance = unknown
        months_loan_duration <= 47
        savings_balance = unknown
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.918]

    Rule 81/19: (9, lift 1.7)
        checking_balance = 1 - 200 DM
        employment_duration in {4 - 7 years, unemployed}
        housing = rent
        ->  class no  [0.909]

    Rule 81/20: (8.5, lift 1.7)
        checking_balance = unknown
        months_loan_duration > 28
        months_loan_duration <= 47
        credit_history in {good, perfect}
        phone = FALSE
        ->  class no  [0.905]

    Rule 81/21: (5.2, lift 1.6)
        credit_history = poor
        savings_balance = unknown
        years_at_residence <= 3
        ->  class no  [0.862]

    Rule 81/22: (4.4, lift 1.6)
        savings_balance = 500 - 1000 DM
        job = management
        ->  class no  [0.844]

    Rule 81/23: (30.5/5.3, lift 1.5)
        credit_history in {critical, poor, very good}
        savings_balance in {< 100 DM, 100 - 500 DM, unknown}
        percent_of_income <= 1
        ->  class no  [0.807]

    Rule 81/24: (21/4.1, lift 1.5)
        checking_balance = < 0 DM
        purpose in {car, furniture/appliances}
        percent_of_income <= 2
        other_credit = none
        dependents > 1
        ->  class no  [0.780]

    Rule 81/25: (34.5/8.6, lift 1.4)
        credit_history in {good, perfect}
        percent_of_income <= 2
        other_credit = none
        housing = rent
        existing_loans_count <= 1
        ->  class no  [0.739]

    Rule 81/26: (127.1/42.2, lift 1.3)
        credit_history in {critical, poor, very good}
        percent_of_income > 1
        years_at_residence > 3
        ->  class no  [0.665]

    Rule 81/27: (842.4/391.6, lift 1.0)
        months_loan_duration <= 47
        ->  class no  [0.535]

    Default class: no

    -----  Trial 82:  -----

    Rules:

    Rule 82/1: (13.7, lift 2.0)
        credit_history = good
        purpose = furniture/appliances
        percent_of_income <= 3
        existing_loans_count > 1
        job = skilled
        ->  class yes  [0.936]

    Rule 82/2: (7.1, lift 1.9)
        employment_duration = unemployed
        years_at_residence <= 1
        ->  class yes  [0.890]

    Rule 82/3: (9.9/1.5, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        purpose = education
        years_at_residence <= 2
        ->  class yes  [0.786]

    Rule 82/4: (39.8/13.6, lift 1.4)
        checking_balance in {> 200 DM, unknown}
        purpose = business
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, unemployed}
        ->  class yes  [0.651]

    Rule 82/5: (581.6/280, lift 1.1)
        checking_balance in {< 0 DM, 1 - 200 DM}
        ->  class yes  [0.518]

    Rule 82/6: (16, lift 1.8)
        checking_balance = < 0 DM
        months_loan_duration > 22
        months_loan_duration <= 36
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        years_at_residence > 1
        ->  class no  [0.944]

    Rule 82/7: (11.4, lift 1.7)
        purpose = car
        amount <= 7485
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        percent_of_income <= 2
        ->  class no  [0.925]

    Rule 82/8: (10.5, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 33
        credit_history = critical
        employment_duration = > 7 years
        years_at_residence > 3
        job in {management, unskilled}
        ->  class no  [0.920]

    Rule 82/9: (9.2, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = < 1 year
        job = management
        ->  class no  [0.911]

    Rule 82/10: (12.9/0.6, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        years_at_residence <= 1
        other_credit = bank
        ->  class no  [0.895]

    Rule 82/11: (22.3/1.7, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        purpose = furniture/appliances
        amount > 1549
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, 4 - 7 years}
        years_at_residence <= 1
        ->  class no  [0.891]

    Rule 82/12: (5.9, lift 1.6)
        credit_history = good
        purpose in {business, education}
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, 4 - 7 years}
        years_at_residence <= 1
        ->  class no  [0.873]

    Rule 82/13: (10/0.9, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = poor
        employment_duration = 1 - 4 years
        other_credit in {bank, none}
        ->  class no  [0.846]

    Rule 82/14: (11.5/1.5, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {critical, perfect}
        years_at_residence <= 1
        ->  class no  [0.818]

    Rule 82/15: (10.5/1.3, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        savings_balance = unknown
        other_credit = bank
        ->  class no  [0.816]

    Rule 82/16: (8.2/0.9, lift 1.5)
        credit_history = good
        employment_duration = 4 - 7 years
        job = unskilled
        ->  class no  [0.813]

    Rule 82/17: (34.2/6.4, lift 1.5)
        months_loan_duration <= 16
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        percent_of_income > 1
        years_at_residence > 1
        years_at_residence <= 3
        ->  class no  [0.796]

    Rule 82/18: (20.2/3.8, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {critical, poor}
        employment_duration = 4 - 7 years
        years_at_residence > 1
        ->  class no  [0.785]

    Rule 82/19: (28.6/5.8, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        amount > 1275
        employment_duration = unemployed
        years_at_residence > 1
        dependents <= 1
        ->  class no  [0.777]

    Rule 82/20: (37.4/8.8, lift 1.4)
        months_loan_duration <= 33
        credit_history = good
        amount <= 5711
        employment_duration = > 7 years
        years_at_residence > 3
        ->  class no  [0.751]

    Rule 82/21: (38.5/10.4, lift 1.4)
        percent_of_income <= 1
        years_at_residence > 1
        years_at_residence <= 3
        ->  class no  [0.718]

    Rule 82/22: (318.4/121.3, lift 1.2)
        checking_balance in {> 200 DM, unknown}
        ->  class no  [0.618]

    Default class: no

    -----  Trial 83:  -----

    Rules:

    Rule 83/1: (12.3, lift 2.0)
        purpose = car
        savings_balance = < 100 DM
        age <= 25
        housing = own
        job in {skilled, unskilled}
        ->  class yes  [0.930]

    Rule 83/2: (12.2, lift 2.0)
        checking_balance = < 0 DM
        purpose = car
        amount <= 7308
        savings_balance = < 100 DM
        years_at_residence <= 2
        housing = own
        existing_loans_count <= 1
        ->  class yes  [0.930]

    Rule 83/3: (10.4, lift 2.0)
        purpose = furniture/appliances
        amount > 4594
        savings_balance = < 100 DM
        employment_duration = < 1 year
        job in {skilled, unskilled}
        ->  class yes  [0.920]

    Rule 83/4: (9.6, lift 2.0)
        purpose = car
        savings_balance = < 100 DM
        other_credit = bank
        existing_loans_count > 1
        ->  class yes  [0.914]

    Rule 83/5: (15.8/0.8, lift 2.0)
        savings_balance = 500 - 1000 DM
        years_at_residence > 1
        years_at_residence <= 2
        age <= 42
        housing = own
        job in {skilled, unskilled}
        ->  class yes  [0.898]

    Rule 83/6: (7.4, lift 2.0)
        checking_balance = 1 - 200 DM
        credit_history = good
        amount > 888
        amount <= 1316
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        existing_loans_count <= 1
        ->  class yes  [0.894]

    Rule 83/7: (22.3/2.1, lift 1.9)
        months_loan_duration > 10
        savings_balance = < 100 DM
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, unemployed}
        percent_of_income > 1
        housing = own
        existing_loans_count <= 1
        dependents > 1
        ->  class yes  [0.871]

    Rule 83/8: (26.8/3.5, lift 1.9)
        credit_history in {perfect, poor, very good}
        housing = own
        job = management
        ->  class yes  [0.845]

    Rule 83/9: (28/4, lift 1.8)
        purpose = car
        age <= 29
        existing_loans_count > 1
        ->  class yes  [0.833]

    Rule 83/10: (11.3/1.4, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM}
        purpose = furniture/appliances
        employment_duration = < 1 year
        years_at_residence > 2
        housing = own
        ->  class yes  [0.821]

    Rule 83/11: (25.8/4.1, lift 1.8)
        months_loan_duration > 24
        savings_balance in {< 100 DM, 100 - 500 DM, unknown}
        housing = rent
        dependents <= 1
        ->  class yes  [0.818]

    Rule 83/12: (14.2/2.1, lift 1.8)
        purpose in {car, education, furniture/appliances}
        savings_balance = 100 - 500 DM
        existing_loans_count > 1
        job in {skilled, unskilled}
        phone = FALSE
        ->  class yes  [0.810]

    Rule 83/13: (13/1.9, lift 1.8)
        savings_balance = unknown
        percent_of_income > 3
        other_credit = bank
        housing = own
        job in {skilled, unskilled}
        ->  class yes  [0.808]

    Rule 83/14: (9.5/1.4, lift 1.7)
        months_loan_duration > 36
        purpose = furniture/appliances
        savings_balance = < 100 DM
        housing = own
        ->  class yes  [0.788]

    Rule 83/15: (13.1/2.3, lift 1.7)
        credit_history in {perfect, very good}
        savings_balance in {< 100 DM, 100 - 500 DM}
        housing = rent
        dependents <= 1
        ->  class yes  [0.784]

    Rule 83/16: (14.9/2.7, lift 1.7)
        purpose in {car, furniture/appliances}
        amount <= 1484
        savings_balance = 100 - 500 DM
        phone = FALSE
        ->  class yes  [0.778]

    Rule 83/17: (25.8/6.1, lift 1.6)
        savings_balance = < 100 DM
        years_at_residence <= 3
        housing = own
        existing_loans_count > 1
        job = management
        ->  class yes  [0.743]

    Rule 83/18: (13.6/3.3, lift 1.6)
        savings_balance in {> 1000 DM, 100 - 500 DM}
        housing = own
        job = management
        ->  class yes  [0.724]

    Rule 83/19: (66/23.1, lift 1.4)
        percent_of_income > 2
        housing = other
        ->  class yes  [0.646]

    Rule 83/20: (100.1/45.3, lift 1.2)
        housing = other
        ->  class yes  [0.547]

    Rule 83/21: (151.5/74, lift 1.1)
        housing = rent
        ->  class yes  [0.512]

    Rule 83/22: (143.3/70, lift 1.1)
        other_credit = bank
        ->  class yes  [0.511]

    Rule 83/23: (18.9/0.5, lift 1.7)
        amount > 1484
        savings_balance = 100 - 500 DM
        housing = own
        existing_loans_count <= 1
        job in {skilled, unskilled}
        ->  class no  [0.926]

    Rule 83/24: (10.4/0.2, lift 1.6)
        credit_history in {critical, good}
        years_at_residence > 3
        existing_loans_count > 1
        job = management
        ->  class no  [0.899]

    Rule 83/25: (7.1, lift 1.6)
        percent_of_income <= 1
        housing = own
        job in {skilled, unemployed, unskilled}
        dependents > 1
        ->  class no  [0.890]

    Rule 83/26: (6.8, lift 1.6)
        checking_balance in {1 - 200 DM, unknown}
        months_loan_duration <= 24
        credit_history = critical
        housing = rent
        ->  class no  [0.887]

    Rule 83/27: (6.8, lift 1.6)
        savings_balance = unknown
        percent_of_income <= 3
        other_credit = bank
        ->  class no  [0.886]

    Rule 83/28: (6.8, lift 1.6)
        savings_balance = > 1000 DM
        housing = rent
        dependents <= 1
        ->  class no  [0.886]

    Rule 83/29: (5, lift 1.6)
        savings_balance = 500 - 1000 DM
        years_at_residence <= 1
        ->  class no  [0.856]

    Rule 83/30: (15/2.4, lift 1.5)
        housing = rent
        dependents > 1
        ->  class no  [0.799]

    Rule 83/31: (22.8/4.7, lift 1.4)
        percent_of_income > 1
        percent_of_income <= 2
        housing = other
        ->  class no  [0.772]

    Rule 83/32: (37.5/8.2, lift 1.4)
        checking_balance = < 0 DM
        purpose = car
        amount <= 7308
        years_at_residence > 2
        existing_loans_count <= 1
        ->  class no  [0.767]

    Rule 83/33: (42.7/10.5, lift 1.4)
        credit_history in {poor, very good}
        employment_duration = 1 - 4 years
        job in {skilled, unskilled}
        ->  class no  [0.744]

    Rule 83/34: (47/11.6, lift 1.4)
        employment_duration = > 7 years
        existing_loans_count <= 1
        job in {skilled, unskilled}
        dependents <= 1
        ->  class no  [0.742]

    Rule 83/35: (65/19.9, lift 1.3)
        amount > 1597
        savings_balance = unknown
        housing = own
        ->  class no  [0.689]

    Rule 83/36: (194.6/72.5, lift 1.1)
        housing = own
        existing_loans_count > 1
        job in {skilled, unemployed, unskilled}
        ->  class no  [0.626]

    Rule 83/37: (111.7/41.6, lift 1.1)
        savings_balance = unknown
        job in {skilled, unemployed, unskilled}
        ->  class no  [0.625]

    Rule 83/38: (441.8/185.7, lift 1.1)
        months_loan_duration <= 24
        existing_loans_count <= 1
        ->  class no  [0.579]

    Default class: no

    -----  Trial 84:  -----

    Rules:

    Rule 84/1: (11.1, lift 1.9)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = perfect
        percent_of_income > 3
        ->  class yes  [0.924]

    Rule 84/2: (38.9/3.1, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 8
        credit_history = good
        purpose in {car, education, furniture/appliances, renovations}
        amount > 629
        amount <= 1047
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        ->  class yes  [0.900]

    Rule 84/3: (8, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 9
        credit_history = very good
        ->  class yes  [0.900]

    Rule 84/4: (8.3/0.2, lift 1.8)
        checking_balance = unknown
        credit_history = poor
        employment_duration = < 1 year
        ->  class yes  [0.879]

    Rule 84/5: (6.1, lift 1.8)
        checking_balance = unknown
        employment_duration = 4 - 7 years
        years_at_residence <= 2
        age <= 23
        ->  class yes  [0.876]

    Rule 84/6: (15.3/1.3, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 20
        credit_history = poor
        savings_balance = < 100 DM
        ->  class yes  [0.867]

    Rule 84/7: (3.9, lift 1.7)
        checking_balance = < 0 DM
        credit_history = very good
        existing_loans_count > 1
        ->  class yes  [0.829]

    Rule 84/8: (14.8/1.9, lift 1.7)
        months_loan_duration <= 24
        amount > 6361
        employment_duration = > 7 years
        ->  class yes  [0.829]

    Rule 84/9: (20.2/2.9, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = perfect
        purpose in {business, car, car0, renovations}
        years_at_residence > 1
        ->  class yes  [0.822]

    Rule 84/10: (29.4/5.1, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        amount > 8086
        ->  class yes  [0.806]

    Rule 84/11: (5.2/0.6, lift 1.6)
        credit_history = poor
        savings_balance = > 1000 DM
        ->  class yes  [0.778]

    Rule 84/12: (40.9/9.4, lift 1.6)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = good
        amount > 1047
        employment_duration = 1 - 4 years
        percent_of_income > 1
        years_at_residence > 1
        age <= 26
        ->  class yes  [0.758]

    Rule 84/13: (78.3/19, lift 1.5)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 11
        credit_history = good
        amount <= 8086
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        percent_of_income > 1
        age > 30
        housing = own
        existing_loans_count <= 1
        ->  class yes  [0.751]

    Rule 84/14: (93.2/35.9, lift 1.3)
        months_loan_duration > 8
        credit_history = good
        housing = rent
        ->  class yes  [0.613]

    Rule 84/15: (586.4/289.2, lift 1.0)
        savings_balance = < 100 DM
        ->  class yes  [0.507]

    Rule 84/16: (18.3, lift 1.9)
        checking_balance = unknown
        purpose = furniture/appliances
        employment_duration = 1 - 4 years
        job = skilled
        ->  class no  [0.951]

    Rule 84/17: (11.2, lift 1.8)
        credit_history = critical
        age > 61
        ->  class no  [0.924]

    Rule 84/18: (8.4, lift 1.8)
        checking_balance = unknown
        employment_duration = 1 - 4 years
        other_credit = none
        dependents > 1
        ->  class no  [0.903]

    Rule 84/19: (7.8, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = good
        amount <= 8086
        savings_balance = > 1000 DM
        ->  class no  [0.898]

    Rule 84/20: (13.9/1.1, lift 1.7)
        credit_history = critical
        savings_balance = < 100 DM
        age <= 53
        dependents > 1
        ->  class no  [0.869]

    Rule 84/21: (17.7/1.9, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = poor
        savings_balance in {100 - 500 DM, unknown}
        ->  class no  [0.852]

    Rule 84/22: (9.1/0.7, lift 1.6)
        credit_history = good
        amount > 1047
        other_credit = bank
        housing = other
        ->  class no  [0.843]

    Rule 84/23: (4.2, lift 1.6)
        credit_history = good
        years_at_residence <= 1
        housing = other
        ->  class no  [0.839]

    Rule 84/24: (22.3/4.7, lift 1.5)
        amount > 1047
        employment_duration in {4 - 7 years, unemployed}
        housing = rent
        ->  class no  [0.767]

    Rule 84/25: (44/9.8, lift 1.5)
        credit_history = critical
        amount <= 1922
        savings_balance = < 100 DM
        dependents <= 1
        ->  class no  [0.766]

    Rule 84/26: (197.4/78.6, lift 1.2)
        checking_balance = unknown
        months_loan_duration <= 30
        ->  class no  [0.601]

    Rule 84/27: (630.7/298.7, lift 1.0)
        housing = own
        ->  class no  [0.526]

    Default class: yes

    -----  Trial 85:  -----

    Rules:

    Rule 85/1: (10.4, lift 1.9)
        checking_balance = 1 - 200 DM
        employment_duration = < 1 year
        housing = rent
        job = skilled
        ->  class yes  [0.919]

    Rule 85/2: (9.8, lift 1.9)
        checking_balance = < 0 DM
        purpose in {education, renovations}
        savings_balance = < 100 DM
        job = skilled
        ->  class yes  [0.915]

    Rule 85/3: (16.3/0.8, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration > 22
        credit_history in {good, very good}
        savings_balance = unknown
        housing = own
        ->  class yes  [0.903]

    Rule 85/4: (15.7/1, lift 1.8)
        checking_balance = < 0 DM
        amount > 4933
        savings_balance = < 100 DM
        housing = own
        job = skilled
        ->  class yes  [0.888]

    Rule 85/5: (6.8, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM}
        employment_duration = 1 - 4 years
        housing = other
        ->  class yes  [0.886]

    Rule 85/6: (14.1/1, lift 1.8)
        months_loan_duration > 22
        employment_duration = < 1 year
        housing = rent
        ->  class yes  [0.878]

    Rule 85/7: (5.5, lift 1.8)
        age > 54
        housing = other
        dependents > 1
        ->  class yes  [0.867]

    Rule 85/8: (4.8, lift 1.8)
        percent_of_income <= 1
        housing = other
        dependents > 1
        ->  class yes  [0.853]

    Rule 85/9: (4.7, lift 1.8)
        credit_history = good
        employment_duration = 1 - 4 years
        housing = rent
        existing_loans_count > 1
        ->  class yes  [0.851]

    Rule 85/10: (4.3, lift 1.7)
        credit_history in {perfect, very good}
        employment_duration = > 7 years
        housing = other
        ->  class yes  [0.842]

    Rule 85/11: (15.2/1.9, lift 1.7)
        credit_history in {perfect, poor, very good}
        savings_balance = < 100 DM
        housing = own
        job = management
        ->  class yes  [0.831]

    Rule 85/12: (7.4/0.7, lift 1.7)
        amount > 12204
        housing = other
        ->  class yes  [0.822]

    Rule 85/13: (3.3, lift 1.7)
        years_at_residence <= 3
        housing = other
        dependents > 1
        ->  class yes  [0.811]

    Rule 85/14: (24/3.9, lift 1.7)
        checking_balance = 1 - 200 DM
        months_loan_duration > 18
        savings_balance = < 100 DM
        years_at_residence > 1
        other_credit = none
        housing = own
        job = skilled
        ->  class yes  [0.810]

    Rule 85/15: (3.2, lift 1.7)
        other_credit = store
        housing = other
        ->  class yes  [0.807]

    Rule 85/16: (26.9/5.1, lift 1.6)
        savings_balance = 500 - 1000 DM
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years}
        years_at_residence > 1
        other_credit = none
        job in {skilled, unskilled}
        dependents <= 1
        ->  class yes  [0.789]

    Rule 85/17: (8.4/1.3, lift 1.6)
        credit_history = very good
        other_credit = none
        housing = own
        ->  class yes  [0.777]

    Rule 85/18: (11.9/2.4, lift 1.6)
        savings_balance = < 100 DM
        job = unemployed
        ->  class yes  [0.752]

    Rule 85/19: (111.4/46.7, lift 1.2)
        other_credit = bank
        job in {skilled, unskilled}
        ->  class yes  [0.580]

    Rule 85/20: (624.4/306.9, lift 1.0)
        housing = own
        ->  class yes  [0.508]

    Rule 85/21: (17, lift 1.8)
        checking_balance = unknown
        purpose in {furniture/appliances, renovations}
        employment_duration = 1 - 4 years
        job = skilled
        ->  class no  [0.947]

    Rule 85/22: (14.1, lift 1.8)
        checking_balance = 1 - 200 DM
        savings_balance = unknown
        employment_duration in {< 1 year, > 7 years, 4 - 7 years}
        housing = own
        ->  class no  [0.938]

    Rule 85/23: (13.2, lift 1.8)
        checking_balance = 1 - 200 DM
        months_loan_duration > 18
        savings_balance = unknown
        housing = own
        ->  class no  [0.934]

    Rule 85/24: (12.9, lift 1.8)
        checking_balance = unknown
        employment_duration = 4 - 7 years
        phone = TRUE
        ->  class no  [0.933]

    Rule 85/25: (12.8, lift 1.8)
        checking_balance = unknown
        credit_history = critical
        savings_balance = < 100 DM
        other_credit = none
        housing = own
        ->  class no  [0.932]

    Rule 85/26: (12.4, lift 1.8)
        purpose in {business, renovations}
        savings_balance = 100 - 500 DM
        housing = own
        ->  class no  [0.931]

    Rule 85/27: (9.4, lift 1.8)
        months_loan_duration <= 8
        savings_balance = < 100 DM
        years_at_residence <= 2
        ->  class no  [0.912]

    Rule 85/28: (9.4, lift 1.8)
        credit_history in {perfect, poor}
        savings_balance = unknown
        housing = own
        ->  class no  [0.912]

    Rule 85/29: (9, lift 1.8)
        checking_balance = unknown
        amount <= 1474
        savings_balance = < 100 DM
        housing = own
        job = skilled
        ->  class no  [0.909]

    Rule 85/30: (8.9, lift 1.8)
        credit_history = critical
        savings_balance = unknown
        other_credit = none
        housing = own
        ->  class no  [0.908]

    Rule 85/31: (8.5, lift 1.8)
        savings_balance = 100 - 500 DM
        housing = own
        job = skilled
        phone = TRUE
        ->  class no  [0.905]

    Rule 85/32: (8.3, lift 1.8)
        checking_balance = unknown
        employment_duration = 1 - 4 years
        percent_of_income <= 1
        ->  class no  [0.903]

    Rule 85/33: (7.7, lift 1.7)
        savings_balance = 100 - 500 DM
        age > 42
        housing = own
        ->  class no  [0.897]

    Rule 85/34: (7.7, lift 1.7)
        savings_balance = < 100 DM
        years_at_residence <= 1
        job = management
        ->  class no  [0.896]

    Rule 85/35: (24.4/2, lift 1.7)
        amount <= 12204
        employment_duration in {< 1 year, 4 - 7 years, unemployed}
        housing = other
        dependents <= 1
        ->  class no  [0.886]

    Rule 85/36: (5.5, lift 1.7)
        credit_history = critical
        other_credit = bank
        job = management
        ->  class no  [0.867]

    Rule 85/37: (4.8, lift 1.7)
        savings_balance = 500 - 1000 DM
        other_credit = none
        dependents > 1
        ->  class no  [0.853]

    Rule 85/38: (21.6/2.7, lift 1.6)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 18
        savings_balance = < 100 DM
        years_at_residence > 2
        job = skilled
        ->  class no  [0.844]

    Rule 85/39: (4.2, lift 1.6)
        savings_balance = 500 - 1000 DM
        years_at_residence <= 1
        ->  class no  [0.838]

    Rule 85/40: (14/1.7, lift 1.6)
        savings_balance = > 1000 DM
        age <= 52
        housing = own
        ->  class no  [0.831]

    Rule 85/41: (9.8/1, lift 1.6)
        credit_history = good
        employment_duration = 1 - 4 years
        housing = rent
        existing_loans_count <= 1
        phone = TRUE
        ->  class no  [0.831]

    Rule 85/42: (3.5, lift 1.6)
        savings_balance = 500 - 1000 DM
        job = management
        ->  class no  [0.817]

    Rule 85/43: (8.6/1.1, lift 1.5)
        employment_duration = 1 - 4 years
        housing = rent
        dependents > 1
        ->  class no  [0.796]

    Rule 85/44: (17.5/3.4, lift 1.5)
        checking_balance = < 0 DM
        credit_history in {critical, good, perfect}
        purpose in {business, car0}
        ->  class no  [0.773]

    Rule 85/45: (48.8/13.2, lift 1.4)
        employment_duration in {> 7 years, 4 - 7 years}
        housing = rent
        ->  class no  [0.720]

    Rule 85/46: (75.9/25.1, lift 1.3)
        purpose = car
        amount > 1388
        amount <= 3031
        ->  class no  [0.665]

    Rule 85/47: (351.7/159.3, lift 1.1)
        years_at_residence > 3
        ->  class no  [0.547]

    Default class: no

    -----  Trial 86:  -----

    Rules:

    Rule 86/1: (12.8, lift 2.0)
        checking_balance = < 0 DM
        savings_balance = < 100 DM
        housing in {other, rent}
        job = skilled
        phone = TRUE
        ->  class yes  [0.933]

    Rule 86/2: (13.5, lift 2.0)
        amount > 7596
        savings_balance = < 100 DM
        percent_of_income > 2
        ->  class yes  [0.929]

    Rule 86/3: (10.9, lift 2.0)
        checking_balance = unknown
        purpose in {business, renovations}
        employment_duration = < 1 year
        ->  class yes  [0.923]

    Rule 86/4: (8.8, lift 2.0)
        checking_balance = < 0 DM
        purpose in {education, renovations}
        savings_balance = < 100 DM
        job = skilled
        ->  class yes  [0.907]

    Rule 86/5: (11.8/0.3, lift 2.0)
        checking_balance = 1 - 200 DM
        credit_history = good
        savings_balance = < 100 DM
        percent_of_income > 2
        age > 35
        existing_loans_count <= 1
        ->  class yes  [0.905]

    Rule 86/6: (8.5, lift 2.0)
        checking_balance = unknown
        purpose = furniture/appliances
        amount > 4594
        employment_duration = < 1 year
        ->  class yes  [0.905]

    Rule 86/7: (20.6/1.5, lift 1.9)
        checking_balance = < 0 DM
        credit_history in {good, poor, very good}
        savings_balance = < 100 DM
        job = skilled
        phone = TRUE
        ->  class yes  [0.888]

    Rule 86/8: (6.4, lift 1.9)
        checking_balance = < 0 DM
        percent_of_income <= 1
        job = management
        ->  class yes  [0.881]

    Rule 86/9: (5.7, lift 1.9)
        checking_balance = < 0 DM
        percent_of_income <= 2
        age <= 21
        job = skilled
        ->  class yes  [0.869]

    Rule 86/10: (6.7/0.2, lift 1.9)
        employment_duration = 4 - 7 years
        age <= 22
        phone = FALSE
        ->  class yes  [0.865]

    Rule 86/11: (4.9, lift 1.9)
        checking_balance = unknown
        purpose = furniture/appliances
        employment_duration = < 1 year
        age > 41
        ->  class yes  [0.855]

    Rule 86/12: (17.2/2, lift 1.8)
        checking_balance = < 0 DM
        savings_balance = < 100 DM
        percent_of_income > 2
        years_at_residence > 3
        housing = rent
        job = skilled
        ->  class yes  [0.844]

    Rule 86/13: (13.3/1.5, lift 1.8)
        checking_balance = 1 - 200 DM
        credit_history = critical
        savings_balance = < 100 DM
        percent_of_income <= 2
        years_at_residence > 1
        housing = own
        ->  class yes  [0.840]

    Rule 86/14: (8.2/0.6, lift 1.8)
        checking_balance = 1 - 200 DM
        savings_balance = 100 - 500 DM
        job = management
        ->  class yes  [0.840]

    Rule 86/15: (19.9/2.6, lift 1.8)
        checking_balance = 1 - 200 DM
        credit_history = critical
        savings_balance = < 100 DM
        years_at_residence > 1
        age <= 33
        housing = own
        ->  class yes  [0.836]

    Rule 86/16: (36/7.7, lift 1.7)
        checking_balance = < 0 DM
        savings_balance = < 100 DM
        percent_of_income > 2
        age > 28
        job = skilled
        phone = FALSE
        ->  class yes  [0.772]

    Rule 86/17: (13.5/2.7, lift 1.6)
        checking_balance = unknown
        employment_duration = unemployed
        percent_of_income > 2
        ->  class yes  [0.759]

    Rule 86/18: (29.1/7.9, lift 1.5)
        checking_balance = < 0 DM
        savings_balance = unknown
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        ->  class yes  [0.713]

    Rule 86/19: (7.9/1.9, lift 1.5)
        checking_balance = > 200 DM
        dependents > 1
        ->  class yes  [0.704]

    Rule 86/20: (45.7/19.2, lift 1.3)
        credit_history = perfect
        ->  class yes  [0.577]

    Rule 86/21: (230/111.1, lift 1.1)
        months_loan_duration > 26
        ->  class yes  [0.517]

    Rule 86/22: (11.6, lift 1.7)
        checking_balance = unknown
        months_loan_duration > 24
        employment_duration = > 7 years
        ->  class no  [0.927]

    Rule 86/23: (11.3, lift 1.7)
        checking_balance = unknown
        employment_duration = 4 - 7 years
        phone = TRUE
        ->  class no  [0.925]

    Rule 86/24: (7.7, lift 1.7)
        checking_balance = unknown
        employment_duration = 1 - 4 years
        percent_of_income <= 1
        ->  class no  [0.896]

    Rule 86/25: (5.8, lift 1.6)
        checking_balance = unknown
        employment_duration = 1 - 4 years
        other_credit = none
        dependents > 1
        ->  class no  [0.872]

    Rule 86/26: (9.6/0.7, lift 1.6)
        savings_balance = 100 - 500 DM
        housing = own
        job in {skilled, unskilled}
        phone = TRUE
        ->  class no  [0.853]

    Rule 86/27: (42.4/6.3, lift 1.5)
        checking_balance = unknown
        credit_history in {critical, perfect}
        other_credit = none
        ->  class no  [0.835]

    Rule 86/28: (12.7/1.9, lift 1.5)
        housing = own
        existing_loans_count <= 1
        job = unskilled
        phone = TRUE
        ->  class no  [0.805]

    Rule 86/29: (58.4/16.5, lift 1.3)
        checking_balance = < 0 DM
        purpose in {business, car, furniture/appliances}
        percent_of_income <= 2
        age > 21
        phone = FALSE
        ->  class no  [0.711]

    Rule 86/30: (65/19.2, lift 1.3)
        checking_balance = > 200 DM
        dependents <= 1
        ->  class no  [0.698]

    Rule 86/31: (839.1/380.6, lift 1.0)
        months_loan_duration <= 42
        ->  class no  [0.546]

    Default class: no

    -----  Trial 87:  -----

    Rules:

    Rule 87/1: (15.2, lift 1.9)
        credit_history = good
        purpose = furniture/appliances
        percent_of_income <= 3
        existing_loans_count > 1
        job = skilled
        ->  class yes  [0.942]

    Rule 87/2: (9.5, lift 1.8)
        checking_balance = < 0 DM
        purpose = car
        amount <= 1484
        age <= 36
        housing = own
        job = skilled
        ->  class yes  [0.913]

    Rule 87/3: (9.5, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        employment_duration = > 7 years
        years_at_residence <= 3
        housing = rent
        ->  class yes  [0.913]

    Rule 87/4: (14.4/0.5, lift 1.8)
        purpose = car
        savings_balance = < 100 DM
        percent_of_income > 1
        other_credit in {bank, store}
        job = skilled
        ->  class yes  [0.910]

    Rule 87/5: (8.7, lift 1.8)
        checking_balance = < 0 DM
        months_loan_duration > 22
        months_loan_duration <= 36
        purpose = car
        savings_balance = < 100 DM
        percent_of_income > 1
        age <= 36
        job = skilled
        ->  class yes  [0.906]

    Rule 87/6: (9.5/0.1, lift 1.8)
        age <= 28
        housing = own
        job = unemployed
        ->  class yes  [0.902]

    Rule 87/7: (21.6/2, lift 1.7)
        percent_of_income > 3
        age > 27
        age <= 38
        other_credit = none
        job = unskilled
        ->  class yes  [0.873]

    Rule 87/8: (23.8/2.6, lift 1.7)
        percent_of_income > 3
        job = unskilled
        dependents > 1
        ->  class yes  [0.861]

    Rule 87/9: (22.9/3.8, lift 1.6)
        credit_history = good
        amount > 4473
        housing = own
        job = management
        ->  class yes  [0.807]

    Rule 87/10: (49/12.9, lift 1.4)
        employment_duration = < 1 year
        age <= 33
        housing = rent
        ->  class yes  [0.726]

    Rule 87/11: (39.9/12.9, lift 1.3)
        purpose = business
        savings_balance = < 100 DM
        years_at_residence > 1
        ->  class yes  [0.667]

    Rule 87/12: (21.4/7.5, lift 1.3)
        job = unemployed
        ->  class yes  [0.636]

    Rule 87/13: (47.5/17.9, lift 1.2)
        other_credit = store
        ->  class yes  [0.618]

    Rule 87/14: (768/368, lift 1.0)
        years_at_residence > 1
        ->  class yes  [0.521]

    Rule 87/15: (16, lift 1.9)
        checking_balance = unknown
        months_loan_duration > 9
        credit_history = good
        purpose = furniture/appliances
        housing = own
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.944]

    Rule 87/16: (11.9, lift 1.9)
        checking_balance in {> 200 DM, 1 - 200 DM, unknown}
        employment_duration = 4 - 7 years
        housing = rent
        ->  class no  [0.928]

    Rule 87/17: (10.4, lift 1.9)
        months_loan_duration > 42
        savings_balance = unknown
        other_credit = none
        ->  class no  [0.920]

    Rule 87/18: (9.6, lift 1.9)
        checking_balance = unknown
        percent_of_income <= 3
        housing = own
        job = unskilled
        ->  class no  [0.914]

    Rule 87/19: (9.5, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration <= 16
        purpose = furniture/appliances
        savings_balance = < 100 DM
        housing = own
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.913]

    Rule 87/20: (9.1, lift 1.8)
        savings_balance = unknown
        employment_duration in {< 1 year, 4 - 7 years}
        housing = own
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.910]

    Rule 87/21: (9.5/0.1, lift 1.8)
        months_loan_duration > 36
        purpose = car
        age <= 36
        job = skilled
        ->  class no  [0.902]

    Rule 87/22: (6.3, lift 1.8)
        purpose = car
        percent_of_income <= 1
        housing = own
        job = skilled
        ->  class no  [0.880]

    Rule 87/23: (16.6/1.3, lift 1.8)
        savings_balance in {> 1000 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        percent_of_income <= 3
        housing = own
        job = unskilled
        ->  class no  [0.878]

    Rule 87/24: (5.1, lift 1.7)
        age > 28
        housing = own
        job = unemployed
        ->  class no  [0.859]

    Rule 87/25: (26.2/3.1, lift 1.7)
        months_loan_duration <= 14
        percent_of_income <= 3
        other_credit in {bank, none}
        housing = own
        job = unskilled
        ->  class no  [0.856]

    Rule 87/26: (4.9, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        employment_duration = > 7 years
        housing = rent
        ->  class no  [0.854]

    Rule 87/27: (4.7, lift 1.7)
        employment_duration = < 1 year
        age > 33
        housing = rent
        ->  class no  [0.848]

    Rule 87/28: (15.2/1.6, lift 1.7)
        purpose = car
        age > 36
        housing = own
        job = skilled
        ->  class no  [0.847]

    Rule 87/29: (4.1, lift 1.7)
        checking_balance = > 200 DM
        purpose in {business, furniture/appliances}
        housing = rent
        ->  class no  [0.837]

    Rule 87/30: (4.1, lift 1.7)
        savings_balance = < 100 DM
        percent_of_income <= 3
        other_credit = store
        job = unskilled
        ->  class no  [0.835]

    Rule 87/31: (10.9/1.4, lift 1.7)
        purpose = education
        savings_balance = unknown
        ->  class no  [0.818]

    Rule 87/32: (3.2, lift 1.6)
        savings_balance = unknown
        other_credit = bank
        housing = other
        ->  class no  [0.807]

    Rule 87/33: (17.2/2.9, lift 1.6)
        credit_history = good
        amount <= 4473
        housing = own
        job = management
        ->  class no  [0.798]

    Rule 87/34: (14.5/2.4, lift 1.6)
        purpose = furniture/appliances
        savings_balance in {> 1000 DM, 100 - 500 DM, 500 - 1000 DM}
        housing = own
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.791]

    Rule 87/35: (8.9/1.4, lift 1.6)
        credit_history = poor
        percent_of_income <= 1
        ->  class no  [0.782]

    Rule 87/36: (18.5/3.6, lift 1.6)
        checking_balance = > 200 DM
        purpose = furniture/appliances
        other_credit in {bank, none}
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.778]

    Rule 87/37: (20.3/3.9, lift 1.6)
        credit_history = critical
        purpose = furniture/appliances
        percent_of_income > 3
        other_credit = none
        job = skilled
        ->  class no  [0.778]

    Rule 87/38: (30.6/7, lift 1.5)
        months_loan_duration <= 22
        purpose = car
        amount > 1484
        age <= 36
        other_credit = none
        job = skilled
        ->  class no  [0.756]

    Rule 87/39: (20.2/5, lift 1.5)
        savings_balance = < 100 DM
        percent_of_income <= 2
        housing = other
        ->  class no  [0.729]

    Rule 87/40: (28.5/7.7, lift 1.5)
        savings_balance = < 100 DM
        percent_of_income > 3
        other_credit = none
        housing = own
        job = unskilled
        dependents <= 1
        ->  class no  [0.716]

    Rule 87/41: (81.9/29.9, lift 1.3)
        savings_balance in {< 100 DM, unknown}
        years_at_residence > 1
        housing = own
        existing_loans_count > 1
        job = skilled
        ->  class no  [0.632]

    Rule 87/42: (79/32.4, lift 1.2)
        purpose = business
        housing = own
        ->  class no  [0.588]

    Rule 87/43: (623.5/302.8, lift 1.0)
        housing = own
        ->  class no  [0.514]

    Default class: no

    -----  Trial 88:  -----

    Rules:

    Rule 88/1: (12, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 22
        credit_history = very good
        purpose = car
        ->  class yes  [0.928]

    Rule 88/2: (9/0.7, lift 1.8)
        checking_balance = unknown
        credit_history = poor
        percent_of_income > 3
        age <= 33
        dependents <= 1
        ->  class yes  [0.845]

    Rule 88/3: (4.4, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        savings_balance = unknown
        job = management
        ->  class yes  [0.843]

    Rule 88/4: (23/3.2, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 7
        credit_history = good
        purpose = furniture/appliances
        amount > 727
        amount <= 1388
        existing_loans_count <= 1
        job = skilled
        phone = FALSE
        ->  class yes  [0.833]

    Rule 88/5: (14.1/2.1, lift 1.8)
        credit_history = very good
        purpose = furniture/appliances
        savings_balance = < 100 DM
        job in {skilled, unskilled}
        ->  class yes  [0.807]

    Rule 88/6: (13.8/2.3, lift 1.7)
        credit_history = critical
        purpose = car
        other_credit in {bank, store}
        ->  class yes  [0.789]

    Rule 88/7: (25.4/4.9, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        amount > 1620
        age <= 31
        existing_loans_count > 1
        ->  class yes  [0.785]

    Rule 88/8: (19.2/3.9, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        ->  class yes  [0.771]

    Rule 88/9: (33.7/7.2, lift 1.7)
        credit_history = good
        purpose = car
        amount <= 2613
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        years_at_residence > 1
        age <= 33
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.770]

    Rule 88/10: (35.5/10.8, lift 1.5)
        credit_history = good
        purpose = furniture/appliances
        amount > 3979
        other_credit = none
        ->  class yes  [0.684]

    Rule 88/11: (22.9/7.5, lift 1.4)
        purpose = renovations
        ->  class yes  [0.660]

    Rule 88/12: (50.4/19.9, lift 1.3)
        credit_history = good
        existing_loans_count > 1
        ->  class yes  [0.601]

    Rule 88/13: (578.7/284.8, lift 1.1)
        checking_balance in {< 0 DM, 1 - 200 DM}
        ->  class yes  [0.508]

    Rule 88/14: (12.4, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 18
        purpose = business
        ->  class no  [0.931]

    Rule 88/15: (11.3, lift 1.7)
        years_at_residence <= 1
        age <= 44
        other_credit = bank
        ->  class no  [0.925]

    Rule 88/16: (10.5, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        savings_balance = > 1000 DM
        age <= 44
        ->  class no  [0.920]

    Rule 88/17: (37.5/2.6, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        credit_history = critical
        other_credit = none
        housing = own
        existing_loans_count <= 2
        ->  class no  [0.909]

    Rule 88/18: (15.6/0.7, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        savings_balance in {< 100 DM, 500 - 1000 DM, unknown}
        age > 31
        age <= 59
        existing_loans_count > 1
        ->  class no  [0.901]

    Rule 88/19: (7.6, lift 1.6)
        checking_balance in {> 200 DM, unknown}
        credit_history = good
        years_at_residence <= 1
        age <= 33
        ->  class no  [0.895]

    Rule 88/20: (20/1.5, lift 1.6)
        checking_balance in {> 200 DM, unknown}
        credit_history = good
        age <= 33
        other_credit = none
        phone = TRUE
        ->  class no  [0.887]

    Rule 88/21: (12.2/0.6, lift 1.6)
        purpose = furniture/appliances
        amount > 1388
        other_credit = none
        job = skilled
        dependents > 1
        ->  class no  [0.886]

    Rule 88/22: (17.1/1.3, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = business
        housing = own
        phone = FALSE
        ->  class no  [0.882]

    Rule 88/23: (6.5, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        purpose = car
        other_credit = none
        existing_loans_count <= 1
        ->  class no  [0.882]

    Rule 88/24: (24.6/2.2, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 15
        credit_history = good
        amount > 1388
        job = skilled
        phone = FALSE
        ->  class no  [0.880]

    Rule 88/25: (59.6/6.7, lift 1.6)
        checking_balance in {> 200 DM, unknown}
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        age > 44
        ->  class no  [0.874]

    Rule 88/26: (32.2/3.6, lift 1.6)
        checking_balance in {> 200 DM, unknown}
        credit_history = good
        purpose = furniture/appliances
        amount <= 3979
        other_credit = none
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.866]

    Rule 88/27: (38.9/5, lift 1.6)
        credit_history = critical
        purpose = car
        age > 29
        other_credit = none
        existing_loans_count > 1
        ->  class no  [0.854]

    Rule 88/28: (10.7/1, lift 1.6)
        purpose = furniture/appliances
        savings_balance = > 1000 DM
        ->  class no  [0.847]

    Rule 88/29: (76.8/12.2, lift 1.5)
        checking_balance in {> 200 DM, unknown}
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        age > 33
        other_credit = none
        dependents <= 1
        ->  class no  [0.832]

    Rule 88/30: (24.1/4, lift 1.5)
        checking_balance in {> 200 DM, unknown}
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        other_credit = none
        dependents > 1
        ->  class no  [0.809]

    Rule 88/31: (13.6/2, lift 1.5)
        checking_balance = unknown
        credit_history = poor
        percent_of_income <= 3
        other_credit = none
        ->  class no  [0.805]

    Rule 88/32: (27.1/5.5, lift 1.4)
        purpose = car
        amount > 1721
        employment_duration = 4 - 7 years
        ->  class no  [0.775]

    Rule 88/33: (37.3/11.9, lift 1.2)
        dependents > 1
        phone = TRUE
        ->  class no  [0.673]

    Rule 88/34: (133.2/52.4, lift 1.1)
        savings_balance = unknown
        ->  class no  [0.605]

    Rule 88/35: (264.5/105.2, lift 1.1)
        months_loan_duration <= 22
        purpose = furniture/appliances
        ->  class no  [0.602]

    Default class: yes

    -----  Trial 89:  -----

    Rules:

    Rule 89/1: (10.2, lift 2.1)
        checking_balance = 1 - 200 DM
        credit_history = good
        purpose = furniture/appliances
        employment_duration in {< 1 year, 1 - 4 years}
        housing in {other, rent}
        job = skilled
        ->  class yes  [0.918]

    Rule 89/2: (8, lift 2.0)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 9
        purpose = car
        employment_duration in {> 7 years, 4 - 7 years, unemployed}
        ->  class yes  [0.900]

    Rule 89/3: (7.3, lift 2.0)
        checking_balance = < 0 DM
        amount > 7596
        percent_of_income > 2
        ->  class yes  [0.893]

    Rule 89/4: (15.2/1.3, lift 1.9)
        checking_balance = < 0 DM
        credit_history in {perfect, poor, very good}
        savings_balance = < 100 DM
        other_credit = none
        job = skilled
        ->  class yes  [0.868]

    Rule 89/5: (4.9, lift 1.9)
        checking_balance = 1 - 200 DM
        purpose = car
        employment_duration in {< 1 year, 1 - 4 years}
        housing = rent
        phone = FALSE
        ->  class yes  [0.856]

    Rule 89/6: (4.5, lift 1.9)
        checking_balance = < 0 DM
        percent_of_income <= 1
        job = management
        ->  class yes  [0.847]

    Rule 89/7: (21.7/2.7, lift 1.9)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 13
        credit_history = good
        purpose = furniture/appliances
        employment_duration in {< 1 year, 1 - 4 years}
        age <= 31
        job = skilled
        ->  class yes  [0.846]

    Rule 89/8: (7.5/0.6, lift 1.9)
        checking_balance = 1 - 200 DM
        months_loan_duration > 36
        purpose = furniture/appliances
        dependents <= 1
        ->  class yes  [0.833]

    Rule 89/9: (23.9/3.4, lift 1.9)
        checking_balance = < 0 DM
        credit_history = good
        percent_of_income > 1
        age <= 24
        other_credit = none
        job = skilled
        ->  class yes  [0.831]

    Rule 89/10: (13.9/2.1, lift 1.8)
        checking_balance = 1 - 200 DM
        purpose = car
        employment_duration in {< 1 year, 1 - 4 years}
        years_at_residence > 3
        ->  class yes  [0.802]

    Rule 89/11: (10.9/1.7, lift 1.8)
        checking_balance = unknown
        employment_duration = < 1 year
        other_credit = none
        existing_loans_count > 1
        job = skilled
        ->  class yes  [0.789]

    Rule 89/12: (23.2/4.8, lift 1.7)
        checking_balance = < 0 DM
        months_loan_duration <= 18
        savings_balance = < 100 DM
        percent_of_income > 2
        existing_loans_count > 1
        job = skilled
        ->  class yes  [0.769]

    Rule 89/13: (33/7.2, lift 1.7)
        checking_balance = < 0 DM
        savings_balance = < 100 DM
        job = skilled
        phone = TRUE
        ->  class yes  [0.767]

    Rule 89/14: (21.3/5.1, lift 1.7)
        checking_balance = 1 - 200 DM
        years_at_residence > 1
        dependents > 1
        ->  class yes  [0.740]

    Rule 89/15: (23.4/5.9, lift 1.6)
        checking_balance = > 200 DM
        amount <= 4308
        percent_of_income > 3
        age > 24
        ->  class yes  [0.727]

    Rule 89/16: (25.1/6.9, lift 1.6)
        checking_balance = unknown
        purpose in {business, car, education}
        percent_of_income > 1
        other_credit = bank
        ->  class yes  [0.708]

    Rule 89/17: (154.9/71, lift 1.2)
        checking_balance = 1 - 200 DM
        employment_duration in {< 1 year, 1 - 4 years}
        ->  class yes  [0.541]

    Rule 89/18: (309.4/154.8, lift 1.1)
        checking_balance = < 0 DM
        ->  class yes  [0.500]

    Rule 89/19: (36.8/2.6, lift 1.6)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 36
        purpose = furniture/appliances
        employment_duration in {> 7 years, 4 - 7 years, unemployed}
        job in {skilled, unemployed, unskilled}
        dependents <= 1
        ->  class no  [0.907]

    Rule 89/20: (8, lift 1.6)
        checking_balance = > 200 DM
        amount > 4308
        ->  class no  [0.900]

    Rule 89/21: (6.9, lift 1.6)
        checking_balance = > 200 DM
        age <= 24
        ->  class no  [0.888]

    Rule 89/22: (67.2/9.5, lift 1.5)
        checking_balance = unknown
        years_at_residence > 1
        age > 32
        other_credit = none
        ->  class no  [0.848]

    Rule 89/23: (4, lift 1.5)
        checking_balance = 1 - 200 DM
        years_at_residence <= 1
        dependents > 1
        ->  class no  [0.833]

    Rule 89/24: (43.4/7.2, lift 1.5)
        checking_balance = unknown
        credit_history = critical
        other_credit = none
        ->  class no  [0.819]

    Rule 89/25: (3.2, lift 1.5)
        checking_balance = unknown
        purpose = car0
        ->  class no  [0.808]

    Rule 89/26: (28.9/5.6, lift 1.4)
        checking_balance = < 0 DM
        amount <= 7596
        savings_balance = < 100 DM
        percent_of_income > 1
        job = management
        ->  class no  [0.785]

    Rule 89/27: (27.6/5.9, lift 1.4)
        savings_balance = < 100 DM
        existing_loans_count > 1
        job = unskilled
        ->  class no  [0.766]

    Rule 89/28: (96.4/26.8, lift 1.3)
        checking_balance = 1 - 200 DM
        months_loan_duration > 9
        employment_duration in {> 7 years, 4 - 7 years, unemployed}
        dependents <= 1
        ->  class no  [0.717]

    Rule 89/29: (84/24.3, lift 1.3)
        checking_balance = unknown
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.706]

    Rule 89/30: (183.2/71.6, lift 1.1)
        savings_balance in {> 1000 DM, 100 - 500 DM, 500 - 1000 DM}
        ->  class no  [0.608]

    Rule 89/31: (459.3/194.1, lift 1.0)
        percent_of_income <= 3
        ->  class no  [0.577]

    Default class: yes

    -----  Trial 90:  -----

    Rules:

    Rule 90/1: (15.9, lift 2.1)
        months_loan_duration > 8
        amount <= 1264
        dependents > 1
        phone = FALSE
        ->  class yes  [0.940]

    Rule 90/2: (9.4/0.2, lift 2.0)
        purpose = education
        amount > 1455
        housing = own
        phone = TRUE
        ->  class yes  [0.894]

    Rule 90/3: (14.4/0.9, lift 1.9)
        checking_balance in {< 0 DM, unknown}
        months_loan_duration > 8
        savings_balance = unknown
        employment_duration = 1 - 4 years
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.886]

    Rule 90/4: (6.6, lift 1.9)
        credit_history = good
        savings_balance = 100 - 500 DM
        existing_loans_count > 1
        phone = FALSE
        ->  class yes  [0.883]

    Rule 90/5: (6.2, lift 1.9)
        months_loan_duration <= 8
        amount > 4057
        existing_loans_count <= 1
        ->  class yes  [0.878]

    Rule 90/6: (6, lift 1.9)
        purpose in {business, renovations}
        employment_duration = < 1 year
        percent_of_income > 3
        phone = FALSE
        ->  class yes  [0.875]

    Rule 90/7: (10/0.6, lift 1.9)
        months_loan_duration > 8
        amount <= 1345
        savings_balance = 100 - 500 DM
        existing_loans_count <= 1
        ->  class yes  [0.870]

    Rule 90/8: (5.1, lift 1.9)
        months_loan_duration > 27
        credit_history = good
        savings_balance = 100 - 500 DM
        phone = FALSE
        ->  class yes  [0.860]

    Rule 90/9: (5.1, lift 1.9)
        months_loan_duration > 30
        employment_duration = < 1 year
        percent_of_income > 3
        phone = FALSE
        ->  class yes  [0.859]

    Rule 90/10: (28.5/3.6, lift 1.9)
        months_loan_duration > 16
        amount <= 4686
        percent_of_income > 1
        dependents > 1
        phone = FALSE
        ->  class yes  [0.848]

    Rule 90/11: (4.5, lift 1.9)
        months_loan_duration > 45
        savings_balance = < 100 DM
        employment_duration = > 7 years
        phone = FALSE
        ->  class yes  [0.846]

    Rule 90/12: (14.1/1.6, lift 1.8)
        savings_balance = < 100 DM
        employment_duration = unemployed
        age <= 33
        phone = FALSE
        ->  class yes  [0.840]

    Rule 90/13: (23.1/3, lift 1.8)
        purpose = business
        amount > 1721
        savings_balance = < 100 DM
        years_at_residence > 1
        phone = TRUE
        ->  class yes  [0.840]

    Rule 90/14: (33.1/5.3, lift 1.8)
        credit_history in {critical, good, perfect}
        amount > 10875
        ->  class yes  [0.819]

    Rule 90/15: (8/0.9, lift 1.8)
        months_loan_duration > 8
        age > 52
        dependents > 1
        ->  class yes  [0.809]

    Rule 90/16: (20.8/4.6, lift 1.7)
        credit_history = very good
        purpose = car
        amount <= 10875
        age > 28
        ->  class yes  [0.755]

    Rule 90/17: (18.9/4.3, lift 1.6)
        months_loan_duration > 8
        purpose = car
        age <= 28
        phone = TRUE
        ->  class yes  [0.747]

    Rule 90/18: (24/5.6, lift 1.6)
        months_loan_duration > 8
        savings_balance = 500 - 1000 DM
        employment_duration in {< 1 year, 1 - 4 years}
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.745]

    Rule 90/19: (44.2/13.9, lift 1.5)
        months_loan_duration > 8
        savings_balance = < 100 DM
        employment_duration = < 1 year
        percent_of_income <= 3
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.678]

    Rule 90/20: (41.4/13.5, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 8
        purpose = furniture/appliances
        years_at_residence <= 3
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.666]

    Rule 90/21: (8, lift 1.7)
        employment_duration = unemployed
        age > 33
        dependents <= 1
        phone = FALSE
        ->  class no  [0.900]

    Rule 90/22: (8.2/0.4, lift 1.6)
        credit_history = critical
        purpose = furniture/appliances
        years_at_residence <= 3
        phone = TRUE
        ->  class no  [0.864]

    Rule 90/23: (31.8/9, lift 1.3)
        credit_history in {critical, poor}
        savings_balance = 100 - 500 DM
        ->  class no  [0.704]

    Rule 90/24: (863.3/382, lift 1.0)
        amount <= 10875
        ->  class no  [0.557]

    Default class: no

    -----  Trial 91:  -----

    Rules:

    Rule 91/1: (19.4, lift 1.9)
        months_loan_duration <= 47
        credit_history = very good
        savings_balance = < 100 DM
        age > 23
        age <= 35
        ->  class yes  [0.953]

    Rule 91/2: (15.7/0.7, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = poor
        savings_balance = < 100 DM
        employment_duration in {< 1 year, > 7 years, 4 - 7 years, unemployed}
        percent_of_income > 1
        ->  class yes  [0.902]

    Rule 91/3: (7.3, lift 1.8)
        purpose in {education, renovations}
        savings_balance = > 1000 DM
        ->  class yes  [0.892]

    Rule 91/4: (21.8/1.7, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        savings_balance = < 100 DM
        employment_duration = > 7 years
        percent_of_income > 1
        age <= 61
        existing_loans_count > 1
        job = skilled
        ->  class yes  [0.886]

    Rule 91/5: (13.4/0.9, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = perfect
        savings_balance = < 100 DM
        years_at_residence > 2
        ->  class yes  [0.876]

    Rule 91/6: (21.3/2.3, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        purpose = furniture/appliances
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, unemployed}
        age <= 44
        existing_loans_count > 1
        phone = TRUE
        ->  class yes  [0.860]

    Rule 91/7: (15.8/1.6, lift 1.7)
        checking_balance = unknown
        savings_balance = < 100 DM
        age <= 22
        job = skilled
        phone = FALSE
        ->  class yes  [0.856]

    Rule 91/8: (4.9, lift 1.7)
        checking_balance = 1 - 200 DM
        savings_balance = 500 - 1000 DM
        job = unskilled
        ->  class yes  [0.855]

    Rule 91/9: (18.7/2.3, lift 1.7)
        checking_balance in {> 200 DM, unknown}
        purpose = furniture/appliances
        age <= 44
        existing_loans_count <= 1
        job = unskilled
        ->  class yes  [0.840]

    Rule 91/10: (25/3.6, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 47
        savings_balance = < 100 DM
        ->  class yes  [0.830]

    Rule 91/11: (11.8/1.6, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        age > 36
        job = management
        ->  class yes  [0.813]

    Rule 91/12: (21.4/3.7, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 47
        credit_history = good
        purpose = car
        savings_balance = < 100 DM
        employment_duration in {> 7 years, 4 - 7 years}
        ->  class yes  [0.798]

    Rule 91/13: (28.3/5.4, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        savings_balance = 100 - 500 DM
        employment_duration in {< 1 year, 1 - 4 years}
        ->  class yes  [0.790]

    Rule 91/14: (39.1/9.4, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        purpose = car
        savings_balance = < 100 DM
        years_at_residence <= 2
        ->  class yes  [0.746]

    Rule 91/15: (48.7/14.7, lift 1.4)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 22
        purpose = furniture/appliances
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        job = skilled
        ->  class yes  [0.691]

    Rule 91/16: (48/16.1, lift 1.3)
        purpose = education
        age <= 44
        ->  class yes  [0.658]

    Rule 91/17: (713.7/350.1, lift 1.0)
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        age <= 44
        ->  class yes  [0.509]

    Rule 91/18: (23, lift 1.9)
        checking_balance in {> 200 DM, unknown}
        purpose = furniture/appliances
        amount <= 3979
        age > 22
        other_credit = none
        existing_loans_count <= 1
        job in {management, skilled}
        ->  class no  [0.960]

    Rule 91/19: (15.1, lift 1.9)
        checking_balance in {> 200 DM, unknown}
        purpose = furniture/appliances
        existing_loans_count <= 1
        phone = TRUE
        ->  class no  [0.941]

    Rule 91/20: (12.7, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {critical, perfect, poor}
        savings_balance = unknown
        ->  class no  [0.932]

    Rule 91/21: (25.4/0.9, lift 1.8)
        checking_balance in {> 200 DM, unknown}
        months_loan_duration > 13
        purpose = car
        other_credit in {none, store}
        housing = own
        ->  class no  [0.932]

    Rule 91/22: (12.4, lift 1.8)
        credit_history = critical
        age > 61
        ->  class no  [0.930]

    Rule 91/23: (10.2, lift 1.8)
        months_loan_duration <= 15
        credit_history = good
        purpose = furniture/appliances
        amount > 1316
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        job = skilled
        ->  class no  [0.918]

    Rule 91/24: (8.9, lift 1.8)
        checking_balance in {> 200 DM, unknown}
        purpose = furniture/appliances
        other_credit = bank
        job in {management, skilled}
        phone = FALSE
        ->  class no  [0.908]

    Rule 91/25: (8.2, lift 1.8)
        checking_balance in {> 200 DM, unknown}
        savings_balance = > 1000 DM
        age <= 44
        ->  class no  [0.902]

    Rule 91/26: (8.1, lift 1.8)
        checking_balance in {> 200 DM, unknown}
        purpose = car
        percent_of_income <= 1
        age <= 44
        ->  class no  [0.901]

    Rule 91/27: (6.7, lift 1.7)
        credit_history = very good
        age <= 23
        ->  class no  [0.886]

    Rule 91/28: (6.6, lift 1.7)
        credit_history = critical
        employment_duration = > 7 years
        other_credit = none
        existing_loans_count <= 1
        job = skilled
        ->  class no  [0.884]

    Rule 91/29: (6.3, lift 1.7)
        months_loan_duration > 40
        months_loan_duration <= 47
        purpose = car
        ->  class no  [0.879]

    Rule 91/30: (4.5, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = poor
        percent_of_income <= 1
        ->  class no  [0.846]

    Rule 91/31: (16/1.9, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 47
        purpose = furniture/appliances
        savings_balance = < 100 DM
        age <= 36
        job = management
        ->  class no  [0.838]

    Rule 91/32: (52.1/8, lift 1.6)
        checking_balance in {> 200 DM, unknown}
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        age > 44
        ->  class no  [0.833]

    Rule 91/33: (9.8/1.1, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose in {business, car, furniture/appliances}
        savings_balance = > 1000 DM
        ->  class no  [0.826]

    Rule 91/34: (24.5/4.7, lift 1.5)
        checking_balance in {> 200 DM, unknown}
        purpose = furniture/appliances
        existing_loans_count > 1
        phone = FALSE
        ->  class no  [0.783]

    Rule 91/35: (29.4/6.8, lift 1.5)
        credit_history in {good, very good}
        savings_balance = unknown
        employment_duration in {< 1 year, 4 - 7 years}
        ->  class no  [0.752]

    Rule 91/36: (28/6.8, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        savings_balance = 100 - 500 DM
        employment_duration in {> 7 years, 4 - 7 years, unemployed}
        existing_loans_count <= 3
        ->  class no  [0.741]

    Rule 91/37: (353.6/161.2, lift 1.1)
        months_loan_duration <= 15
        ->  class no  [0.544]

    Default class: no

    -----  Trial 92:  -----

    Rules:

    Rule 92/1: (12.1/0.2, lift 2.0)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        percent_of_income <= 2
        years_at_residence > 1
        age > 46
        dependents > 1
        ->  class yes  [0.915]

    Rule 92/2: (17.3/0.8, lift 2.0)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history in {good, poor}
        percent_of_income > 2
        years_at_residence > 1
        other_credit = store
        ->  class yes  [0.905]

    Rule 92/3: (5.9, lift 1.9)
        checking_balance = unknown
        credit_history = good
        purpose = furniture/appliances
        employment_duration = 1 - 4 years
        job = unskilled
        ->  class yes  [0.873]

    Rule 92/4: (5.7, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        amount > 9283
        percent_of_income <= 2
        years_at_residence > 1
        ->  class yes  [0.870]

    Rule 92/5: (14.1/1.1, lift 1.9)
        checking_balance = < 0 DM
        purpose = car
        savings_balance = < 100 DM
        percent_of_income > 2
        percent_of_income <= 3
        ->  class yes  [0.870]

    Rule 92/6: (15.8/1.6, lift 1.9)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history = perfect
        percent_of_income > 2
        years_at_residence > 1
        ->  class yes  [0.855]

    Rule 92/7: (8.6/0.7, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        employment_duration in {> 7 years, unemployed}
        years_at_residence <= 1
        other_credit = none
        ->  class yes  [0.835]

    Rule 92/8: (20.8/2.9, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        amount > 1221
        savings_balance = < 100 DM
        employment_duration = < 1 year
        percent_of_income > 2
        years_at_residence > 1
        other_credit = none
        ->  class yes  [0.828]

    Rule 92/9: (17.4/2.5, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 13
        credit_history = good
        savings_balance = unknown
        percent_of_income > 2
        years_at_residence > 1
        other_credit = none
        ->  class yes  [0.822]

    Rule 92/10: (4.2/0.1, lift 1.8)
        checking_balance = unknown
        credit_history = poor
        years_at_residence <= 1
        ->  class yes  [0.814]

    Rule 92/11: (7.3/0.7, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history in {poor, very good}
        years_at_residence <= 1
        other_credit in {none, store}
        ->  class yes  [0.813]

    Rule 92/12: (33.3/6, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 33
        purpose = car
        percent_of_income > 2
        years_at_residence > 1
        age <= 36
        phone = FALSE
        ->  class yes  [0.802]

    Rule 92/13: (15.2/2.6, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 33
        purpose = car
        savings_balance = < 100 DM
        percent_of_income > 3
        ->  class yes  [0.789]

    Rule 92/14: (16/3, lift 1.7)
        checking_balance = unknown
        months_loan_duration <= 30
        credit_history = good
        purpose = car
        employment_duration = 1 - 4 years
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.777]

    Rule 92/15: (17.9/3.9, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = car
        amount <= 1455
        percent_of_income <= 2
        dependents <= 1
        ->  class yes  [0.754]

    Rule 92/16: (186.3/81.2, lift 1.2)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = furniture/appliances
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years}
        years_at_residence > 1
        dependents <= 1
        ->  class yes  [0.563]

    Rule 92/17: (83.2/36.6, lift 1.2)
        other_credit = bank
        phone = FALSE
        ->  class yes  [0.559]

    Rule 92/18: (487.7/251.5, lift 1.1)
        savings_balance = < 100 DM
        years_at_residence > 1
        ->  class yes  [0.484]

    Rule 92/19: (21.5, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = car
        amount > 1455
        amount <= 9283
        percent_of_income <= 2
        years_at_residence > 1
        years_at_residence <= 3
        dependents <= 1
        ->  class no  [0.957]

    Rule 92/20: (16.2, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration <= 27
        credit_history = good
        purpose = furniture/appliances
        percent_of_income <= 2
        years_at_residence > 3
        age > 21
        ->  class no  [0.945]

    Rule 92/21: (9.1, lift 1.7)
        credit_history = good
        purpose = furniture/appliances
        percent_of_income <= 2
        years_at_residence > 1
        other_credit = none
        phone = TRUE
        ->  class no  [0.910]

    Rule 92/22: (8.5, lift 1.7)
        checking_balance = unknown
        purpose = car
        employment_duration = < 1 year
        ->  class no  [0.904]

    Rule 92/23: (7.3, lift 1.6)
        checking_balance = unknown
        credit_history = critical
        other_credit = bank
        phone = TRUE
        ->  class no  [0.893]

    Rule 92/24: (22.5/1.6, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        amount <= 1221
        savings_balance = < 100 DM
        percent_of_income > 2
        years_at_residence > 1
        other_credit = none
        ->  class no  [0.892]

    Rule 92/25: (6.7, lift 1.6)
        employment_duration in {1 - 4 years, 4 - 7 years}
        percent_of_income <= 2
        years_at_residence > 1
        other_credit = store
        dependents <= 1
        ->  class no  [0.885]

    Rule 92/26: (14.6/1.7, lift 1.5)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        years_at_residence <= 1
        other_credit = bank
        ->  class no  [0.840]

    Rule 92/27: (11.9/1.4, lift 1.5)
        credit_history in {perfect, poor}
        purpose = furniture/appliances
        percent_of_income <= 2
        ->  class no  [0.831]

    Rule 92/28: (38.6/8.9, lift 1.4)
        checking_balance = unknown
        employment_duration = > 7 years
        dependents <= 1
        ->  class no  [0.757]

    Rule 92/29: (106.5/30.4, lift 1.3)
        credit_history in {critical, good, perfect}
        employment_duration in {< 1 year, 1 - 4 years, 4 - 7 years}
        years_at_residence <= 1
        ->  class no  [0.710]

    Rule 92/30: (118/37.1, lift 1.2)
        months_loan_duration <= 33
        percent_of_income > 3
        phone = TRUE
        ->  class no  [0.682]

    Rule 92/31: (700.3/308.7, lift 1.0)
        other_credit = none
        ->  class no  [0.559]

    Default class: no

    -----  Trial 93:  -----

    Rules:

    Rule 93/1: (14.7, lift 2.0)
        checking_balance = < 0 DM
        credit_history = good
        purpose = car
        amount <= 8133
        savings_balance = < 100 DM
        years_at_residence <= 2
        job in {skilled, unskilled}
        ->  class yes  [0.940]

    Rule 93/2: (12.3, lift 2.0)
        credit_history = very good
        amount > 4530
        amount <= 7629
        ->  class yes  [0.930]

    Rule 93/3: (9.6, lift 2.0)
        months_loan_duration > 24
        purpose = car
        amount <= 4308
        savings_balance = < 100 DM
        job in {skilled, unskilled}
        ->  class yes  [0.909]

    Rule 93/4: (9.6, lift 2.0)
        months_loan_duration <= 10
        credit_history = good
        purpose = furniture/appliances
        amount > 1316
        savings_balance = < 100 DM
        employment_duration = < 1 year
        job = skilled
        ->  class yes  [0.909]

    Rule 93/5: (12/0.7, lift 1.9)
        credit_history = perfect
        housing in {other, rent}
        ->  class yes  [0.880]

    Rule 93/6: (11.8/0.7, lift 1.9)
        credit_history = good
        purpose in {car, furniture/appliances}
        amount <= 1597
        savings_balance = unknown
        employment_duration = 1 - 4 years
        ->  class yes  [0.878]

    Rule 93/7: (11.7/0.7, lift 1.9)
        credit_history = good
        purpose = furniture/appliances
        amount <= 2241
        savings_balance = < 100 DM
        other_credit = store
        ->  class yes  [0.872]

    Rule 93/8: (7/0.4, lift 1.8)
        checking_balance = < 0 DM
        months_loan_duration > 16
        credit_history = poor
        ->  class yes  [0.846]

    Rule 93/9: (15.7/1.9, lift 1.8)
        credit_history = poor
        age > 46
        ->  class yes  [0.838]

    Rule 93/10: (16.2/2.1, lift 1.8)
        months_loan_duration > 10
        amount > 1316
        amount <= 3441
        savings_balance = < 100 DM
        employment_duration = < 1 year
        percent_of_income <= 2
        other_credit = none
        ->  class yes  [0.829]

    Rule 93/11: (3.6, lift 1.8)
        checking_balance = 1 - 200 DM
        credit_history = critical
        dependents > 1
        ->  class yes  [0.821]

    Rule 93/12: (10.8/1.4, lift 1.8)
        credit_history = good
        purpose = renovations
        savings_balance = < 100 DM
        ->  class yes  [0.809]

    Rule 93/13: (11.2/1.6, lift 1.7)
        credit_history = good
        purpose = education
        savings_balance = < 100 DM
        existing_loans_count <= 1
        job in {skilled, unskilled}
        ->  class yes  [0.800]

    Rule 93/14: (16/2.7, lift 1.7)
        purpose = furniture/appliances
        amount > 1316
        employment_duration = < 1 year
        job in {skilled, unskilled}
        phone = TRUE
        ->  class yes  [0.795]

    Rule 93/15: (40.7/7.8, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = very good
        amount <= 7629
        age > 23
        ->  class yes  [0.793]

    Rule 93/16: (2.8, lift 1.7)
        credit_history = poor
        existing_loans_count > 3
        ->  class yes  [0.791]

    Rule 93/17: (13.7/2.3, lift 1.7)
        amount > 11328
        age <= 33
        ->  class yes  [0.786]

    Rule 93/18: (9.1/1.5, lift 1.7)
        months_loan_duration > 16
        credit_history = poor
        employment_duration = < 1 year
        ->  class yes  [0.772]

    Rule 93/19: (34.4/7.7, lift 1.6)
        credit_history = good
        amount > 8133
        ->  class yes  [0.761]

    Rule 93/20: (14.2/3, lift 1.6)
        credit_history = perfect
        age > 33
        ->  class yes  [0.755]

    Rule 93/21: (27.4/6.7, lift 1.6)
        checking_balance = < 0 DM
        credit_history = critical
        amount > 2122
        age <= 60
        dependents <= 1
        ->  class yes  [0.738]

    Rule 93/22: (46.2/18.8, lift 1.3)
        credit_history = good
        savings_balance = 100 - 500 DM
        ->  class yes  [0.589]

    Rule 93/23: (123.9/54.4, lift 1.2)
        months_loan_duration > 16
        months_loan_duration <= 18
        ->  class yes  [0.560]

    Rule 93/24: (17.1, lift 1.8)
        credit_history = perfect
        amount > 2445
        amount <= 11328
        age <= 33
        housing = own
        ->  class no  [0.948]

    Rule 93/25: (10.2, lift 1.7)
        purpose = furniture/appliances
        amount <= 1316
        savings_balance = < 100 DM
        employment_duration = < 1 year
        other_credit = none
        ->  class no  [0.918]

    Rule 93/26: (10.1, lift 1.7)
        credit_history = critical
        age > 60
        ->  class no  [0.917]

    Rule 93/27: (8.7, lift 1.7)
        credit_history = good
        amount > 2241
        amount <= 8133
        other_credit = store
        ->  class no  [0.907]

    Rule 93/28: (6.7, lift 1.6)
        purpose = education
        savings_balance = unknown
        employment_duration = 1 - 4 years
        ->  class no  [0.885]

    Rule 93/29: (4.8, lift 1.6)
        credit_history = very good
        amount > 7629
        ->  class no  [0.852]

    Rule 93/30: (9.5/0.8, lift 1.6)
        checking_balance in {> 200 DM, unknown}
        credit_history = very good
        amount <= 4530
        ->  class no  [0.844]

    Rule 93/31: (18.2/2.3, lift 1.6)
        purpose = furniture/appliances
        employment_duration = 4 - 7 years
        years_at_residence > 3
        other_credit = none
        ->  class no  [0.835]

    Rule 93/32: (35/5.2, lift 1.5)
        months_loan_duration <= 24
        purpose = car
        amount > 1386
        amount <= 4308
        years_at_residence > 2
        existing_loans_count <= 1
        ->  class no  [0.833]

    Rule 93/33: (833.8/373.4, lift 1.0)
        amount <= 8133
        ->  class no  [0.552]

    Default class: no

    -----  Trial 94:  -----

    Rules:

    Rule 94/1: (15.5, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 36
        purpose = furniture/appliances
        savings_balance = < 100 DM
        dependents <= 1
        ->  class yes  [0.943]

    Rule 94/2: (14.8, lift 2.0)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        months_loan_duration > 15
        months_loan_duration <= 21
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        job = skilled
        phone = FALSE
        ->  class yes  [0.940]

    Rule 94/3: (7.5, lift 1.9)
        checking_balance = unknown
        amount > 2096
        employment_duration = 1 - 4 years
        percent_of_income > 1
        other_credit = bank
        ->  class yes  [0.895]

    Rule 94/4: (7.4, lift 1.9)
        checking_balance = unknown
        credit_history = good
        employment_duration = 1 - 4 years
        other_credit = none
        existing_loans_count > 1
        ->  class yes  [0.893]

    Rule 94/5: (6.6, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        amount <= 983
        savings_balance = 100 - 500 DM
        ->  class yes  [0.884]

    Rule 94/6: (14.3/1, lift 1.9)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        purpose = business
        savings_balance = < 100 DM
        other_credit = none
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.878]

    Rule 94/7: (8.2/0.3, lift 1.9)
        checking_balance = 1 - 200 DM
        savings_balance = 100 - 500 DM
        years_at_residence > 2
        age <= 29
        ->  class yes  [0.874]

    Rule 94/8: (16.4/1.4, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 7
        amount <= 1289
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        existing_loans_count <= 1
        job = skilled
        phone = FALSE
        ->  class yes  [0.867]

    Rule 94/9: (9.5/0.9, lift 1.8)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        savings_balance = 500 - 1000 DM
        percent_of_income > 3
        ->  class yes  [0.831]

    Rule 94/10: (12/1.6, lift 1.8)
        checking_balance = unknown
        amount > 4594
        employment_duration = < 1 year
        ->  class yes  [0.816]

    Rule 94/11: (15.2/2.3, lift 1.7)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        credit_history in {critical, good}
        purpose in {car, furniture/appliances}
        savings_balance = unknown
        housing in {other, rent}
        job = skilled
        ->  class yes  [0.806]

    Rule 94/12: (2.5, lift 1.7)
        savings_balance = 100 - 500 DM
        existing_loans_count > 3
        ->  class yes  [0.779]

    Rule 94/13: (11.1/2, lift 1.7)
        employment_duration = 4 - 7 years
        age <= 22
        ->  class yes  [0.769]

    Rule 94/14: (24/6.1, lift 1.6)
        purpose = car
        employment_duration = 1 - 4 years
        percent_of_income > 1
        age <= 27
        ->  class yes  [0.728]

    Rule 94/15: (78.5/26, lift 1.4)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        amount <= 5045
        savings_balance = < 100 DM
        dependents <= 1
        ->  class yes  [0.665]

    Rule 94/16: (33.3/12, lift 1.4)
        employment_duration = unemployed
        percent_of_income > 2
        ->  class yes  [0.631]

    Rule 94/17: (54.1/20.3, lift 1.3)
        credit_history in {perfect, poor, very good}
        savings_balance = < 100 DM
        existing_loans_count <= 1
        ->  class yes  [0.620]

    Rule 94/18: (647.9/323.1, lift 1.1)
        checking_balance in {< 0 DM, > 200 DM, 1 - 200 DM}
        ->  class yes  [0.501]

    Rule 94/19: (10.5, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {perfect, poor}
        savings_balance = unknown
        ->  class no  [0.920]

    Rule 94/20: (24.8/1.2, lift 1.7)
        checking_balance = unknown
        employment_duration = 4 - 7 years
        age > 22
        ->  class no  [0.918]

    Rule 94/21: (8.3, lift 1.7)
        checking_balance = unknown
        employment_duration = 1 - 4 years
        percent_of_income <= 1
        ->  class no  [0.903]

    Rule 94/22: (6.2, lift 1.6)
        months_loan_duration <= 47
        purpose = education
        savings_balance = < 100 DM
        age > 37
        dependents <= 1
        ->  class no  [0.878]

    Rule 94/23: (7.8/0.4, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = business
        savings_balance = < 100 DM
        other_credit in {bank, store}
        ->  class no  [0.858]

    Rule 94/24: (8.4/0.6, lift 1.6)
        months_loan_duration <= 47
        purpose = car0
        savings_balance = < 100 DM
        dependents <= 1
        ->  class no  [0.850]

    Rule 94/25: (31.9/5.8, lift 1.5)
        checking_balance = unknown
        amount <= 4594
        employment_duration = < 1 year
        ->  class no  [0.800]

    Rule 94/26: (54/15.8, lift 1.3)
        checking_balance = unknown
        employment_duration = > 7 years
        ->  class no  [0.701]

    Rule 94/27: (90.5/34.3, lift 1.2)
        savings_balance = 100 - 500 DM
        dependents <= 1
        ->  class no  [0.618]

    Rule 94/28: (745.6/334.5, lift 1.0)
        months_loan_duration <= 47
        dependents <= 1
        ->  class no  [0.551]

    Default class: no

    -----  Trial 95:  -----

    Rules:

    Rule 95/1: (20.9, lift 2.0)
        checking_balance = < 0 DM
        months_loan_duration > 16
        months_loan_duration <= 45
        credit_history = good
        savings_balance = < 100 DM
        percent_of_income > 1
        years_at_residence <= 3
        other_credit in {bank, none}
        housing = own
        job = skilled
        ->  class yes  [0.956]

    Rule 95/2: (13, lift 2.0)
        checking_balance = < 0 DM
        credit_history in {poor, very good}
        savings_balance = < 100 DM
        percent_of_income > 1
        job = skilled
        ->  class yes  [0.933]

    Rule 95/3: (9.3, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration > 42
        years_at_residence > 3
        ->  class yes  [0.912]

    Rule 95/4: (7.2, lift 1.9)
        checking_balance = < 0 DM
        credit_history = good
        savings_balance = < 100 DM
        years_at_residence <= 3
        job = skilled
        phone = TRUE
        ->  class yes  [0.892]

    Rule 95/5: (7.1, lift 1.9)
        months_loan_duration > 8
        years_at_residence <= 3
        age > 57
        job = skilled
        ->  class yes  [0.883]

    Rule 95/6: (13.7/0.9, lift 1.9)
        checking_balance = unknown
        months_loan_duration > 8
        years_at_residence <= 3
        age > 32
        other_credit = bank
        housing = own
        ->  class yes  [0.877]

    Rule 95/7: (18.9/2.7, lift 1.8)
        checking_balance = > 200 DM
        months_loan_duration > 8
        amount <= 4308
        savings_balance = < 100 DM
        percent_of_income > 1
        years_at_residence <= 3
        age > 24
        ->  class yes  [0.822]

    Rule 95/8: (25.1/4.6, lift 1.7)
        checking_balance = < 0 DM
        savings_balance = < 100 DM
        percent_of_income > 1
        years_at_residence > 3
        housing in {other, rent}
        job = skilled
        dependents <= 1
        ->  class yes  [0.794]

    Rule 95/9: (39.2/8.3, lift 1.7)
        months_loan_duration > 8
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, unemployed}
        percent_of_income > 1
        years_at_residence <= 3
        housing = rent
        dependents <= 1
        ->  class yes  [0.774]

    Rule 95/10: (40.8/9.9, lift 1.6)
        checking_balance = 1 - 200 DM
        months_loan_duration > 8
        savings_balance in {< 100 DM, 100 - 500 DM, unknown}
        employment_duration in {> 7 years, 1 - 4 years, 4 - 7 years}
        percent_of_income > 1
        years_at_residence <= 3
        housing = own
        existing_loans_count <= 1
        job = skilled
        ->  class yes  [0.745]

    Rule 95/11: (39.2/10.8, lift 1.5)
        checking_balance = 1 - 200 DM
        months_loan_duration > 20
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM}
        years_at_residence > 3
        ->  class yes  [0.714]

    Rule 95/12: (245.3/113.7, lift 1.1)
        amount > 4139
        ->  class yes  [0.536]

    Rule 95/13: (340.1/162.8, lift 1.1)
        months_loan_duration > 8
        amount <= 2073
        ->  class yes  [0.521]

    Rule 95/14: (588.7/302.8, lift 1.0)
        savings_balance = < 100 DM
        ->  class yes  [0.486]

    Rule 95/15: (17.8, lift 1.8)
        checking_balance = 1 - 200 DM
        months_loan_duration > 20
        savings_balance in {> 1000 DM, unknown}
        percent_of_income > 1
        ->  class no  [0.949]

    Rule 95/16: (17, lift 1.8)
        checking_balance = unknown
        purpose = furniture/appliances
        years_at_residence > 3
        age > 32
        ->  class no  [0.947]

    Rule 95/17: (16.3, lift 1.8)
        checking_balance = unknown
        savings_balance in {> 1000 DM, unknown}
        percent_of_income > 1
        years_at_residence <= 3
        other_credit = none
        existing_loans_count <= 2
        ->  class no  [0.945]

    Rule 95/18: (15.9, lift 1.8)
        months_loan_duration > 13
        amount <= 9436
        years_at_residence <= 3
        housing = other
        dependents <= 1
        ->  class no  [0.944]

    Rule 95/19: (10.9, lift 1.7)
        checking_balance = unknown
        years_at_residence > 2
        years_at_residence <= 3
        other_credit = none
        housing = own
        existing_loans_count <= 2
        ->  class no  [0.922]

    Rule 95/20: (13.6/0.4, lift 1.7)
        checking_balance = unknown
        months_loan_duration > 21
        purpose = car
        years_at_residence > 3
        ->  class no  [0.908]

    Rule 95/21: (8.5, lift 1.7)
        checking_balance = < 0 DM
        purpose in {business, education, furniture/appliances}
        savings_balance = < 100 DM
        percent_of_income > 1
        years_at_residence <= 3
        housing = own
        job = unskilled
        ->  class no  [0.905]

    Rule 95/22: (8.5, lift 1.7)
        checking_balance = > 200 DM
        savings_balance in {> 1000 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        housing = own
        existing_loans_count <= 2
        dependents <= 1
        ->  class no  [0.905]

    Rule 95/23: (7.5, lift 1.7)
        checking_balance = 1 - 200 DM
        savings_balance in {> 1000 DM, 500 - 1000 DM}
        age <= 48
        housing = own
        existing_loans_count <= 1
        ->  class no  [0.894]

    Rule 95/24: (14.4/1, lift 1.7)
        checking_balance = < 0 DM
        months_loan_duration <= 42
        percent_of_income > 1
        years_at_residence > 3
        job in {management, skilled}
        dependents > 1
        ->  class no  [0.881]

    Rule 95/25: (6.2, lift 1.7)
        checking_balance = > 200 DM
        age <= 24
        ->  class no  [0.878]

    Rule 95/26: (5.7, lift 1.6)
        employment_duration = 4 - 7 years
        years_at_residence <= 3
        housing = rent
        ->  class no  [0.869]

    Rule 95/27: (38.7/4.6, lift 1.6)
        checking_balance = 1 - 200 DM
        months_loan_duration > 8
        months_loan_duration <= 20
        years_at_residence > 3
        ->  class no  [0.863]

    Rule 95/28: (5.2, lift 1.6)
        checking_balance = > 200 DM
        amount > 4308
        ->  class no  [0.861]

    Rule 95/29: (18.3/2.2, lift 1.6)
        checking_balance = unknown
        purpose = furniture/appliances
        percent_of_income > 1
        years_at_residence <= 3
        age > 27
        other_credit = none
        ->  class no  [0.845]

    Rule 95/30: (14.8/1.7, lift 1.6)
        checking_balance = unknown
        purpose = furniture/appliances
        years_at_residence > 3
        phone = TRUE
        ->  class no  [0.842]

    Rule 95/31: (51.6/8.8, lift 1.5)
        months_loan_duration > 8
        amount > 2073
        amount <= 10722
        percent_of_income <= 1
        age <= 35
        ->  class no  [0.817]

    Rule 95/32: (28/4.6, lift 1.5)
        checking_balance = unknown
        purpose in {business, car, renovations}
        percent_of_income > 1
        years_at_residence <= 3
        other_credit = none
        housing = own
        existing_loans_count <= 2
        ->  class no  [0.815]

    Rule 95/33: (54.4/10.5, lift 1.5)
        months_loan_duration <= 8
        amount <= 4139
        ->  class no  [0.796]

    Rule 95/34: (30.6/5.9, lift 1.5)
        checking_balance = < 0 DM
        months_loan_duration <= 42
        amount <= 5804
        savings_balance = < 100 DM
        years_at_residence > 3
        other_credit = none
        housing = own
        ->  class no  [0.789]

    Rule 95/35: (22.2/4.6, lift 1.4)
        checking_balance = 1 - 200 DM
        employment_duration = < 1 year
        other_credit = none
        housing = own
        job = skilled
        ->  class no  [0.767]

    Rule 95/36: (71.4/23.7, lift 1.2)
        checking_balance = < 0 DM
        credit_history = critical
        ->  class no  [0.664]

    Rule 95/37: (838.9/405, lift 1.0)
        months_loan_duration > 8
        ->  class no  [0.517]

    Default class: no

    -----  Trial 96:  -----

    Rules:

    Rule 96/1: (6.3, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        employment_duration = unemployed
        dependents > 1
        ->  class yes  [0.879]

    Rule 96/2: (6.2, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = critical
        purpose = furniture/appliances
        age > 42
        existing_loans_count <= 1
        ->  class yes  [0.878]

    Rule 96/3: (6.1, lift 2.0)
        checking_balance = unknown
        purpose = furniture/appliances
        amount > 4594
        employment_duration = < 1 year
        ->  class yes  [0.877]

    Rule 96/4: (9.4/0.5, lift 2.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 10
        purpose = education
        ->  class yes  [0.866]

    Rule 96/5: (4.6, lift 2.0)
        checking_balance = unknown
        purpose = furniture/appliances
        employment_duration = < 1 year
        age > 41
        other_credit = none
        ->  class yes  [0.849]

    Rule 96/6: (17.5/2.2, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 36
        purpose = furniture/appliances
        ->  class yes  [0.838]

    Rule 96/7: (18.5/2.7, lift 1.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 18
        purpose = business
        savings_balance in {< 100 DM, 100 - 500 DM}
        employment_duration in {< 1 year, > 7 years, 1 - 4 years, unemployed}
        years_at_residence > 1
        ->  class yes  [0.818]

    Rule 96/8: (2.9, lift 1.8)
        credit_history = poor
        purpose = car
        employment_duration = unemployed
        ->  class yes  [0.795]

    Rule 96/9: (16.2/2.8, lift 1.8)
        checking_balance = < 0 DM
        purpose = education
        age <= 42
        ->  class yes  [0.788]

    Rule 96/10: (14.4/2.6, lift 1.8)
        purpose = car
        amount <= 1721
        employment_duration = 4 - 7 years
        ->  class yes  [0.782]

    Rule 96/11: (8.7/1.5, lift 1.8)
        checking_balance in {> 200 DM, unknown}
        purpose in {business, renovations}
        employment_duration = < 1 year
        ->  class yes  [0.768]

    Rule 96/12: (26.9/5.9, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        employment_duration = < 1 year
        age <= 34
        ->  class yes  [0.763]

    Rule 96/13: (14.4/3.1, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = car
        employment_duration = > 7 years
        other_credit in {bank, store}
        ->  class yes  [0.752]

    Rule 96/14: (23.7/5.4, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        savings_balance = < 100 DM
        age > 30
        existing_loans_count <= 1
        job = unskilled
        phone = FALSE
        ->  class yes  [0.752]

    Rule 96/15: (14.1/3.2, lift 1.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = renovations
        savings_balance in {< 100 DM, > 1000 DM}
        ->  class yes  [0.740]

    Rule 96/16: (116.2/58.2, lift 1.2)
        age <= 23
        ->  class yes  [0.500]

    Rule 96/17: (588.8/306.8, lift 1.1)
        checking_balance in {< 0 DM, 1 - 200 DM}
        ->  class yes  [0.479]

    Rule 96/18: (12.5, lift 1.6)
        checking_balance in {> 200 DM, unknown}
        employment_duration = 4 - 7 years
        years_at_residence > 3
        ->  class no  [0.931]

    Rule 96/19: (12.2, lift 1.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        purpose = furniture/appliances
        age <= 30
        job = unskilled
        phone = FALSE
        ->  class no  [0.930]

    Rule 96/20: (11.7/0.6, lift 1.6)
        purpose = business
        employment_duration = 4 - 7 years
        years_at_residence > 1
        ->  class no  [0.886]

    Rule 96/21: (5.6, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = business
        savings_balance = unknown
        ->  class no  [0.868]

    Rule 96/22: (17.1/1.6, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {critical, good, very good}
        purpose = car
        amount <= 12389
        employment_duration = unemployed
        dependents <= 1
        ->  class no  [0.862]

    Rule 96/23: (12.8/1.2, lift 1.5)
        purpose = education
        age > 42
        ->  class no  [0.848]

    Rule 96/24: (16.6/2.1, lift 1.5)
        purpose = car
        employment_duration = < 1 year
        age > 34
        ->  class no  [0.833]

    Rule 96/25: (40.8/7.7, lift 1.4)
        months_loan_duration <= 36
        credit_history = critical
        purpose = furniture/appliances
        savings_balance = < 100 DM
        age <= 42
        ->  class no  [0.796]

    Rule 96/26: (77.4/20.2, lift 1.3)
        checking_balance in {> 200 DM, unknown}
        employment_duration = > 7 years
        ->  class no  [0.733]

    Rule 96/27: (74.9/20.6, lift 1.3)
        checking_balance in {> 200 DM, unknown}
        employment_duration = 1 - 4 years
        other_credit = none
        job in {skilled, unskilled}
        ->  class no  [0.719]

    Rule 96/28: (788.5/328.3, lift 1.0)
        amount <= 6948
        ->  class no  [0.583]

    Default class: no

    -----  Trial 97:  -----

    Rules:

    Rule 97/1: (14, lift 2.0)
        checking_balance = < 0 DM
        months_loan_duration > 22
        purpose = car
        percent_of_income > 2
        age <= 46
        job in {skilled, unskilled}
        dependents <= 1
        ->  class yes  [0.938]

    Rule 97/2: (12.9, lift 2.0)
        checking_balance = < 0 DM
        amount <= 3573
        savings_balance = < 100 DM
        years_at_residence > 1
        other_credit = bank
        job = skilled
        ->  class yes  [0.933]

    Rule 97/3: (10.4, lift 2.0)
        checking_balance = < 0 DM
        purpose = car
        savings_balance = < 100 DM
        percent_of_income > 2
        percent_of_income <= 3
        dependents <= 1
        ->  class yes  [0.920]

    Rule 97/4: (26.1/1.5, lift 1.9)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        amount > 2394
        amount <= 3573
        savings_balance = < 100 DM
        years_at_residence > 1
        dependents <= 1
        ->  class yes  [0.912]

    Rule 97/5: (16.8/0.8, lift 1.9)
        checking_balance = 1 - 200 DM
        purpose in {business, car, education}
        savings_balance = < 100 DM
        employment_duration = > 7 years
        age > 30
        ->  class yes  [0.902]

    Rule 97/6: (8.2, lift 1.9)
        checking_balance = unknown
        amount > 2096
        employment_duration = 1 - 4 years
        percent_of_income > 1
        other_credit = bank
        ->  class yes  [0.902]

    Rule 97/7: (28.7/2.2, lift 1.9)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        amount > 1743
        amount <= 3573
        savings_balance = < 100 DM
        years_at_residence > 1
        job = skilled
        dependents <= 1
        ->  class yes  [0.897]

    Rule 97/8: (7.6, lift 1.9)
        employment_duration = unemployed
        years_at_residence <= 1
        ->  class yes  [0.896]

    Rule 97/9: (7.4, lift 1.9)
        checking_balance = < 0 DM
        months_loan_duration > 27
        credit_history = good
        savings_balance = unknown
        ->  class yes  [0.893]

    Rule 97/10: (6.6, lift 1.9)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        amount > 5771
        savings_balance = < 100 DM
        dependents <= 1
        ->  class yes  [0.884]

    Rule 97/11: (17.1/1.3, lift 1.9)
        checking_balance = unknown
        credit_history in {critical, good}
        amount > 709
        savings_balance in {100 - 500 DM, 500 - 1000 DM, unknown}
        employment_duration = > 7 years
        age <= 35
        ->  class yes  [0.882]

    Rule 97/12: (5.1, lift 1.8)
        checking_balance = 1 - 200 DM
        employment_duration = < 1 year
        other_credit = store
        ->  class yes  [0.859]

    Rule 97/13: (7.6/0.4, lift 1.8)
        checking_balance = 1 - 200 DM
        credit_history = good
        savings_balance = < 100 DM
        percent_of_income <= 1
        phone = FALSE
        ->  class yes  [0.850]

    Rule 97/14: (4.5, lift 1.8)
        checking_balance = unknown
        months_loan_duration > 30
        employment_duration = < 1 year
        ->  class yes  [0.845]

    Rule 97/15: (12.1/1.2, lift 1.8)
        checking_balance = unknown
        purpose in {business, car, furniture/appliances}
        employment_duration = unemployed
        percent_of_income > 2
        ->  class yes  [0.842]

    Rule 97/16: (3.9, lift 1.8)
        checking_balance = unknown
        employment_duration = 1 - 4 years
        existing_loans_count > 2
        ->  class yes  [0.829]

    Rule 97/17: (3.8, lift 1.8)
        checking_balance = < 0 DM
        credit_history = very good
        savings_balance = unknown
        ->  class yes  [0.828]

    Rule 97/18: (3.7, lift 1.7)
        checking_balance = 1 - 200 DM
        months_loan_duration <= 8
        employment_duration = unemployed
        ->  class yes  [0.825]

    Rule 97/19: (18.2/2.6, lift 1.7)
        purpose = car
        amount <= 960
        percent_of_income > 3
        age <= 46
        ->  class yes  [0.820]

    Rule 97/20: (20/3.1, lift 1.7)
        checking_balance = 1 - 200 DM
        purpose = furniture/appliances
        amount > 1721
        years_at_residence > 1
        years_at_residence <= 3
        other_credit = none
        phone = FALSE
        ->  class yes  [0.815]

    Rule 97/21: (18.5/2.8, lift 1.7)
        checking_balance = 1 - 200 DM
        employment_duration = < 1 year
        housing = rent
        ->  class yes  [0.813]

    Rule 97/22: (13.8/2.5, lift 1.7)
        checking_balance = < 0 DM
        credit_history = good
        savings_balance = unknown
        phone = FALSE
        ->  class yes  [0.782]

    Rule 97/23: (6.3/0.9, lift 1.6)
        employment_duration = unemployed
        dependents > 1
        ->  class yes  [0.765]

    Rule 97/24: (16.2/3.5, lift 1.6)
        checking_balance = 1 - 200 DM
        credit_history = good
        employment_duration = 1 - 4 years
        years_at_residence > 3
        ->  class yes  [0.755]

    Rule 97/25: (28.6/7.4, lift 1.5)
        checking_balance = 1 - 200 DM
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM}
        employment_duration = < 1 year
        years_at_residence > 1
        ->  class yes  [0.725]

    Rule 97/26: (596.5/304.9, lift 1.0)
        savings_balance = < 100 DM
        ->  class yes  [0.489]

    Rule 97/27: (17.1, lift 1.8)
        checking_balance = 1 - 200 DM
        employment_duration = < 1 year
        years_at_residence <= 1
        other_credit in {bank, none}
        housing = own
        ->  class no  [0.948]

    Rule 97/28: (13.6, lift 1.8)
        checking_balance = unknown
        credit_history in {critical, perfect}
        employment_duration = 1 - 4 years
        other_credit = none
        ->  class no  [0.936]

    Rule 97/29: (10.6, lift 1.7)
        checking_balance = 1 - 200 DM
        employment_duration = 4 - 7 years
        housing = rent
        ->  class no  [0.921]

    Rule 97/30: (9.1, lift 1.7)
        checking_balance = 1 - 200 DM
        employment_duration = > 7 years
        age <= 30
        ->  class no  [0.910]

    Rule 97/31: (9.4/0.3, lift 1.7)
        purpose in {car0, education}
        employment_duration = unemployed
        ->  class no  [0.891]

    Rule 97/32: (12.4/1.4, lift 1.6)
        employment_duration = 4 - 7 years
        housing = own
        dependents > 1
        ->  class no  [0.836]

    Rule 97/33: (35.5/5.7, lift 1.6)
        checking_balance = unknown
        employment_duration = > 7 years
        age > 35
        ->  class no  [0.821]

    Rule 97/34: (7.9/1.1, lift 1.5)
        employment_duration = < 1 year
        housing = other
        ->  class no  [0.784]

    Rule 97/35: (34.1/8.3, lift 1.4)
        checking_balance = unknown
        employment_duration = 4 - 7 years
        ->  class no  [0.742]

    Rule 97/36: (37.7/10.6, lift 1.3)
        credit_history in {critical, perfect, poor}
        savings_balance = unknown
        ->  class no  [0.709]

    Rule 97/37: (110.4/34.7, lift 1.3)
        checking_balance = unknown
        amount <= 2096
        ->  class no  [0.682]

    Rule 97/38: (386.3/164.2, lift 1.1)
        credit_history in {critical, good}
        amount <= 2394
        ->  class no  [0.575]

    Rule 97/39: (321.2/141.2, lift 1.1)
        percent_of_income <= 2
        ->  class no  [0.560]

    Default class: no

    -----  Trial 98:  -----

    Rules:

    Rule 98/1: (22.4/1.6, lift 1.8)
        months_loan_duration > 18
        credit_history = good
        purpose in {business, car0, furniture/appliances}
        housing = own
        existing_loans_count > 1
        dependents <= 1
        ->  class yes  [0.892]

    Rule 98/2: (6.9, lift 1.8)
        credit_history = good
        percent_of_income > 3
        years_at_residence > 1
        other_credit = store
        phone = FALSE
        ->  class yes  [0.888]

    Rule 98/3: (9.5/0.4, lift 1.8)
        months_loan_duration > 30
        purpose = furniture/appliances
        employment_duration = 1 - 4 years
        years_at_residence > 1
        existing_loans_count <= 1
        dependents <= 1
        ->  class yes  [0.882]

    Rule 98/4: (17.3/1.3, lift 1.8)
        credit_history = good
        housing = rent
        existing_loans_count > 1
        dependents <= 1
        ->  class yes  [0.880]

    Rule 98/5: (9.8/0.9, lift 1.7)
        credit_history = good
        purpose = furniture/appliances
        percent_of_income > 3
        existing_loans_count <= 1
        dependents > 1
        ->  class yes  [0.842]

    Rule 98/6: (10.8/1.1, lift 1.7)
        credit_history = perfect
        savings_balance = < 100 DM
        age > 33
        housing = own
        ->  class yes  [0.837]

    Rule 98/7: (29.3/4.2, lift 1.7)
        credit_history = poor
        savings_balance in {< 100 DM, > 1000 DM, 100 - 500 DM}
        percent_of_income > 2
        housing in {other, own}
        job in {management, skilled}
        ->  class yes  [0.835]

    Rule 98/8: (16/2.4, lift 1.6)
        credit_history = critical
        amount > 7763
        ->  class yes  [0.810]

    Rule 98/9: (28.1/5.7, lift 1.6)
        credit_history = critical
        other_credit = bank
        job in {skilled, unskilled}
        ->  class yes  [0.777]

    Rule 98/10: (34.6/7.7, lift 1.6)
        credit_history = good
        employment_duration in {< 1 year, 4 - 7 years, unemployed}
        percent_of_income > 3
        years_at_residence > 1
        other_credit = none
        dependents <= 1
        phone = FALSE
        ->  class yes  [0.762]

    Rule 98/11: (117.1/31.8, lift 1.5)
        credit_history = good
        amount <= 1386
        savings_balance in {< 100 DM, 100 - 500 DM, 500 - 1000 DM, unknown}
        existing_loans_count <= 1
        phone = FALSE
        ->  class yes  [0.725]

    Rule 98/12: (44.7/12.9, lift 1.4)
        months_loan_duration <= 39
        credit_history = very good
        age > 23
        ->  class yes  [0.701]

    Rule 98/13: (96.7/39.6, lift 1.2)
        purpose in {business, education}
        savings_balance = < 100 DM
        ->  class yes  [0.589]

    Rule 98/14: (591.9/287.7, lift 1.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        ->  class yes  [0.514]

    Rule 98/15: (28, lift 1.9)
        checking_balance = unknown
        credit_history = good
        purpose in {car, furniture/appliances}
        existing_loans_count <= 1
        phone = TRUE
        ->  class no  [0.967]

    Rule 98/16: (21.3, lift 1.9)
        credit_history = good
        amount > 1386
        employment_duration in {4 - 7 years, unemployed}
        percent_of_income <= 3
        phone = FALSE
        ->  class no  [0.957]

    Rule 98/17: (11.8, lift 1.8)
        months_loan_duration <= 18
        credit_history = good
        housing = own
        existing_loans_count > 1
        ->  class no  [0.927]

    Rule 98/18: (9.3, lift 1.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = good
        amount <= 11328
        other_credit = bank
        existing_loans_count <= 1
        phone = TRUE
        ->  class no  [0.912]

    Rule 98/19: (9.1, lift 1.8)
        credit_history = good
        purpose = furniture/appliances
        amount > 1386
        percent_of_income > 3
        other_credit = bank
        existing_loans_count <= 1
        ->  class no  [0.910]

    Rule 98/20: (8.4, lift 1.8)
        credit_history = good
        purpose in {car, education}
        housing = own
        existing_loans_count > 1
        ->  class no  [0.904]

    Rule 98/21: (8.2, lift 1.8)
        credit_history = very good
        age <= 23
        ->  class no  [0.902]

    Rule 98/22: (7.5, lift 1.8)
        credit_history = good
        savings_balance = > 1000 DM
        existing_loans_count <= 1
        ->  class no  [0.894]

    Rule 98/23: (7.2, lift 1.8)
        credit_history = critical
        other_credit = bank
        job in {management, unemployed}
        ->  class no  [0.892]

    Rule 98/24: (7, lift 1.8)
        months_loan_duration <= 8
        credit_history = good
        amount <= 11328
        phone = TRUE
        ->  class no  [0.889]

    Rule 98/25: (6.6, lift 1.7)
        checking_balance = > 200 DM
        credit_history = good
        phone = TRUE
        ->  class no  [0.884]

    Rule 98/26: (6.4, lift 1.7)
        credit_history = critical
        purpose in {car0, renovations}
        dependents <= 1
        ->  class no  [0.881]

    Rule 98/27: (6.2, lift 1.7)
        months_loan_duration <= 20
        credit_history = poor
        percent_of_income <= 2
        ->  class no  [0.878]

    Rule 98/28: (4.4, lift 1.7)
        credit_history = perfect
        savings_balance in {500 - 1000 DM, unknown}
        ->  class no  [0.844]

    Rule 98/29: (19.5/2.5, lift 1.7)
        months_loan_duration <= 30
        credit_history = good
        amount > 1386
        employment_duration = 1 - 4 years
        percent_of_income > 3
        other_credit = none
        existing_loans_count <= 1
        dependents <= 1
        ->  class no  [0.838]

    Rule 98/30: (24.8/3.7, lift 1.6)
        credit_history = good
        purpose = furniture/appliances
        amount > 1386
        years_at_residence <= 1
        dependents <= 1
        phone = FALSE
        ->  class no  [0.823]

    Rule 98/31: (18.4/2.7, lift 1.6)
        credit_history = good
        amount <= 11328
        savings_balance = < 100 DM
        years_at_residence <= 1
        phone = TRUE
        ->  class no  [0.818]

    Rule 98/32: (27.7/4.4, lift 1.6)
        credit_history = good
        amount <= 11328
        savings_balance = < 100 DM
        employment_duration in {1 - 4 years, unemployed}
        other_credit = none
        existing_loans_count <= 1
        phone = TRUE
        ->  class no  [0.817]

    Rule 98/33: (10.6/1.3, lift 1.6)
        credit_history = poor
        housing = rent
        ->  class no  [0.815]

    Rule 98/34: (10.5/1.5, lift 1.6)
        credit_history = poor
        housing = own
        job = unskilled
        ->  class no  [0.801]

    Rule 98/35: (17.1/3, lift 1.6)
        credit_history = perfect
        amount <= 10297
        age <= 33
        housing = own
        ->  class no  [0.791]

    Rule 98/36: (25.6/5.3, lift 1.5)
        credit_history = critical
        amount <= 7763
        savings_balance in {> 1000 DM, 100 - 500 DM, unknown}
        other_credit in {none, store}
        ->  class no  [0.770]

    Rule 98/37: (69.9/16, lift 1.5)
        credit_history = good
        amount > 1386
        amount <= 3976
        percent_of_income <= 3
        other_credit = none
        existing_loans_count <= 1
        phone = FALSE
        ->  class no  [0.763]

    Rule 98/38: (91.2/21.8, lift 1.5)
        months_loan_duration <= 36
        credit_history = critical
        purpose in {car, furniture/appliances}
        amount <= 7763
        savings_balance = < 100 DM
        other_credit in {none, store}
        ->  class no  [0.756]

    Default class: no

    -----  Trial 99:  -----

    Rules:

    Rule 99/1: (12.4/0.7, lift 2.0)
        checking_balance in {1 - 200 DM, unknown}
        months_loan_duration <= 9
        credit_history = good
        amount > 909
        employment_duration in {< 1 year, 1 - 4 years}
        age <= 32
        housing = own
        job = skilled
        ->  class yes  [0.884]

    Rule 99/2: (6.3, lift 2.0)
        years_at_residence <= 3
        other_credit = store
        job = management
        ->  class yes  [0.880]

    Rule 99/3: (7.9/0.3, lift 2.0)
        months_loan_duration > 42
        credit_history = good
        employment_duration in {< 1 year, 1 - 4 years}
        job = skilled
        ->  class yes  [0.871]

    Rule 99/4: (18.6/2.4, lift 1.9)
        checking_balance in {> 200 DM, 1 - 200 DM, unknown}
        credit_history in {good, perfect, poor, very good}
        employment_duration in {< 1 year, 1 - 4 years, unemployed}
        years_at_residence <= 3
        other_credit = none
        housing = own
        job = management
        ->  class yes  [0.835]

    Rule 99/5: (11.7/1.4, lift 1.9)
        checking_balance = < 0 DM
        employment_duration = > 7 years
        job = management
        dependents <= 1
        ->  class yes  [0.827]

    Rule 99/6: (3.8, lift 1.9)
        employment_duration = > 7 years
        age > 54
        dependents > 1
        ->  class yes  [0.827]

    Rule 99/7: (16.3/2.3, lift 1.8)
        checking_balance in {> 200 DM, 1 - 200 DM, unknown}
        amount <= 1264
        employment_duration in {> 7 years, 4 - 7 years}
        dependents > 1
        ->  class yes  [0.820]

    Rule 99/8: (41.6/6.9, lift 1.8)
        checking_balance in {> 200 DM, 1 - 200 DM, unknown}
        months_loan_duration > 7
        amount > 909
        employment_duration in {< 1 year, 1 - 4 years, unemployed}
        percent_of_income > 1
        age <= 33
        housing = rent
        ->  class yes  [0.819]

    Rule 99/9: (14.9/2.6, lift 1.8)
        employment_duration in {1 - 4 years, unemployed}
        age > 36
        age <= 52
        housing = other
        ->  class yes  [0.787]

    Rule 99/10: (32.4/7.1, lift 1.7)
        checking_balance = < 0 DM
        months_loan_duration > 11
        savings_balance = < 100 DM
        job = skilled
        dependents <= 1
        phone = TRUE
        ->  class yes  [0.765]

    Rule 99/11: (24.1/6.4, lift 1.6)
        checking_balance in {> 200 DM, 1 - 200 DM}
        other_credit = none
        housing = own
        job = management
        ->  class yes  [0.718]

    Rule 99/12: (50.1/14.5, lift 1.6)
        checking_balance in {> 200 DM, 1 - 200 DM, unknown}
        purpose in {business, car, furniture/appliances}
        amount > 909
        employment_duration in {< 1 year, 1 - 4 years}
        age <= 44
        other_credit in {bank, none}
        job = unskilled
        ->  class yes  [0.703]

    Rule 99/13: (36.8/11.4, lift 1.5)
        checking_balance in {> 200 DM, 1 - 200 DM, unknown}
        years_at_residence <= 2
        other_credit = bank
        job in {skilled, unskilled}
        ->  class yes  [0.680]

    Rule 99/14: (43.9/18.8, lift 1.3)
        credit_history = perfect
        ->  class yes  [0.569]

    Rule 99/15: (295.4/141.7, lift 1.2)
        checking_balance = < 0 DM
        ->  class yes  [0.520]

    Rule 99/16: (19.3, lift 1.7)
        employment_duration in {> 7 years, 4 - 7 years}
        other_credit = none
        job = unskilled
        dependents <= 1
        ->  class no  [0.953]

    Rule 99/17: (15.3, lift 1.7)
        checking_balance in {> 200 DM, 1 - 200 DM, unknown}
        months_loan_duration <= 9
        credit_history = good
        age > 32
        ->  class no  [0.942]

    Rule 99/18: (9.6, lift 1.6)
        checking_balance = < 0 DM
        months_loan_duration > 21
        months_loan_duration <= 27
        savings_balance = < 100 DM
        employment_duration = 1 - 4 years
        dependents <= 1
        phone = FALSE
        ->  class no  [0.914]

    Rule 99/19: (8.5, lift 1.6)
        checking_balance = < 0 DM
        purpose in {business, car0, education}
        job = unskilled
        dependents <= 1
        ->  class no  [0.905]

    Rule 99/20: (38.2/5.1, lift 1.5)
        checking_balance in {> 200 DM, 1 - 200 DM, unknown}
        amount <= 909
        savings_balance in {< 100 DM, > 1000 DM, 100 - 500 DM, unknown}
        employment_duration in {< 1 year, 1 - 4 years, unemployed}
        ->  class no  [0.849]

    Rule 99/21: (34.6/6.9, lift 1.4)
        checking_balance = < 0 DM
        savings_balance = < 100 DM
        percent_of_income > 1
        job = management
        ->  class no  [0.783]

    Rule 99/22: (80.6/20.6, lift 1.3)
        months_loan_duration > 11
        purpose in {business, car, furniture/appliances}
        amount <= 5595
        employment_duration in {< 1 year, > 7 years}
        existing_loans_count <= 1
        job = skilled
        dependents <= 1
        ->  class no  [0.738]

    Rule 99/23: (603.6/246.1, lift 1.1)
        checking_balance in {> 200 DM, 1 - 200 DM, unknown}
        ->  class no  [0.592]

    Default class: no


    Evaluation on training data (900 cases):

    Trial           Rules     
    -----     ----------------
            No      Errors

       0        30  115(12.8%)
       1        22  215(23.9%)
       2        34  169(18.8%)
       3        22  177(19.7%)
       4        25  223(24.8%)
       5        36  164(18.2%)
       6        29  169(18.8%)
       7        43  183(20.3%)
       8        34  182(20.2%)
       9        26  214(23.8%)
      10        21  186(20.7%)
      11        37  203(22.6%)
      12        31  157(17.4%)
      13        37  186(20.7%)
      14        30  208(23.1%)
      15        33  169(18.8%)
      16        33  159(17.7%)
      17        35  226(25.1%)
      18        40  193(21.4%)
      19        34  181(20.1%)
      20        29  157(17.4%)
      21        39  182(20.2%)
      22        32  200(22.2%)
      23        37  181(20.1%)
      24        25  176(19.6%)
      25        28  179(19.9%)
      26        47  175(19.4%)
      27        23  242(26.9%)
      28        20  267(29.7%)
      29        33  183(20.3%)
      30        40  160(17.8%)
      31        25  199(22.1%)
      32        25  171(19.0%)
      33        33  256(28.4%)
      34        29  168(18.7%)
      35        32  164(18.2%)
      36        31  255(28.3%)
      37        31  185(20.6%)
      38        36  198(22.0%)
      39        28  174(19.3%)
      40        27  146(16.2%)
      41        31  209(23.2%)
      42        31  175(19.4%)
      43        22  190(21.1%)
      44        27  169(18.8%)
      45        36  232(25.8%)
      46        32  138(15.3%)
      47        33  191(21.2%)
      48        37  154(17.1%)
      49        24  177(19.7%)
      50        28  195(21.7%)
      51        35  177(19.7%)
      52        33  196(21.8%)
      53        27  208(23.1%)
      54        27  279(31.0%)
      55        22  187(20.8%)
      56        31  211(23.4%)
      57        40  157(17.4%)
      58        30  161(17.9%)
      59        27  160(17.8%)
      60        34  195(21.7%)
      61        34  189(21.0%)
      62        41  166(18.4%)
      63        28  183(20.3%)
      64        27  182(20.2%)
      65        31  161(17.9%)
      66        40  233(25.9%)
      67        37  152(16.9%)
      68        32  200(22.2%)
      69        23  168(18.7%)
      70        34  163(18.1%)
      71        31  270(30.0%)
      72        27  176(19.6%)
      73        32  189(21.0%)
      74        27  161(17.9%)
      75        36  176(19.6%)
      76        32  174(19.3%)
      77        38  170(18.9%)
      78        24  289(32.1%)
      79        34  178(19.8%)
      80        31  149(16.6%)
      81        27  213(23.7%)
      82        22  180(20.0%)
      83        38  175(19.4%)
      84        27  176(19.6%)
      85        47  217(24.1%)
      86        31  159(17.7%)
      87        43  208(23.1%)
      88        35  166(18.4%)
      89        31  199(22.1%)
      90        24  167(18.6%)
      91        37  187(20.8%)
      92        31  186(20.7%)
      93        33  178(19.8%)
      94        28  191(21.2%)
      95        37  163(18.1%)
      96        28  185(20.6%)
      97        39  156(17.3%)
      98        38  183(20.3%)
      99        23  209(23.2%)
    boost             4( 0.4%)   <<


           (a)   (b)    <-classified as
          ----  ----
           266     4    (a): class yes
                 630    (b): class no


        Attribute usage:

        100.00% checking_balance
        100.00% months_loan_duration
        100.00% credit_history
        100.00% purpose
        100.00% amount
        100.00% savings_balance
        100.00% employment_duration
        100.00% percent_of_income
        100.00% years_at_residence
        100.00% age
        100.00% other_credit
        100.00% housing
        100.00% job
        100.00% dependents
        100.00% phone
         99.89% existing_loans_count


    Time: 1.1 secs

#### Model Predictors

     [1] "checking_balance"     "credit_history"       "savings_balance"      "purpose"              "employment_duration"  "months_loan_duration" "amount"              
     [8] "job"                  "percent_of_income"    "other_credit"         "age"                  "years_at_residence"   "housing"              "phone"               
    [15] "dependents"           "existing_loans_count"

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/c50_model_grouped_categories-1.png" width="750px" />

#### Variable Importance

    C5.0 variable importance

     Overall
      100.00
      100.00
      100.00
      100.00
      100.00
      100.00
      100.00
      100.00
      100.00
      100.00
      100.00
      100.00
      100.00
      100.00
      100.00
       99.89

### c50\_rules\_model\_independent\_categories

#### Model Summary


    Call:
    C5.0.default(x = x, y = y, rules = TRUE, weights = wts)


    C5.0 [Release 2.07 GPL Edition]     Tue Jun 27 22:09:33 2017
    -------------------------------

    Class specified by attribute `outcome'

    Read 900 cases (36 attributes) from undefined.data

    Rules:

    Rule 1: (21/1, lift 3.0)
        checking_balanceunknown <= 0
        months_loan_duration > 36
        credit_historyvery good <= 0
        amount <= 7980
        employment_durationunemployed <= 0
        years_at_residence > 1
        dependents <= 1
        ->  class yes  [0.913]

    Rule 2: (9, lift 3.0)
        checking_balance> 200 DM <= 0
        checking_balance1 - 200 DM <= 0
        checking_balanceunknown <= 0
        credit_historyvery good <= 0
        employment_duration1 - 4 years > 0
        years_at_residence > 1
        age <= 24
        jobskilled > 0
        phoneTRUE <= 0
        ->  class yes  [0.909]

    Rule 3: (9, lift 3.0)
        checking_balanceunknown <= 0
        purposefurniture/appliances > 0
        savings_balance100 - 500 DM > 0
        employment_duration> 7 years <= 0
        years_at_residence > 1
        phoneTRUE <= 0
        ->  class yes  [0.909]

    Rule 4: (8, lift 3.0)
        checking_balance> 200 DM <= 0
        checking_balance1 - 200 DM <= 0
        checking_balanceunknown <= 0
        months_loan_duration > 11
        months_loan_duration <= 36
        savings_balance100 - 500 DM <= 0
        employment_duration4 - 7 years > 0
        jobskilled > 0
        phoneTRUE <= 0
        ->  class yes  [0.900]

    Rule 5: (7, lift 3.0)
        checking_balanceunknown <= 0
        months_loan_duration > 11
        other_creditnone <= 0
        dependents > 1
        phoneTRUE <= 0
        ->  class yes  [0.889]

    Rule 6: (7, lift 3.0)
        checking_balanceunknown <= 0
        months_loan_duration > 11
        months_loan_duration <= 36
        credit_historygood > 0
        percent_of_income > 3
        dependents > 1
        phoneTRUE <= 0
        ->  class yes  [0.889]

    Rule 7: (27/3, lift 2.9)
        checking_balanceunknown <= 0
        credit_historyperfect <= 0
        credit_historypoor <= 0
        credit_historyvery good <= 0
        amount > 7980
        ->  class yes  [0.862]

    Rule 8: (5, lift 2.9)
        checking_balanceunknown <= 0
        months_loan_duration <= 11
        purposeeducation > 0
        age <= 36
        ->  class yes  [0.857]

    Rule 9: (18/2, lift 2.8)
        checking_balance> 200 DM <= 0
        checking_balanceunknown <= 0
        months_loan_duration > 11
        credit_historyperfect <= 0
        purposefurniture/appliances > 0
        employment_duration> 7 years <= 0
        years_at_residence > 1
        age > 30
        dependents <= 1
        phoneTRUE <= 0
        ->  class yes  [0.850]

    Rule 10: (4, lift 2.8)
        checking_balanceunknown <= 0
        employment_durationunemployed > 0
        dependents > 1
        ->  class yes  [0.833]

    Rule 11: (14/3, lift 2.5)
        checking_balanceunknown > 0
        purposefurniture/appliances <= 0
        savings_balance> 1000 DM <= 0
        employment_durationunemployed <= 0
        percent_of_income > 1
        years_at_residence > 1
        age <= 43
        other_creditnone <= 0
        ->  class yes  [0.750]

    Rule 12: (36/10, lift 2.4)
        checking_balance> 200 DM <= 0
        checking_balanceunknown <= 0
        months_loan_duration > 11
        months_loan_duration <= 36
        purposecar > 0
        employment_duration> 7 years <= 0
        percent_of_income > 2
        phoneTRUE <= 0
        ->  class yes  [0.711]

    Rule 13: (33/10, lift 2.3)
        checking_balanceunknown <= 0
        credit_historyvery good > 0
        amount <= 7980
        savings_balance500 - 1000 DM <= 0
        ->  class yes  [0.686]

    Rule 14: (33/10, lift 2.3)
        checking_balanceunknown <= 0
        credit_historyperfect > 0
        ->  class yes  [0.686]

    Rule 15: (35/12, lift 2.2)
        checking_balanceunknown <= 0
        credit_historyvery good > 0
        amount <= 7980
        ->  class yes  [0.649]

    Rule 16: (43/17, lift 2.0)
        checking_balance> 200 DM <= 0
        checking_balanceunknown <= 0
        months_loan_duration > 11
        months_loan_duration <= 36
        credit_historyvery good <= 0
        purposecar > 0
        employment_duration> 7 years <= 0
        dependents <= 1
        phoneTRUE <= 0
        ->  class yes  [0.600]

    Rule 17: (95/51, lift 1.5)
        purposefurniture/appliances <= 0
        other_creditnone <= 0
        ->  class yes  [0.464]

    Rule 18: (301/25, lift 1.3)
        checking_balanceunknown > 0
        other_creditnone > 0
        ->  class no  [0.914]

    Rule 19: (861/246, lift 1.0)
        credit_historyperfect <= 0
        ->  class no  [0.714]

    Default class: no


    Evaluation on training data (900 cases):

                Rules     
          ----------------
            No      Errors

            19  144(16.0%)   <<


           (a)   (b)    <-classified as
          ----  ----
           164   106    (a): class yes
            38   592    (b): class no


        Attribute usage:

         99.33% credit_historyperfect
         57.44% checking_balanceunknown
         44.44% other_creditnone
         14.78% credit_historyvery good
         13.33% purposefurniture/appliances
         11.89% months_loan_duration
         10.78% phoneTRUE
         10.44% dependents
          9.22% amount
          9.00% checking_balance> 200 DM
          8.33% employment_duration> 7 years
          7.22% years_at_residence
          6.22% percent_of_income
          5.56% purposecar
          5.11% age
          4.33% employment_durationunemployed
          3.67% savings_balance500 - 1000 DM
          3.00% credit_historypoor
          1.89% checking_balance1 - 200 DM
          1.89% savings_balance100 - 500 DM
          1.89% jobskilled
          1.56% savings_balance> 1000 DM
          1.00% employment_duration1 - 4 years
          0.89% employment_duration4 - 7 years
          0.78% credit_historygood
          0.56% purposeeducation


    Time: 0.0 secs

#### Model Predictors

     [1] "checking_balanceunknown"       "months_loan_duration"          "phoneTRUE"                     "credit_historyvery"            "dependents"                   
     [6] "checking_balance>"             "years_at_residence"            "age"                           "amount"                        "credit_historyperfect"        
    [11] "employment_duration>"          "other_creditnone"              "purposefurniture/appliances"   "employment_durationunemployed" "percent_of_income"            
    [16] "checking_balance1"             "jobskilled"                    "purposecar"                    "savings_balance100"            "credit_historygood"           
    [21] "credit_historypoor"            "employment_duration1"          "employment_duration4"          "purposeeducation"              "savings_balance>"             
    [26] "savings_balance500"           

#### Variable Importance

    C5.0Rules variable importance

      only 20 most important variables shown (out of 44)

     Overall
       99.33
       57.44
       44.44
       14.78
       13.33
       11.89
       10.78
       10.44
        9.22
        9.00
        8.33
        7.22
        6.22
        5.56
        5.11
        4.33
        3.67
        3.00
        1.89
        1.89

### c50\_rules\_model\_grouped\_categories

#### Model Summary


    Call:
    C5.0.default(x = x, y = y, rules = TRUE, weights = wts)


    C5.0 [Release 2.07 GPL Edition]     Tue Jun 27 22:10:06 2017
    -------------------------------

    Class specified by attribute `outcome'

    Read 900 cases (17 attributes) from undefined.data

    Rules:

    Rule 1: (10, lift 3.1)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 36
        amount <= 7980
        employment_duration in {< 1 year, > 7 years}
        ->  class yes  [0.917]

    Rule 2: (10, lift 3.1)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 36
        credit_history in {critical, good}
        employment_duration = 1 - 4 years
        ->  class yes  [0.917]

    Rule 3: (8, lift 3.0)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {critical, good}
        purpose = car
        other_credit in {bank, store}
        phone = FALSE
        ->  class yes  [0.900]

    Rule 4: (7, lift 3.0)
        credit_history in {good, poor}
        amount <= 2101
        other_credit = store
        phone = FALSE
        ->  class yes  [0.889]

    Rule 5: (6, lift 2.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 11
        purpose = car
        amount <= 5179
        savings_balance = < 100 DM
        employment_duration = > 7 years
        age <= 50
        phone = FALSE
        ->  class yes  [0.875]

    Rule 6: (6, lift 2.9)
        checking_balance = unknown
        purpose = business
        employment_duration in {< 1 year, 1 - 4 years, unemployed}
        percent_of_income > 1
        other_credit in {bank, store}
        ->  class yes  [0.875]

    Rule 7: (6, lift 2.9)
        checking_balance in {> 200 DM, unknown}
        months_loan_duration > 16
        purpose = car
        percent_of_income > 1
        other_credit = bank
        phone = FALSE
        ->  class yes  [0.875]

    Rule 8: (27/3, lift 2.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {critical, good}
        amount > 7980
        ->  class yes  [0.862]

    Rule 9: (5, lift 2.9)
        checking_balance = < 0 DM
        credit_history in {good, poor}
        purpose = education
        phone = FALSE
        ->  class yes  [0.857]

    Rule 10: (12/1, lift 2.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 16
        purpose = furniture/appliances
        savings_balance = < 100 DM
        percent_of_income > 1
        years_at_residence <= 3
        other_credit = none
        existing_loans_count <= 1
        job = skilled
        phone = FALSE
        ->  class yes  [0.857]

    Rule 11: (5, lift 2.9)
        checking_balance = < 0 DM
        purpose = furniture/appliances
        employment_duration in {1 - 4 years, 4 - 7 years}
        years_at_residence > 3
        housing = rent
        job = skilled
        ->  class yes  [0.857]

    Rule 12: (5, lift 2.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration <= 36
        savings_balance = < 100 DM
        job = unemployed
        phone = FALSE
        ->  class yes  [0.857]

    Rule 13: (5, lift 2.9)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history = poor
        savings_balance = < 100 DM
        percent_of_income > 3
        phone = TRUE
        ->  class yes  [0.857]

    Rule 14: (17/2, lift 2.8)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 36
        credit_history in {critical, good, poor}
        purpose in {business, car0, education, furniture/appliances}
        amount <= 7980
        age <= 37
        ->  class yes  [0.842]

    Rule 15: (4, lift 2.8)
        checking_balance = < 0 DM
        credit_history = critical
        savings_balance = < 100 DM
        housing = rent
        phone = TRUE
        ->  class yes  [0.833]

    Rule 16: (15/2, lift 2.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 11
        months_loan_duration <= 36
        purpose = car
        amount <= 5179
        savings_balance = < 100 DM
        employment_duration in {< 1 year, 4 - 7 years}
        phone = FALSE
        ->  class yes  [0.824]

    Rule 17: (9/1, lift 2.7)
        checking_balance = < 0 DM
        credit_history = good
        savings_balance = unknown
        job = skilled
        phone = FALSE
        ->  class yes  [0.818]

    Rule 18: (8/1, lift 2.7)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 11
        amount > 1829
        age > 30
        other_credit = none
        job = unskilled
        phone = FALSE
        ->  class yes  [0.800]

    Rule 19: (12/2, lift 2.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        purpose = furniture/appliances
        savings_balance = 100 - 500 DM
        phone = FALSE
        ->  class yes  [0.786]

    Rule 20: (7/1, lift 2.6)
        checking_balance in {< 0 DM, 1 - 200 DM}
        months_loan_duration > 11
        credit_history = good
        savings_balance = < 100 DM
        percent_of_income > 2
        percent_of_income <= 3
        phone = TRUE
        ->  class yes  [0.778]

    Rule 21: (46/11, lift 2.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        credit_history in {perfect, very good}
        savings_balance in {< 100 DM, > 1000 DM}
        ->  class yes  [0.750]

    Rule 22: (9/2, lift 2.4)
        checking_balance = 1 - 200 DM
        credit_history in {perfect, very good}
        savings_balance = 100 - 500 DM
        ->  class yes  [0.727]

    Rule 23: (481/267, lift 1.5)
        checking_balance in {< 0 DM, 1 - 200 DM}
        ->  class yes  [0.445]

    Rule 24: (100/6, lift 1.3)
        checking_balance in {> 200 DM, unknown}
        age > 44
        ->  class no  [0.931]

    Rule 25: (62/4, lift 1.3)
        checking_balance in {> 200 DM, unknown}
        months_loan_duration <= 16
        purpose = car
        ->  class no  [0.922]

    Rule 26: (20/1, lift 1.3)
        checking_balance = 1 - 200 DM
        savings_balance = unknown
        housing in {other, own}
        phone = FALSE
        ->  class no  [0.909]

    Rule 27: (19/1, lift 1.3)
        months_loan_duration > 16
        amount <= 7980
        savings_balance = 100 - 500 DM
        phone = TRUE
        ->  class no  [0.905]

    Rule 28: (349/35, lift 1.3)
        checking_balance in {> 200 DM, unknown}
        other_credit = none
        ->  class no  [0.897]

    Rule 29: (147/18, lift 1.2)
        months_loan_duration <= 11
        credit_history in {critical, good, poor}
        amount <= 7980
        ->  class no  [0.872]

    Rule 30: (740/244, lift 1.0)
        months_loan_duration > 11
        ->  class no  [0.670]

    Default class: no


    Evaluation on training data (900 cases):

                Rules     
          ----------------
            No      Errors

            30  115(12.8%)   <<


           (a)   (b)    <-classified as
          ----  ----
           177    93    (a): class yes
            22   608    (b): class no


        Attribute usage:

         98.78% months_loan_duration
         95.89% checking_balance
         44.00% other_credit
         31.78% credit_history
         27.78% amount
         18.00% savings_balance
         16.44% purpose
         15.67% phone
         14.56% age
          5.67% employment_duration
          4.33% job
          4.00% percent_of_income
          3.22% housing
          1.89% years_at_residence
          1.33% existing_loans_count


    Time: 0.0 secs

#### Model Predictors

     [1] "checking_balance"     "phone"                "months_loan_duration" "credit_history"       "savings_balance"      "purpose"              "amount"              
     [8] "other_credit"         "employment_duration"  "percent_of_income"    "job"                  "age"                  "housing"              "years_at_residence"  
    [15] "existing_loans_count"

#### Variable Importance

    C5.0Rules variable importance

     Overall
       98.78
       95.89
       44.00
       31.78
       27.78
       18.00
       16.44
       15.67
       14.56
        5.67
        4.33
        4.00
        3.22
        1.89
        1.33
        0.00

### rf\_independent\_categories

#### Model Summary

                    Length Class      Mode     
    call               4   -none-     call     
    type               1   -none-     character
    predicted        900   factor     numeric  
    err.rate        1500   -none-     numeric  
    confusion          6   -none-     numeric  
    votes           1800   matrix     numeric  
    oob.times        900   -none-     numeric  
    classes            2   -none-     character
    importance        35   -none-     numeric  
    importanceSD       0   -none-     NULL     
    localImportance    0   -none-     NULL     
    proximity          0   -none-     NULL     
    ntree              1   -none-     numeric  
    mtry               1   -none-     numeric  
    forest            14   -none-     list     
    y                900   factor     numeric  
    test               0   -none-     NULL     
    inbag              0   -none-     NULL     
    xNames            35   -none-     character
    problemType        1   -none-     character
    tuneValue          1   data.frame list     
    obsLevels          2   -none-     character
    param              0   -none-     list     

#### Model Predictors

     [1] "checking_balance> 200 DM"       "checking_balance1 - 200 DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historyperfect"          "credit_historypoor"             "credit_historyvery good"        "purposecar"                     "purposecar0"                   
    [11] "purposeeducation"               "purposefurniture/appliances"    "purposerenovations"             "amount"                         "savings_balance> 1000 DM"      
    [16] "savings_balance100 - 500 DM"    "savings_balance500 - 1000 DM"   "savings_balanceunknown"         "employment_duration> 7 years"   "employment_duration1 - 4 years"
    [21] "employment_duration4 - 7 years" "employment_durationunemployed"  "percent_of_income"              "years_at_residence"             "age"                           
    [26] "other_creditnone"               "other_creditstore"              "housingown"                     "housingrent"                    "existing_loans_count"          
    [31] "jobskilled"                     "jobunemployed"                  "jobunskilled"                   "dependents"                     "phoneTRUE"                     

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/rf_independent_categories-1.png" width="750px" />

#### Variable Importance

    rf variable importance

      only 20 most important variables shown (out of 35)

     Overall
      67.721
      47.403
      39.892
      35.856
      17.353
      17.345
       8.923
       7.946
       7.858
       7.476
       7.152
       6.923
       6.866
       6.572
       6.348
       6.192
       5.973
       5.786
       5.729
       5.728

### rf\_grouped\_categories

#### Model Summary

                    Length Class      Mode     
    call               4   -none-     call     
    type               1   -none-     character
    predicted        900   factor     numeric  
    err.rate        1500   -none-     numeric  
    confusion          6   -none-     numeric  
    votes           1800   matrix     numeric  
    oob.times        900   -none-     numeric  
    classes            2   -none-     character
    importance        16   -none-     numeric  
    importanceSD       0   -none-     NULL     
    localImportance    0   -none-     NULL     
    proximity          0   -none-     NULL     
    ntree              1   -none-     numeric  
    mtry               1   -none-     numeric  
    forest            14   -none-     list     
    y                900   factor     numeric  
    test               0   -none-     NULL     
    inbag              0   -none-     NULL     
    xNames            16   -none-     character
    problemType        1   -none-     character
    tuneValue          1   data.frame list     
    obsLevels          2   -none-     character
    param              0   -none-     list     

#### Model Predictors

     [1] "checking_balance"     "months_loan_duration" "credit_history"       "purpose"              "amount"               "savings_balance"      "employment_duration" 
     [8] "percent_of_income"    "years_at_residence"   "age"                  "other_credit"         "housing"              "existing_loans_count" "job"                 
    [15] "dependents"           "phone"               

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/rf_grouped_categories-1.png" width="750px" />

#### Variable Importance

    rf variable importance

     Overall
      66.435
      48.143
      46.651
      39.509
      24.959
      24.503
      21.505
      20.702
      17.551
      17.461
      11.133
      10.789
       9.414
       8.494
       5.351
       5.066

### adaboost\_independent\_categories

#### Model Summary

                       Length Class      Mode     
    formula              3    formula    call     
    trees              350    -none-     list     
    weights            350    -none-     numeric  
    classnames           2    -none-     character
    dependent_variable   1    -none-     character
    call                 4    -none-     call     
    xNames              35    -none-     character
    problemType          1    -none-     character
    tuneValue            2    data.frame list     
    obsLevels            2    -none-     character
    param                0    -none-     list     

#### Model Predictors

    Loading required package: fastAdaboost

     [1] "checking_balanceunknown"        "months_loan_duration"           "employment_durationunemployed"  "years_at_residence"             "employment_duration4 - 7 years"
     [6] "percent_of_income"              "amount"                         "credit_historyvery good"        "purposefurniture/appliances"    "purposecar"                    
    [11] "phoneTRUE"                      "jobunskilled"                   "existing_loans_count"           "housingrent"                    "age"                           
    [16] "checking_balance1 - 200 DM"     "other_creditnone"               "savings_balance500 - 1000 DM"   "savings_balanceunknown"         "savings_balance> 1000 DM"      
    [21] "purposecar0"                    "credit_historyperfect"          "dependents"                     "employment_duration> 7 years"   "employment_duration1 - 4 years"
    [26] "savings_balance100 - 500 DM"    "credit_historypoor"             "purposeeducation"               "checking_balance> 200 DM"       "jobskilled"                    
    [31] "credit_historygood"             "purposerenovations"             "housingown"                     "jobunemployed"                  "other_creditstore"             

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/adaboost_independent_categories-1.png" width="750px" />

#### Variable Importance

    ROC curve variable importance

     Importance
         0.6906
         0.6195
         0.6146
         0.5906
         0.5700
         0.5533
         0.5470
         0.5279
         0.5275
         0.5258
         0.5252
         0.5204
         0.5193
         0.5105
         0.5065
         0.5063

### adaboost\_grouped\_categories

#### Model Summary

                       Length Class      Mode     
    formula              3    formula    call     
    trees              250    -none-     list     
    weights            250    -none-     numeric  
    classnames           2    -none-     character
    dependent_variable   1    -none-     character
    call                 4    -none-     call     
    xNames              16    -none-     character
    problemType          1    -none-     character
    tuneValue            2    data.frame list     
    obsLevels            2    -none-     character
    param                0    -none-     list     

#### Model Predictors

     [1] "checking_balance"     "months_loan_duration" "employment_duration"  "savings_balance"      "age"                  "percent_of_income"    "credit_history"      
     [8] "housing"              "amount"               "purpose"              "existing_loans_count" "years_at_residence"   "other_credit"         "job"                 
    [15] "dependents"           "phone"               

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/adaboost_grouped_categories-1.png" width="750px" />

#### Variable Importance

    ROC curve variable importance

     Importance
         0.6906
         0.6195
         0.6146
         0.5906
         0.5700
         0.5533
         0.5470
         0.5279
         0.5275
         0.5258
         0.5252
         0.5204
         0.5193
         0.5105
         0.5065
         0.5063

### adabag\_independent\_categories

> gives errors with current code/data

### adabag\_grouped\_categories

### gbm\_independent\_categories (stochastic gradient boosting)

> if this takes a long time to run, consider tune\_length rather than tune\_grid

#### Model Summary

                      Length Class      Mode     
    initF                1   -none-     numeric  
    fit                900   -none-     numeric  
    train.error       2525   -none-     numeric  
    valid.error       2525   -none-     numeric  
    oobag.improve     2525   -none-     numeric  
    trees             2525   -none-     list     
    c.splits             0   -none-     list     
    bag.fraction         1   -none-     numeric  
    distribution         1   -none-     list     
    interaction.depth    1   -none-     numeric  
    n.minobsinnode       1   -none-     numeric  
    num.classes          1   -none-     numeric  
    n.trees              1   -none-     numeric  
    nTrain               1   -none-     numeric  
    train.fraction       1   -none-     numeric  
    response.name        1   -none-     character
    shrinkage            1   -none-     numeric  
    var.levels          35   -none-     list     
    var.monotone        35   -none-     numeric  
    var.names           35   -none-     character
    var.type            35   -none-     numeric  
    verbose              1   -none-     logical  
    data                 6   -none-     list     
    xNames              35   -none-     character
    problemType          1   -none-     character
    tuneValue            4   data.frame list     
    obsLevels            2   -none-     character
    param                0   -none-     list     

#### Model Predictors

    Loading required package: gbm

    Loading required package: splines

    Loaded gbm 2.1.3

     [1] "checking_balance> 200 DM"       "checking_balance1 - 200 DM"     "checking_balanceunknown"        "months_loan_duration"           "credit_historygood"            
     [6] "credit_historyperfect"          "credit_historypoor"             "credit_historyvery good"        "purposecar"                     "purposecar0"                   
    [11] "purposeeducation"               "purposefurniture/appliances"    "purposerenovations"             "amount"                         "savings_balance> 1000 DM"      
    [16] "savings_balance100 - 500 DM"    "savings_balance500 - 1000 DM"   "savings_balanceunknown"         "employment_duration> 7 years"   "employment_duration1 - 4 years"
    [21] "employment_duration4 - 7 years" "employment_durationunemployed"  "percent_of_income"              "years_at_residence"             "age"                           
    [26] "other_creditnone"               "other_creditstore"              "housingown"                     "housingrent"                    "existing_loans_count"          
    [31] "jobskilled"                     "jobunemployed"                  "jobunskilled"                   "dependents"                     "phoneTRUE"                     

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/gbm_independent_categories-1.png" width="750px" />

#### Variable Importance

    gbm variable importance

      only 20 most important variables shown (out of 35)

     Overall
     1844.15
     1011.63
      854.65
      637.43
      335.40
      267.94
      169.40
      164.26
      163.39
      136.36
      132.41
      128.59
      121.37
      118.36
      116.16
      112.88
      106.02
      100.04
       94.18
       94.15

### gbm\_grouped\_categories (stochastic gradient boosting)

<img src="predictive_analysis_classification_files/figure-markdown_github/gbm_grouped_categories-1.png" width="750px" />

#### Model Summary

                      var    rel.inf
         checking_balance 19.9720164
                   amount 17.7267958
     months_loan_duration 11.0940848
           credit_history 10.1869895
                      age  8.4820503
          savings_balance  6.9920109
                  purpose  5.6413317
      employment_duration  5.5737383
             other_credit  3.8793641
        percent_of_income  3.8173197
                  housing  2.0966105
       years_at_residence  1.5550416
               dependents  1.3723316
                    phone  0.7319289
                      job  0.5807290
     existing_loans_count  0.2976568

#### Model Predictors

     [1] "checking_balance"     "months_loan_duration" "credit_history"       "purpose"              "amount"               "savings_balance"      "employment_duration" 
     [8] "percent_of_income"    "years_at_residence"   "age"                  "other_credit"         "housing"              "existing_loans_count" "job"                 
    [15] "dependents"           "phone"               

#### Model Tuning Grid Performance

<img src="predictive_analysis_classification_files/figure-markdown_github/gbm_grouped_categories-2.png" width="750px" />

#### Variable Importance

    gbm variable importance

     Overall
      72.282
      64.156
      40.151
      36.868
      30.698
      25.305
      20.417
      20.172
      14.040
      13.816
       7.588
       5.628
       4.967
       2.649
       2.102
       1.077

### All Models on Page 550 that are classification or both regression and classification

### Models used for spot-check.Rmd

Resamples & Top Models
======================

Resamples
---------

    ## 
    ## Call:
    ## summary.resamples(object = resamples)
    ## 
    ## Models: glm_no_pre_process, glm_basic_processing, glm_yeojohnson, logistic_regression_stepwise_backward, linear_discriminant_analsysis, linear_discriminant_analsysis_remove_collinear_skew, partial_least_squares_discriminant_analysis, partial_least_squares_discriminant_analysis_skew, glmnet_lasso_ridge, sparse_lda, regularized_discriminant_analysis, mixture_discriminant_analysis, neural_network_spatial_rc, neural_network_spatial_rc_skew, flexible_discriminant_analsysis, svm_linear, svm_polynomial, svm_radial, k_nearest_neighbors, naive_bayes, rpart_independent_categories, rpart_grouped_categories, treebag_independent_categories, treebag_grouped_categories, c50_model_independent_categories, c50_model_grouped_categories, c50_rules_model_independent_categories, c50_rules_model_grouped_categories, rf_independent_categories, rf_grouped_categories, adaboost_independent_categories, adaboost_grouped_categories, gbm_independent_categories, gbm_grouped_categories 
    ## Number of resamples: 30 
    ## 
    ## ROC 
    ##                                                          Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## glm_no_pre_process                                  0.6748971 0.7308936 0.7598471 0.7628258 0.7839506 0.8536155    0
    ## glm_basic_processing                                0.6801881 0.7225162 0.7595532 0.7594748 0.7914462 0.8677249    0
    ## glm_yeojohnson                                      0.6737213 0.7191358 0.7507349 0.7523026 0.7839506 0.8565550    0
    ## logistic_regression_stepwise_backward               0.6878307 0.7264844 0.7757202 0.7644523 0.7962963 0.8641975    0
    ## linear_discriminant_analsysis                       0.6937096 0.7182540 0.7645503 0.7610817 0.7948266 0.8665491    0
    ## linear_discriminant_analsysis_remove_collinear_skew 0.6719577 0.7175191 0.7495591 0.7528709 0.7835097 0.8553792    0
    ## partial_least_squares_discriminant_analysis         0.6978248 0.7260435 0.7592593 0.7627670 0.7979130 0.8689006    0
    ## partial_least_squares_discriminant_analysis_skew    0.6778366 0.7150206 0.7527925 0.7544974 0.7851264 0.8559671    0
    ## glmnet_lasso_ridge                                  0.6760729 0.7244268 0.7583774 0.7628258 0.7974721 0.8589065    0
    ## sparse_lda                                          0.6931217 0.7273663 0.7589653 0.7644327 0.7992357 0.8689006    0
    ## regularized_discriminant_analysis                   0.6737213 0.7341270 0.7604350 0.7652361 0.7943857 0.8424456    0
    ## mixture_discriminant_analysis                       0.6937096 0.7182540 0.7645503 0.7610817 0.7948266 0.8665491    0
    ## neural_network_spatial_rc                           0.6984127 0.7273663 0.7601411 0.7632569 0.7907113 0.8471487    0
    ## neural_network_spatial_rc_skew                      0.6913580 0.7244268 0.7522046 0.7576916 0.7946796 0.8447972    0
    ## flexible_discriminant_analsysis                     0.6543210 0.7176661 0.7475015 0.7490790 0.7824074 0.8289242    0
    ## svm_linear                                          0.6819518 0.7214874 0.7544092 0.7583480 0.7836567 0.8512640    0
    ## svm_polynomial                                      0.6954733 0.7382422 0.7624927 0.7631981 0.7846855 0.8436214    0
    ## svm_radial                                          0.7001764 0.7355967 0.7554380 0.7620811 0.7858613 0.8483245    0
    ## k_nearest_neighbors                                 0.6860670 0.7212669 0.7532334 0.7519106 0.7781452 0.8500882    0
    ## naive_bayes                                         0.6666667 0.7026749 0.7375073 0.7360768 0.7642563 0.8500882    0
    ## rpart_independent_categories                        0.5881834 0.6874633 0.7203116 0.7099157 0.7365520 0.7721928    0
    ## rpart_grouped_categories                            0.6334509 0.6829071 0.7170782 0.7158044 0.7395650 0.8059965    0
    ## treebag_independent_categories                      0.6601999 0.7275867 0.7586714 0.7522438 0.7756467 0.8353909    0
    ## treebag_grouped_categories                          0.6587302 0.7337596 0.7646972 0.7624535 0.7939447 0.8518519    0
    ## c50_model_independent_categories                    0.6578483 0.7408877 0.7604350 0.7616892 0.7838036 0.8447972    0
    ## c50_model_grouped_categories                        0.6813639 0.7316285 0.7533804 0.7564766 0.7758671 0.8442093    0
    ## c50_rules_model_independent_categories              0.5005879 0.6249265 0.6482951 0.6603665 0.7062022 0.7716049    0
    ## c50_rules_model_grouped_categories                  0.5881834 0.6363904 0.6651969 0.6643935 0.6912845 0.7513228    0
    ## rf_independent_categories                           0.6751911 0.7414021 0.7723398 0.7647952 0.7860817 0.8665491    0
    ## rf_grouped_categories                               0.7101705 0.7544827 0.7830688 0.7819910 0.8084215 0.8962375    0
    ## adaboost_independent_categories                     0.6637272 0.7181070 0.7480894 0.7477366 0.7799824 0.8477366    0
    ## adaboost_grouped_categories                         0.6931217 0.7244268 0.7583774 0.7572996 0.7890947 0.8835979    0
    ## gbm_independent_categories                          0.6807760 0.7408877 0.7771899 0.7688615 0.7840976 0.8594944    0
    ## gbm_grouped_categories                              0.7166373 0.7504409 0.7698413 0.7759945 0.8007055 0.8694885    0
    ## 
    ## Sens 
    ##                                                           Min.   1st Qu.    Median        Mean   3rd Qu.       Max. NA's
    ## glm_no_pre_process                                  0.25925926 0.4074074 0.4814815 0.453086420 0.5185185 0.55555556    0
    ## glm_basic_processing                                0.22222222 0.3333333 0.4074074 0.411111111 0.4814815 0.62962963    0
    ## glm_yeojohnson                                      0.25925926 0.3055556 0.3888889 0.397530864 0.4722222 0.66666667    0
    ## logistic_regression_stepwise_backward               0.22222222 0.3796296 0.4259259 0.411111111 0.4444444 0.55555556    0
    ## linear_discriminant_analsysis                       0.25925926 0.3703704 0.4444444 0.424691358 0.4814815 0.59259259    0
    ## linear_discriminant_analsysis_remove_collinear_skew 0.22222222 0.3703704 0.4444444 0.425925926 0.4814815 0.62962963    0
    ## partial_least_squares_discriminant_analysis         0.22222222 0.3333333 0.4074074 0.388888889 0.4722222 0.51851852    0
    ## partial_least_squares_discriminant_analysis_skew    0.22222222 0.3333333 0.3703704 0.395061728 0.4814815 0.59259259    0
    ## glmnet_lasso_ridge                                  0.18518519 0.2222222 0.2777778 0.287654321 0.3333333 0.48148148    0
    ## sparse_lda                                          0.25925926 0.3425926 0.4444444 0.422222222 0.4814815 0.62962963    0
    ## regularized_discriminant_analysis                   0.33333333 0.4444444 0.5000000 0.496296296 0.5555556 0.62962963    0
    ## mixture_discriminant_analysis                       0.25925926 0.3703704 0.4444444 0.425925926 0.4814815 0.59259259    0
    ## neural_network_spatial_rc                           0.07407407 0.1851852 0.2592593 0.251851852 0.3240741 0.44444444    0
    ## neural_network_spatial_rc_skew                      0.07407407 0.1851852 0.2592593 0.249382716 0.3240741 0.48148148    0
    ## flexible_discriminant_analsysis                     0.22222222 0.3055556 0.3703704 0.398765432 0.4814815 0.55555556    0
    ## svm_linear                                          0.25925926 0.3703704 0.4074074 0.414814815 0.4814815 0.51851852    0
    ## svm_polynomial                                      0.14814815 0.2962963 0.3333333 0.322222222 0.3703704 0.48148148    0
    ## svm_radial                                          0.14814815 0.2592593 0.3333333 0.317283951 0.3703704 0.51851852    0
    ## k_nearest_neighbors                                 0.00000000 0.0000000 0.0000000 0.009876543 0.0000000 0.07407407    0
    ## naive_bayes                                         0.00000000 0.0000000 0.0000000 0.001234568 0.0000000 0.03703704    0
    ## rpart_independent_categories                        0.14814815 0.2592593 0.3333333 0.334567901 0.4074074 0.51851852    0
    ## rpart_grouped_categories                            0.25925926 0.3333333 0.4074074 0.401234568 0.4444444 0.55555556    0
    ## treebag_independent_categories                      0.33333333 0.4166667 0.4814815 0.470370370 0.5185185 0.62962963    0
    ## treebag_grouped_categories                          0.29629630 0.4074074 0.4444444 0.474074074 0.5462963 0.66666667    0
    ## c50_model_independent_categories                    0.33333333 0.4074074 0.4444444 0.462962963 0.5185185 0.66666667    0
    ## c50_model_grouped_categories                        0.29629630 0.4074074 0.4629630 0.462962963 0.5185185 0.66666667    0
    ## c50_rules_model_independent_categories              0.22222222 0.4074074 0.4259259 0.435802469 0.5092593 0.59259259    0
    ## c50_rules_model_grouped_categories                  0.18518519 0.3333333 0.3518519 0.385185185 0.4444444 0.62962963    0
    ## rf_independent_categories                           0.25925926 0.3703704 0.4259259 0.422222222 0.4814815 0.55555556    0
    ## rf_grouped_categories                               0.25925926 0.3796296 0.4444444 0.439506173 0.4814815 0.55555556    0
    ## adaboost_independent_categories                     0.22222222 0.4074074 0.4444444 0.454320988 0.5185185 0.62962963    0
    ## adaboost_grouped_categories                         0.22222222 0.4074074 0.4444444 0.450617284 0.4814815 0.59259259    0
    ## gbm_independent_categories                          0.29629630 0.4166667 0.4814815 0.477777778 0.5462963 0.62962963    0
    ## gbm_grouped_categories                              0.18518519 0.3703704 0.4259259 0.418518519 0.4814815 0.55555556    0
    ## 
    ## Spec 
    ##                                                          Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## glm_no_pre_process                                  0.7460317 0.8253968 0.8730159 0.8592593 0.8888889 0.9523810    0
    ## glm_basic_processing                                0.7619048 0.8253968 0.8650794 0.8613757 0.8888889 0.9523810    0
    ## glm_yeojohnson                                      0.7777778 0.8253968 0.8571429 0.8571429 0.8888889 0.9523810    0
    ## logistic_regression_stepwise_backward               0.7936508 0.8571429 0.8730159 0.8772487 0.9047619 0.9523810    0
    ## linear_discriminant_analsysis                       0.7619048 0.8293651 0.8571429 0.8576720 0.8888889 0.9523810    0
    ## linear_discriminant_analsysis_remove_collinear_skew 0.7936508 0.8095238 0.8412698 0.8513228 0.8888889 0.9523810    0
    ## partial_least_squares_discriminant_analysis         0.7936508 0.8412698 0.8730159 0.8756614 0.9047619 0.9523810    0
    ## partial_least_squares_discriminant_analysis_skew    0.8095238 0.8293651 0.8730159 0.8730159 0.9047619 0.9523810    0
    ## glmnet_lasso_ridge                                  0.8412698 0.8928571 0.9206349 0.9137566 0.9365079 0.9682540    0
    ## sparse_lda                                          0.7619048 0.8253968 0.8571429 0.8592593 0.8849206 0.9523810    0
    ## regularized_discriminant_analysis                   0.7301587 0.8095238 0.8492063 0.8375661 0.8690476 0.9206349    0
    ## mixture_discriminant_analysis                       0.7619048 0.8293651 0.8571429 0.8576720 0.8888889 0.9523810    0
    ## neural_network_spatial_rc                           0.8730159 0.9047619 0.9365079 0.9312169 0.9523810 0.9841270    0
    ## neural_network_spatial_rc_skew                      0.8571429 0.9047619 0.9285714 0.9264550 0.9523810 0.9841270    0
    ## flexible_discriminant_analsysis                     0.7936508 0.8253968 0.8730159 0.8608466 0.8888889 0.9206349    0
    ## svm_linear                                          0.7777778 0.8571429 0.8730159 0.8746032 0.9047619 0.9523810    0
    ## svm_polynomial                                      0.8253968 0.8888889 0.9206349 0.9089947 0.9365079 0.9682540    0
    ## svm_radial                                          0.8095238 0.8888889 0.9047619 0.9031746 0.9325397 0.9523810    0
    ## k_nearest_neighbors                                 0.9841270 1.0000000 1.0000000 0.9994709 1.0000000 1.0000000    0
    ## naive_bayes                                         0.9841270 1.0000000 1.0000000 0.9984127 1.0000000 1.0000000    0
    ## rpart_independent_categories                        0.7777778 0.8412698 0.8809524 0.8730159 0.9047619 0.9682540    0
    ## rpart_grouped_categories                            0.7460317 0.8253968 0.8492063 0.8513228 0.8888889 0.9365079    0
    ## treebag_independent_categories                      0.7301587 0.8015873 0.8412698 0.8465608 0.8888889 0.9523810    0
    ## treebag_grouped_categories                          0.7777778 0.8253968 0.8571429 0.8523810 0.8888889 0.9365079    0
    ## c50_model_independent_categories                    0.7301587 0.8293651 0.8571429 0.8550265 0.8888889 0.9365079    0
    ## c50_model_grouped_categories                        0.7460317 0.8571429 0.8730159 0.8714286 0.9007937 0.9523810    0
    ## c50_rules_model_independent_categories              0.7301587 0.8134921 0.8253968 0.8354497 0.8690476 0.9047619    0
    ## c50_rules_model_grouped_categories                  0.7142857 0.8412698 0.8730159 0.8708995 0.9007937 0.9523810    0
    ## rf_independent_categories                           0.8253968 0.8571429 0.8888889 0.8883598 0.9206349 0.9523810    0
    ## rf_grouped_categories                               0.8253968 0.8730159 0.8968254 0.8968254 0.9325397 0.9523810    0
    ## adaboost_independent_categories                     0.7619048 0.8412698 0.8492063 0.8592593 0.8888889 0.9523810    0
    ## adaboost_grouped_categories                         0.7777778 0.8412698 0.8571429 0.8682540 0.8888889 0.9682540    0
    ## gbm_independent_categories                          0.7936508 0.8253968 0.8571429 0.8661376 0.9047619 0.9682540    0
    ## gbm_grouped_categories                              0.8253968 0.8571429 0.8968254 0.8936508 0.9206349 0.9682540    0

<img src="predictive_analysis_classification_files/figure-markdown_github/resamples_regression-1.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/resamples_regression-2.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/resamples_regression-3.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/resamples_regression-4.png" width="750px" />

Train Top Models on Entire Training Dataset & Predict on Test Set
-----------------------------------------------------------------

> after using cross-validation to tune, we will take the highest ranked models, retrain the models (with the final tuning parameters) on the entire training set, and predict using the test set.

### Random Forest (rf\_grouped\_categories)

> Model Processing: \` \`

                    Length Class      Mode     
    call               4   -none-     call     
    type               1   -none-     character
    predicted        900   factor     numeric  
    err.rate        1500   -none-     numeric  
    confusion          6   -none-     numeric  
    votes           1800   matrix     numeric  
    oob.times        900   -none-     numeric  
    classes            2   -none-     character
    importance        16   -none-     numeric  
    importanceSD       0   -none-     NULL     
    localImportance    0   -none-     NULL     
    proximity          0   -none-     NULL     
    ntree              1   -none-     numeric  
    mtry               1   -none-     numeric  
    forest            14   -none-     list     
    y                900   factor     numeric  
    test               0   -none-     NULL     
    inbag              0   -none-     NULL     
    xNames            16   -none-     character
    problemType        1   -none-     character
    tuneValue          1   data.frame list     
    obsLevels          2   -none-     character
    param              0   -none-     list     

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-1.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-2.png" width="750px" />

    Confusion Matrix and Statistics

              Reference
    Prediction yes no
           yes  18  4
           no   12 66
                                              
                   Accuracy : 0.84            
                     95% CI : (0.7532, 0.9057)
        No Information Rate : 0.7             
        P-Value [Acc > NIR] : 0.0009689       
                                              
                      Kappa : 0.5876          
     Mcnemar's Test P-Value : 0.0801183       
                                              
                Sensitivity : 0.6000          
                Specificity : 0.9429          
             Pos Pred Value : 0.8182          
             Neg Pred Value : 0.8462          
                 Prevalence : 0.3000          
             Detection Rate : 0.1800          
       Detection Prevalence : 0.2200          
          Balanced Accuracy : 0.7714          
                                              
           'Positive' Class : yes             
                                              

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-3.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-4.png" width="750px" />NULL

      threshold specificity sensitivity 
      0.4890000   0.9142857   0.6333333 

      threshold specificity sensitivity 
      0.4890000   0.9142857   0.6333333 

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-5.png" width="750px" />

NOTE: when tuning final model, alternative cutoff points should be determined using an evaluation dataset (APL pg 425).

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-6.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-7.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-8.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-9.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-10.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-11.png" width="750px" />

### Stochastic Gradient Boosting (gbm\_grouped\_categories)

> Model Processing: \` \`

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-12.png" width="750px" />`var    rel.inf      checking_balance 20.2991981                amount 14.7668276  months_loan_duration 12.9758115        credit_history  9.1126882   employment_duration  7.7445948       savings_balance  7.1323685                   age  6.9311927               purpose  5.4043114          other_credit  3.6950532     percent_of_income  3.0209027               housing  2.8956110    years_at_residence  1.9670349            dependents  1.4892067                 phone  0.9052488  existing_loans_count  0.8825967                   job  0.7773533`

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-13.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-14.png" width="750px" />

    Confusion Matrix and Statistics

              Reference
    Prediction yes no
           yes  16  8
           no   14 62
                                              
                   Accuracy : 0.78            
                     95% CI : (0.6861, 0.8567)
        No Information Rate : 0.7             
        P-Value [Acc > NIR] : 0.04787         
                                              
                      Kappa : 0.4444          
     Mcnemar's Test P-Value : 0.28642         
                                              
                Sensitivity : 0.5333          
                Specificity : 0.8857          
             Pos Pred Value : 0.6667          
             Neg Pred Value : 0.8158          
                 Prevalence : 0.3000          
             Detection Rate : 0.1600          
       Detection Prevalence : 0.2400          
          Balanced Accuracy : 0.7095          
                                              
           'Positive' Class : yes             
                                              

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-15.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-16.png" width="750px" />NULL

      threshold specificity sensitivity 
      0.2810120   0.7142857   0.7666667 

      threshold specificity sensitivity 
      0.2810120   0.7142857   0.7666667 

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-17.png" width="750px" />

NOTE: when tuning final model, alternative cutoff points should be determined using an evaluation dataset (APL pg 425).

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-18.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-19.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-20.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-21.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-22.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-23.png" width="750px" />

### Stochastic Gradient Boosting (gbm\_independent\_categories)

> Model Processing: \` \`

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-24.png" width="750px" />`var     rel.inf                          amount 24.50924462                             age 13.99074860            months_loan_duration 11.48396109         checking_balanceunknown  8.46060673               percent_of_income  4.21441883              years_at_residence  3.51606589          savings_balanceunknown  2.35678482         credit_historyvery good  2.05957224                other_creditnone  1.87888148           credit_historyperfect  1.75432501                       phoneTRUE  1.67061557                      housingown  1.62837285  employment_duration4 - 7 years  1.59821125            existing_loans_count  1.57737056        checking_balance> 200 DM  1.57583990     purposefurniture/appliances  1.50093489      checking_balance1 - 200 DM  1.39118928              credit_historypoor  1.36232219   employment_durationunemployed  1.35822574              credit_historygood  1.28202896  employment_duration1 - 4 years  1.24699943                      purposecar  1.06998968                purposeeducation  0.98712940                      dependents  0.97553656                     housingrent  0.91442823    employment_duration> 7 years  0.89525717               other_creditstore  0.82081094                    jobunskilled  0.76224815              purposerenovations  0.71093790                      jobskilled  0.70136546     savings_balance100 - 500 DM  0.63422718        savings_balance> 1000 DM  0.56647347    savings_balance500 - 1000 DM  0.34235700                     purposecar0  0.13406462                   jobunemployed  0.06845431`

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-25.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-26.png" width="750px" />

    Confusion Matrix and Statistics

              Reference
    Prediction yes no
           yes  14 10
           no   16 60
                                              
                   Accuracy : 0.74            
                     95% CI : (0.6427, 0.8226)
        No Information Rate : 0.7             
        P-Value [Acc > NIR] : 0.2244          
                                              
                      Kappa : 0.3434          
     Mcnemar's Test P-Value : 0.3268          
                                              
                Sensitivity : 0.4667          
                Specificity : 0.8571          
             Pos Pred Value : 0.5833          
             Neg Pred Value : 0.7895          
                 Prevalence : 0.3000          
             Detection Rate : 0.1400          
       Detection Prevalence : 0.2400          
          Balanced Accuracy : 0.6619          
                                              
           'Positive' Class : yes             
                                              

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-27.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-28.png" width="750px" />NULL

      threshold specificity sensitivity 
      0.2470993   0.6857143   0.7666667 

      threshold specificity sensitivity 
      0.2470993   0.6857143   0.7666667 

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-29.png" width="750px" />

NOTE: when tuning final model, alternative cutoff points should be determined using an evaluation dataset (APL pg 425).

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-30.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-31.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-32.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-33.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-34.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-35.png" width="750px" />

### Regularized Discriminant Analysis (regularized\_discriminant\_analysis)

> Model Processing: `nzv; center; scale`

                   Length Class      Mode     
    call              5   -none-     call     
    regularization    2   -none-     numeric  
    classes           2   -none-     character
    prior             2   -none-     numeric  
    error.rate        1   -none-     numeric  
    varnames         29   -none-     character
    means            58   -none-     numeric  
    covariances    1682   -none-     numeric  
    covpooled       841   -none-     numeric  
    converged         1   -none-     logical  
    iter              1   -none-     numeric  
    xNames           29   -none-     character
    problemType       1   -none-     character
    tuneValue         2   data.frame list     
    obsLevels         2   -none-     character
    param             0   -none-     list     

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-36.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-37.png" width="750px" />

    Confusion Matrix and Statistics

              Reference
    Prediction yes no
           yes  16 10
           no   14 60
                                              
                   Accuracy : 0.76            
                     95% CI : (0.6643, 0.8398)
        No Information Rate : 0.7             
        P-Value [Acc > NIR] : 0.1136          
                                              
                      Kappa : 0.4059          
     Mcnemar's Test P-Value : 0.5403          
                                              
                Sensitivity : 0.5333          
                Specificity : 0.8571          
             Pos Pred Value : 0.6154          
             Neg Pred Value : 0.8108          
                 Prevalence : 0.3000          
             Detection Rate : 0.1600          
       Detection Prevalence : 0.2600          
          Balanced Accuracy : 0.6952          
                                              
           'Positive' Class : yes             
                                              

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-38.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-39.png" width="750px" />NULL

      threshold specificity sensitivity 
      0.2707901   0.6714286   0.7333333 

      threshold specificity sensitivity 
      0.2482568   0.6428571   0.7666667 

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-40.png" width="750px" />

NOTE: when tuning final model, alternative cutoff points should be determined using an evaluation dataset (APL pg 425).

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-41.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-42.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-43.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-44.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-45.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-46.png" width="750px" />

### Random Forest (rf\_independent\_categories)

> Model Processing: \` \`

                    Length Class      Mode     
    call               4   -none-     call     
    type               1   -none-     character
    predicted        900   factor     numeric  
    err.rate        1500   -none-     numeric  
    confusion          6   -none-     numeric  
    votes           1800   matrix     numeric  
    oob.times        900   -none-     numeric  
    classes            2   -none-     character
    importance        35   -none-     numeric  
    importanceSD       0   -none-     NULL     
    localImportance    0   -none-     NULL     
    proximity          0   -none-     NULL     
    ntree              1   -none-     numeric  
    mtry               1   -none-     numeric  
    forest            14   -none-     list     
    y                900   factor     numeric  
    test               0   -none-     NULL     
    inbag              0   -none-     NULL     
    xNames            35   -none-     character
    problemType        1   -none-     character
    tuneValue          1   data.frame list     
    obsLevels          2   -none-     character
    param              0   -none-     list     

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-47.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-48.png" width="750px" />

    Confusion Matrix and Statistics

              Reference
    Prediction yes no
           yes  16  5
           no   14 65
                                              
                   Accuracy : 0.81            
                     95% CI : (0.7193, 0.8816)
        No Information Rate : 0.7             
        P-Value [Acc > NIR] : 0.008887        
                                              
                      Kappa : 0.5052          
     Mcnemar's Test P-Value : 0.066457        
                                              
                Sensitivity : 0.5333          
                Specificity : 0.9286          
             Pos Pred Value : 0.7619          
             Neg Pred Value : 0.8228          
                 Prevalence : 0.3000          
             Detection Rate : 0.1600          
       Detection Prevalence : 0.2100          
          Balanced Accuracy : 0.7310          
                                              
           'Positive' Class : yes             
                                              

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-49.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-50.png" width="750px" />NULL

      threshold specificity sensitivity 
          0.372       0.800       0.700 

      threshold specificity sensitivity 
          0.372       0.800       0.700 

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-51.png" width="750px" />

NOTE: when tuning final model, alternative cutoff points should be determined using an evaluation dataset (APL pg 425).

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-52.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-53.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-54.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-55.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-56.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-57.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-58.png" width="750px" />
