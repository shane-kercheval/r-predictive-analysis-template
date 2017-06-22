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
        -   [Random Forest](#random-forest)
        -   [Neural Network](#neural-network)
        -   [Ada Boost](#ada-boost)
        -   [All Models on Page 550 that are classification or both regression and classification](#all-models-on-page-550-that-are-classification-or-both-regression-and-classification)
        -   [Models used for spot-check.Rmd](#models-used-for-spot-check.rmd)
    -   [Resamples & Top Models](#resamples-top-models)
    -   [Resamples](#resamples)
    -   [Train Top Models on Entire Training Dataset & Predict on Test Set](#train-top-models-on-entire-training-dataset-predict-on-test-set)
        -   [Regularized Discriminant Analysis (regularized\_discriminant\_analysis)](#regularized-discriminant-analysis-regularized_discriminant_analysis)
        -   [Generalized Linear Model (logistic\_regression\_stepwise\_backward)](#generalized-linear-model-logistic_regression_stepwise_backward)
        -   [Sparse Linear Discriminant Analysis (sparse\_lda)](#sparse-linear-discriminant-analysis-sparse_lda)
        -   [glmnet (glmnet\_lasso\_ridge)](#glmnet-glmnet_lasso_ridge)
        -   [Generalized Linear Model (glm\_no\_pre\_process)](#generalized-linear-model-glm_no_pre_process)
        -   [Partial Least Squares (partial\_least\_squares\_discriminant\_analysis)](#partial-least-squares-partial_least_squares_discriminant_analysis)

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

#### TURN BACK ON

### Random Forest

### Neural Network

### Ada Boost

### All Models on Page 550 that are classification or both regression and classification

### Models used for spot-check.Rmd

Resamples & Top Models
----------------------

Resamples
---------

    ## 
    ## Call:
    ## summary.resamples(object = resamples)
    ## 
    ## Models: glm_no_pre_process, glm_basic_processing, glm_yeojohnson, logistic_regression_stepwise_backward, linear_discriminant_analsysis, linear_discriminant_analsysis_remove_collinear_skew, partial_least_squares_discriminant_analysis, partial_least_squares_discriminant_analysis_skew, glmnet_lasso_ridge, sparse_lda, regularized_discriminant_analysis, mixture_discriminant_analysis 
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
    ## 
    ## Sens 
    ##                                                          Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## glm_no_pre_process                                  0.2592593 0.4074074 0.4814815 0.4530864 0.5185185 0.5555556    0
    ## glm_basic_processing                                0.2222222 0.3333333 0.4074074 0.4111111 0.4814815 0.6296296    0
    ## glm_yeojohnson                                      0.2592593 0.3055556 0.3888889 0.3975309 0.4722222 0.6666667    0
    ## logistic_regression_stepwise_backward               0.2222222 0.3796296 0.4259259 0.4111111 0.4444444 0.5555556    0
    ## linear_discriminant_analsysis                       0.2592593 0.3703704 0.4444444 0.4246914 0.4814815 0.5925926    0
    ## linear_discriminant_analsysis_remove_collinear_skew 0.2222222 0.3703704 0.4444444 0.4259259 0.4814815 0.6296296    0
    ## partial_least_squares_discriminant_analysis         0.2222222 0.3333333 0.4074074 0.3888889 0.4722222 0.5185185    0
    ## partial_least_squares_discriminant_analysis_skew    0.2222222 0.3333333 0.3703704 0.3950617 0.4814815 0.5925926    0
    ## glmnet_lasso_ridge                                  0.1851852 0.2222222 0.2777778 0.2876543 0.3333333 0.4814815    0
    ## sparse_lda                                          0.2592593 0.3425926 0.4444444 0.4222222 0.4814815 0.6296296    0
    ## regularized_discriminant_analysis                   0.3333333 0.4444444 0.5000000 0.4962963 0.5555556 0.6296296    0
    ## mixture_discriminant_analysis                       0.2592593 0.3703704 0.4444444 0.4259259 0.4814815 0.5925926    0
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

<img src="predictive_analysis_classification_files/figure-markdown_github/resamples_regression-1.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/resamples_regression-2.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/resamples_regression-3.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/resamples_regression-4.png" width="750px" />

Train Top Models on Entire Training Dataset & Predict on Test Set
-----------------------------------------------------------------

> after using cross-validation to tune, we'll take the highest ranked models, retrain the models (with the final tuning parameters) on the entire training set, and predict using the test set.

### Regularized Discriminant Analysis (regularized\_discriminant\_analysis)

> Model Processing: nzv; center; scale

> Model Formula: `target ~ checking_balance + months_loan_duration + credit_history + purpose + amount + savings_balance + employment_duration + percent_of_income + years_at_residence + age + other_credit + housing + existing_loans_count + job + dependents + phone`

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

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-1.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-2.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-3.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-4.png" width="750px" />

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
                                              

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-5.png" width="750px" />NULL <img src="predictive_analysis_classification_files/figure-markdown_github/top_models-6.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-7.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-8.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-9.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-10.png" width="750px" />

### Generalized Linear Model (logistic\_regression\_stepwise\_backward)

> Model Processing: nzv; center; scale

> Model Formula: `target ~ checking_balance + credit_history + savings_balance + employment_duration + other_credit + housing + phone + months_loan_duration + amount + percent_of_income + existing_loans_count`


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

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-11.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-12.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-13.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-14.png" width="750px" />

    Confusion Matrix and Statistics

              Reference
    Prediction yes no
           yes  14  9
           no   16 61
                                              
                   Accuracy : 0.75            
                     95% CI : (0.6534, 0.8312)
        No Information Rate : 0.7             
        P-Value [Acc > NIR] : 0.1631          
                                              
                      Kappa : 0.3622          
     Mcnemar's Test P-Value : 0.2301          
                                              
                Sensitivity : 0.4667          
                Specificity : 0.8714          
             Pos Pred Value : 0.6087          
             Neg Pred Value : 0.7922          
                 Prevalence : 0.3000          
             Detection Rate : 0.1400          
       Detection Prevalence : 0.2300          
          Balanced Accuracy : 0.6690          
                                              
           'Positive' Class : yes             
                                              

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-15.png" width="750px" />NULL <img src="predictive_analysis_classification_files/figure-markdown_github/top_models-16.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-17.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-18.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-19.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-20.png" width="750px" />

### Sparse Linear Discriminant Analysis (sparse\_lda)

> Model Processing: nzv; center; scale

> Model Formula: `target ~ checking_balance + months_loan_duration + credit_history + purpose + amount + savings_balance + employment_duration + percent_of_income + years_at_residence + age + other_credit + housing + existing_loans_count + job + dependents + phone`

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

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-21.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-22.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-23.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-24.png" width="750px" />

    Confusion Matrix and Statistics

              Reference
    Prediction yes no
           yes  14  9
           no   16 61
                                              
                   Accuracy : 0.75            
                     95% CI : (0.6534, 0.8312)
        No Information Rate : 0.7             
        P-Value [Acc > NIR] : 0.1631          
                                              
                      Kappa : 0.3622          
     Mcnemar's Test P-Value : 0.2301          
                                              
                Sensitivity : 0.4667          
                Specificity : 0.8714          
             Pos Pred Value : 0.6087          
             Neg Pred Value : 0.7922          
                 Prevalence : 0.3000          
             Detection Rate : 0.1400          
       Detection Prevalence : 0.2300          
          Balanced Accuracy : 0.6690          
                                              
           'Positive' Class : yes             
                                              

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-25.png" width="750px" />NULL <img src="predictive_analysis_classification_files/figure-markdown_github/top_models-26.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-27.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-28.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-29.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-30.png" width="750px" />

### glmnet (glmnet\_lasso\_ridge)

> Model Processing: nzv; center; scale

> Model Formula: `target ~ checking_balance + months_loan_duration + credit_history + purpose + amount + savings_balance + employment_duration + percent_of_income + years_at_residence + age + other_credit + housing + existing_loans_count + job + dependents + phone`

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

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-31.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-32.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-33.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-34.png" width="750px" />

    Confusion Matrix and Statistics

              Reference
    Prediction yes no
           yes  12  5
           no   18 65
                                              
                   Accuracy : 0.77            
                     95% CI : (0.6751, 0.8483)
        No Information Rate : 0.7             
        P-Value [Acc > NIR] : 0.07553         
                                              
                      Kappa : 0.375           
     Mcnemar's Test P-Value : 0.01234         
                                              
                Sensitivity : 0.4000          
                Specificity : 0.9286          
             Pos Pred Value : 0.7059          
             Neg Pred Value : 0.7831          
                 Prevalence : 0.3000          
             Detection Rate : 0.1200          
       Detection Prevalence : 0.1700          
          Balanced Accuracy : 0.6643          
                                              
           'Positive' Class : yes             
                                              

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-35.png" width="750px" />NULL <img src="predictive_analysis_classification_files/figure-markdown_github/top_models-36.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-37.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-38.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-39.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-40.png" width="750px" />

### Generalized Linear Model (glm\_no\_pre\_process)

> Model Processing: NA

> Model Formula: `target ~ checking_balance + months_loan_duration + credit_history + purpose + amount + savings_balance + employment_duration + percent_of_income + years_at_residence + age + other_credit + housing + existing_loans_count + job + dependents + phone`


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

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-41.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-42.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-43.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-44.png" width="750px" />

    Confusion Matrix and Statistics

              Reference
    Prediction yes no
           yes  14  9
           no   16 61
                                              
                   Accuracy : 0.75            
                     95% CI : (0.6534, 0.8312)
        No Information Rate : 0.7             
        P-Value [Acc > NIR] : 0.1631          
                                              
                      Kappa : 0.3622          
     Mcnemar's Test P-Value : 0.2301          
                                              
                Sensitivity : 0.4667          
                Specificity : 0.8714          
             Pos Pred Value : 0.6087          
             Neg Pred Value : 0.7922          
                 Prevalence : 0.3000          
             Detection Rate : 0.1400          
       Detection Prevalence : 0.2300          
          Balanced Accuracy : 0.6690          
                                              
           'Positive' Class : yes             
                                              

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-45.png" width="750px" />NULL <img src="predictive_analysis_classification_files/figure-markdown_github/top_models-46.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-47.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-48.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-49.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-50.png" width="750px" />

### Partial Least Squares (partial\_least\_squares\_discriminant\_analysis)

> Model Processing: nzv; center; scale

> Model Formula: `target ~ checking_balance + months_loan_duration + credit_history + purpose + amount + savings_balance + employment_duration + percent_of_income + years_at_residence + age + other_credit + housing + existing_loans_count + job + dependents + phone`

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-51.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-52.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-53.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-54.png" width="750px" />

    Confusion Matrix and Statistics

              Reference
    Prediction yes no
           yes  13  6
           no   17 64
                                              
                   Accuracy : 0.77            
                     95% CI : (0.6751, 0.8483)
        No Information Rate : 0.7             
        P-Value [Acc > NIR] : 0.07553         
                                              
                      Kappa : 0.3883          
     Mcnemar's Test P-Value : 0.03706         
                                              
                Sensitivity : 0.4333          
                Specificity : 0.9143          
             Pos Pred Value : 0.6842          
             Neg Pred Value : 0.7901          
                 Prevalence : 0.3000          
             Detection Rate : 0.1300          
       Detection Prevalence : 0.1900          
          Balanced Accuracy : 0.6738          
                                              
           'Positive' Class : yes             
                                              

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-55.png" width="750px" />NULL <img src="predictive_analysis_classification_files/figure-markdown_github/top_models-56.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-57.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-58.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-59.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-60.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-61.png" width="750px" />
