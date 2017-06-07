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
    -   [Class Balance](#class-balance)
        -   [Training Data](#training-data)
        -   [Test](#test)
        -   [Logistic Regression - no pre processing](#logistic-regression---no-pre-processing)
        -   [Logistic Regression - basic pre-processing](#logistic-regression---basic-pre-processing)
        -   [Logistic Regression - remove collinear data - based on caret's recommendation](#logistic-regression---remove-collinear-data---based-on-carets-recommendation)
        -   [Logistic Regression - remove collinear data - based on calculation](#logistic-regression---remove-collinear-data---based-on-calculation)
        -   [Logistic Regression - remove collinear data - based on calculation](#logistic-regression---remove-collinear-data---based-on-calculation-1)
        -   [Linear Discriminant Analysis](#linear-discriminant-analysis)
        -   [Linear Discriminant Analysis - Remove Collinear Data Based on Caret's Recommendation](#linear-discriminant-analysis---remove-collinear-data-based-on-carets-recommendation)
        -   [Partial Least Squares Discriminant Analysis (PLSDA)](#partial-least-squares-discriminant-analysis-plsda)
        -   [glmnet (LASSO and RIDGE)](#glmnet-lasso-and-ridge)
        -   [Sparse LDA](#sparse-lda)
        -   [Nearest Shrunken Centroids](#nearest-shrunken-centroids)
        -   [Random Forest](#random-forest)
        -   [Neural Network](#neural-network)
        -   [Ada Boost](#ada-boost)
        -   [All Models on Page 550 that are classification or both regression and classification](#all-models-on-page-550-that-are-classification-or-both-regression-and-classification)
        -   [Models used for spot-check.Rmd](#models-used-for-spot-check.rmd)
    -   [Resamples & Top Models](#resamples-top-models)
        -   [Resamples](#resamples)
-   [Train Top Models on Entire Training Dataset & Predict on Test Set](#train-top-models-on-entire-training-dataset-predict-on-test-set)

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

tuning_nearest_shrunken_centroids_shrinkage_threshold <- data.frame(.threshold = 0:25)
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

Class Balance
-------------

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

### Logistic Regression - no pre processing

> NOTE that for logistic regression (GLM), caret's `train()` (because of `glm()`) uses the second-level factor value as the success/postive event but `resamples()` uses the first-level as the success event. The result is either the `sensitivity` and `specificity` for `resamples()` will be reversed (and so I would be unable to compare apples to apples with other models), or I need to keep the first-level factor as the positive event (the default approach), which will mean that THE COEFFICIENTS WILL BE REVERSED, MAKIN THE MODEL RELATIVE TO THE NEGATIVE EVENT. I chose the latter, in order to compare models below, but this means that when using the logistic model to explain the data, the reader needs to mentally reverse the direction/sign of the coefficients, or correct the problem in the final stages of model building.

``` r
if(refresh_models)
{
    check_data(classification_train, validate_n_p = TRUE)
    set.seed(custom_seed)
    #model_glm_no_pre_processing <- train(target ~ ., data=classification_train %>% mutate(target = factor(target, levels = c('no', 'yes'))), method='glm', metric=metric, trControl=train_control)
    model_glm_no_pre_processing <- train(target ~ ., data=classification_train, method='glm', metric=metric, trControl=train_control)
    saveRDS(model_glm_no_pre_processing, file = './classification_data/model_glm_no_pre_processing.RDS')
} else{
    model_glm_no_pre_processing <- readRDS('./classification_data/model_glm_no_pre_processing.RDS')
}
summary(model_glm_no_pre_processing)
```

    ## 
    ## Call:
    ## NULL
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.6364  -0.7956   0.4122   0.7664   1.8864  
    ## 
    ## Coefficients:
    ##                                     Estimate  Std. Error z value          Pr(>|z|)    
    ## (Intercept)                       1.67233497  0.93888105   1.781          0.074880 .  
    ## `checking_balance> 200 DM`        0.91202032  0.36959975   2.468          0.013603 *  
    ## `checking_balance1 - 200 DM`      0.36146669  0.21675128   1.668          0.095384 .  
    ## checking_balanceunknown           1.69919542  0.23318347   7.287 0.000000000000317 ***
    ## months_loan_duration             -0.01909034  0.00930934  -2.051          0.040300 *  
    ## credit_historygood               -0.83854132  0.26057700  -3.218          0.001291 ** 
    ## credit_historyperfect            -1.18300647  0.42783664  -2.765          0.005691 ** 
    ## credit_historypoor               -0.70792254  0.34560901  -2.048          0.040527 *  
    ## `credit_historyvery good`        -1.43539610  0.42810208  -3.353          0.000800 ***
    ## purposecar                       -0.14052327  0.32598325  -0.431          0.666414    
    ## purposecar0                       0.63233919  0.81457907   0.776          0.437585    
    ## purposeeducation                 -0.58632030  0.43971583  -1.333          0.182398    
    ## `purposefurniture/appliances`     0.16610147  0.31881865   0.521          0.602373    
    ## purposerenovations               -0.68269967  0.60731250  -1.124          0.260957    
    ## amount                           -0.00013829  0.00004389  -3.151          0.001627 ** 
    ## `savings_balance> 1000 DM`        1.03432320  0.51321912   2.015          0.043867 *  
    ## `savings_balance100 - 500 DM`     0.13185558  0.28429005   0.464          0.642786    
    ## `savings_balance500 - 1000 DM`    0.27415120  0.41264732   0.664          0.506452    
    ## savings_balanceunknown            0.90758459  0.26502755   3.424          0.000616 ***
    ## `employment_duration> 7 years`    0.51216659  0.29605002   1.730          0.083630 .  
    ## `employment_duration1 - 4 years`  0.16207344  0.23846600   0.680          0.496726    
    ## `employment_duration4 - 7 years`  0.92790647  0.30112909   3.081          0.002060 ** 
    ## employment_durationunemployed     0.14840842  0.43655991   0.340          0.733894    
    ## percent_of_income                -0.34774866  0.08869354  -3.921 0.000088259512881 ***
    ## years_at_residence               -0.00385951  0.08729784  -0.044          0.964736    
    ## age                               0.01108220  0.00927862   1.194          0.232329    
    ## other_creditnone                  0.52544326  0.24108458   2.179          0.029295 *  
    ## other_creditstore                 0.12816587  0.42389741   0.302          0.762384    
    ## housingown                        0.27205220  0.30231677   0.900          0.368178    
    ## housingrent                      -0.25445634  0.34509987  -0.737          0.460915    
    ## existing_loans_count             -0.33507655  0.19199533  -1.745          0.080944 .  
    ## jobskilled                        0.04584693  0.28901415   0.159          0.873959    
    ## jobunemployed                     0.09476193  0.65453976   0.145          0.884887    
    ## jobunskilled                      0.14669504  0.35145618   0.417          0.676392    
    ## dependents                       -0.11052559  0.24712936  -0.447          0.654703    
    ## phoneTRUE                         0.41866313  0.20925782   2.001          0.045424 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1099.56  on 899  degrees of freedom
    ## Residual deviance:  857.01  on 864  degrees of freedom
    ## AIC: 929.01
    ## 
    ## Number of Fisher Scoring iterations: 5

> NOTE: "Logistic regression does not make many of the key assumptions of linear regression and general linear models that are based on ordinary least squares algorithms – particularly regarding linearity, normality, homoscedasticity, and measurement level." [link](http://www.statisticssolutions.com/assumptions-of-logistic-regression/)

### Logistic Regression - basic pre-processing

> NOTE that for logistic regression (GLM), caret's `train()` (because of `glm()`) uses the second-level factor value as the success/postive event but `resamples()` uses the first-level as the success event. The result is either the `sensitivity` and `specificity` for `resamples()` will be reversed (and so I would be unable to compare apples to apples with other models), or I need to keep the first-level factor as the positive event (the default approach), which will mean that THE COEFFICIENTS WILL BE REVERSED, MAKIN THE MODEL RELATIVE TO THE NEGATIVE EVENT. I chose the latter, in order to compare models below, but this means that when using the logistic model to explain the data, the reader needs to mentally reverse the direction/sign of the coefficients, or correct the problem in the final stages of model building.

``` r
if(refresh_models)
{
    check_data(classification_train, validate_n_p = TRUE)
    set.seed(custom_seed)
    model_glm_basic_processing <- train(target ~ ., data=classification_train, method='glm', metric=metric, preProc=c('nzv', 'center', 'scale', 'knnImpute'), trControl=train_control)
    saveRDS(model_glm_basic_processing, file = './classification_data/model_glm_basic_processing.RDS')
} else{
    model_glm_basic_processing <- readRDS('./classification_data/model_glm_basic_processing.RDS')
}
summary(model_glm_basic_processing)
```

    ## 
    ## Call:
    ## NULL
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.5895  -0.8436   0.4207   0.7751   2.0452  
    ## 
    ## Coefficients:
    ##                                   Estimate Std. Error z value             Pr(>|z|)    
    ## (Intercept)                       1.147190   0.094081  12.194 < 0.0000000000000002 ***
    ## `checking_balance> 200 DM`        0.238392   0.090948   2.621             0.008763 ** 
    ## `checking_balance1 - 200 DM`      0.169782   0.093537   1.815             0.069503 .  
    ## checking_balanceunknown           0.864977   0.112309   7.702   0.0000000000000134 ***
    ## months_loan_duration             -0.243480   0.109843  -2.217             0.026649 *  
    ## credit_historygood               -0.296156   0.119650  -2.475             0.013317 *  
    ## credit_historypoor               -0.109460   0.090283  -1.212             0.225354    
    ## `credit_historyvery good`        -0.241627   0.089351  -2.704             0.006846 ** 
    ## purposecar                        0.034689   0.130295   0.266             0.790058    
    ## purposeeducation                 -0.088180   0.099152  -0.889             0.373819    
    ## `purposefurniture/appliances`     0.180758   0.135377   1.335             0.181803    
    ## amount                           -0.400086   0.120926  -3.309             0.000938 ***
    ## `savings_balance100 - 500 DM`     0.015658   0.085541   0.183             0.854758    
    ## `savings_balance500 - 1000 DM`    0.054567   0.098771   0.552             0.580635    
    ## savings_balanceunknown            0.329990   0.100635   3.279             0.001041 ** 
    ## `employment_duration> 7 years`    0.255840   0.126875   2.016             0.043749 *  
    ## `employment_duration1 - 4 years`  0.099657   0.111016   0.898             0.369353    
    ## `employment_duration4 - 7 years`  0.371148   0.112825   3.290             0.001003 ** 
    ## employment_durationunemployed     0.050138   0.097707   0.513             0.607846    
    ## percent_of_income                -0.383384   0.097775  -3.921   0.0000881507786376 ***
    ## years_at_residence                0.005518   0.095818   0.058             0.954075    
    ## age                               0.128240   0.104123   1.232             0.218093    
    ## other_creditnone                  0.187232   0.083529   2.242             0.024992 *  
    ## housingown                        0.151433   0.135406   1.118             0.263413    
    ## housingrent                      -0.084313   0.130666  -0.645             0.518761    
    ## existing_loans_count             -0.178839   0.108980  -1.641             0.100790    
    ## jobskilled                        0.007380   0.130238   0.057             0.954813    
    ## jobunskilled                      0.044701   0.131371   0.340             0.733655    
    ## dependents                       -0.041299   0.086738  -0.476             0.633977    
    ## phoneTRUE                         0.222771   0.099246   2.245             0.024791 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1099.56  on 899  degrees of freedom
    ## Residual deviance:  872.53  on 870  degrees of freedom
    ## AIC: 932.53
    ## 
    ## Number of Fisher Scoring iterations: 5

> NOTE: "Logistic regression does not make many of the key assumptions of linear regression and general linear models that are based on ordinary least squares algorithms – particularly regarding linearity, normality, homoscedasticity, and measurement level." [link](http://www.statisticssolutions.com/assumptions-of-logistic-regression/)

### Logistic Regression - remove collinear data - based on caret's recommendation

> NOTE that for logistic regression (GLM), caret's `train()` (because of `glm()`) uses the second-level factor value as the success/postive event but `resamples()` uses the first-level as the success event. The result is either the `sensitivity` and `specificity` for `resamples()` will be reversed (and so I would be unable to compare apples to apples with other models), or I need to keep the first-level factor as the positive event (the default approach), which will mean that THE COEFFICIENTS WILL BE REVERSED, MAKIN THE MODEL RELATIVE TO THE NEGATIVE EVENT. I chose the latter, in order to compare models below, but this means that when using the logistic model to explain the data, the reader needs to mentally reverse the direction/sign of the coefficients, or correct the problem in the final stages of model building.

``` r
if(refresh_models)
{
    check_data(classification_train[, recommended_columns_caret], validate_n_p = TRUE)
    set.seed(custom_seed)
    glm_remove_collinearity_caret <- train(target ~ ., data = classification_train[, recommended_columns_caret], method = 'glm', metric=metric, preProc=c('nzv', 'center', 'scale', 'knnImpute'), trControl = train_control)
    saveRDS(glm_remove_collinearity_caret, file = './classification_data/glm_remove_collinearity_caret.RDS')
} else{
    glm_remove_collinearity_caret <- readRDS('./classification_data/glm_remove_collinearity_caret.RDS')
}
summary(glm_remove_collinearity_caret)
```

    ## 
    ## Call:
    ## NULL
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.5895  -0.8436   0.4207   0.7751   2.0452  
    ## 
    ## Coefficients:
    ##                                   Estimate Std. Error z value             Pr(>|z|)    
    ## (Intercept)                       1.147190   0.094081  12.194 < 0.0000000000000002 ***
    ## `checking_balance> 200 DM`        0.238392   0.090948   2.621             0.008763 ** 
    ## `checking_balance1 - 200 DM`      0.169782   0.093537   1.815             0.069503 .  
    ## checking_balanceunknown           0.864977   0.112309   7.702   0.0000000000000134 ***
    ## months_loan_duration             -0.243480   0.109843  -2.217             0.026649 *  
    ## credit_historygood               -0.296156   0.119650  -2.475             0.013317 *  
    ## credit_historypoor               -0.109460   0.090283  -1.212             0.225354    
    ## `credit_historyvery good`        -0.241627   0.089351  -2.704             0.006846 ** 
    ## purposecar                        0.034689   0.130295   0.266             0.790058    
    ## purposeeducation                 -0.088180   0.099152  -0.889             0.373819    
    ## `purposefurniture/appliances`     0.180758   0.135377   1.335             0.181803    
    ## amount                           -0.400086   0.120926  -3.309             0.000938 ***
    ## `savings_balance100 - 500 DM`     0.015658   0.085541   0.183             0.854758    
    ## `savings_balance500 - 1000 DM`    0.054567   0.098771   0.552             0.580635    
    ## savings_balanceunknown            0.329990   0.100635   3.279             0.001041 ** 
    ## `employment_duration> 7 years`    0.255840   0.126875   2.016             0.043749 *  
    ## `employment_duration1 - 4 years`  0.099657   0.111016   0.898             0.369353    
    ## `employment_duration4 - 7 years`  0.371148   0.112825   3.290             0.001003 ** 
    ## employment_durationunemployed     0.050138   0.097707   0.513             0.607846    
    ## percent_of_income                -0.383384   0.097775  -3.921   0.0000881507786376 ***
    ## years_at_residence                0.005518   0.095818   0.058             0.954075    
    ## age                               0.128240   0.104123   1.232             0.218093    
    ## other_creditnone                  0.187232   0.083529   2.242             0.024992 *  
    ## housingown                        0.151433   0.135406   1.118             0.263413    
    ## housingrent                      -0.084313   0.130666  -0.645             0.518761    
    ## existing_loans_count             -0.178839   0.108980  -1.641             0.100790    
    ## jobskilled                        0.007380   0.130238   0.057             0.954813    
    ## jobunskilled                      0.044701   0.131371   0.340             0.733655    
    ## dependents                       -0.041299   0.086738  -0.476             0.633977    
    ## phoneTRUE                         0.222771   0.099246   2.245             0.024791 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1099.56  on 899  degrees of freedom
    ## Residual deviance:  872.53  on 870  degrees of freedom
    ## AIC: 932.53
    ## 
    ## Number of Fisher Scoring iterations: 5

> NOTE: "Logistic regression does not make many of the key assumptions of linear regression and general linear models that are based on ordinary least squares algorithms – particularly regarding linearity, normality, homoscedasticity, and measurement level." [link](http://www.statisticssolutions.com/assumptions-of-logistic-regression/)

### Logistic Regression - remove collinear data - based on calculation

> NOTE that for logistic regression (GLM), caret's `train()` (because of `glm()`) uses the second-level factor value as the success/postive event but `resamples()` uses the first-level as the success event. The result is either the `sensitivity` and `specificity` for `resamples()` will be reversed (and so I would be unable to compare apples to apples with other models), or I need to keep the first-level factor as the positive event (the default approach), which will mean that THE COEFFICIENTS WILL BE REVERSED, MAKIN THE MODEL RELATIVE TO THE NEGATIVE EVENT. I chose the latter, in order to compare models below, but this means that when using the logistic model to explain the data, the reader needs to mentally reverse the direction/sign of the coefficients, or correct the problem in the final stages of model building.

``` r
if(refresh_models)
{
    check_data(classification_train[, recommended_columns_custom], validate_n_p = TRUE)
    set.seed(custom_seed)
    glm_remove_collinearity_custom <- train(target ~ ., data = classification_train[, recommended_columns_custom], method = 'glm', metric=metric, preProc=c('nzv', 'center', 'scale', 'knnImpute'), trControl = train_control)
    saveRDS(glm_remove_collinearity_custom, file = './classification_data/glm_remove_collinearity_custom.RDS')
} else{
    glm_remove_collinearity_custom <- readRDS('./classification_data/glm_remove_collinearity_custom.RDS')
}
summary(glm_remove_collinearity_custom)
```

    ## 
    ## Call:
    ## NULL
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.5895  -0.8436   0.4207   0.7751   2.0452  
    ## 
    ## Coefficients:
    ##                                   Estimate Std. Error z value             Pr(>|z|)    
    ## (Intercept)                       1.147190   0.094081  12.194 < 0.0000000000000002 ***
    ## months_loan_duration             -0.243480   0.109843  -2.217             0.026649 *  
    ## amount                           -0.400086   0.120926  -3.309             0.000938 ***
    ## percent_of_income                -0.383384   0.097775  -3.921   0.0000881507786376 ***
    ## years_at_residence                0.005518   0.095818   0.058             0.954075    
    ## age                               0.128240   0.104123   1.232             0.218093    
    ## existing_loans_count             -0.178839   0.108980  -1.641             0.100790    
    ## dependents                       -0.041299   0.086738  -0.476             0.633977    
    ## `checking_balance> 200 DM`        0.238392   0.090948   2.621             0.008763 ** 
    ## `checking_balance1 - 200 DM`      0.169782   0.093537   1.815             0.069503 .  
    ## checking_balanceunknown           0.864977   0.112309   7.702   0.0000000000000134 ***
    ## credit_historygood               -0.296156   0.119650  -2.475             0.013317 *  
    ## credit_historypoor               -0.109460   0.090283  -1.212             0.225354    
    ## `credit_historyvery good`        -0.241627   0.089351  -2.704             0.006846 ** 
    ## purposecar                        0.034689   0.130295   0.266             0.790058    
    ## purposeeducation                 -0.088180   0.099152  -0.889             0.373819    
    ## `purposefurniture/appliances`     0.180758   0.135377   1.335             0.181803    
    ## `savings_balance100 - 500 DM`     0.015658   0.085541   0.183             0.854758    
    ## `savings_balance500 - 1000 DM`    0.054567   0.098771   0.552             0.580635    
    ## savings_balanceunknown            0.329990   0.100635   3.279             0.001041 ** 
    ## `employment_duration> 7 years`    0.255840   0.126875   2.016             0.043749 *  
    ## `employment_duration1 - 4 years`  0.099657   0.111016   0.898             0.369353    
    ## `employment_duration4 - 7 years`  0.371148   0.112825   3.290             0.001003 ** 
    ## employment_durationunemployed     0.050138   0.097707   0.513             0.607846    
    ## other_creditnone                  0.187232   0.083529   2.242             0.024992 *  
    ## housingown                        0.151433   0.135406   1.118             0.263413    
    ## housingrent                      -0.084313   0.130666  -0.645             0.518761    
    ## jobskilled                        0.007380   0.130238   0.057             0.954813    
    ## jobunskilled                      0.044701   0.131371   0.340             0.733655    
    ## phoneTRUE                         0.222771   0.099246   2.245             0.024791 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1099.56  on 899  degrees of freedom
    ## Residual deviance:  872.53  on 870  degrees of freedom
    ## AIC: 932.53
    ## 
    ## Number of Fisher Scoring iterations: 5

> NOTE: "Logistic regression does not make many of the key assumptions of linear regression and general linear models that are based on ordinary least squares algorithms – particularly regarding linearity, normality, homoscedasticity, and measurement level." [link](http://www.statisticssolutions.com/assumptions-of-logistic-regression/)

### Logistic Regression - remove collinear data - based on calculation

> NOTE that for logistic regression (GLM), caret's `train()` (because of `glm()`) uses the second-level factor value as the success/postive event but `resamples()` uses the first-level as the success event. The result is either the `sensitivity` and `specificity` for `resamples()` will be reversed (and so I would be unable to compare apples to apples with other models), or I need to keep the first-level factor as the positive event (the default approach), which will mean that THE COEFFICIENTS WILL BE REVERSED, MAKIN THE MODEL RELATIVE TO THE NEGATIVE EVENT. I chose the latter, in order to compare models below, but this means that when using the logistic model to explain the data, the reader needs to mentally reverse the direction/sign of the coefficients, or correct the problem in the final stages of model building.

``` r
if(refresh_models)
{
    check_data(classification_train, validate_n_p = TRUE)
    set.seed(custom_seed)
    
    set.seed(custom_seed)
    pre_processed_numeric_data <- preProcess(classification_train, method = c('nzv', 'center', 'scale', 'knnImpute')) # ignores non-numeric data
    columns_not_in_preprocessed_data <- colnames(classification_train)[!(colnames(classification_train) %in% colnames(pre_processed_numeric_data$data))] # figure out which columns we need to add back in (i.e. all (non-numeric) that are in classification_train but are NOT in pre_processed_numeric_data
    pre_processed_classification_train <- cbind(classification_train[, columns_not_in_preprocessed_data], pre_processed_numeric_data$data)

    set.seed(custom_seed)
    stepwise_glm_model <- step(glm(family = binomial, formula = target ~ ., data = pre_processed_classification_train), direction="backward", trace=0) # do stepwise regression (glm doesn't like factor target variables), then use the formula in train in order to take advantage of k-fold Cross Validation
    # stepwise_glm_model$formula gives the `optimal` formula (need to do this because coefficients will have factor variable names, not original columns. The formula will exclude columns not statistically significant)
    # now feed this back into training to do cross validation
    logistic_regression_stepwise_backward <- train(stepwise_glm_model$formula, data = classification_train, method = 'glm', metric=metric, preProc=c('nzv', 'center', 'scale', 'knnImpute'), trControl = train_control)
    saveRDS(logistic_regression_stepwise_backward, file = './classification_data/logistic_regression_stepwise_backward.RDS')
} else{
    logistic_regression_stepwise_backward <- readRDS('./classification_data/logistic_regression_stepwise_backward.RDS')
}
summary(logistic_regression_stepwise_backward)
```

    ## 
    ## Call:
    ## NULL
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.5554  -0.8800   0.4347   0.7691   2.0064  
    ## 
    ## Coefficients:
    ##                                   Estimate Std. Error z value             Pr(>|z|)    
    ## (Intercept)                       1.131640   0.092838  12.189 < 0.0000000000000002 ***
    ## `checking_balance> 200 DM`        0.245993   0.089350   2.753             0.005903 ** 
    ## `checking_balance1 - 200 DM`      0.165115   0.091617   1.802             0.071508 .  
    ## checking_balanceunknown           0.859391   0.111245   7.725   0.0000000000000112 ***
    ## credit_historygood               -0.283603   0.117919  -2.405             0.016169 *  
    ## credit_historypoor               -0.120046   0.088977  -1.349             0.177278    
    ## `credit_historyvery good`        -0.246000   0.088881  -2.768             0.005645 ** 
    ## `savings_balance100 - 500 DM`     0.001352   0.084900   0.016             0.987294    
    ## `savings_balance500 - 1000 DM`    0.070428   0.098308   0.716             0.473743    
    ## savings_balanceunknown            0.312059   0.098696   3.162             0.001568 ** 
    ## `employment_duration> 7 years`    0.295568   0.116962   2.527             0.011502 *  
    ## `employment_duration1 - 4 years`  0.103760   0.109846   0.945             0.344866    
    ## `employment_duration4 - 7 years`  0.372998   0.111061   3.359             0.000784 ***
    ## employment_durationunemployed     0.059406   0.089472   0.664             0.506715    
    ## other_creditnone                  0.183429   0.082681   2.219             0.026519 *  
    ## housingown                        0.178234   0.123668   1.441             0.149520    
    ## housingrent                      -0.081868   0.122738  -0.667             0.504761    
    ## phoneTRUE                         0.210035   0.091554   2.294             0.021785 *  
    ## months_loan_duration             -0.251555   0.107446  -2.341             0.019221 *  
    ## amount                           -0.409812   0.117708  -3.482             0.000498 ***
    ## percent_of_income                -0.372375   0.096322  -3.866             0.000111 ***
    ## existing_loans_count             -0.180763   0.106837  -1.692             0.090654 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1099.56  on 899  degrees of freedom
    ## Residual deviance:  879.81  on 878  degrees of freedom
    ## AIC: 923.81
    ## 
    ## Number of Fisher Scoring iterations: 5

> NOTE: "Logistic regression does not make many of the key assumptions of linear regression and general linear models that are based on ordinary least squares algorithms – particularly regarding linearity, normality, homoscedasticity, and measurement level." [link](http://www.statisticssolutions.com/assumptions-of-logistic-regression/)

### Linear Discriminant Analysis

``` r
if(refresh_models)
{
    check_data(classification_train, validate_n_p = TRUE)
    set.seed(custom_seed)
    linear_discriminant_analsysis <- train(target ~ ., data = classification_train, method = 'lda', metric=metric, preProc=c('nzv', 'center', 'scale'), trControl = train_control)
    saveRDS(linear_discriminant_analsysis, file = './classification_data/linear_discriminant_analsysis.RDS')
} else{
    linear_discriminant_analsysis <- readRDS('./classification_data/linear_discriminant_analsysis.RDS')
}
```

    ## Loading required package: MASS

    ## 
    ## Attaching package: 'MASS'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     select

``` r
summary(linear_discriminant_analsysis)
```

    ##             Length Class      Mode     
    ## prior        2     -none-     numeric  
    ## counts       2     -none-     numeric  
    ## means       58     -none-     numeric  
    ## scaling     29     -none-     numeric  
    ## lev          2     -none-     character
    ## svd          1     -none-     numeric  
    ## N            1     -none-     numeric  
    ## call         3     -none-     call     
    ## xNames      29     -none-     character
    ## problemType  1     -none-     character
    ## tuneValue    1     data.frame list     
    ## obsLevels    2     -none-     character
    ## param        0     -none-     list

``` r
varImp(linear_discriminant_analsysis, scale = FALSE)
```

    ## ROC curve variable importance
    ## 
    ##                      Importance
    ## checking_balance         0.6906
    ## credit_history           0.6195
    ## months_loan_duration     0.6146
    ## savings_balance          0.5906
    ## age                      0.5700
    ## amount                   0.5533
    ## percent_of_income        0.5470
    ## employment_duration      0.5279
    ## phone                    0.5275
    ## purpose                  0.5258
    ## other_credit             0.5252
    ## existing_loans_count     0.5204
    ## job                      0.5193
    ## housing                  0.5105
    ## years_at_residence       0.5065
    ## dependents               0.5063

### Linear Discriminant Analysis - Remove Collinear Data Based on Caret's Recommendation

``` r
if(refresh_models)
{
    check_data(classification_train[, recommended_columns_caret], validate_n_p = TRUE)
    set.seed(custom_seed)
    linear_discriminant_analsysis_remove_collinear <- train(target ~ ., data = classification_train[, recommended_columns_caret], method = 'lda', metric=metric, preProc=c('nzv', 'center', 'scale'), trControl = train_control)
    saveRDS(linear_discriminant_analsysis_remove_collinear, file = './classification_data/linear_discriminant_analsysis_remove_collinear.RDS')
} else{
    linear_discriminant_analsysis_remove_collinear <- readRDS('./classification_data/linear_discriminant_analsysis_remove_collinear.RDS')
}
summary(linear_discriminant_analsysis_remove_collinear)
```

    ##             Length Class      Mode     
    ## prior        2     -none-     numeric  
    ## counts       2     -none-     numeric  
    ## means       58     -none-     numeric  
    ## scaling     29     -none-     numeric  
    ## lev          2     -none-     character
    ## svd          1     -none-     numeric  
    ## N            1     -none-     numeric  
    ## call         3     -none-     call     
    ## xNames      29     -none-     character
    ## problemType  1     -none-     character
    ## tuneValue    1     data.frame list     
    ## obsLevels    2     -none-     character
    ## param        0     -none-     list

``` r
varImp(linear_discriminant_analsysis_remove_collinear, scale = FALSE)
```

    ## ROC curve variable importance
    ## 
    ##                      Importance
    ## checking_balance         0.6906
    ## credit_history           0.6195
    ## months_loan_duration     0.6146
    ## savings_balance          0.5906
    ## age                      0.5700
    ## amount                   0.5533
    ## percent_of_income        0.5470
    ## employment_duration      0.5279
    ## phone                    0.5275
    ## purpose                  0.5258
    ## other_credit             0.5252
    ## existing_loans_count     0.5204
    ## job                      0.5193
    ## housing                  0.5105
    ## years_at_residence       0.5065
    ## dependents               0.5063

### Partial Least Squares Discriminant Analysis (PLSDA)

``` r
if(refresh_models)
{
    check_data(classification_train, validate_n_p = TRUE)
    set.seed(custom_seed)
    partial_least_squares_discriminant_analysis <- train(target ~ ., data = classification_train, method = 'pls', metric=metric, preProc=c('nzv', 'center', 'scale'), trControl = train_control, # "performance of PLS is affected when including predictors that contain little or no predictive information" i.e. remove NZV
                                                        tuneGrid = expand.grid(.ncomp = tuning_number_of_latent_variables_to_retain))
    saveRDS(partial_least_squares_discriminant_analysis, file = './classification_data/partial_least_squares_discriminant_analysis.RDS')
} else{
    partial_least_squares_discriminant_analysis <- readRDS('./classification_data/partial_least_squares_discriminant_analysis.RDS')
}
```

    ## Loading required package: pls

    ## 
    ## Attaching package: 'pls'

    ## The following object is masked from 'package:caret':
    ## 
    ##     R2

    ## The following object is masked from 'package:corrplot':
    ## 
    ##     corrplot

    ## The following object is masked from 'package:stats':
    ## 
    ##     loadings

``` r
partial_least_squares_discriminant_analysis
```

    ## Partial Least Squares 
    ## 
    ## 900 samples
    ##  16 predictor
    ##   2 classes: 'yes', 'no' 
    ## 
    ## Pre-processing: centered (29), scaled (29), remove (6) 
    ## Resampling: Cross-Validated (10 fold, repeated 3 times) 
    ## Summary of sample sizes: 810, 810, 810, 810, 810, 810, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   ncomp  ROC        Sens       Spec     
    ##    1     0.7442485  0.2888889  0.9068783
    ##    2     0.7555752  0.3493827  0.8941799
    ##    3     0.7583578  0.3777778  0.8846561
    ##    4     0.7627670  0.3888889  0.8756614
    ##    5     0.7619048  0.3962963  0.8798942
    ##    6     0.7611405  0.3888889  0.8777778
    ##    7     0.7612777  0.3864198  0.8777778
    ##    8     0.7611797  0.3851852  0.8783069
    ##    9     0.7610425  0.3839506  0.8777778
    ##   10     0.7610817  0.3851852  0.8783069
    ## 
    ## ROC was used to select the optimal model using  the largest value.
    ## The final value used for the model was ncomp = 4.

``` r
#summary(partial_least_squares_discriminant_analysis)
plot(partial_least_squares_discriminant_analysis, top = 20, scales = list(y = list(cex = 0.95)))
```

<img src="predictive_analysis_classification_files/figure-markdown_github/partial_least_squares_discriminant_analysis-1.png" width="750px" />

``` r
varImp(partial_least_squares_discriminant_analysis, scale = FALSE)
```

    ## pls variable importance
    ## 
    ##   only 20 most important variables shown (out of 29)
    ## 
    ##                                Overall
    ## checking_balanceunknown        0.07640
    ## months_loan_duration           0.04441
    ## amount                         0.03569
    ## checking_balance1 - 200 DM     0.03420
    ## housingown                     0.03307
    ## savings_balanceunknown         0.03221
    ## credit_historyvery good        0.02983
    ## other_creditnone               0.02795
    ## housingrent                    0.02299
    ## percent_of_income              0.02299
    ## age                            0.02162
    ## employment_duration4 - 7 years 0.02091
    ## employment_duration> 7 years   0.01912
    ## purposefurniture/appliances    0.01669
    ## savings_balance500 - 1000 DM   0.01606
    ## phoneTRUE                      0.01457
    ## checking_balance> 200 DM       0.01339
    ## purposeeducation               0.01202
    ## employment_durationunemployed  0.01105
    ## credit_historygood             0.01049

### glmnet (LASSO and RIDGE)

``` r
if(refresh_models)
{
    check_data(classification_train, validate_n_p = TRUE)
    set.seed(custom_seed)
    glmnet_lasso_ridge <- train(target ~ ., data = classification_train, method = 'glmnet', metric=metric, preProc=c('nzv', 'center', 'scale'), trControl = train_control,
                                                        tuneGrid = expand.grid(alpha = tuning_glmnet_alpha, lambda = tuning_glmnet_lambda))
    saveRDS(glmnet_lasso_ridge, file = './classification_data/glmnet_lasso_ridge.RDS')
} else{
    glmnet_lasso_ridge <- readRDS('./classification_data/glmnet_lasso_ridge.RDS')
}
```

    ## Loading required package: glmnet

    ## Loading required package: Matrix

    ## 
    ## Attaching package: 'Matrix'

    ## The following object is masked from 'package:tidyr':
    ## 
    ##     expand

    ## Loaded glmnet 2.0-10

``` r
#glmnet_lasso_ridge
summary(glmnet_lasso_ridge)
```

    ##             Length Class      Mode     
    ## a0            66   -none-     numeric  
    ## beta        1914   dgCMatrix  S4       
    ## df            66   -none-     numeric  
    ## dim            2   -none-     numeric  
    ## lambda        66   -none-     numeric  
    ## dev.ratio     66   -none-     numeric  
    ## nulldev        1   -none-     numeric  
    ## npasses        1   -none-     numeric  
    ## jerr           1   -none-     numeric  
    ## offset         1   -none-     logical  
    ## classnames     2   -none-     character
    ## call           5   -none-     call     
    ## nobs           1   -none-     numeric  
    ## lambdaOpt      1   -none-     numeric  
    ## xNames        29   -none-     character
    ## problemType    1   -none-     character
    ## tuneValue      2   data.frame list     
    ## obsLevels      2   -none-     character
    ## param          0   -none-     list

``` r
plot(glmnet_lasso_ridge, top = 20, scales = list(y = list(cex = 0.95)))
```

<img src="predictive_analysis_classification_files/figure-markdown_github/glmnet_lasso_ridge-1.png" width="750px" />

``` r
plot(glmnet_lasso_ridge, plotType = 'level')
```

<img src="predictive_analysis_classification_files/figure-markdown_github/glmnet_lasso_ridge-2.png" width="750px" />

``` r
varImp(glmnet_lasso_ridge, scale = FALSE)
```

    ## glmnet variable importance
    ## 
    ##   only 20 most important variables shown (out of 29)
    ## 
    ##                                Overall
    ## checking_balanceunknown        0.65776
    ## amount                         0.22805
    ## percent_of_income              0.22683
    ## months_loan_duration           0.21889
    ## savings_balanceunknown         0.21305
    ## employment_duration4 - 7 years 0.17986
    ## housingown                     0.14292
    ## other_creditnone               0.13728
    ## checking_balance> 200 DM       0.13057
    ## credit_historyvery good        0.12062
    ## phoneTRUE                      0.10410
    ## employment_duration> 7 years   0.10060
    ## age                            0.08770
    ## purposefurniture/appliances    0.07662
    ## credit_historygood             0.05505
    ## purposeeducation               0.04112
    ## housingrent                    0.03713
    ## savings_balance500 - 1000 DM   0.01972
    ## checking_balance1 - 200 DM     0.01924
    ## jobskilled                     0.00000

### Sparse LDA

``` r
if(refresh_models)
{
    check_data(classification_train, validate_n_p = FALSE)
    set.seed(custom_seed)
    sparse_lda <- train(target ~ ., data = classification_train, method = 'sparseLDA', metric=metric, preProc=c('nzv', 'center', 'scale'), trControl = train_control,
                                                        tuneLength = 5)
    saveRDS(sparse_lda, file = './classification_data/sparse_lda.RDS')
} else{
    sparse_lda <- readRDS('./classification_data/sparse_lda.RDS')
}
```

    ## Loading required package: sparseLDA

``` r
#sparse_lda
summary(sparse_lda)
```

    ##             Length Class      Mode     
    ## call         5     -none-     call     
    ## beta        22     -none-     numeric  
    ## theta        2     -none-     numeric  
    ## varNames    22     -none-     character
    ## varIndex    22     -none-     numeric  
    ## origP        1     -none-     numeric  
    ## rss          1     -none-     numeric  
    ## fit          8     lda        list     
    ## classes      2     -none-     character
    ## lambda       1     -none-     numeric  
    ## stop         1     -none-     numeric  
    ## xNames      29     -none-     character
    ## problemType  1     -none-     character
    ## tuneValue    2     data.frame list     
    ## obsLevels    2     -none-     character
    ## param        0     -none-     list

``` r
plot(sparse_lda, top = 20, scales = list(y = list(cex = 0.95)))
```

<img src="predictive_analysis_classification_files/figure-markdown_github/sparse_lda-1.png" width="750px" />

``` r
plot(sparse_lda, plotType = 'level')
```

<img src="predictive_analysis_classification_files/figure-markdown_github/sparse_lda-2.png" width="750px" />

``` r
varImp(sparse_lda, scale = FALSE)
```

    ## ROC curve variable importance
    ## 
    ##                      Importance
    ## checking_balance         0.6906
    ## credit_history           0.6195
    ## months_loan_duration     0.6146
    ## savings_balance          0.5906
    ## age                      0.5700
    ## amount                   0.5533
    ## percent_of_income        0.5470
    ## employment_duration      0.5279
    ## phone                    0.5275
    ## purpose                  0.5258
    ## other_credit             0.5252
    ## existing_loans_count     0.5204
    ## job                      0.5193
    ## housing                  0.5105
    ## years_at_residence       0.5065
    ## dependents               0.5063

### Nearest Shrunken Centroids

> Poor performance and varImp() gives error

### Random Forest

### Neural Network

### Ada Boost

### All Models on Page 550 that are classification or both regression and classification

### Models used for spot-check.Rmd

Resamples & Top Models
----------------------

### Resamples

    ## 
    ## Call:
    ## summary.resamples(object = resamples)
    ## 
    ## Models: model_glm_no_pre_processing, model_glm_basic_processing, glm_remove_collinearity_caret, glm_remove_collinearity_custom, logistic_regression_stepwise_backward, linear_discriminant_analsysis, linear_discriminant_analsysis_remove_collinear, partial_least_squares_discriminant_analysis, glmnet_lasso_ridge, sparse_lda 
    ## Number of resamples: 30 
    ## 
    ## ROC 
    ##                                                     Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## model_glm_no_pre_processing                    0.6748971 0.7308936 0.7598471 0.7628258 0.7839506 0.8536155    0
    ## model_glm_basic_processing                     0.6801881 0.7225162 0.7595532 0.7594748 0.7914462 0.8677249    0
    ## glm_remove_collinearity_caret                  0.6801881 0.7225162 0.7595532 0.7594748 0.7914462 0.8677249    0
    ## glm_remove_collinearity_custom                 0.6801881 0.7225162 0.7595532 0.7594748 0.7914462 0.8677249    0
    ## logistic_regression_stepwise_backward          0.6878307 0.7264844 0.7757202 0.7644523 0.7962963 0.8641975    0
    ## linear_discriminant_analsysis                  0.6937096 0.7182540 0.7645503 0.7610817 0.7948266 0.8665491    0
    ## linear_discriminant_analsysis_remove_collinear 0.6937096 0.7182540 0.7645503 0.7610817 0.7948266 0.8665491    0
    ## partial_least_squares_discriminant_analysis    0.6978248 0.7260435 0.7592593 0.7627670 0.7979130 0.8689006    0
    ## glmnet_lasso_ridge                             0.6760729 0.7244268 0.7583774 0.7628258 0.7974721 0.8589065    0
    ## sparse_lda                                     0.6931217 0.7273663 0.7589653 0.7644327 0.7992357 0.8689006    0
    ## 
    ## Sens 
    ##                                                     Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## model_glm_no_pre_processing                    0.2592593 0.4074074 0.4814815 0.4530864 0.5185185 0.5555556    0
    ## model_glm_basic_processing                     0.2222222 0.3333333 0.4074074 0.4111111 0.4814815 0.6296296    0
    ## glm_remove_collinearity_caret                  0.2222222 0.3333333 0.4074074 0.4111111 0.4814815 0.6296296    0
    ## glm_remove_collinearity_custom                 0.2222222 0.3333333 0.4074074 0.4111111 0.4814815 0.6296296    0
    ## logistic_regression_stepwise_backward          0.2222222 0.3796296 0.4259259 0.4111111 0.4444444 0.5555556    0
    ## linear_discriminant_analsysis                  0.2592593 0.3703704 0.4444444 0.4246914 0.4814815 0.5925926    0
    ## linear_discriminant_analsysis_remove_collinear 0.2592593 0.3703704 0.4444444 0.4246914 0.4814815 0.5925926    0
    ## partial_least_squares_discriminant_analysis    0.2222222 0.3333333 0.4074074 0.3888889 0.4722222 0.5185185    0
    ## glmnet_lasso_ridge                             0.1851852 0.2222222 0.2777778 0.2876543 0.3333333 0.4814815    0
    ## sparse_lda                                     0.2592593 0.3425926 0.4444444 0.4222222 0.4814815 0.6296296    0
    ## 
    ## Spec 
    ##                                                     Min.   1st Qu.    Median      Mean   3rd Qu.     Max. NA's
    ## model_glm_no_pre_processing                    0.7460317 0.8253968 0.8730159 0.8592593 0.8888889 0.952381    0
    ## model_glm_basic_processing                     0.7619048 0.8253968 0.8650794 0.8613757 0.8888889 0.952381    0
    ## glm_remove_collinearity_caret                  0.7619048 0.8253968 0.8650794 0.8613757 0.8888889 0.952381    0
    ## glm_remove_collinearity_custom                 0.7619048 0.8253968 0.8650794 0.8613757 0.8888889 0.952381    0
    ## logistic_regression_stepwise_backward          0.7936508 0.8571429 0.8730159 0.8772487 0.9047619 0.952381    0
    ## linear_discriminant_analsysis                  0.7619048 0.8293651 0.8571429 0.8576720 0.8888889 0.952381    0
    ## linear_discriminant_analsysis_remove_collinear 0.7619048 0.8293651 0.8571429 0.8576720 0.8888889 0.952381    0
    ## partial_least_squares_discriminant_analysis    0.7936508 0.8412698 0.8730159 0.8756614 0.9047619 0.952381    0
    ## glmnet_lasso_ridge                             0.8412698 0.8928571 0.9206349 0.9137566 0.9365079 0.968254    0
    ## sparse_lda                                     0.7619048 0.8253968 0.8571429 0.8592593 0.8849206 0.952381    0

<img src="predictive_analysis_classification_files/figure-markdown_github/resamples_regression-1.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/resamples_regression-2.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/resamples_regression-3.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/resamples_regression-4.png" width="750px" />

Train Top Models on Entire Training Dataset & Predict on Test Set
=================================================================

> after using cross-validation to tune, we'll take the highest ranked models, retrain the models (with the final tuning parameters) on the entire training set, and predict using the test set.

    ## 
    ## 
    ## ### Generalized Linear Model (glm_remove_collinearity_custom)
    ## 
    ## 
    ## 
    ## Pre-Processing: `nzv, center, scale, knnImpute`
    ## 
    ## 
    ## Call:
    ## NULL
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.5895  -0.8436   0.4207   0.7751   2.0452  
    ## 
    ## Coefficients:
    ##                                   Estimate Std. Error z value             Pr(>|z|)    
    ## (Intercept)                       1.147190   0.094081  12.194 < 0.0000000000000002 ***
    ## `checking_balance> 200 DM`        0.238392   0.090948   2.621             0.008763 ** 
    ## `checking_balance1 - 200 DM`      0.169782   0.093537   1.815             0.069503 .  
    ## checking_balanceunknown           0.864977   0.112309   7.702   0.0000000000000134 ***
    ## months_loan_duration             -0.243480   0.109843  -2.217             0.026649 *  
    ## credit_historygood               -0.296156   0.119650  -2.475             0.013317 *  
    ## credit_historypoor               -0.109460   0.090283  -1.212             0.225354    
    ## `credit_historyvery good`        -0.241627   0.089351  -2.704             0.006846 ** 
    ## purposecar                        0.034689   0.130295   0.266             0.790058    
    ## purposeeducation                 -0.088180   0.099152  -0.889             0.373819    
    ## `purposefurniture/appliances`     0.180758   0.135377   1.335             0.181803    
    ## amount                           -0.400086   0.120926  -3.309             0.000938 ***
    ## `savings_balance100 - 500 DM`     0.015658   0.085541   0.183             0.854758    
    ## `savings_balance500 - 1000 DM`    0.054567   0.098771   0.552             0.580635    
    ## savings_balanceunknown            0.329990   0.100635   3.279             0.001041 ** 
    ## `employment_duration> 7 years`    0.255840   0.126875   2.016             0.043749 *  
    ## `employment_duration1 - 4 years`  0.099657   0.111016   0.898             0.369353    
    ## `employment_duration4 - 7 years`  0.371148   0.112825   3.290             0.001003 ** 
    ## employment_durationunemployed     0.050138   0.097707   0.513             0.607846    
    ## percent_of_income                -0.383384   0.097775  -3.921   0.0000881507786376 ***
    ## years_at_residence                0.005518   0.095818   0.058             0.954075    
    ## age                               0.128240   0.104123   1.232             0.218093    
    ## other_creditnone                  0.187232   0.083529   2.242             0.024992 *  
    ## housingown                        0.151433   0.135406   1.118             0.263413    
    ## housingrent                      -0.084313   0.130666  -0.645             0.518761    
    ## existing_loans_count             -0.178839   0.108980  -1.641             0.100790    
    ## jobskilled                        0.007380   0.130238   0.057             0.954813    
    ## jobunskilled                      0.044701   0.131371   0.340             0.733655    
    ## dependents                       -0.041299   0.086738  -0.476             0.633977    
    ## phoneTRUE                         0.222771   0.099246   2.245             0.024791 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1099.56  on 899  degrees of freedom
    ## Residual deviance:  872.53  on 870  degrees of freedom
    ## AIC: 932.53
    ## 
    ## Number of Fisher Scoring iterations: 5

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-1.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-2.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-3.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-4.png" width="750px" />

    ## ```
    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction yes no
    ##        yes  14  8
    ##        no   16 62
    ##                                           
    ##                Accuracy : 0.76            
    ##                  95% CI : (0.6643, 0.8398)
    ##     No Information Rate : 0.7             
    ##     P-Value [Acc > NIR] : 0.1136          
    ##                                           
    ##                   Kappa : 0.3814          
    ##  Mcnemar's Test P-Value : 0.1530          
    ##                                           
    ##             Sensitivity : 0.4667          
    ##             Specificity : 0.8857          
    ##          Pos Pred Value : 0.6364          
    ##          Neg Pred Value : 0.7949          
    ##              Prevalence : 0.3000          
    ##          Detection Rate : 0.1400          
    ##    Detection Prevalence : 0.2200          
    ##       Balanced Accuracy : 0.6762          
    ##                                           
    ##        'Positive' Class : yes             
    ##                                           
    ## ```

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-5.png" width="750px" />

    ## NULL

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-6.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-7.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-8.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-9.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-10.png" width="750px" />

    ## 
    ## 
    ## ### Generalized Linear Model (glm_remove_collinearity_caret)
    ## 
    ## 
    ## 
    ## Pre-Processing: `nzv, center, scale, knnImpute`
    ## 
    ## 
    ## Call:
    ## NULL
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.5895  -0.8436   0.4207   0.7751   2.0452  
    ## 
    ## Coefficients:
    ##                                   Estimate Std. Error z value             Pr(>|z|)    
    ## (Intercept)                       1.147190   0.094081  12.194 < 0.0000000000000002 ***
    ## `checking_balance> 200 DM`        0.238392   0.090948   2.621             0.008763 ** 
    ## `checking_balance1 - 200 DM`      0.169782   0.093537   1.815             0.069503 .  
    ## checking_balanceunknown           0.864977   0.112309   7.702   0.0000000000000134 ***
    ## months_loan_duration             -0.243480   0.109843  -2.217             0.026649 *  
    ## credit_historygood               -0.296156   0.119650  -2.475             0.013317 *  
    ## credit_historypoor               -0.109460   0.090283  -1.212             0.225354    
    ## `credit_historyvery good`        -0.241627   0.089351  -2.704             0.006846 ** 
    ## purposecar                        0.034689   0.130295   0.266             0.790058    
    ## purposeeducation                 -0.088180   0.099152  -0.889             0.373819    
    ## `purposefurniture/appliances`     0.180758   0.135377   1.335             0.181803    
    ## amount                           -0.400086   0.120926  -3.309             0.000938 ***
    ## `savings_balance100 - 500 DM`     0.015658   0.085541   0.183             0.854758    
    ## `savings_balance500 - 1000 DM`    0.054567   0.098771   0.552             0.580635    
    ## savings_balanceunknown            0.329990   0.100635   3.279             0.001041 ** 
    ## `employment_duration> 7 years`    0.255840   0.126875   2.016             0.043749 *  
    ## `employment_duration1 - 4 years`  0.099657   0.111016   0.898             0.369353    
    ## `employment_duration4 - 7 years`  0.371148   0.112825   3.290             0.001003 ** 
    ## employment_durationunemployed     0.050138   0.097707   0.513             0.607846    
    ## percent_of_income                -0.383384   0.097775  -3.921   0.0000881507786376 ***
    ## years_at_residence                0.005518   0.095818   0.058             0.954075    
    ## age                               0.128240   0.104123   1.232             0.218093    
    ## other_creditnone                  0.187232   0.083529   2.242             0.024992 *  
    ## housingown                        0.151433   0.135406   1.118             0.263413    
    ## housingrent                      -0.084313   0.130666  -0.645             0.518761    
    ## existing_loans_count             -0.178839   0.108980  -1.641             0.100790    
    ## jobskilled                        0.007380   0.130238   0.057             0.954813    
    ## jobunskilled                      0.044701   0.131371   0.340             0.733655    
    ## dependents                       -0.041299   0.086738  -0.476             0.633977    
    ## phoneTRUE                         0.222771   0.099246   2.245             0.024791 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1099.56  on 899  degrees of freedom
    ## Residual deviance:  872.53  on 870  degrees of freedom
    ## AIC: 932.53
    ## 
    ## Number of Fisher Scoring iterations: 5

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-11.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-12.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-13.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-14.png" width="750px" />

    ## ```
    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction yes no
    ##        yes  14  8
    ##        no   16 62
    ##                                           
    ##                Accuracy : 0.76            
    ##                  95% CI : (0.6643, 0.8398)
    ##     No Information Rate : 0.7             
    ##     P-Value [Acc > NIR] : 0.1136          
    ##                                           
    ##                   Kappa : 0.3814          
    ##  Mcnemar's Test P-Value : 0.1530          
    ##                                           
    ##             Sensitivity : 0.4667          
    ##             Specificity : 0.8857          
    ##          Pos Pred Value : 0.6364          
    ##          Neg Pred Value : 0.7949          
    ##              Prevalence : 0.3000          
    ##          Detection Rate : 0.1400          
    ##    Detection Prevalence : 0.2200          
    ##       Balanced Accuracy : 0.6762          
    ##                                           
    ##        'Positive' Class : yes             
    ##                                           
    ## ```

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-15.png" width="750px" />

    ## NULL

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-16.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-17.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-18.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-19.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-20.png" width="750px" />

    ## 
    ## 
    ## ### Generalized Linear Model (model_glm_basic_processing)
    ## 
    ## 
    ## 
    ## Pre-Processing: `nzv, center, scale, knnImpute`
    ## 
    ## 
    ## Call:
    ## NULL
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.5895  -0.8436   0.4207   0.7751   2.0452  
    ## 
    ## Coefficients:
    ##                                   Estimate Std. Error z value             Pr(>|z|)    
    ## (Intercept)                       1.147190   0.094081  12.194 < 0.0000000000000002 ***
    ## `checking_balance> 200 DM`        0.238392   0.090948   2.621             0.008763 ** 
    ## `checking_balance1 - 200 DM`      0.169782   0.093537   1.815             0.069503 .  
    ## checking_balanceunknown           0.864977   0.112309   7.702   0.0000000000000134 ***
    ## months_loan_duration             -0.243480   0.109843  -2.217             0.026649 *  
    ## credit_historygood               -0.296156   0.119650  -2.475             0.013317 *  
    ## credit_historypoor               -0.109460   0.090283  -1.212             0.225354    
    ## `credit_historyvery good`        -0.241627   0.089351  -2.704             0.006846 ** 
    ## purposecar                        0.034689   0.130295   0.266             0.790058    
    ## purposeeducation                 -0.088180   0.099152  -0.889             0.373819    
    ## `purposefurniture/appliances`     0.180758   0.135377   1.335             0.181803    
    ## amount                           -0.400086   0.120926  -3.309             0.000938 ***
    ## `savings_balance100 - 500 DM`     0.015658   0.085541   0.183             0.854758    
    ## `savings_balance500 - 1000 DM`    0.054567   0.098771   0.552             0.580635    
    ## savings_balanceunknown            0.329990   0.100635   3.279             0.001041 ** 
    ## `employment_duration> 7 years`    0.255840   0.126875   2.016             0.043749 *  
    ## `employment_duration1 - 4 years`  0.099657   0.111016   0.898             0.369353    
    ## `employment_duration4 - 7 years`  0.371148   0.112825   3.290             0.001003 ** 
    ## employment_durationunemployed     0.050138   0.097707   0.513             0.607846    
    ## percent_of_income                -0.383384   0.097775  -3.921   0.0000881507786376 ***
    ## years_at_residence                0.005518   0.095818   0.058             0.954075    
    ## age                               0.128240   0.104123   1.232             0.218093    
    ## other_creditnone                  0.187232   0.083529   2.242             0.024992 *  
    ## housingown                        0.151433   0.135406   1.118             0.263413    
    ## housingrent                      -0.084313   0.130666  -0.645             0.518761    
    ## existing_loans_count             -0.178839   0.108980  -1.641             0.100790    
    ## jobskilled                        0.007380   0.130238   0.057             0.954813    
    ## jobunskilled                      0.044701   0.131371   0.340             0.733655    
    ## dependents                       -0.041299   0.086738  -0.476             0.633977    
    ## phoneTRUE                         0.222771   0.099246   2.245             0.024791 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1099.56  on 899  degrees of freedom
    ## Residual deviance:  872.53  on 870  degrees of freedom
    ## AIC: 932.53
    ## 
    ## Number of Fisher Scoring iterations: 5

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-21.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-22.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-23.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-24.png" width="750px" />

    ## ```
    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction yes no
    ##        yes  14  8
    ##        no   16 62
    ##                                           
    ##                Accuracy : 0.76            
    ##                  95% CI : (0.6643, 0.8398)
    ##     No Information Rate : 0.7             
    ##     P-Value [Acc > NIR] : 0.1136          
    ##                                           
    ##                   Kappa : 0.3814          
    ##  Mcnemar's Test P-Value : 0.1530          
    ##                                           
    ##             Sensitivity : 0.4667          
    ##             Specificity : 0.8857          
    ##          Pos Pred Value : 0.6364          
    ##          Neg Pred Value : 0.7949          
    ##              Prevalence : 0.3000          
    ##          Detection Rate : 0.1400          
    ##    Detection Prevalence : 0.2200          
    ##       Balanced Accuracy : 0.6762          
    ##                                           
    ##        'Positive' Class : yes             
    ##                                           
    ## ```

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-25.png" width="750px" />

    ## NULL

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-26.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-27.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-28.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-29.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-30.png" width="750px" />

    ## 
    ## 
    ## ### Linear Discriminant Analysis (linear_discriminant_analsysis_remove_collinear)
    ## 
    ## 
    ## 
    ## Pre-Processing: `nzv, center, scale`
    ## 
    ##             Length Class      Mode     
    ## prior        2     -none-     numeric  
    ## counts       2     -none-     numeric  
    ## means       58     -none-     numeric  
    ## scaling     29     -none-     numeric  
    ## lev          2     -none-     character
    ## svd          1     -none-     numeric  
    ## N            1     -none-     numeric  
    ## call         3     -none-     call     
    ## xNames      29     -none-     character
    ## problemType  1     -none-     character
    ## tuneValue    1     data.frame list     
    ## obsLevels    2     -none-     character
    ## param        0     -none-     list

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-31.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-32.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-33.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-34.png" width="750px" />

    ## ```
    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction yes no
    ##        yes  13  8
    ##        no   17 62
    ##                                           
    ##                Accuracy : 0.75            
    ##                  95% CI : (0.6534, 0.8312)
    ##     No Information Rate : 0.7             
    ##     P-Value [Acc > NIR] : 0.1631          
    ##                                           
    ##                   Kappa : 0.349           
    ##  Mcnemar's Test P-Value : 0.1096          
    ##                                           
    ##             Sensitivity : 0.4333          
    ##             Specificity : 0.8857          
    ##          Pos Pred Value : 0.6190          
    ##          Neg Pred Value : 0.7848          
    ##              Prevalence : 0.3000          
    ##          Detection Rate : 0.1300          
    ##    Detection Prevalence : 0.2100          
    ##       Balanced Accuracy : 0.6595          
    ##                                           
    ##        'Positive' Class : yes             
    ##                                           
    ## ```

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-35.png" width="750px" />

    ## NULL

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-36.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-37.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-38.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-39.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-40.png" width="750px" />

    ## 
    ## 
    ## ### Linear Discriminant Analysis (linear_discriminant_analsysis)
    ## 
    ## 
    ## 
    ## Pre-Processing: `nzv, center, scale`
    ## 
    ##             Length Class      Mode     
    ## prior        2     -none-     numeric  
    ## counts       2     -none-     numeric  
    ## means       58     -none-     numeric  
    ## scaling     29     -none-     numeric  
    ## lev          2     -none-     character
    ## svd          1     -none-     numeric  
    ## N            1     -none-     numeric  
    ## call         3     -none-     call     
    ## xNames      29     -none-     character
    ## problemType  1     -none-     character
    ## tuneValue    1     data.frame list     
    ## obsLevels    2     -none-     character
    ## param        0     -none-     list

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-41.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-42.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-43.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-44.png" width="750px" />

    ## ```
    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction yes no
    ##        yes  13  8
    ##        no   17 62
    ##                                           
    ##                Accuracy : 0.75            
    ##                  95% CI : (0.6534, 0.8312)
    ##     No Information Rate : 0.7             
    ##     P-Value [Acc > NIR] : 0.1631          
    ##                                           
    ##                   Kappa : 0.349           
    ##  Mcnemar's Test P-Value : 0.1096          
    ##                                           
    ##             Sensitivity : 0.4333          
    ##             Specificity : 0.8857          
    ##          Pos Pred Value : 0.6190          
    ##          Neg Pred Value : 0.7848          
    ##              Prevalence : 0.3000          
    ##          Detection Rate : 0.1300          
    ##    Detection Prevalence : 0.2100          
    ##       Balanced Accuracy : 0.6595          
    ##                                           
    ##        'Positive' Class : yes             
    ##                                           
    ## ```

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-45.png" width="750px" />

    ## NULL

<img src="predictive_analysis_classification_files/figure-markdown_github/top_models-46.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-47.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-48.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-49.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-50.png" width="750px" /><img src="predictive_analysis_classification_files/figure-markdown_github/top_models-51.png" width="750px" />
