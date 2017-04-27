-   [Tuning Parameters](#tuning-parameters)
-   [Dataset](#dataset)
    -   [Summary](#summary)
    -   [Skewness](#skewness)
    -   [Outliers](#outliers)
    -   [Correlation & Collinearity](#correlation-collinearity)
        -   [Collinearity Removal](#collinearity-removal)
    -   [Graphs](#graphs)
        -   [cement](#cement)
        -   [slag](#slag)
        -   [ash](#ash)
        -   [water](#water)
        -   [superplastic](#superplastic)
        -   [coarseagg](#coarseagg)
        -   [fineagg](#fineagg)
        -   [age](#age)
        -   [random1](#random1)
        -   [random2](#random2)
-   [Spot-Check](#spot-check)
    -   [Linear Regression - no pre processing](#linear-regression---no-pre-processing)
    -   [Linear Regression - basic pre processing](#linear-regression---basic-pre-processing)
    -   [Linear Regression - basic pre processing with medianImpute](#linear-regression---basic-pre-processing-with-medianimpute)
    -   [Linear Regression - basic pre processing](#linear-regression---basic-pre-processing-1)
    -   [Linear Regression - skewness - YeoJohnson](#linear-regression---skewness---yeojohnson)
    -   [Linear Regression - skewness - BoxCox](#linear-regression---skewness---boxcox)
    -   [Linear Regression - remove collinear data - based on caret's recommendation](#linear-regression---remove-collinear-data---based-on-carets-recommendation)
    -   [Linear Regression - remove collinear data - based on calculation](#linear-regression---remove-collinear-data---based-on-calculation)
    -   [Robust Linear Regression](#robust-linear-regression)
    -   [Linear Regression - spatial sign](#linear-regression---spatial-sign)
    -   [Linear Regression - principal components analysis](#linear-regression---principal-components-analysis)
    -   [Principal Component Regression](#principal-component-regression)
    -   [Partial Least Squares](#partial-least-squares)
    -   [Ridge Regression](#ridge-regression)
    -   [ridge & lasso combo](#ridge-lasso-combo)
    -   [neural network - basic](#neural-network---basic)
    -   [neural network - basic - removing correlated predictors](#neural-network---basic---removing-correlated-predictors)
    -   [neural network - model averaging - removing correlated predictors](#neural-network---model-averaging---removing-correlated-predictors)
    -   [neural network - model averaging - PCA](#neural-network---model-averaging---pca)
    -   [MARS (Multivariate Adaptive Regression Splines)](#mars-multivariate-adaptive-regression-splines)
    -   [SVM - Support Vector Machine - Radial](#svm---support-vector-machine---radial)
    -   [SVM - Support Vector Machine - Linear](#svm---support-vector-machine---linear)
    -   [SVM - Support Vector Machine - Polynomial](#svm---support-vector-machine---polynomial)
    -   [CART - Classification and Regression Tree - Tuning over maximum depth](#cart---classification-and-regression-tree---tuning-over-maximum-depth)
    -   [CART - Classification and Regression Tree - Tuning over maximum depth](#cart---classification-and-regression-tree---tuning-over-maximum-depth-1)
    -   [Conditional Inference Tree](#conditional-inference-tree)
    -   [Conditional Inference Tree - Tuning over maximum depth](#conditional-inference-tree---tuning-over-maximum-depth)
    -   [Model Trees - M5](#model-trees---m5)
    -   [Model Trees - M5 Rules](#model-trees---m5-rules)
    -   [Bagged Trees](#bagged-trees)
    -   [Random Forest](#random-forest)
    -   [Random Forest - Conditional Inference](#random-forest---conditional-inference)
    -   [Boosting](#boosting)
    -   [Cubist](#cubist)
    -   [Resamples & Top Models](#resamples-top-models)
        -   [Resamples](#resamples)
        -   [Top Models](#top-models)
-   [Train Top Models on Entire Dataset & Predict on Test Set](#train-top-models-on-entire-dataset-predict-on-test-set)
    -   [ensemble\_boosting](#ensemble_boosting)
    -   [ensemble\_cubist](#ensemble_cubist)
    -   [ensemble\_random\_forest](#ensemble_random_forest)
    -   [nlm\_neur\_net\_averaging\_pca](#nlm_neur_net_averaging_pca)
    -   [nlm\_mars](#nlm_mars)

Tuning Parameters
=================

``` r
# train/test set
training_percentage <- 0.80

# cross validation
cross_validation_num_folds <- 10
cross_validation_num_repeats <- 3

# tuning
tuning_ridge_lambda <- seq(0, 0.1, length = 15) # Weight Decay
tuning_enet_lambda = c(0, 0.01, 0.1) # Weight Decay
tuning_enet_fraction <- seq(0.05, 1, length = 20) # Fraction of Full Solution

tuning_neural_network_decay <- c(0, 0.01, 0.1) # weight decay
tuning_neural_network_size <- c(1, 3, 5, 7, 9, 11, 13) # number of hidden units
tuning_neural_network_bag <- c(FALSE, TRUE) # bagging
parameter_neural_network_linout <- TRUE # use the linear relationship between the hidden units and the prediction (APM pg 162) i.e. linear output units
parameter_neural_network_trace <- FALSE
parameter_neural_network_max_num_weights <- 13 * (ncol(regression_dataset)) + 13 + 1 # The maximum allowable number of weights. There is no intrinsic limit in the code, but increasing MaxNWts will probably allow fits that are very slow and time-consuming.
parameter_neural_network_max_iterations <- 1000 # maximum number of iterations. Default 100

tuning_mars_degree <- 1:2 # Product Degree (of features that are added to the model)
tuning_mars_nprune <- 2:38 # number of terms retained

tuning_svm_poly_degree <- 1:2 # Polynomial Degree
tuning_svm_poly_scale <- c(0.01, 0.005, 0.001) # Scale
tuning_svm_cost <- 2^(-4:10) # Cost

tuning_ctree_mincriterion <- sort(c(0.95, seq(0.75, 0.99, length = 2)))

tuning_treebag_nbagg <- 25 # number of decision trees voting in the ensemble (default for some packages is 25)

parameter_random_forest_ntree <- 1000 # the number of bootstrap samples. althought he default is 500, at least 1,000 bootstrap samples should be used (and perhaps more depending on the number of predictors and the values of mtry).
tuning_random_forest_mtry <- unique(floor(seq(10, ncol(regression_dataset) - 1, length = 10))) # number of predictors that are randomly sampled as randidates for each split (default in regression is number of predictors divded by 3)
if(length(tuning_random_forest_mtry) == 1)
{
    tuning_random_forest_mtry <- c(round(tuning_random_forest_mtry / 2), tuning_random_forest_mtry - 1)
}

tuning_boosting_interaction_depth <- seq(1, 7, by = 2) # Boosting Iterations
tuning_boosting_n_trees <- seq(100, 1000, by = 50) # Max Tree Depth
tuning_boosting_shrinkage <- c(0.01, 0.1) # Shrinkage
tuning_boosting_min_obs_in_node <- 10 # Min. Terminal Node Size

tuning_cubist_committees <- c(1:10, 20, 50, 75, 100)
tuning_cubist_neighbors <- c(0, 1, 5, 9)
```

Dataset
=======

> Assumes the dataset has factors for strings; logical for TRUE/FALSE; `target` for outcome variable

Summary
-------

> Total predictors: `10`

> Total data-points/rows: `1030`

> Number of training data-points: `824`

Rule of thumbs for dimensions (Probabilistic and Statistical Modeling in Computer Science; pg 430):

> r &lt; sqrt(n); where r is the number of predictors and sqrt(n) is the square root of the sample size (`32`): `TRUE`

> r &lt; sqrt(n\_t); where r is the number of predictors and sqrt(n\_t) is the square root of the training set size (`29`): `TRUE`

    ##      target          cement           slag            ash             water        superplastic      coarseagg         fineagg           age          random1     random2   
    ##  Min.   : 2.33   Min.   :102.0   Min.   :  0.0   Min.   :  0.00   Min.   :121.8   Min.   : 0.000   Min.   : 801.0   Min.   :594.0   Min.   :  1.00   FALSE:506   FALSE:543  
    ##  1st Qu.:23.71   1st Qu.:192.4   1st Qu.:  0.0   1st Qu.:  0.00   1st Qu.:164.9   1st Qu.: 0.000   1st Qu.: 932.0   1st Qu.:731.0   1st Qu.:  7.00   TRUE :524   TRUE :487  
    ##  Median :34.45   Median :272.9   Median : 22.0   Median :  0.00   Median :185.0   Median : 6.400   Median : 968.0   Median :779.5   Median : 28.00                          
    ##  Mean   :35.82   Mean   :281.2   Mean   : 73.9   Mean   : 54.19   Mean   :181.6   Mean   : 6.205   Mean   : 972.9   Mean   :773.6   Mean   : 45.66                          
    ##  3rd Qu.:46.13   3rd Qu.:350.0   3rd Qu.:142.9   3rd Qu.:118.30   3rd Qu.:192.0   3rd Qu.:10.200   3rd Qu.:1029.4   3rd Qu.:824.0   3rd Qu.: 56.00                          
    ##  Max.   :82.60   Max.   :540.0   Max.   :359.4   Max.   :200.10   Max.   :247.0   Max.   :32.200   Max.   :1145.0   Max.   :992.6   Max.   :365.00

Skewness
--------

Note: `Box-Cox` can only be applied to sets (i.e. predictors) where all values are `> 0`. So some/most/all? `NA`s will be from that limiation.

| column       |  boxcox\_skewness|
|:-------------|-----------------:|
| target       |         0.4157636|
| cement       |         0.5079982|
| slag         |                NA|
| ash          |                NA|
| water        |         0.0744112|
| superplastic |                NA|
| coarseagg    |        -0.0401027|
| fineagg      |        -0.2522732|
| age          |         3.2596617|
| random1      |                NA|
| random2      |                NA|

Outliers
--------

| columns      | lower\_outlier\_count | upper\_outlier\_count |
|:-------------|:----------------------|:----------------------|
| target       | 0                     | 0                     |
| cement       | 0                     | 0                     |
| slag         | 0                     | 0                     |
| ash          | 0                     | 0                     |
| water        | 5                     | 2                     |
| superplastic | 0                     | 5                     |
| coarseagg    | 0                     | 0                     |
| fineagg      | 0                     | 0                     |
| age          | 0                     | 59                    |

Correlation & Collinearity
--------------------------

<img src="predictive_analysis_regression_files/figure-markdown_github/correlation-1.png" width="750px" />

### Collinearity Removal

#### Caret's `findCorrelation`

Shows caret's recommendation of removing collinear columns based on correlation threshold of `0.9`

> columns recommended for removal: \`\`

> final columns recommended: `target, cement, slag, ash, water, superplastic, coarseagg, fineagg, age, random1, random2`

#### Heuristic

This method is described in APM pg 47 as the following steps

-   calculate the correlation matrix of predictors
-   determine the two predictors associated with the largest absolute pairwise correlation (call them predictors `A` and `B`)
-   Determine the average correlation between `A` and the other variables.
    -   Do the same for `B`
-   If `A` has a larger average correlation, remove it; otherwise, remove predcitor `B`
-   Repeat until no absolute correlations are above the threshold (`0.9`)

> columns recommended for removal: \`\`

> final columns recommended: `target, random1, random2, cement, slag, ash, water, superplastic, coarseagg, fineagg, age`

Graphs
------

<img src="predictive_analysis_regression_files/figure-markdown_github/graphs-1.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-2.png" width="750px" />

### cement

<img src="predictive_analysis_regression_files/figure-markdown_github/graphs-3.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-4.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-5.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-6.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-7.png" width="750px" />

### slag

<img src="predictive_analysis_regression_files/figure-markdown_github/graphs-8.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-9.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-10.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-11.png" width="750px" />

Error with boxplot.

### ash

<img src="predictive_analysis_regression_files/figure-markdown_github/graphs-12.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-13.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-14.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-15.png" width="750px" />

Error with boxplot.

### water

<img src="predictive_analysis_regression_files/figure-markdown_github/graphs-16.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-17.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-18.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-19.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-20.png" width="750px" />

### superplastic

<img src="predictive_analysis_regression_files/figure-markdown_github/graphs-21.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-22.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-23.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-24.png" width="750px" />

Error with boxplot.

### coarseagg

<img src="predictive_analysis_regression_files/figure-markdown_github/graphs-25.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-26.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-27.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-28.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-29.png" width="750px" />

### fineagg

<img src="predictive_analysis_regression_files/figure-markdown_github/graphs-30.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-31.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-32.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-33.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-34.png" width="750px" />

### age

<img src="predictive_analysis_regression_files/figure-markdown_github/graphs-35.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-36.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-37.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-38.png" width="750px" />

Error with boxplot.

### random1

<img src="predictive_analysis_regression_files/figure-markdown_github/graphs-39.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-40.png" width="750px" />

### random2

<img src="predictive_analysis_regression_files/figure-markdown_github/graphs-41.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/graphs-42.png" width="750px" />

Spot-Check
==========

<img src="predictive_analysis_regression_files/figure-markdown_github/spot_check_prepare_numeric-1.png" width="750px" />

-   Note: e.g. if there are rare values at the target extremes (lows/highs), the train and especially the test set might not be training/testing on them. Is the test set representative? If the test set doesn't have as extreme values, it can even predict better (e.g. lower RMSE higher Rsquared) than the average Cross Validation given on training because it's not using those extreme values.

> used `80%` of data for `training` set (`826`), and `20%` for `test` set (`204`).

### Linear Regression - no pre processing

``` r
if(refresh_models)
{
    set.seed(seed)
    lm_no_pre_processing <- train(target ~ ., data = regression_train, method = 'lm', trControl = train_control)
    saveRDS(lm_no_pre_processing, file = './regression_data/lm_no_pre_processing.RDS')
} else{
    lm_no_pre_processing <- readRDS('./regression_data/lm_no_pre_processing.RDS')
}
summary(lm_no_pre_processing)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -27.882  -6.431   0.735   6.936  34.283 
    ## 
    ## Coefficients:
    ##                Estimate Std. Error t value             Pr(>|t|)    
    ## (Intercept)  -23.524282  28.495943  -0.826             0.409312    
    ## cement         0.118779   0.009166  12.958 < 0.0000000000000002 ***
    ## slag           0.105370   0.011037   9.547 < 0.0000000000000002 ***
    ## ash            0.091045   0.013620   6.685      0.0000000000429 ***
    ## water         -0.153178   0.043074  -3.556             0.000398 ***
    ## superplastic   0.252151   0.101542   2.483             0.013220 *  
    ## coarseagg      0.019197   0.010104   1.900             0.057802 .  
    ## fineagg        0.020103   0.011595   1.734             0.083349 .  
    ## age            0.114094   0.006113  18.664 < 0.0000000000000002 ***
    ## random1TRUE    0.290822   0.729389   0.399             0.690204    
    ## random2TRUE   -0.258564   0.732620  -0.353             0.724232    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 10.44 on 815 degrees of freedom
    ## Multiple R-squared:  0.6103, Adjusted R-squared:  0.6055 
    ## F-statistic: 127.6 on 10 and 815 DF,  p-value: < 0.00000000000000022

``` r
plot(lm_no_pre_processing$finalModel)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_no_pre_processing-1.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_no_pre_processing-2.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_no_pre_processing-3.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_no_pre_processing-4.png" width="750px" />

### Linear Regression - basic pre processing

``` r
if(refresh_models)
{
    set.seed(seed)
    lm_basic_pre_processing <- train(target ~ ., data = regression_train, method = 'lm', trControl = train_control, preProc=c('center', 'scale', 'knnImpute'))
    saveRDS(lm_basic_pre_processing, file = './regression_data/lm_basic_pre_processing.RDS')
} else{
    lm_basic_pre_processing <- readRDS('./regression_data/lm_basic_pre_processing.RDS')
}
summary(lm_basic_pre_processing)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -27.882  -6.431   0.735   6.936  34.283 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value             Pr(>|t|)    
    ## (Intercept)   35.6050     0.3632  98.027 < 0.0000000000000002 ***
    ## cement        12.5118     0.9655  12.958 < 0.0000000000000002 ***
    ## slag           9.0555     0.9485   9.547 < 0.0000000000000002 ***
    ## ash            5.8400     0.8736   6.685      0.0000000000429 ***
    ## water         -3.2206     0.9056  -3.556             0.000398 ***
    ## superplastic   1.5227     0.6132   2.483             0.013220 *  
    ## coarseagg      1.4982     0.7886   1.900             0.057802 .  
    ## fineagg        1.5839     0.9136   1.734             0.083349 .  
    ## age            7.1341     0.3822  18.664 < 0.0000000000000002 ***
    ## random1TRUE    0.1455     0.3648   0.399             0.690204    
    ## random2TRUE   -0.1291     0.3657  -0.353             0.724232    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 10.44 on 815 degrees of freedom
    ## Multiple R-squared:  0.6103, Adjusted R-squared:  0.6055 
    ## F-statistic: 127.6 on 10 and 815 DF,  p-value: < 0.00000000000000022

``` r
plot(lm_basic_pre_processing$finalModel)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_basic_pre_processing-1.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_basic_pre_processing-2.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_basic_pre_processing-3.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_basic_pre_processing-4.png" width="750px" />

### Linear Regression - basic pre processing with medianImpute

``` r
if(refresh_models)
{
    set.seed(seed)
    lm_median_impute <- train(target ~ ., data = regression_train, method = 'lm', trControl = train_control, preProc=c('center', 'scale', 'medianImpute'))
    saveRDS(lm_median_impute, file = './regression_data/lm_median_impute.RDS')
} else{
    lm_median_impute <- readRDS('./regression_data/lm_median_impute.RDS')
}
summary(lm_median_impute)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -27.882  -6.431   0.735   6.936  34.283 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value             Pr(>|t|)    
    ## (Intercept)   35.6050     0.3632  98.027 < 0.0000000000000002 ***
    ## cement        12.5118     0.9655  12.958 < 0.0000000000000002 ***
    ## slag           9.0555     0.9485   9.547 < 0.0000000000000002 ***
    ## ash            5.8400     0.8736   6.685      0.0000000000429 ***
    ## water         -3.2206     0.9056  -3.556             0.000398 ***
    ## superplastic   1.5227     0.6132   2.483             0.013220 *  
    ## coarseagg      1.4982     0.7886   1.900             0.057802 .  
    ## fineagg        1.5839     0.9136   1.734             0.083349 .  
    ## age            7.1341     0.3822  18.664 < 0.0000000000000002 ***
    ## random1TRUE    0.1455     0.3648   0.399             0.690204    
    ## random2TRUE   -0.1291     0.3657  -0.353             0.724232    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 10.44 on 815 degrees of freedom
    ## Multiple R-squared:  0.6103, Adjusted R-squared:  0.6055 
    ## F-statistic: 127.6 on 10 and 815 DF,  p-value: < 0.00000000000000022

``` r
plot(lm_median_impute$finalModel)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_impute-1.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_impute-2.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_impute-3.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_impute-4.png" width="750px" />

### Linear Regression - basic pre processing

``` r
if(refresh_models)
{
    set.seed(seed)
    lm_near_zero_variance <- train(target ~ ., data = regression_train, method = 'lm', trControl = train_control, preProc=c('nzv', 'center', 'scale', 'knnImpute')) # APM pg 550
    saveRDS(lm_near_zero_variance, file = './regression_data/lm_near_zero_variance.RDS')
} else{
    lm_near_zero_variance <- readRDS('./regression_data/lm_near_zero_variance.RDS')
}
summary(lm_near_zero_variance)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -27.882  -6.431   0.735   6.936  34.283 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value             Pr(>|t|)    
    ## (Intercept)   35.6050     0.3632  98.027 < 0.0000000000000002 ***
    ## cement        12.5118     0.9655  12.958 < 0.0000000000000002 ***
    ## slag           9.0555     0.9485   9.547 < 0.0000000000000002 ***
    ## ash            5.8400     0.8736   6.685      0.0000000000429 ***
    ## water         -3.2206     0.9056  -3.556             0.000398 ***
    ## superplastic   1.5227     0.6132   2.483             0.013220 *  
    ## coarseagg      1.4982     0.7886   1.900             0.057802 .  
    ## fineagg        1.5839     0.9136   1.734             0.083349 .  
    ## age            7.1341     0.3822  18.664 < 0.0000000000000002 ***
    ## random1TRUE    0.1455     0.3648   0.399             0.690204    
    ## random2TRUE   -0.1291     0.3657  -0.353             0.724232    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 10.44 on 815 degrees of freedom
    ## Multiple R-squared:  0.6103, Adjusted R-squared:  0.6055 
    ## F-statistic: 127.6 on 10 and 815 DF,  p-value: < 0.00000000000000022

``` r
plot(lm_near_zero_variance$finalModel)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_basic-1.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_basic-2.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_basic-3.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_basic-4.png" width="750px" />

### Linear Regression - skewness - YeoJohnson

``` r
if(refresh_models)
{
    set.seed(seed)
    lm_skewness_y <- train(target ~ ., data = regression_train, method = 'lm', trControl = train_control, preProc=c('YeoJohnson', 'center', 'scale', 'knnImpute')) # (caret docs: 'The Yeo-Johnson transformation is similar to the Box-Cox model but can accommodate predictors with zero and/or negative values (while the predictors values for the Box-Cox transformation must be strictly positive.) ')
    saveRDS(lm_skewness_y, file = './regression_data/lm_skewness_y.RDS')
} else{
    lm_skewness_y <- readRDS('./regression_data/lm_skewness_y.RDS')
}
summary(lm_skewness_y)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -21.9000  -4.7851   0.1101   4.4590  26.8109 
    ## 
    ## Coefficients:
    ##                 Estimate  Std. Error t value             Pr(>|t|)    
    ## (Intercept)  35.60497579  0.25605085 139.054 < 0.0000000000000002 ***
    ## cement        9.07110625  0.40922811  22.166 < 0.0000000000000002 ***
    ## slag          5.45715642  0.42840696  12.738 < 0.0000000000000002 ***
    ## ash           1.07145671  0.41888601   2.558             0.010711 *  
    ## water        -5.05962119  0.51171433  -9.888 < 0.0000000000000002 ***
    ## superplastic  1.74322992  0.48030663   3.629             0.000302 ***
    ## coarseagg    -0.32859819  0.43075814  -0.763             0.445781    
    ## fineagg      -1.22625817  0.44103594  -2.780             0.005554 ** 
    ## age          10.01404491  0.26101831  38.365 < 0.0000000000000002 ***
    ## random1TRUE  -0.00002926  0.25721843   0.000             0.999909    
    ## random2TRUE  -0.14107014  0.25748691  -0.548             0.583929    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 7.359 on 815 degrees of freedom
    ## Multiple R-squared:  0.8063, Adjusted R-squared:  0.804 
    ## F-statistic: 339.3 on 10 and 815 DF,  p-value: < 0.00000000000000022

``` r
plot(lm_skewness_y$finalModel)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_YeoJohnson-1.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_YeoJohnson-2.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_YeoJohnson-3.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_YeoJohnson-4.png" width="750px" />

### Linear Regression - skewness - BoxCox

``` r
if(refresh_models)
{
    set.seed(seed)
    lm_skewness_bc <- train(target ~ ., data = regression_train, method = 'lm', trControl = train_control, preProc=c('BoxCox', 'center', 'scale', 'knnImpute')) # (caret docs: 'The Yeo-Johnson transformation is similar to the Box-Cox model but can accommodate predictors with zero and/or negative values (while the predictors values for the Box-Cox transformation must be strictly positive.) ')
    saveRDS(lm_skewness_bc, file = './regression_data/lm_skewness_bc.RDS')
} else{
    lm_skewness_bc <- readRDS('./regression_data/lm_skewness_bc.RDS')
}
summary(lm_skewness_bc)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -20.9589  -4.2744  -0.2363   3.8977  27.4090 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value             Pr(>|t|)    
    ## (Intercept)  35.60498    0.24605 144.704 < 0.0000000000000002 ***
    ## cement       13.17055    0.61037  21.578 < 0.0000000000000002 ***
    ## slag          9.31782    0.61333  15.192 < 0.0000000000000002 ***
    ## ash           5.37323    0.56737   9.470 < 0.0000000000000002 ***
    ## water        -3.96743    0.57629  -6.884      0.0000000000116 ***
    ## superplastic  0.17385    0.41601   0.418               0.6761    
    ## coarseagg     1.23272    0.50238   2.454               0.0143 *  
    ## fineagg       1.65975    0.57186   2.902               0.0038 ** 
    ## age          10.35398    0.25163  41.148 < 0.0000000000000002 ***
    ## random1TRUE  -0.08115    0.24724  -0.328               0.7428    
    ## random2TRUE   0.09622    0.24736   0.389               0.6974    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 7.072 on 815 degrees of freedom
    ## Multiple R-squared:  0.8212, Adjusted R-squared:  0.819 
    ## F-statistic: 374.2 on 10 and 815 DF,  p-value: < 0.00000000000000022

``` r
plot(lm_skewness_bc$finalModel)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_BoxCox-1.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_BoxCox-2.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_BoxCox-3.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_BoxCox-4.png" width="750px" />

### Linear Regression - remove collinear data - based on caret's recommendation

``` r
if(refresh_models)
{
    set.seed(seed)
    lm_remove_collinearity_caret <- train(target ~ ., data = regression_train[, recommended_columns_caret], method = 'lm', trControl = train_control, preProc=c('nzv', 'center', 'scale', 'knnImpute')) # APM pg 550
    saveRDS(lm_remove_collinearity_caret, file = './regression_data/lm_remove_collinearity_caret.RDS')
} else{
    lm_remove_collinearity_caret <- readRDS('./regression_data/lm_remove_collinearity_caret.RDS')
}
summary(lm_remove_collinearity_caret)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -27.882  -6.431   0.735   6.936  34.283 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value             Pr(>|t|)    
    ## (Intercept)   35.6050     0.3632  98.027 < 0.0000000000000002 ***
    ## cement        12.5118     0.9655  12.958 < 0.0000000000000002 ***
    ## slag           9.0555     0.9485   9.547 < 0.0000000000000002 ***
    ## ash            5.8400     0.8736   6.685      0.0000000000429 ***
    ## water         -3.2206     0.9056  -3.556             0.000398 ***
    ## superplastic   1.5227     0.6132   2.483             0.013220 *  
    ## coarseagg      1.4982     0.7886   1.900             0.057802 .  
    ## fineagg        1.5839     0.9136   1.734             0.083349 .  
    ## age            7.1341     0.3822  18.664 < 0.0000000000000002 ***
    ## random1TRUE    0.1455     0.3648   0.399             0.690204    
    ## random2TRUE   -0.1291     0.3657  -0.353             0.724232    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 10.44 on 815 degrees of freedom
    ## Multiple R-squared:  0.6103, Adjusted R-squared:  0.6055 
    ## F-statistic: 127.6 on 10 and 815 DF,  p-value: < 0.00000000000000022

``` r
plot(lm_remove_collinearity_caret$finalModel)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_collinear_caret-1.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_collinear_caret-2.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_collinear_caret-3.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_collinear_caret-4.png" width="750px" />

### Linear Regression - remove collinear data - based on calculation

``` r
if(refresh_models)
{
    set.seed(seed)
    lm_remove_collinearity_custom <- train(target ~ ., data = regression_train[, recommended_columns_custom], method = 'lm', trControl = train_control, preProc=c('nzv', 'center', 'scale', 'knnImpute')) # APM pg 550
    saveRDS(lm_remove_collinearity_custom, file = './regression_data/lm_remove_collinearity_custom.RDS')
} else{
    lm_remove_collinearity_custom <- readRDS('./regression_data/lm_remove_collinearity_custom.RDS')
}
summary(lm_remove_collinearity_custom)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -27.882  -6.431   0.735   6.936  34.283 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value             Pr(>|t|)    
    ## (Intercept)   35.6050     0.3632  98.027 < 0.0000000000000002 ***
    ## random1TRUE    0.1455     0.3648   0.399             0.690204    
    ## random2TRUE   -0.1291     0.3657  -0.353             0.724232    
    ## cement        12.5118     0.9655  12.958 < 0.0000000000000002 ***
    ## slag           9.0555     0.9485   9.547 < 0.0000000000000002 ***
    ## ash            5.8400     0.8736   6.685      0.0000000000429 ***
    ## water         -3.2206     0.9056  -3.556             0.000398 ***
    ## superplastic   1.5227     0.6132   2.483             0.013220 *  
    ## coarseagg      1.4982     0.7886   1.900             0.057802 .  
    ## fineagg        1.5839     0.9136   1.734             0.083349 .  
    ## age            7.1341     0.3822  18.664 < 0.0000000000000002 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 10.44 on 815 degrees of freedom
    ## Multiple R-squared:  0.6103, Adjusted R-squared:  0.6055 
    ## F-statistic: 127.6 on 10 and 815 DF,  p-value: < 0.00000000000000022

``` r
plot(lm_remove_collinearity_custom$finalModel)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_collinear_calc-1.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_collinear_calc-2.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_collinear_calc-3.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_collinear_calc-4.png" width="750px" />

### Robust Linear Regression

``` r
if(refresh_models)
{
    set.seed(seed)
    lm_robust <- train(target ~ ., data = regression_train, method = 'rlm', trControl = train_control, preProc=c('YeoJohnson', 'center', 'scale', 'knnImpute', 'pca')) # by default uses the 'Huber' approach. (APM pg 130.) 'The Huber function uses the squared residuals when they are 'small' and the simple difference between the observed and predicted values values when the residuals are above a threshold.' (APM pg 109f)
    saveRDS(lm_robust, file = './regression_data/lm_robust.RDS')
} else{
    lm_robust <- readRDS('./regression_data/lm_robust.RDS')
}
summary(lm_robust)
```

    ## 
    ## Call: rlm(formula = .outcome ~ ., data = dat)
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -22.785  -4.727   0.157   4.611  28.796 
    ## 
    ## Coefficients:
    ##             Value    Std. Error t value 
    ## (Intercept)  35.6294   0.2686   132.6594
    ## PC1          -1.0231   0.1815    -5.6385
    ## PC2           2.9418   0.2237    13.1521
    ## PC3          -4.9719   0.2451   -20.2848
    ## PC4           8.1779   0.2516    32.5051
    ## PC5          -5.6748   0.2650   -21.4169
    ## PC6           2.8722   0.2724    10.5441
    ## PC7           3.3483   0.2742    12.2106
    ## PC8           7.3719   0.3000    24.5692
    ## 
    ## Residual standard error: 6.91 on 817 degrees of freedom

``` r
plot(lm_robust$finalModel)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/robust_linear-1.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/robust_linear-2.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/robust_linear-3.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/robust_linear-4.png" width="750px" />

### Linear Regression - spatial sign

``` r
if(refresh_models)
{
    set.seed(seed)
    lm_spatial_sign <- train(target ~ ., data = regression_train, method = 'lm', trControl = train_control, preProc=c('center', 'scale', 'knnImpute', 'spatialSign'))
    saveRDS(lm_spatial_sign, file = './regression_data/lm_spatial_sign.RDS')
} else{
    lm_spatial_sign <- readRDS('./regression_data/lm_spatial_sign.RDS')
}
summary(lm_spatial_sign)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -31.190  -5.417   0.541   6.153  34.370 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value             Pr(>|t|)    
    ## (Intercept)   37.2957     0.3383 110.237 < 0.0000000000000002 ***
    ## cement        37.4566     2.7175  13.783 < 0.0000000000000002 ***
    ## slag          25.0943     2.6437   9.492 < 0.0000000000000002 ***
    ## ash           13.7834     2.3924   5.761         0.0000000118 ***
    ## water         -8.6842     2.5269  -3.437             0.000619 ***
    ## superplastic   8.4372     1.8926   4.458         0.0000094308 ***
    ## coarseagg      4.7899     2.1551   2.223             0.026519 *  
    ## fineagg        4.5265     2.5285   1.790             0.073798 .  
    ## age           32.3782     1.4260  22.706 < 0.0000000000000002 ***
    ## random1TRUE    0.4679     0.9478   0.494             0.621703    
    ## random2TRUE   -0.1009     0.9510  -0.106             0.915559    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 9.613 on 815 degrees of freedom
    ## Multiple R-squared:  0.6696, Adjusted R-squared:  0.6655 
    ## F-statistic: 165.1 on 10 and 815 DF,  p-value: < 0.00000000000000022

``` r
plot(lm_spatial_sign$finalModel)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_spatial_sign-1.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_spatial_sign-2.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_spatial_sign-3.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_spatial_sign-4.png" width="750px" />

### Linear Regression - principal components analysis

``` r
if(refresh_models)
{
    set.seed(seed)
    lm_pca <- train(target ~ ., data = regression_train, method = 'lm', trControl = train_control, preProc=c('YeoJohnson', 'center', 'scale', 'knnImpute', 'pca')) # (APM pg 37 'to help PCA avoid summarizing distributional differences and predictor scale informatino, it is best to first transform skewed predictors and then center and scale the predictors prior to performing PCA. Centering and scaling enables PCA to find the underlying relationships in the data without being influenced by the original measurement scales.')
    saveRDS(lm_pca, file = './regression_data/lm_pca.RDS')
} else{
    lm_pca <- readRDS('./regression_data/lm_pca.RDS')
}
summary(lm_pca)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -22.556  -4.848   0.162   4.730  28.033 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value             Pr(>|t|)    
    ## (Intercept)  35.6050     0.2682 132.764 < 0.0000000000000002 ***
    ## PC1          -1.0340     0.1812  -5.707         0.0000000161 ***
    ## PC2           2.8851     0.2233  12.918 < 0.0000000000000002 ***
    ## PC3          -5.2914     0.2447 -21.620 < 0.0000000000000002 ***
    ## PC4           8.0245     0.2512  31.942 < 0.0000000000000002 ***
    ## PC5          -5.6012     0.2646 -21.170 < 0.0000000000000002 ***
    ## PC6           2.8076     0.2720  10.322 < 0.0000000000000002 ***
    ## PC7           3.5493     0.2738  12.963 < 0.0000000000000002 ***
    ## PC8           7.3946     0.2996  24.681 < 0.0000000000000002 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 7.708 on 817 degrees of freedom
    ## Multiple R-squared:  0.787,  Adjusted R-squared:  0.7849 
    ## F-statistic: 377.4 on 8 and 817 DF,  p-value: < 0.00000000000000022

``` r
plot(lm_pca$finalModel)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_PCA-1.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_PCA-2.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_PCA-3.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/linear_regression_PCA-4.png" width="750px" />

### Principal Component Regression

``` r
if(refresh_models)
{
    set.seed(seed)
    lm_pcr <- train(target ~ ., data = regression_train, method = 'pcr', trControl = train_control, tuneLength = 20, preProc = c('center', 'scale', 'knnImpute'))
    saveRDS(lm_pcr, file = './regression_data/lm_pcr.RDS')
} else{
    lm_pcr <- readRDS('./regression_data/lm_pcr.RDS')
}
summary(lm_pcr)
```

    ##               Length Class      Mode     
    ## coefficients    90   -none-     numeric  
    ## scores        7434   scores     numeric  
    ## loadings        90   loadings   numeric  
    ## Yloadings        9   loadings   numeric  
    ## projection      90   -none-     numeric  
    ## Xmeans          10   -none-     numeric  
    ## Ymeans           1   -none-     numeric  
    ## fitted.values 7434   -none-     numeric  
    ## residuals     7434   -none-     numeric  
    ## Xvar             9   -none-     numeric  
    ## Xtotvar          1   -none-     numeric  
    ## fit.time         1   -none-     numeric  
    ## ncomp            1   -none-     numeric  
    ## method           1   -none-     character
    ## call             4   -none-     call     
    ## terms            3   terms      call     
    ## model           11   data.frame list     
    ## xNames          10   -none-     character
    ## problemType      1   -none-     character
    ## tuneValue        1   data.frame list     
    ## obsLevels        1   -none-     logical

``` r
plot(lm_pcr)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/principal_component_PCR-1.png" width="750px" />

### Partial Least Squares

``` r
if(refresh_models)
{
    set.seed(seed)
    lm_pls <- train(target ~ ., data = regression_train, method = 'pls', trControl = train_control, preProc = c('center', 'scale', 'knnImpute'), tuneLength = 14)
    saveRDS(lm_pls, file = './regression_data/lm_pls.RDS')
} else{
    lm_pls <- readRDS('./regression_data/lm_pls.RDS')
}
summary(lm_pls)
```

    ##                 Length Class      Mode     
    ## coefficients      80   -none-     numeric  
    ## scores          6608   scores     numeric  
    ## loadings          80   loadings   numeric  
    ## loading.weights   80   loadings   numeric  
    ## Yscores         6608   scores     numeric  
    ## Yloadings          8   loadings   numeric  
    ## projection        80   -none-     numeric  
    ## Xmeans            10   -none-     numeric  
    ## Ymeans             1   -none-     numeric  
    ## fitted.values   6608   -none-     numeric  
    ## residuals       6608   -none-     numeric  
    ## Xvar               8   -none-     numeric  
    ## Xtotvar            1   -none-     numeric  
    ## fit.time           1   -none-     numeric  
    ## ncomp              1   -none-     numeric  
    ## method             1   -none-     character
    ## call               5   -none-     call     
    ## terms              3   terms      call     
    ## model             11   data.frame list     
    ## xNames            10   -none-     character
    ## problemType        1   -none-     character
    ## tuneValue          1   data.frame list     
    ## obsLevels          1   -none-     logical  
    ## bestIter           1   data.frame list

``` r
# loadings(lm_pls$finalModel)
# scores(lm_pls$finalModel)
plot(lm_pls)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/partial_least_squares-1.png" width="750px" />

``` r
# Compare # of components of PCR vs. PLS
xyplot(RMSE ~ ncomp, data = rbind(lm_pls$results %>% mutate(Model = 'PLS'), lm_pcr$results %>% mutate(Model = 'PCR')), xlab = '# Components', ylab = 'RMSE (Cross-Validation)', auto.key = list(columns = 2), groups = Model, type = c('o', 'g'))
```

<img src="predictive_analysis_regression_files/figure-markdown_github/partial_least_squares-2.png" width="750px" />

### Ridge Regression

``` r
if(refresh_models)
{
    set.seed(seed)
    lm_ridge <- train(target ~ ., data = regression_train, method = 'ridge', tuneGrid = expand.grid(lambda = tuning_ridge_lambda) , trControl = train_control, preProc = c('center', 'scale', 'knnImpute'))
    saveRDS(lm_ridge, file = './regression_data/lm_ridge.RDS')
} else{
    lm_ridge <- readRDS('./regression_data/lm_ridge.RDS')
}
summary(lm_ridge)
```

    ##             Length Class      Mode     
    ## call          4    -none-     call     
    ## actions      13    -none-     list     
    ## allset       10    -none-     numeric  
    ## beta.pure   130    -none-     numeric  
    ## vn           10    -none-     character
    ## mu            1    -none-     numeric  
    ## normx        10    -none-     numeric  
    ## meanx        10    -none-     numeric  
    ## lambda        1    -none-     numeric  
    ## L1norm       13    -none-     numeric  
    ## penalty      13    -none-     numeric  
    ## df           13    -none-     numeric  
    ## Cp           13    -none-     numeric  
    ## sigma2        1    -none-     numeric  
    ## xNames       10    -none-     character
    ## problemType   1    -none-     character
    ## tuneValue     1    data.frame list     
    ## obsLevels     1    -none-     logical

``` r
plot(lm_ridge)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/ridge_regression-1.png" width="750px" />

### ridge & lasso combo

``` r
if(refresh_models)
{
    set.seed(seed)
    lm_enet <- train(target ~ ., data = regression_train, method = 'enet', tuneGrid = expand.grid(lambda = tuning_enet_lambda, fraction = tuning_enet_fraction), trControl = train_control, preProc = c('center', 'scale', 'knnImpute'))
    saveRDS(lm_enet, file = './regression_data/lm_enet.RDS')
} else{
    lm_enet <- readRDS('./regression_data/lm_enet.RDS')
}
#lm_enet$finalModel
summary(lm_enet)
```

    ##             Length Class      Mode     
    ## call          4    -none-     call     
    ## actions      13    -none-     list     
    ## allset       10    -none-     numeric  
    ## beta.pure   130    -none-     numeric  
    ## vn           10    -none-     character
    ## mu            1    -none-     numeric  
    ## normx        10    -none-     numeric  
    ## meanx        10    -none-     numeric  
    ## lambda        1    -none-     numeric  
    ## L1norm       13    -none-     numeric  
    ## penalty      13    -none-     numeric  
    ## df           13    -none-     numeric  
    ## Cp           13    -none-     numeric  
    ## sigma2        1    -none-     numeric  
    ## xNames       10    -none-     character
    ## problemType   1    -none-     character
    ## tuneValue     2    data.frame list     
    ## obsLevels     1    -none-     logical

``` r
plot(lm_enet)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/ridge_lasso-1.png" width="750px" />

### neural network - basic

### neural network - basic - removing correlated predictors

### neural network - model averaging - removing correlated predictors

### neural network - model averaging - PCA

``` r
if(refresh_models)
{
    set.seed(seed)
    nlm_neur_net_averaging_pca <- train(target ~ ., data = regression_train, method = 'avNNet',
                            tuneGrid = expand.grid(decay = tuning_neural_network_decay, size = tuning_neural_network_size, bag = tuning_neural_network_bag),
                            trControl = train_control,
                            preProc = c('nzv', 'YeoJohnson', 'center', 'scale', 'knnImpute', 'pca'),
                            linout = parameter_neural_network_linout,
                            trace = parameter_neural_network_trace,
                            MaxNWts = parameter_neural_network_max_num_weights,
                            maxit = parameter_neural_network_max_iterations)
    saveRDS(nlm_neur_net_averaging_pca, file = './regression_data/nlm_neur_net_averaging_pca.RDS')
} else{
    nlm_neur_net_averaging_pca <- readRDS('./regression_data/nlm_neur_net_averaging_pca.RDS')
}
summary(nlm_neur_net_averaging_pca$finalModel)
```

    ##             Length Class      Mode     
    ## model       5      -none-     list     
    ## repeats     1      -none-     numeric  
    ## bag         1      -none-     logical  
    ## seeds       5      -none-     numeric  
    ## names       8      -none-     character
    ## terms       3      terms      call     
    ## coefnames   8      -none-     character
    ## xlevels     0      -none-     list     
    ## xNames      8      -none-     character
    ## problemType 1      -none-     character
    ## tuneValue   3      data.frame list     
    ## obsLevels   1      -none-     logical

``` r
plot(nlm_neur_net_averaging_pca)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/neural_network_averaging_PCA-1.png" width="750px" />

### MARS (Multivariate Adaptive Regression Splines)

``` r
if(refresh_models)
{
    set.seed(seed)
    nlm_mars <- train(target ~ ., data = regression_train, method = 'earth', tuneGrid = expand.grid(degree = tuning_mars_degree, nprune = tuning_mars_nprune), trControl = train_control)
    saveRDS(nlm_mars, file = './regression_data/nlm_mars.RDS')
} else{
    nlm_mars <- readRDS('./regression_data/nlm_mars.RDS')
}
#nlm_mars$finalModel
summary(nlm_mars$finalModel)
```

    ##                   Length Class      Mode     
    ## rss                   1  -none-     numeric  
    ## rsq                   1  -none-     numeric  
    ## gcv                   1  -none-     numeric  
    ## grsq                  1  -none-     numeric  
    ## bx                15694  -none-     numeric  
    ## dirs                200  -none-     numeric  
    ## cuts                200  -none-     numeric  
    ## selected.terms       19  -none-     numeric  
    ## prune.terms         400  -none-     numeric  
    ## fitted.values       826  -none-     numeric  
    ## residuals           826  -none-     numeric  
    ## coefficients         19  -none-     numeric  
    ## rss.per.response      1  -none-     numeric  
    ## rsq.per.response      1  -none-     numeric  
    ## gcv.per.response      1  -none-     numeric  
    ## grsq.per.response     1  -none-     numeric  
    ## rss.per.subset       20  -none-     numeric  
    ## gcv.per.subset       20  -none-     numeric  
    ## leverages           826  -none-     numeric  
    ## pmethod               1  -none-     character
    ## nprune                1  -none-     numeric  
    ## penalty               1  -none-     numeric  
    ## nk                    1  -none-     numeric  
    ## thresh                1  -none-     numeric  
    ## termcond              1  -none-     numeric  
    ## weights               0  -none-     NULL     
    ## call                  6  -none-     call     
    ## namesx.org           10  -none-     character
    ## namesx               10  -none-     character
    ## x                  8260  -none-     numeric  
    ## y                   826  -none-     numeric  
    ## xNames               10  -none-     character
    ## problemType           1  -none-     character
    ## tuneValue             2  data.frame list     
    ## obsLevels             1  -none-     logical

``` r
plot(nlm_mars)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/mars-1.png" width="750px" />

``` r
require(earth)
```

    ## Loading required package: earth

    ## Loading required package: plotmo

    ## Loading required package: plotrix

    ## 
    ## Attaching package: 'plotrix'

    ## The following object is masked from 'package:scales':
    ## 
    ##     rescale

    ## The following object is masked from 'package:psych':
    ## 
    ##     rescale

    ## Loading required package: TeachingDemos

    ## 
    ## Attaching package: 'TeachingDemos'

    ## The following object is masked _by_ '.GlobalEnv':
    ## 
    ##     outliers

    ## The following objects are masked from 'package:Hmisc':
    ## 
    ##     cnvrt.coords, subplot

``` r
plotmo(nlm_mars$finalModel)
```

    ##  plotmo grid:    cement slag ash water superplastic coarseagg fineagg age random1TRUE random2TRUE
    ##                   275.1   20   0 185.7          6.1       968   780.1  28           1           0

<img src="predictive_analysis_regression_files/figure-markdown_github/mars-2.png" width="750px" />

``` r
#marsImp <- varImp(nlm_mars, scale = FALSE)
#plot(marsImp, top = 25)
```

### SVM - Support Vector Machine - Radial

``` r
if(refresh_models)
{
    set.seed(seed)
    nlm_svm_radial <- train(target ~ ., data = regression_train, method = 'svmRadial', preProc = c('center', 'scale', 'knnImpute'), tuneLength = 14, trControl = train_control) # tuneLength tunes C (sigma is chosen automatically)
    saveRDS(nlm_svm_radial, file = './regression_data/nlm_svm_radial.RDS')
} else{
    nlm_svm_radial <- readRDS('./regression_data/nlm_svm_radial.RDS')
}
nlm_svm_radial$finalModel
```

    ## Support Vector Machine object of class "ksvm" 
    ## 
    ## SV type: eps-svr  (regression) 
    ##  parameter : epsilon = 0.1  cost C = 16 
    ## 
    ## Gaussian Radial Basis kernel function. 
    ##  Hyperparameter : sigma =  0.0756661229831825 
    ## 
    ## Number of Support Vectors : 607 
    ## 
    ## Objective Function Value : -1441.983 
    ## Training error : 0.049895

``` r
plot(nlm_svm_radial, scales = list(x = list(log = 2)))
```

<img src="predictive_analysis_regression_files/figure-markdown_github/svm_radial-1.png" width="750px" />

### SVM - Support Vector Machine - Linear

``` r
if(refresh_models)
{
    set.seed(seed)
    nlm_svm_linear <- train(target ~ ., data = regression_train, method = 'svmLinear', preProc = c('center', 'scale', 'knnImpute'), tuneGrid = data.frame(C=tuning_svm_cost), trControl = train_control)
    saveRDS(nlm_svm_linear, file = './regression_data/nlm_svm_linear.RDS')
} else{
    nlm_svm_linear <- readRDS('./regression_data/nlm_svm_linear.RDS')
}
nlm_svm_linear$finalModel
```

    ## Support Vector Machine object of class "ksvm" 
    ## 
    ## SV type: eps-svr  (regression) 
    ##  parameter : epsilon = 0.1  cost C = 0.0625 
    ## 
    ## Linear (vanilla) kernel function. 
    ## 
    ## Number of Support Vectors : 721 
    ## 
    ## Objective Function Value : -21.1551 
    ## Training error : 0.415165

``` r
plot(nlm_svm_linear, scales = list(x = list(log = 2)))
```

<img src="predictive_analysis_regression_files/figure-markdown_github/svm_linear-1.png" width="750px" />

### SVM - Support Vector Machine - Polynomial

``` r
if(refresh_models)
{
    set.seed(seed)
    nlm_svm_poly <- train(target ~ ., data = regression_train, method = 'svmPoly', preProc = c('center', 'scale', 'knnImpute'), tuneGrid = expand.grid(degree = tuning_svm_poly_degree, scale = tuning_svm_poly_scale, C = tuning_svm_cost), trControl = train_control)
    saveRDS(nlm_svm_poly, file = './regression_data/nlm_svm_poly.RDS')
} else{
    nlm_svm_poly <- readRDS('./regression_data/nlm_svm_poly.RDS')
}
nlm_svm_poly$finalModel
```

    ## Support Vector Machine object of class "ksvm" 
    ## 
    ## SV type: eps-svr  (regression) 
    ##  parameter : epsilon = 0.1  cost C = 16 
    ## 
    ## Polynomial kernel function. 
    ##  Hyperparameters : degree =  2  scale =  0.01  offset =  1 
    ## 
    ## Number of Support Vectors : 660 
    ## 
    ## Objective Function Value : -3607.035 
    ## Training error : 0.213308

``` r
plot(nlm_svm_poly, scales = list(x = list(log = 2),  between = list(x = .5, y = 1)))
```

<img src="predictive_analysis_regression_files/figure-markdown_github/svm_poly-1.png" width="750px" />

### CART - Classification and Regression Tree - Tuning over maximum depth

``` r
if(refresh_models)
{
    set.seed(seed)
    tree_cart <- train(target ~ ., data = regression_train, method = 'rpart', tuneLength = 25, trControl = train_control) # tuneLength tunes `cp` (Complexity Parameter)
    saveRDS(tree_cart, file = './regression_data/tree_cart.RDS')
} else{
    tree_cart <- readRDS('./regression_data/tree_cart.RDS')
}
tree_cart$finalModel
```

    ## n= 826 
    ## 
    ## node), split, n, deviance, yval
    ##       * denotes terminal node
    ## 
    ##   1) root 826 227892.8000 35.60498  
    ##     2) age< 21 264  41973.5400 23.28186  
    ##       4) cement< 354.5 187  15073.1200 18.29551  
    ##         8) age< 10.5 142   7464.1190 15.32063  
    ##          16) superplastic< 8 118   3544.2550 13.67864 *
    ##          17) superplastic>=8 24   2037.5160 23.39375  
    ##            34) ash>=60.35 16    398.4923 18.66250 *
    ##            35) ash< 60.35 8    564.5572 32.85625 *
    ##         9) age>=10.5 45   2386.7830 27.68289  
    ##          18) superplastic< 7.95 26    483.4598 23.53346 *
    ##          19) superplastic>=7.95 19    843.0722 33.36105 *
    ##       5) cement>=354.5 77  10959.2600 35.39156  
    ##        10) water>=183.4 30   3733.9160 28.21033  
    ##          20) cement< 396.5 13    528.9561 19.54923 *
    ##          21) cement>=396.5 17   1484.0350 34.83353 *
    ##        11) water< 183.4 47   4690.7340 39.97532  
    ##          22) age< 5 25    891.4632 32.95080 *
    ##          23) age>=5 22   1163.8590 47.95773 *
    ##     3) age>=21 562 126995.7000 41.39377  
    ##       6) cement< 352.5 433  71901.1000 37.15898  
    ##        12) cement< 164.8 98   8669.9670 25.83520  
    ##          24) slag< 115.3 28    375.9911 14.45750 *
    ##          25) slag>=115.3 70   3219.4510 30.38629 *
    ##        13) cement>=164.8 335  46988.6600 40.47161  
    ##          26) water>=176 217  20892.1600 36.52152  
    ##            52) slag< 13 115   6013.0390 31.65322  
    ##             104) coarseagg>=1086.5 12    159.7631 19.76417 *
    ##             105) coarseagg< 1086.5 103   3959.4670 33.03835 *
    ##            53) slag>=13 102   9080.6630 42.01029  
    ##             106) cement< 294 74   4825.7090 39.16689  
    ##               212) age< 42 50   2534.6270 36.27100 *
    ##               213) age>=42 24    998.2114 45.20000 *
    ##             107) cement>=294 28   2075.4860 49.52500 *
    ##          27) water< 176 118  16483.9900 47.73576  
    ##            54) slag< 47.65 69   5731.4790 42.27319  
    ##             108) cement< 203.35 26   1358.4060 35.97154 *
    ##             109) cement>=203.35 43   2716.3020 46.08349 *
    ##            55) slag>=47.65 49   5794.2470 55.42796  
    ##             110) water>=162.5 31   2141.4850 50.85032  
    ##               220) cement< 311.65 24   1081.1300 47.98333 *
    ##               221) cement>=311.65 7    186.7258 60.68000 *
    ##             111) water< 162.5 18   1884.4130 63.31167 *
    ##       7) cement>=352.5 129  21264.9600 55.60822  
    ##        14) water>=183.05 55   5928.0170 46.05655  
    ##          28) cement< 477.5 39   3426.4250 42.45154 *
    ##          29) cement>=477.5 16    759.3060 54.84375 *
    ##        15) water< 183.05 74   6589.5320 62.70743  
    ##          30) slag< 170.1 62   4544.0740 60.61387  
    ##            60) superplastic>=3.85 53   3256.8960 59.04170 *
    ##            61) superplastic< 3.85 9    384.7234 69.87222 *
    ##          31) slag>=170.1 12    369.6911 73.52417 *

``` r
plot(tree_cart) # Plot the tuning results
```

<img src="predictive_analysis_regression_files/figure-markdown_github/cart-1.png" width="750px" />

``` r
party_tree <- as.party(tree_cart$finalModel)
plot(party_tree)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/cart-2.png" width="750px" />

### CART - Classification and Regression Tree - Tuning over maximum depth

``` r
if(refresh_models)
{
    set.seed(seed)
    tree_cart2 <- train(target ~ ., data = regression_train, method = 'rpart2', tuneLength = 25, trControl = train_control) # tuneLength tunes `maxdepth` (Max Tree Depth)
    saveRDS(tree_cart2, file = './regression_data/tree_cart2.RDS')
} else{
    tree_cart2 <- readRDS('./regression_data/tree_cart2.RDS')
}
tree_cart2$finalModel
```

    ## n= 826 
    ## 
    ## node), split, n, deviance, yval
    ##       * denotes terminal node
    ## 
    ##  1) root 826 227892.8000 35.60498  
    ##    2) age< 21 264  41973.5400 23.28186  
    ##      4) cement< 354.5 187  15073.1200 18.29551  
    ##        8) age< 10.5 142   7464.1190 15.32063 *
    ##        9) age>=10.5 45   2386.7830 27.68289 *
    ##      5) cement>=354.5 77  10959.2600 35.39156  
    ##       10) water>=183.4 30   3733.9160 28.21033 *
    ##       11) water< 183.4 47   4690.7340 39.97532  
    ##         22) age< 5 25    891.4632 32.95080 *
    ##         23) age>=5 22   1163.8590 47.95773 *
    ##    3) age>=21 562 126995.7000 41.39377  
    ##      6) cement< 352.5 433  71901.1000 37.15898  
    ##       12) cement< 164.8 98   8669.9670 25.83520  
    ##         24) slag< 115.3 28    375.9911 14.45750 *
    ##         25) slag>=115.3 70   3219.4510 30.38629 *
    ##       13) cement>=164.8 335  46988.6600 40.47161  
    ##         26) water>=176 217  20892.1600 36.52152  
    ##           52) slag< 13 115   6013.0390 31.65322 *
    ##           53) slag>=13 102   9080.6630 42.01029 *
    ##         27) water< 176 118  16483.9900 47.73576  
    ##           54) slag< 47.65 69   5731.4790 42.27319 *
    ##           55) slag>=47.65 49   5794.2470 55.42796 *
    ##      7) cement>=352.5 129  21264.9600 55.60822  
    ##       14) water>=183.05 55   5928.0170 46.05655 *
    ##       15) water< 183.05 74   6589.5320 62.70743 *

``` r
plot(tree_cart2)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/cart2-1.png" width="750px" />

``` r
party_tree <- as.party(tree_cart2$finalModel)
plot(party_tree)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/cart2-2.png" width="750px" />

### Conditional Inference Tree

``` r
if(refresh_models)
{
    set.seed(seed)
    tree_cond_inference <- train(target ~ ., data = regression_train, method = 'ctree', tuneGrid = data.frame(mincriterion = tuning_ctree_mincriterion), trControl = train_control)
    saveRDS(tree_cond_inference, file = './regression_data/tree_cond_inference.RDS')
} else{
    tree_cond_inference <- readRDS('./regression_data/tree_cond_inference.RDS')
}
tree_cond_inference$finalModel
```

    ## 
    ##   Conditional inference tree with 51 terminal nodes
    ## 
    ## Response:  .outcome 
    ## Inputs:  cement, slag, ash, water, superplastic, coarseagg, fineagg, age, random1TRUE, random2TRUE 
    ## Number of observations:  826 
    ## 
    ## 1) cement <= 350; criterion = 1, statistic = 192.037
    ##   2) age <= 7; criterion = 1, statistic = 98.204
    ##     3) cement <= 255; criterion = 1, statistic = 35.892
    ##       4) cement <= 153; criterion = 0.999, statistic = 15.606
    ##         5)*  weights = 15 
    ##       4) cement > 153
    ##         6) fineagg <= 694.1; criterion = 0.954, statistic = 7.999
    ##           7)*  weights = 7 
    ##         6) fineagg > 694.1
    ##           8) superplastic <= 6.4; criterion = 0.999, statistic = 14.766
    ##             9)*  weights = 49 
    ##           8) superplastic > 6.4
    ##             10)*  weights = 21 
    ##     3) cement > 255
    ##       11) slag <= 15; criterion = 1, statistic = 22.361
    ##         12) water <= 186; criterion = 0.964, statistic = 8.714
    ##           13)*  weights = 12 
    ##         12) water > 186
    ##           14) age <= 3; criterion = 0.834, statistic = 5.603
    ##             15)*  weights = 13 
    ##           14) age > 3
    ##             16)*  weights = 10 
    ##       11) slag > 15
    ##         17)*  weights = 15 
    ##   2) age > 7
    ##     18) cement <= 164.6; criterion = 1, statistic = 66.154
    ##       19) slag <= 114.6; criterion = 1, statistic = 65.907
    ##         20)*  weights = 28 
    ##       19) slag > 114.6
    ##         21) slag <= 183.9; criterion = 0.998, statistic = 14.284
    ##           22) ash <= 116; criterion = 0.979, statistic = 9.463
    ##             23)*  weights = 27 
    ##           22) ash > 116
    ##             24)*  weights = 14 
    ##         21) slag > 183.9
    ##           25) coarseagg <= 1001.8; criterion = 0.849, statistic = 11.285
    ##             26)*  weights = 20 
    ##           25) coarseagg > 1001.8
    ##             27)*  weights = 9 
    ##     18) cement > 164.6
    ##       28) slag <= 13.6; criterion = 1, statistic = 70.125
    ##         29) superplastic <= 8.7; criterion = 1, statistic = 46.977
    ##           30) age <= 28; criterion = 1, statistic = 20.001
    ##             31) cement <= 296; criterion = 0.999, statistic = 16.425
    ##               32) superplastic <= 1.7; criterion = 1, statistic = 18.085
    ##                 33)*  weights = 19 
    ##               32) superplastic > 1.7
    ##                 34) age <= 14; criterion = 0.902, statistic = 6.588
    ##                   35)*  weights = 19 
    ##                 34) age > 14
    ##                   36)*  weights = 14 
    ##             31) cement > 296
    ##               37)*  weights = 29 
    ##           30) age > 28
    ##             38) water <= 192.9; criterion = 1, statistic = 17.457
    ##               39) superplastic <= 6.4; criterion = 0.972, statistic = 8.942
    ##                 40)*  weights = 32 
    ##               39) superplastic > 6.4
    ##                 41)*  weights = 14 
    ##             38) water > 192.9
    ##               42) fineagg <= 812; criterion = 0.958, statistic = 8.168
    ##                 43)*  weights = 13 
    ##               42) fineagg > 812
    ##                 44)*  weights = 7 
    ##         29) superplastic > 8.7
    ##           45) age <= 28; criterion = 1, statistic = 29.987
    ##             46) water <= 160.6; criterion = 0.917, statistic = 6.901
    ##               47)*  weights = 19 
    ##             46) water > 160.6
    ##               48) cement <= 304.8; criterion = 0.926, statistic = 7.104
    ##                 49)*  weights = 18 
    ##               48) cement > 304.8
    ##                 50)*  weights = 7 
    ##           45) age > 28
    ##             51)*  weights = 22 
    ##       28) slag > 13.6
    ##         52) cement <= 273; criterion = 1, statistic = 23.949
    ##           53) age <= 28; criterion = 0.996, statistic = 12.314
    ##             54)*  weights = 54 
    ##           53) age > 28
    ##             55) superplastic <= 5.7; criterion = 0.994, statistic = 11.829
    ##               56)*  weights = 20 
    ##             55) superplastic > 5.7
    ##               57)*  weights = 22 
    ##         52) cement > 273
    ##           58) water <= 162; criterion = 0.971, statistic = 8.855
    ##             59)*  weights = 13 
    ##           58) water > 162
    ##             60) slag <= 145; criterion = 0.888, statistic = 6.344
    ##               61) slag <= 137.2; criterion = 0.845, statistic = 5.724
    ##                 62) age <= 28; criterion = 0.983, statistic = 9.868
    ##                   63)*  weights = 17 
    ##                 62) age > 28
    ##                   64)*  weights = 7 
    ##               61) slag > 137.2
    ##                 65)*  weights = 13 
    ##             60) slag > 145
    ##               66) superplastic <= 8.5; criterion = 0.984, statistic = 9.911
    ##                 67)*  weights = 12 
    ##               66) superplastic > 8.5
    ##                 68)*  weights = 9 
    ## 1) cement > 350
    ##   69) slag <= 97.1; criterion = 1, statistic = 26.881
    ##     70) cement <= 516; criterion = 1, statistic = 18.361
    ##       71) superplastic <= 0; criterion = 1, statistic = 16.795
    ##         72) age <= 14; criterion = 0.999, statistic = 14.415
    ##           73) age <= 3; criterion = 0.914, statistic = 6.834
    ##             74)*  weights = 7 
    ##           73) age > 3
    ##             75)*  weights = 14 
    ##         72) age > 14
    ##           76) cement <= 475; criterion = 0.921, statistic = 7.001
    ##             77) age <= 56; criterion = 0.967, statistic = 8.583
    ##               78)*  weights = 17 
    ##             77) age > 56
    ##               79)*  weights = 15 
    ##           76) cement > 475
    ##             80)*  weights = 8 
    ##       71) superplastic > 0
    ##         81) age <= 7; criterion = 1, statistic = 36.168
    ##           82) age <= 3; criterion = 0.997, statistic = 13.237
    ##             83)*  weights = 13 
    ##           82) age > 3
    ##             84)*  weights = 12 
    ##         81) age > 7
    ##           85) age <= 28; criterion = 0.951, statistic = 7.857
    ##             86) ash <= 97; criterion = 0.762, statistic = 4.901
    ##               87)*  weights = 18 
    ##             86) ash > 97
    ##               88)*  weights = 9 
    ##           85) age > 28
    ##             89)*  weights = 15 
    ##     70) cement > 516
    ##       90) age <= 14; criterion = 0.894, statistic = 6.446
    ##         91)*  weights = 8 
    ##       90) age > 14
    ##         92)*  weights = 15 
    ##   69) slag > 97.1
    ##     93) age <= 3; criterion = 1, statistic = 25.647
    ##       94)*  weights = 13 
    ##     93) age > 3
    ##       95) age <= 7; criterion = 0.994, statistic = 11.737
    ##         96)*  weights = 10 
    ##       95) age > 7
    ##         97) slag <= 187; criterion = 0.939, statistic = 7.462
    ##           98) coarseagg <= 852.1; criterion = 0.993, statistic = 11.465
    ##             99)*  weights = 13 
    ##           98) coarseagg > 852.1
    ##             100)*  weights = 7 
    ##         97) slag > 187
    ##           101)*  weights = 12

``` r
plot(tree_cond_inference)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/conditional_inference-1.png" width="750px" />

``` r
plot(tree_cond_inference$finalModel)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/conditional_inference-2.png" width="750px" />

### Conditional Inference Tree - Tuning over maximum depth

``` r
if(refresh_models)
{
    set.seed(seed)
    tree_cond_inference2 <- train(target ~ ., data = regression_train, method = 'ctree2', tuneLength = 25, trControl = train_control) # tuneLength tunes `maxdepth` (Max Tree Depth)
    saveRDS(tree_cond_inference2, file = './regression_data/tree_cond_inference2.RDS')
} else{
    tree_cond_inference2 <- readRDS('./regression_data/tree_cond_inference2.RDS')
}
tree_cond_inference2$finalModel
```

    ## 
    ##   Conditional inference tree with 66 terminal nodes
    ## 
    ## Response:  .outcome 
    ## Inputs:  cement, slag, ash, water, superplastic, coarseagg, fineagg, age, random1TRUE, random2TRUE 
    ## Number of observations:  826 
    ## 
    ## 1) cement <= 350; criterion = 1, statistic = 192.037
    ##   2) age <= 7; criterion = 1, statistic = 98.204
    ##     3) cement <= 255; criterion = 1, statistic = 35.892
    ##       4) cement <= 153; criterion = 0.999, statistic = 15.606
    ##         5)*  weights = 15 
    ##       4) cement > 153
    ##         6) fineagg <= 694.1; criterion = 0.954, statistic = 7.999
    ##           7)*  weights = 7 
    ##         6) fineagg > 694.1
    ##           8) superplastic <= 6.4; criterion = 0.999, statistic = 14.766
    ##             9) fineagg <= 757.7; criterion = 0.596, statistic = 3.825
    ##               10)*  weights = 14 
    ##             9) fineagg > 757.7
    ##               11) ash <= 118.3; criterion = 0.187, statistic = 2.029
    ##                 12) age <= 3; criterion = 0.844, statistic = 5.717
    ##                   13)*  weights = 11 
    ##                 12) age > 3
    ##                   14)*  weights = 17 
    ##               11) ash > 118.3
    ##                 15)*  weights = 7 
    ##           8) superplastic > 6.4
    ##             16) water <= 164.8; criterion = 0.747, statistic = 4.779
    ##               17)*  weights = 12 
    ##             16) water > 164.8
    ##               18)*  weights = 9 
    ##     3) cement > 255
    ##       19) slag <= 15; criterion = 1, statistic = 22.361
    ##         20) water <= 186; criterion = 0.964, statistic = 8.714
    ##           21)*  weights = 12 
    ##         20) water > 186
    ##           22) age <= 3; criterion = 0.834, statistic = 5.603
    ##             23)*  weights = 13 
    ##           22) age > 3
    ##             24)*  weights = 10 
    ##       19) slag > 15
    ##         25)*  weights = 15 
    ##   2) age > 7
    ##     26) cement <= 164.6; criterion = 1, statistic = 66.154
    ##       27) slag <= 114.6; criterion = 1, statistic = 65.907
    ##         28)*  weights = 28 
    ##       27) slag > 114.6
    ##         29) slag <= 183.9; criterion = 0.998, statistic = 14.284
    ##           30) ash <= 116; criterion = 0.979, statistic = 9.463
    ##             31) coarseagg <= 958.2; criterion = 0.364, statistic = 2.767
    ##               32)*  weights = 18 
    ##             31) coarseagg > 958.2
    ##               33)*  weights = 9 
    ##           30) ash > 116
    ##             34)*  weights = 14 
    ##         29) slag > 183.9
    ##           35) coarseagg <= 1001.8; criterion = 0.849, statistic = 11.285
    ##             36) cement <= 152.6; criterion = 0.078, statistic = 9.272
    ##               37)*  weights = 10 
    ##             36) cement > 152.6
    ##               38)*  weights = 10 
    ##           35) coarseagg > 1001.8
    ##             39)*  weights = 9 
    ##     26) cement > 164.6
    ##       40) slag <= 13.6; criterion = 1, statistic = 70.125
    ##         41) superplastic <= 8.7; criterion = 1, statistic = 46.977
    ##           42) age <= 28; criterion = 1, statistic = 20.001
    ##             43) cement <= 296; criterion = 0.999, statistic = 16.425
    ##               44) superplastic <= 1.7; criterion = 1, statistic = 18.085
    ##                 45)*  weights = 19 
    ##               44) superplastic > 1.7
    ##                 46) age <= 14; criterion = 0.902, statistic = 6.588
    ##                   47)*  weights = 19 
    ##                 46) age > 14
    ##                   48)*  weights = 14 
    ##             43) cement > 296
    ##               49)*  weights = 29 
    ##           42) age > 28
    ##             50) water <= 192.9; criterion = 1, statistic = 17.457
    ##               51) superplastic <= 6.4; criterion = 0.972, statistic = 8.942
    ##                 52) superplastic <= 4.5; criterion = 0.157, statistic = 1.893
    ##                   53) cement <= 310; criterion = 0.188, statistic = 2.032
    ##                     54)*  weights = 13 
    ##                   53) cement > 310
    ##                     55)*  weights = 9 
    ##                 52) superplastic > 4.5
    ##                   56)*  weights = 10 
    ##               51) superplastic > 6.4
    ##                 57)*  weights = 14 
    ##             50) water > 192.9
    ##               58) fineagg <= 812; criterion = 0.958, statistic = 8.168
    ##                 59)*  weights = 13 
    ##               58) fineagg > 812
    ##                 60)*  weights = 7 
    ##         41) superplastic > 8.7
    ##           61) age <= 28; criterion = 1, statistic = 29.987
    ##             62) water <= 160.6; criterion = 0.917, statistic = 6.901
    ##               63)*  weights = 19 
    ##             62) water > 160.6
    ##               64) cement <= 304.8; criterion = 0.926, statistic = 7.104
    ##                 65)*  weights = 18 
    ##               64) cement > 304.8
    ##                 66)*  weights = 7 
    ##           61) age > 28
    ##             67) age <= 56; criterion = 0.488, statistic = 3.3
    ##               68)*  weights = 10 
    ##             67) age > 56
    ##               69)*  weights = 12 
    ##       40) slag > 13.6
    ##         70) cement <= 273; criterion = 1, statistic = 23.949
    ##           71) age <= 28; criterion = 0.996, statistic = 12.314
    ##             72) age <= 14; criterion = 0.442, statistic = 3.097
    ##               73)*  weights = 7 
    ##             72) age > 14
    ##               74) superplastic <= 6.5; criterion = 0.879, statistic = 6.198
    ##                 75) slag <= 181.9; criterion = 0.74, statistic = 4.73
    ##                   76) cement <= 238.2; criterion = 0.946, statistic = 7.694
    ##                     77)*  weights = 15 
    ##                   76) cement > 238.2
    ##                     78)*  weights = 10 
    ##                 75) slag > 181.9
    ##                   79)*  weights = 9 
    ##               74) superplastic > 6.5
    ##                 80)*  weights = 13 
    ##           71) age > 28
    ##             81) superplastic <= 5.7; criterion = 0.994, statistic = 11.829
    ##               82) fineagg <= 749.1; criterion = 0.321, statistic = 2.592
    ##                 83)*  weights = 12 
    ##               82) fineagg > 749.1
    ##                 84)*  weights = 8 
    ##             81) superplastic > 5.7
    ##               85) water <= 164.8; criterion = 0.694, statistic = 4.404
    ##                 86)*  weights = 10 
    ##               85) water > 164.8
    ##                 87)*  weights = 12 
    ##         70) cement > 273
    ##           88) water <= 162; criterion = 0.971, statistic = 8.855
    ##             89)*  weights = 13 
    ##           88) water > 162
    ##             90) slag <= 145; criterion = 0.888, statistic = 6.344
    ##               91) slag <= 137.2; criterion = 0.845, statistic = 5.724
    ##                 92) age <= 28; criterion = 0.983, statistic = 9.868
    ##                   93)*  weights = 17 
    ##                 92) age > 28
    ##                   94)*  weights = 7 
    ##               91) slag > 137.2
    ##                 95)*  weights = 13 
    ##             90) slag > 145
    ##               96) superplastic <= 8.5; criterion = 0.984, statistic = 9.911
    ##                 97)*  weights = 12 
    ##               96) superplastic > 8.5
    ##                 98)*  weights = 9 
    ## 1) cement > 350
    ##   99) slag <= 97.1; criterion = 1, statistic = 26.881
    ##     100) cement <= 516; criterion = 1, statistic = 18.361
    ##       101) superplastic <= 0; criterion = 1, statistic = 16.795
    ##         102) age <= 14; criterion = 0.999, statistic = 14.415
    ##           103) age <= 3; criterion = 0.914, statistic = 6.834
    ##             104)*  weights = 7 
    ##           103) age > 3
    ##             105)*  weights = 14 
    ##         102) age > 14
    ##           106) cement <= 475; criterion = 0.921, statistic = 7.001
    ##             107) age <= 56; criterion = 0.967, statistic = 8.583
    ##               108)*  weights = 17 
    ##             107) age > 56
    ##               109)*  weights = 15 
    ##           106) cement > 475
    ##             110)*  weights = 8 
    ##       101) superplastic > 0
    ##         111) age <= 7; criterion = 1, statistic = 36.168
    ##           112) age <= 3; criterion = 0.997, statistic = 13.237
    ##             113)*  weights = 13 
    ##           112) age > 3
    ##             114)*  weights = 12 
    ##         111) age > 7
    ##           115) age <= 28; criterion = 0.951, statistic = 7.857
    ##             116) ash <= 97; criterion = 0.762, statistic = 4.901
    ##               117)*  weights = 18 
    ##             116) ash > 97
    ##               118)*  weights = 9 
    ##           115) age > 28
    ##             119)*  weights = 15 
    ##     100) cement > 516
    ##       120) age <= 14; criterion = 0.894, statistic = 6.446
    ##         121)*  weights = 8 
    ##       120) age > 14
    ##         122)*  weights = 15 
    ##   99) slag > 97.1
    ##     123) age <= 3; criterion = 1, statistic = 25.647
    ##       124)*  weights = 13 
    ##     123) age > 3
    ##       125) age <= 7; criterion = 0.994, statistic = 11.737
    ##         126)*  weights = 10 
    ##       125) age > 7
    ##         127) slag <= 187; criterion = 0.939, statistic = 7.462
    ##           128) coarseagg <= 852.1; criterion = 0.993, statistic = 11.465
    ##             129)*  weights = 13 
    ##           128) coarseagg > 852.1
    ##             130)*  weights = 7 
    ##         127) slag > 187
    ##           131)*  weights = 12

``` r
plot(tree_cond_inference2) # Plot the tuning results
```

<img src="predictive_analysis_regression_files/figure-markdown_github/conditional_inference2-1.png" width="750px" />

``` r
plot(tree_cond_inference2$finalModel)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/conditional_inference2-2.png" width="750px" />

### Model Trees - M5

> This model is failing (not only for me, try in the future, seems like a problem in RWeka): <https://github.com/topepo/caret/issues/618>

### Model Trees - M5 Rules

> This model is failing (not only for me, try in the future, seems like a problem in RWeka): <https://github.com/topepo/caret/issues/618>

### Bagged Trees

``` r
if(refresh_models)
{
    set.seed(seed)
    ensemble_bagged_tree <- train(target ~ ., data = regression_train, method = 'treebag', nbagg = tuning_treebag_nbagg, trControl = train_control)
    saveRDS(ensemble_bagged_tree, file = './regression_data/ensemble_bagged_tree.RDS')
} else{
    ensemble_bagged_tree <- readRDS('./regression_data/ensemble_bagged_tree.RDS')
}
```

    ## Warning: namespace 'ipred' is not available and has been replaced
    ## by .GlobalEnv when processing object 'terminal'

``` r
ensemble_bagged_tree
```

    ## Bagged CART 
    ## 
    ## 826 samples
    ##  10 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold, repeated 3 times) 
    ## Summary of sample sizes: 742, 745, 743, 743, 743, 742, ... 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared 
    ##   7.609265  0.8008812

``` r
summary(ensemble_bagged_tree)
```

    ##             Length Class      Mode     
    ## y           826    -none-     numeric  
    ## X             0    -none-     NULL     
    ## mtrees       25    -none-     list     
    ## OOB           1    -none-     logical  
    ## comb          1    -none-     logical  
    ## xNames       10    -none-     character
    ## problemType   1    -none-     character
    ## tuneValue     1    data.frame list     
    ## obsLevels     1    -none-     logical

### Random Forest

``` r
if(refresh_models)
{
    set.seed(seed)
    ensemble_random_forest <- train(target ~ ., data = regression_train, method = 'rf', tuneGrid = data.frame(mtry = tuning_random_forest_mtry), ntree = parameter_random_forest_ntree, trControl = train_control, preProc=c('knnImpute'))
    saveRDS(ensemble_random_forest, file = './regression_data/ensemble_random_forest.RDS')
} else{
    ensemble_random_forest <- readRDS('./regression_data/ensemble_random_forest.RDS')
}
ensemble_random_forest
```

    ## Random Forest 
    ## 
    ## 826 samples
    ##  10 predictor
    ## 
    ## Pre-processing: nearest neighbor imputation (10), centered (10), scaled (10) 
    ## Resampling: Cross-Validated (10 fold, repeated 3 times) 
    ## Summary of sample sizes: 742, 745, 743, 743, 743, 742, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE      Rsquared 
    ##   5     4.967075  0.9177331
    ##   9     4.966603  0.9143650
    ## 
    ## RMSE was used to select the optimal model using  the smallest value.
    ## The final value used for the model was mtry = 9.

``` r
#ensemble_random_forest$finalModel
summary(ensemble_random_forest$finalModel)
```

    ##                 Length Class      Mode     
    ## call               6   -none-     call     
    ## type               1   -none-     character
    ## predicted        826   -none-     numeric  
    ## mse             1000   -none-     numeric  
    ## rsq             1000   -none-     numeric  
    ## oob.times        826   -none-     numeric  
    ## importance        20   -none-     numeric  
    ## importanceSD      10   -none-     numeric  
    ## localImportance    0   -none-     NULL     
    ## proximity          0   -none-     NULL     
    ## ntree              1   -none-     numeric  
    ## mtry               1   -none-     numeric  
    ## forest            11   -none-     list     
    ## coefs              0   -none-     NULL     
    ## y                826   -none-     numeric  
    ## test               0   -none-     NULL     
    ## inbag              0   -none-     NULL     
    ## xNames            10   -none-     character
    ## problemType        1   -none-     character
    ## tuneValue          1   data.frame list     
    ## obsLevels          1   -none-     logical

``` r
plot(ensemble_random_forest)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/random_forest-1.png" width="750px" />

### Random Forest - Conditional Inference

### Boosting

``` r
if(refresh_models)
{
    set.seed(seed)
    ensemble_boosting <- train(target ~ ., data = regression_train, method = 'gbm', tuneGrid = expand.grid(interaction.depth = tuning_boosting_interaction_depth, n.trees = tuning_boosting_n_trees, shrinkage = tuning_boosting_shrinkage, n.minobsinnode = tuning_boosting_min_obs_in_node), trControl = train_control, verbose = FALSE)
    saveRDS(ensemble_boosting, file = './regression_data/ensemble_boosting.RDS')
} else{
    ensemble_boosting <- readRDS('./regression_data/ensemble_boosting.RDS')
}
# ensemble_boosting
# ensemble_boosting$finalModel
summary(ensemble_boosting)
```

    ##                   Length Class      Mode     
    ## initF               1    -none-     numeric  
    ## fit               826    -none-     numeric  
    ## train.error       950    -none-     numeric  
    ## valid.error       950    -none-     numeric  
    ## oobag.improve     950    -none-     numeric  
    ## trees             950    -none-     list     
    ## c.splits            0    -none-     list     
    ## bag.fraction        1    -none-     numeric  
    ## distribution        1    -none-     list     
    ## interaction.depth   1    -none-     numeric  
    ## n.minobsinnode      1    -none-     numeric  
    ## num.classes         1    -none-     numeric  
    ## n.trees             1    -none-     numeric  
    ## nTrain              1    -none-     numeric  
    ## train.fraction      1    -none-     numeric  
    ## response.name       1    -none-     character
    ## shrinkage           1    -none-     numeric  
    ## var.levels         10    -none-     list     
    ## var.monotone       10    -none-     numeric  
    ## var.names          10    -none-     character
    ## var.type           10    -none-     numeric  
    ## verbose             1    -none-     logical  
    ## data                6    -none-     list     
    ## xNames             10    -none-     character
    ## problemType         1    -none-     character
    ## tuneValue           4    data.frame list     
    ## obsLevels           1    -none-     logical

``` r
summary(ensemble_boosting$finalModel)
```

    ##                   Length Class      Mode     
    ## initF               1    -none-     numeric  
    ## fit               826    -none-     numeric  
    ## train.error       950    -none-     numeric  
    ## valid.error       950    -none-     numeric  
    ## oobag.improve     950    -none-     numeric  
    ## trees             950    -none-     list     
    ## c.splits            0    -none-     list     
    ## bag.fraction        1    -none-     numeric  
    ## distribution        1    -none-     list     
    ## interaction.depth   1    -none-     numeric  
    ## n.minobsinnode      1    -none-     numeric  
    ## num.classes         1    -none-     numeric  
    ## n.trees             1    -none-     numeric  
    ## nTrain              1    -none-     numeric  
    ## train.fraction      1    -none-     numeric  
    ## response.name       1    -none-     character
    ## shrinkage           1    -none-     numeric  
    ## var.levels         10    -none-     list     
    ## var.monotone       10    -none-     numeric  
    ## var.names          10    -none-     character
    ## var.type           10    -none-     numeric  
    ## verbose             1    -none-     logical  
    ## data                6    -none-     list     
    ## xNames             10    -none-     character
    ## problemType         1    -none-     character
    ## tuneValue           4    data.frame list     
    ## obsLevels           1    -none-     logical

``` r
plot(ensemble_boosting)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/boosting-1.png" width="750px" />

### Cubist

``` r
if(refresh_models)
{
    set.seed(seed)
    ensemble_cubist <- train(target ~ ., data = regression_train, 'cubist', tuneGrid = expand.grid(committees = tuning_cubist_committees, neighbors = tuning_cubist_neighbors), trControl = train_control)
    saveRDS(ensemble_cubist, file = './regression_data/ensemble_cubist.RDS')
} else{
    ensemble_cubist <- readRDS('./regression_data/ensemble_cubist.RDS')
}
# ensemble_cubist
# ensemble_cubist$finalModel
summary(ensemble_cubist)
```

    ##              Length Class      Mode     
    ## data          1     -none-     character
    ## names         1     -none-     character
    ## model         1     -none-     character
    ## output        1     -none-     character
    ## control       6     -none-     list     
    ## committees    1     -none-     numeric  
    ## maxd          1     -none-     numeric  
    ## dims          2     -none-     numeric  
    ## splits        8     data.frame list     
    ## usage         3     data.frame list     
    ## call          4     -none-     call     
    ## coefficients 13     data.frame list     
    ## vars          2     -none-     list     
    ## tuneValue     2     data.frame list     
    ## xNames       10     -none-     character
    ## problemType   1     -none-     character
    ## obsLevels     1     -none-     logical

``` r
summary(ensemble_cubist$finalModel)
```

    ##              Length Class      Mode     
    ## data          1     -none-     character
    ## names         1     -none-     character
    ## model         1     -none-     character
    ## output        1     -none-     character
    ## control       6     -none-     list     
    ## committees    1     -none-     numeric  
    ## maxd          1     -none-     numeric  
    ## dims          2     -none-     numeric  
    ## splits        8     data.frame list     
    ## usage         3     data.frame list     
    ## call          4     -none-     call     
    ## coefficients 13     data.frame list     
    ## vars          2     -none-     list     
    ## tuneValue     2     data.frame list     
    ## xNames       10     -none-     character
    ## problemType   1     -none-     character
    ## obsLevels     1     -none-     logical

``` r
plot(ensemble_cubist)
```

<img src="predictive_analysis_regression_files/figure-markdown_github/cubist-1.png" width="750px" />

``` r
plot(ensemble_cubist, auto.key = list(columns = 4, lines = TRUE))
```

<img src="predictive_analysis_regression_files/figure-markdown_github/cubist-2.png" width="750px" />

Resamples & Top Models
----------------------

### Resamples

    ## 
    ## Call:
    ## summary.resamples(object = resamples)
    ## 
    ## Models: lm_no_pre_processing, lm_basic_pre_processing, lm_median_impute, lm_near_zero_variance, lm_skewness_y, lm_skewness_bc, lm_remove_collinearity_caret, lm_remove_collinearity_custom, lm_robust, lm_spatial_sign, lm_pca, lm_pcr, lm_pls, lm_ridge, lm_enet, nlm_neur_net_averaging_pca, nlm_mars, nlm_svm_radial, nlm_svm_linear, nlm_svm_poly, tree_cart, tree_cart2, tree_cond_inference, tree_cond_inference2, ensemble_bagged_tree, ensemble_random_forest, ensemble_boosting, ensemble_cubist 
    ## Number of resamples: 30 
    ## 
    ## RMSE 
    ##                                   Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## lm_no_pre_processing          8.738478 10.010057 10.497878 10.514757 11.000751 11.940636    0
    ## lm_basic_pre_processing       8.738478 10.010057 10.497878 10.514757 11.000751 11.940636    0
    ## lm_median_impute              8.738478 10.010057 10.497878 10.514757 11.000751 11.940636    0
    ## lm_near_zero_variance         8.738478 10.010057 10.497878 10.514757 11.000751 11.940636    0
    ## lm_skewness_y                 5.885970  6.980959  7.393021  7.388440  7.959172  8.467634    0
    ## lm_skewness_bc                6.051467  6.709106  7.051514  7.138558  7.458175  8.207896    0
    ## lm_remove_collinearity_caret  8.738478 10.010057 10.497878 10.514757 11.000751 11.940636    0
    ## lm_remove_collinearity_custom 8.738478 10.010057 10.497878 10.514757 11.000751 11.940636    0
    ## lm_robust                     6.601819  7.338742  7.788631  7.742115  8.293228  8.646746    0
    ## lm_spatial_sign               8.083810  9.188881  9.579182  9.659868 10.052552 11.115776    0
    ## lm_pca                        6.543975  7.312288  7.783855  7.731158  8.244214  8.591784    0
    ## lm_pcr                        9.020453 10.254945 10.646466 10.735250 11.290506 11.956643    0
    ## lm_pls                        8.739534 10.009050 10.498699 10.514593 10.997821 11.939257    0
    ## lm_ridge                      8.738478 10.010057 10.497878 10.514757 11.000751 11.940636    0
    ## lm_enet                       8.766058 10.011650 10.520798 10.513972 11.003917 11.921695    0
    ## nlm_neur_net_averaging_pca    4.748123  5.184613  6.044048  5.883033  6.397794  7.011654    0
    ## nlm_mars                      5.360653  5.883339  6.238895  6.240566  6.625160  7.333883    0
    ## nlm_svm_radial                5.232939  6.059667  6.501122  6.518945  6.910123  7.645726    0
    ## nlm_svm_linear                8.885056 10.224090 10.932951 10.897490 11.445453 12.950250    0
    ## nlm_svm_poly                  6.645202  7.828922  8.080644  8.134308  8.539647  9.638576    0
    ## tree_cart                     6.711325  7.513569  7.881865  8.092351  8.369132 10.579532    0
    ## tree_cart2                    7.847382  8.719991  9.012310  9.175958  9.695644 10.344805    0
    ## tree_cond_inference           6.534783  7.073133  7.747693  7.806524  8.323706  9.857765    0
    ## tree_cond_inference2          6.536448  7.020205  7.479727  7.578241  7.884452  9.066629    0
    ## ensemble_bagged_tree          6.172881  7.097359  7.686422  7.609265  8.125044  8.613750    0
    ## ensemble_random_forest        3.995251  4.510693  4.932979  4.966603  5.393772  6.079412    0
    ## ensemble_boosting             3.126248  3.685153  4.018325  4.028151  4.356902  5.003080    0
    ## ensemble_cubist               3.853350  4.255797  4.630432  4.639967  4.998612  5.523623    0
    ## 
    ## Rsquared 
    ##                                    Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## lm_no_pre_processing          0.4656696 0.5714508 0.6050429 0.6048799 0.6416742 0.7947447    0
    ## lm_basic_pre_processing       0.4656696 0.5714508 0.6050429 0.6048799 0.6416742 0.7947447    0
    ## lm_median_impute              0.4656696 0.5714508 0.6050429 0.6048799 0.6416742 0.7947447    0
    ## lm_near_zero_variance         0.4656696 0.5714508 0.6050429 0.6048799 0.6416742 0.7947447    0
    ## lm_skewness_y                 0.7387950 0.7890597 0.8045592 0.8046701 0.8252289 0.8662577    0
    ## lm_skewness_bc                0.7380665 0.7984465 0.8274261 0.8182863 0.8380072 0.8713801    0
    ## lm_remove_collinearity_caret  0.4656696 0.5714508 0.6050429 0.6048799 0.6416742 0.7947447    0
    ## lm_remove_collinearity_custom 0.4656696 0.5714508 0.6050429 0.6048799 0.6416742 0.7947447    0
    ## lm_robust                     0.7091638 0.7653943 0.7904017 0.7857616 0.8086358 0.8355487    0
    ## lm_spatial_sign               0.5524285 0.6311870 0.6693305 0.6660099 0.7013985 0.8349695    0
    ## lm_pca                        0.7107165 0.7660678 0.7898693 0.7864924 0.8102328 0.8369624    0
    ## lm_pcr                        0.4804495 0.5511912 0.5921717 0.5856976 0.6167787 0.7717105    0
    ## lm_pls                        0.4658338 0.5714005 0.6051009 0.6048908 0.6417536 0.7946863    0
    ## lm_ridge                      0.4656696 0.5714508 0.6050429 0.6048799 0.6416742 0.7947447    0
    ## lm_enet                       0.4694774 0.5727657 0.6052748 0.6046441 0.6411779 0.7940750    0
    ## nlm_neur_net_averaging_pca    0.8332915 0.8604190 0.8723552 0.8761662 0.8984994 0.9197226    0
    ## nlm_mars                      0.8041402 0.8460351 0.8681909 0.8615293 0.8778465 0.9047973    0
    ## nlm_svm_radial                0.7698679 0.8289878 0.8484217 0.8479704 0.8740063 0.9032133    0
    ## nlm_svm_linear                0.4604173 0.5587927 0.5838388 0.5893012 0.6164380 0.7454254    0
    ## nlm_svm_poly                  0.6658420 0.7437070 0.7717173 0.7638321 0.7888754 0.8460427    0
    ## tree_cart                     0.6258185 0.7540236 0.7856241 0.7679315 0.7993443 0.8320331    0
    ## tree_cart2                    0.5934375 0.6732754 0.6922716 0.7003796 0.7268814 0.8076605    0
    ## tree_cond_inference           0.6823096 0.7592862 0.7932960 0.7839204 0.8211438 0.8484643    0
    ## tree_cond_inference2          0.6970032 0.7829267 0.7961661 0.7969249 0.8264006 0.8430229    0
    ## ensemble_bagged_tree          0.7120208 0.7758003 0.8034596 0.8008812 0.8276941 0.8656369    0
    ## ensemble_random_forest        0.8511594 0.9051965 0.9155719 0.9143650 0.9299679 0.9515721    0
    ## ensemble_boosting             0.8977295 0.9333470 0.9437963 0.9418764 0.9504757 0.9641534    0
    ## ensemble_cubist               0.8808580 0.9107882 0.9257168 0.9234823 0.9363818 0.9482455    0

<img src="predictive_analysis_regression_files/figure-markdown_github/resamples_regression-1.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/resamples_regression-2.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/resamples_regression-3.png" width="750px" />

### Top Models

-   TODO: some models have duplicate results (e.g. different pre-processing, same results) so i'm potentially grabbing dups, and therefore &lt; 5 different modelss

Train Top Models on Entire Dataset & Predict on Test Set
========================================================

``` r
top_x_models <- 5
```

<img src="predictive_analysis_regression_files/figure-markdown_github/unnamed-chunk-2-1.png" width="750px" />

-   Note: e.g. if there are rare values at the target extremes (lows/highs), the train and especially the test set might not be training/testing on them. Is the test set representative? If the test set doesn't have as extreme values, it can even predict better (e.g. lower RMSE higher Rsquared) than the average Cross Validation given on training because it's not using those extreme values.

### ensemble\_boosting

    Loading required package: gbm

    Loading required package: splines

    Loaded gbm 2.1.3

    Loading required package: plyr

    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    You have loaded plyr after dplyr - this is likely to cause problems.
    If you need functions from both plyr and dplyr, please load plyr first, then dplyr:
    library(plyr); library(dplyr)

    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    Attaching package: 'plyr'

    The following objects are masked from 'package:Hmisc':

        is.discrete, summarize

    The following object is masked from 'package:DMwR':

        join

    The following objects are masked from 'package:dplyr':

        arrange, count, desc, failwith, id, mutate, rename, summarise, summarize

    The following object is masked from 'package:purrr':

        compact

> Model RMSE: `4.2679`

> Model MAE: `2.6338`

> Model Correaltion Between Actual & Predicted: `0.9682`

Metrics from Test Data:

         RMSE  Rsquared 
    4.2679390 0.9373662 

Actual Observations:

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
       7.32   23.84   34.37   36.68   45.59   82.60 

Predictios:

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
      6.416  25.088  34.591  36.949  47.251  81.596 

<img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-1.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-2.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-3.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-4.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-5.png" width="750px" />

### ensemble\_cubist

    Loading required package: Cubist

<img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-6.png" width="750px" />

> Model RMSE: `4.8704`

> Model MAE: `3.507`

> Model Correaltion Between Actual & Predicted: `0.9586`

Metrics from Test Data:

         RMSE  Rsquared 
    4.8703605 0.9190064 

Actual Observations:

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
       7.32   23.84   34.37   36.68   45.59   82.60 

Predictios:

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
      7.181  26.357  35.163  37.140  45.896  78.439 

<img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-7.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-8.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-9.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-10.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-11.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-12.png" width="750px" />

### ensemble\_random\_forest

Pre-Processing:

    [1] "knnImpute" "center"    "scale"    

> Model RMSE: `5.1219`

> Model MAE: `3.5463`

> Model Correaltion Between Actual & Predicted: `0.9552`

Metrics from Test Data:

         RMSE  Rsquared 
    5.1219039 0.9124374 

Actual Observations:

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
       7.32   23.84   34.37   36.68   45.59   82.60 

Predictios:

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
      9.266  27.115  35.126  37.091  46.972  77.189 

<img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-13.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-14.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-15.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-16.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-17.png" width="750px" />

### nlm\_neur\_net\_averaging\_pca

Pre-Processing:

    [1] "YeoJohnson" "center"     "scale"      "knnImpute"  "pca"       

    Loading required package: nnet

<img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-18.png" width="750px" />

> Model RMSE: `6.0322`

> Model MAE: `4.5052`

> Model Correaltion Between Actual & Predicted: `0.9356`

Metrics from Test Data:

        RMSE Rsquared 
    6.032193 0.875268 

Actual Observations:

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
       7.32   23.84   34.37   36.68   45.59   82.60 

Predictios:

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
      5.784  23.919  33.670  36.302  46.695  73.387 

<img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-19.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-20.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-21.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-22.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-23.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-24.png" width="750px" />

### nlm\_mars

> Model RMSE: `6.2572`

> Model MAE: `4.7121`

> Model Correaltion Between Actual & Predicted: `0.9331`

Metrics from Test Data:

         RMSE  Rsquared 
    6.2572259 0.8706679 

Actual Observations:

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
       7.32   23.84   34.37   36.68   45.59   82.60 

Predictios:

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
      4.044  27.233  36.494  37.972  46.896  76.117 

<img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-25.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-26.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-27.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-28.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-29.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-30.png" width="750px" /><img src="predictive_analysis_regression_files/figure-markdown_github/determine_best_models-31.png" width="750px" />
