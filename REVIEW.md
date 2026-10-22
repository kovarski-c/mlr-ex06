# 1. Preprocessing inside CV: 
*Locate the full pipeline in the objective:* `mean, sd, _ = cv_scores(build_candidate(family, params), X_train, y_train)` in `objective` in `run_family` in `bayes_compare.py`. 

*Check that imputation and scaling are never fitted on validation/test rows:* `scores = cross_val_score(pipeline, X_train, y_train, cv=common.CV, scoring="accuracy")` and `pipeline.fit(X_train, y_train)`. No problem.

*Confirm the fixed cleaning bounds:* `common.py`, function `clean_readings`: 
    ```
    frame.loc[(frame["charge_pct"] < 0) | (frame["charge_pct"] > 100), "charge_pct"] = np.nan
    frame.loc[frame["distance_km"] <= 0, "distance_km"] = np.nan
    ```
    -> OK

*and safe ratios:* The ratio `wh_per_km` in `common.py` in func. `add_extended_features` is safe because imputation runs before it is computed, so the distance is never 0 or missing at that point.


# 2. Test isolation: 
*Find every use of X_test/y_test:* in `bayes_compare.py`, creating those variables and testing accuracy on them in function `final_test`. 

*Were all model and feature choices fixed before these scores were calculated?* final_test is called only in main, after the loop that runs run_family for all families has finished. It gets pipelines built from the stored bests[family]["parameters"], so no choice depends on the test score.
*No post-trip variables or target-derived features may enter X.*
X is built only from NUMERIC + CATEGORICAL in `common.py`, the target runs_out is not in it, and the engineered features use only `capacity_wh`, `charge_pct`, `distance_km`.


# 3. Fair comparison: 

*Count completed trials, compare folds and data rows across algorithms, and verify the two reference rows. Equal trial counts do not imply equal compute effort.* `trials.csv` contains 5 families, each with 20 completed trials. SVM reaches the biggest cv_mean, knn smallest. Reference rows from `results.csv`:
```
"forest, original features (notebook baseline)",reference,"{""features"": ""original"", ""impute"": ""median"", ""model"": ""RandomForestClassifier(n_estimators=100, random_state=0)""}",,0.917037,0.012351,0.897778,False
majority class,reference,"{""model"": ""DummyClassifier(strategy='most_frequent')""}",,0.612593,0.001048,0.613333,False
```
Dummy classifier has much lower cv_mean than all the other models, Random Forest is by eye better than knn and other models outperform it.


# 4. Search ranges: 

*Inspect logarithmic versus linear ranges, sensible neighbor/leaf bounds and the limited feature/preparation choices.*
`bayes_compare.py`, function `suggest_parameters`. Log=True are svm, logreg and histgb, all of those use parameters that range across several orders. No problem found in bounds and feature/preparation choices. 
*Are all returned parameters applied?* `build_candidate` reads all keys used in `suggest_parameters. 


# 5. Conditional parameters: 
*Gamma only for RBF:* yes.
*Scaling for the distance/linear models, not a meaningless tree scaling search.* The scaler is added for k-NN, SVM and logistic regression: `SCALED_FAMILIES = {"knn", "svm", "logreg"}` after the extra features are made (in `build_candidate`). It is not added for the forest and gradient boosting, and it is not searched at all.
*Check the best parameter dictionaries.*
In `results.csv` (`column parameters`) each best dictionary contains only the parameters of its own family, all inside the searched ranges. The best SVM uses kernel: linear and has no gamma, which confirms that gamma is only applied for RBF. The only value on a bound is forest min_samples_leaf = 1, which is the smallest possible value.


# 6. Reproducibility: 
*Inspect every seed and compare the two full-run CSVs. Record the command and evidence; do not include runtime in a table expected to be byte-identical.*
The split uses seed 42. The CV splitter uses seed 0. Every Optuna sampler uses seed 0 with 5 random start trials. The forest, the baseline forest and gradient boosting use random_state=0. Optuna runs with n_jobs=1, so one trial runs after another. SVM, logistic regression, k-NN and the dummy model have no randomness.
`uv run python bayes_compare.py --trials 20`, 
### Hashes and diff
```
2b07b573203543fdbc604f667d211be537891713f9d82d8558a020dd13aaafdd  run1_results.csv
2b07b573203543fdbc604f667d211be537891713f9d82d8558a020dd13aaafdd  results.csv
8a28d6b9170e681b93a32f9eb41292f8a234d3b2d258fbadade612f29021ef6c  run1_trials.csv
8a28d6b9170e681b93a32f9eb41292f8a234d3b2d258fbadade612f29021ef6c  trials.csv
IDENTICAL
```


# 7. Agreement with your work: 
*Compare the original-feature forest with Task 4 and explain any discrepancy.*
The baseline (CV 0.9170, SD 0.0124) is the same as in notebook cell 22. There are two differences. First, the table in notebook cell 27 shows 0.9193 for the baseline, which is the energy-feature score. The cells were run in a wrong order, and cell 23 is a markdown cell that contains code. Second, the notebook's small Optuna demo found 0.9244 (engineered features, median imputation, 0.5 feature share, leaf size 1). The 20-trial search found only 0.9215. That exact setting with median imputation was never tried in the 20 trials. I checked on training data only, with 10 repeats of the 3-fold CV. Both settings give the same average (0.9216 and 0.9219), so the 0.9244 was luck of the fixed folds.

*Look at convergence and whether extra features helped. Large gains deserve investigation, not automatic acceptance.**
Forest and k-NN stopped improving early (best at trials 6 and 7). Logistic regression stopped at trial 13. SVM improved at trial 18 and gradient boosting at trial 19, so those two curves are still rising and 20 trials is not fully enough for them.
Extra features mostly helped.


# 8. Conclusions: 
*Distinguish validation selection from final testing.*
The SVM was chosen only by CV (0.9422, SD 0.0018). The test accuracies were calculated afterwards and were not used for the choice: gradient boosting 0.9533, SVM 0.9444, logistic regression and forest 0.9378, k-NN 0.9044. Gradient boosting is first on the test, but it was not chosen, because the choice is made on CV. The CV score of the winner is too optimistic, because it is the best of 100 noisy results.

*Compare differences with fold variability without treating it as a formal significance test.*
SVM leads logistic regression by 0.0015 and gradient boosting by 0.0044. Both gaps are smaller than the fold standard deviations of those models (0.011 and 0.013), so it's not clear whether the svm is actually better. The forest (0.9215) is about 2 points behind, and k-NN (0.8881) about 5 points behind, which is bigger than standard deviation. On the test set the top three differ by less than about one standard error (about 1 point), so the different order there does not change the CV choice.

*State what synthetic data cannot establish about real e-bikes.*
The data are simulated trips, not measurements. They cannot show how models would perform on real bikes.