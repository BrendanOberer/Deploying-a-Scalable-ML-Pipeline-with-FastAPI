# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

I used a random forest classifier with 100 decision trees and a random state of 64. The model is developed in scikit learn. It is designed to predict wether a persons income is higher than 50,000 based on employment and demographic information.

## Intended Use

The models intended use is for the demonstration of ML classification, model evaluation, and fastAPI use. As mentioned before it predicts if an individuals annual income is above or below 50,000. It is not designed for financial predictions or other business decisions. 

## Training Data

The model is trained with the provided UCI Adult Census dataset, which contains information about age, education, occupation, marital status, and other demographic details. The data comes from a 1994 census.

## Evaluation Data

The model is tested with the remaining 20% of the census dataset not used in training. Performance is evaulated across individual categories of each categorical feature to identify differences in model performance between groups.

## Metrics

The model is evaluated with precision, recall, and F1 score. Annual income above 50,000 is treated as the positive class. On the test dataset the model achieved a precision of 0.7266, recall of 0.6135 and F1 score of 0.6653. Precision measures the proportion of positive predictions that were true. Recall measures the proportion of positive cases identified by the model. F1 measures a combined metric of precision and recall.

## Ethical Considerations

The dataset has sensitive information within it, such as age sex and race. Historical inequalities in the census might influence the models predictions, potentially leading to different error rates across groups. The model should not be used to make decisions that could affect people, such as hiring, lending or more. 

## Caveats and Recommendations

The dataset is based on older census data that might not reflect modern patterns. Measurements of performance for categories with a small number of observations might be unreliable due to the limited data. The model has also not been through extensive hyperparameter tuning. Improvements could look like additional model tuning, or evaluation on more modern data that may be more relevant depending on use case.