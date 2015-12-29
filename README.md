# whats-cooking		
## Introduction  
This model predicts cuisines of a recipe based on the ingredients used in this recipe. The full decription of this competition can be found [HERE](https://www.kaggle.com/c/whats-cooking).  
## Model  
* Feature Extraction:   
	* Tf-idf features  
	* chi-squared feature selection  
* Estimator: svm with rbf (two hyperparameters: C and gamma)
## Result
0.80457 on the average and of 0.81386 on the Kaggle’s test data (ranked 61th in the competition as compared to the highest score of 0.83216).   

## Run
* Download to your local: $ git clone https://github.com/thanh-ng/whats-cooking.git'''
* utils.py: utility functions for building models
* main.py : Change hyperparameter space in this file for grid search and run this file.   

*Also see my other projects [HERE](http://thanh-ng.github.io/pages/src/)
