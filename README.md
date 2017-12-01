# basketball-training-recognition
This is the source code of our recent work "Towards an Artificial Training Expert System for Basketball" published in ELECO2017.

The dataset contains calculated 153 features in columns and every row is labeled as exercise types. Exercise types are labeled as numbers and their representations as follows:

| Number | Label Name                 |
|:------:|----------------------------|
|    1   | Forward-Backward Dribbling |
|    2   | Left-Right Dribbling       |
|    3   | Regular Dribbling          |
|    4   | Two Hands Dribbling        |
|    5   | Shooting                   |
|    6   | Layup                      |

Feature names with their position in dataset can be seen in the "FeatureNames.csv" file.

How to run code:
1. To use the feature selection methods (Information Gain, Fisher's Score, etc.), "FeatureSelection-Installer.jar" file should be installed. It has instructions about how to install and how to run.
2. "SVM_Kfold.m" file is the main file.
3. The code is ready to select 28 features and run SVM.
4. If you want to run 153 features set, you need to change open commented areas for 153 features and comment out the areas for 28 features.
