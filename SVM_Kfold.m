% Multiclass classification with SVM. 
%
% "FeatureSelection-Installer.jar" needs to be INSTALLED and
% load_fspackage.m needs to be ran.
%
% This script calculates Information Gain, Fisher Score and T-Test
% for feature set. Then, it takes the intersections of best features
% ranked by every algorithm and creates reduced dataset.
% Then, it normalizes data and does k-fold cross validation with SVM.

%Initialization
close all; clc; clear;

init = 54;
mySeed=RandStream.create('mt19937ar','seed',init);
RandStream.setGlobalStream(mySeed);

%% Loading Data
% Loading data and set it to a variable.
data = load('dataset.txt');

% % Choosing only 1 and 2 labels
% data = data(1:74,:);

% % Randomize rows and ordering data according to randomized rows
% order = randperm(size(data,1));
% data = data(order,:);

% Separate data into X and y. Last column is y and other columns are
% features. (Second last column is dribbling or not classification).
X = data(:,1:end-2);
y = data(:,end);

% Information Gain
iGain = fsInfoGain(X, y);
iGain_W = iGain.W;
iGain_fList = iGain.fList;

% Fisher Score
fisherScore = fsFisher(X, y);
fisher_W = fisherScore.W;
fisher_fList = fisherScore.fList;

% t-Test
tTest = fsTtest(X, y);
tTest_W = tTest.W;
tTest_fList = tTest.fList;

% Choosing best 30 features for each algorithm
iGain_top = iGain_fList(1:30);
fisher_top = fisher_fList(1:30);
tTest_top = tTest_fList(1:30)';

% Choosing intersections of best features of each algorithm
intersection = sort([intersect(iGain_top, fisher_top), intersect(iGain_top, tTest_top),...
    intersect(fisher_top, tTest_top)]);

X = X(:, intersection);

%Feature Normalization
prompt = 'Do you want to normalize data? (Yes: 1, No: 0) (Recommendation: Yes) \n';
answer_norm = input(prompt);
if answer_norm == 1
    [X, mu, sigma] = featureNormalize(X);
end

% SVM
% Split training and testing sets
k = 10; % fold number
cvFolds = crossvalind('Kfold', y, k); % Setting indices that determines fold groups

cp = classperf(y); % Performance of classifier

for i = 1:k                                  %# for each fold
    testIdx = (cvFolds == i);                %# get indices of test instances
    trainIdx = ~testIdx;                     %# get indices training instances

    train_data = X(trainIdx,:); % create train set
    train_label = y(trainIdx,:); % create train labels

    test_data = X(testIdx,:); % create test set
    test_label = y(testIdx,:); % create test labels

    % Calling multisvm function to train svm and return test results
    result = multisvm(train_data, train_label, test_data);

    % Calculating correct predictions of test set
    correctPredictions = result == test_label;
    testAccuracy(i) = sum(correctPredictions)/length(correctPredictions);

    % Evaluating classifier performance
    cp = classperf(cp, result, testIdx);
end

Accuracy = mean(testAccuracy)

%# get accuracy
correctRate = cp.CorrectRate

%# get error
errorRate = cp.ErrorRate

%# get confusion matrix
%# columns:actual, rows:predicted, last-row: unclassified instances
confusionMatrix = cp.CountingMatrix

[precision, recall, f1score] = errorMetrics(confusionMatrix)