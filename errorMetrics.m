function [precision, recall, f1score] = errorMetrics(confusionmat)
%Calculation of Precision, Recall and F1 Score for multiclass
%classification.

%   FOR INPUT:
%   CONFUSION MATRIX'S COLUMNS = ACTUAL VALUES.
%   CONFUSION MATRIX'S ROWS = PREDICTED VALUES.

% Function takes confusion matrix as an input. Then, confusion matrix's
%   column length taken. For every class, calculations is done and assigned
%   to an array. Then, average value of every metric calculated and
%   returned.

    cm = confusionmat;
    [~, length] = size(cm);
    
    for i=1:length
        prec(i) = cm(i,i) / sum(cm(i,1:length));
        
        rec(i) = cm(i,i) / sum(cm(1:length,i));
        
        f1(i) = (2 * prec(i) * rec(i)) / (prec(i) + rec(i)); 
    end
    
    precision = mean(prec(:));
    recall = mean(rec(:));
    f1score = mean(f1(:)); 
end

