function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
testValues = [0.01 0.03 0.1 0.3 1 3 10 30];
infinitValue = Inf;
bestError = infinitValue;
C = 1; %give default value 
sigma = 0.3; % give default value;
% loop over test values  for C and sigma then  make tests on validation data set 
for i = 1:length(testValues)
  C_i = testValues(i);
  for j = 1:length(testValues)
    sigma_j = testValues(j);
    %svmPredict requier model that can be returned by svmTrain
    model = svmTrain(X, y , C_i,@(x1, x2) gaussianKernel(x1, x2, sigma_j));
    
    predictionsOnValidationData = svmPredict(model, Xval);
    
    validationError = mean(double(predictionsOnValidationData != yval)); % take the mean of missclassified
    
    if validationError < bestError
      bestError = validationError;
      C = C_i;
      sigma = sigma_j;
    endif
    
  endfor
end

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
