function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
variables_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
variables_length = length(variables_vec);
mean_errors = zeros(variables_length^2,3);
x1 = X(:,1);
x2 = X(:,2);
me = 1;
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

for i = 1:variables_length
	for j = 1:variables_length
		model= svmTrain(X, y, variables_vec(i), @(x1, x2) gaussianKernel(x1, x2, variables_vec(j)));
		predictions = svmPredict(model, Xval);
		temp_me = mean(double(predictions ~= yval));
		if(temp_me < me)
			me = temp_me;
			C = variables_vec(i);
			sigma = variables_vec(j);
		end;
	end;
end;

disp(me);
disp(C);

disp(sigma);


% =========================================================================

end
