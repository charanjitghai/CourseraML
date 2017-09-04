function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

pred = X*Theta';
error = pred - Y;
error2 = error .* error;
J = sum(error2(R==1))/2 ;

theta2 = sum(sum(Theta .* Theta));
X2 = sum(sum(X .* X));

J = J + lambda * (X2 + theta2)/ 2;


%for i = 1:num_movies
%	for k = 1:num_features
%		error_movie_i = error(i,:) .* R(i,:);
%		theta_feature_k = Theta(:,k)';
%		X_grad(i,k) = sum(error_movie_i .* theta_feature_k);
%	end
%end


for i = 1:num_movies
	error_movie_i = error(i,:) .* R(i,:);
	X_grad(i,:) = error_movie_i * Theta;
end;


X_grad = X_grad + lambda*X;


%for j = 1:num_users
%	for k = 1:num_features
%		error_user_j = error(:,j) .* R(:,j);
%		x_feature_k = X(:,k);
%		Theta_grad(j,k) = sum(error_user_j .* x_feature_k);
%	end
%end

for j = 1:num_users
	error_user_j = error(:,j) .* R(:,j);
	Theta_grad(j,:) = error_user_j' * X;
end;

Theta_grad = Theta_grad + lambda*Theta;


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
