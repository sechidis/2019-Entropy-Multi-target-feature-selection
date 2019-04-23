function H= h(X)
% Summary
%    Estimate the entropy H(X) of a categorical variable X

[~,~,X] = unique(X,'rows');
[~,ar,X]=unique(X); arity_X=length(ar);
n = length(X);

p_X = accumarray(X,1,[arity_X 1])/n;

H = -sum( p_X(p_X>0) .* log(p_X(p_X>0)));