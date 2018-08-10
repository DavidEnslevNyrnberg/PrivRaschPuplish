function [likelihood,w_gradient] = rasch_neglog_likelihood_private_TD(w, data, lambda, epsilon)
%return the private negative loglikelihood and gradient of the Rasch model
[N,I] = size(data);
wBeta = w(1:N);
wDelta = w(N+1:end)';

b_norm = gamrnd(I, sqrt(I)/epsilon);
% then direction randomly in d-dimensional space (http://mathworld.wolfram.com/HyperspherePointPicking.html)
bx = normrnd(0, 1, [1,I]);
b = bx/norm(bx)*b_norm;

% Rasch weight matrix
wMatrix = log(1+exp(wBeta*ones(1,I)-ones(N,1)*wDelta));

% matrix for neg-log-lik and calc likelihood
wLikeMatrix = log(1+wMatrix);
likelihood = sum(sum(wLikeMatrix,2))+sum(wDelta.*sum(data,1))-sum(wBeta.*sum(data,2))+lambda*w'*w+b*wDelta';

% matrix for gradient and calc gradient
wGradMatrix = 1./(1+wMatrix.^(-1));
gradBeta = sum(wGradMatrix, 2) - sum(data,2) + 2*lambda*wBeta;
gradDelta = -sum(wGradMatrix,1) + sum(data,1) + 2*lambda*wDelta + b;
w_gradient = [gradBeta;gradDelta'];
end

