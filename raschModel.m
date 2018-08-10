function [raschModel] = raschModel(betaN, deltaI)

% load options
N = length(betaN);
I = length(deltaI);
% prob rasch model form -> 1/(1+exp{deltaI-betaN})
raschModel = 1./(1+exp(ones(N,1)*deltaI-betaN*ones(1,I))); % TODO: check delta-beta term

end