function [raschModel] = raschModel(betaN, deltaI)

N = length(betaN);
I = length(deltaI);

raschModel = 1./(1+exp(ones(N,1)*deltaI-betaN*ones(1,I)));

end