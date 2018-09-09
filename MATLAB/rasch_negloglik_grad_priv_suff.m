function [negloglik, grad ] = rasch_negloglik_grad_priv_suff(w, xd_priv,xn_priv,lam)
% computes the normalized neg log-likelihood of the Rasch model
% based on sufficients statistics sum(X,1) and sum(X,2)
% lam is L2 regularization
    D=size(xn_priv,2);
    N=size(xd_priv,1);
    bet=w(1:N);
    del=w((N+1):(N+D))';
    dum=exp(bet*ones(1,D)-ones(N,1)*del);
    dum1=1./(1 + dum.^(-1) );
    negloglik=sum(sum(log(1+dum)))-sum(xd_priv.*bet)+sum(xn_priv.*del)+lam*(sum(bet.*bet)+sum(del.*del));
    ped=sum(dum1,2);
    pen=sum(dum1,1);
    gradbet=(ped-xd_priv)+2*lam*bet;
    graddel=-(pen-xn_priv)+2*lam*del;
    grad=[gradbet;graddel'];
end

