function [delta,beta] = suff_stat_parameter_est(X,lam,epsi,N,D)
    % sufficient stats
    xd=sum(X,2);
    xn=sum(X,1);
    % initialization
     bet=0.001*randn(N,1);
     del=0.001*randn(1,D);

     w0=[bet;del'];
     % tune parameters
     options = optimoptions('fminunc','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'MaxIter',10^5,'MaxFunEvals',10^5,'TolX',10^-5,'Display','off');
     w=fminunc(@(w) rasch_negloglik_grad_priv_suff(w,xd,xn,lam),w0,options);
     bet=w(1:N);
     del=w((N+1):(N+D))';
%    % private suff stats
     xd_priv=max(zeros(size(xd)),xd+(D/epsi)*sign(randn(size(xd))).*log(rand(size(xd))));
     xd_priv=min(D*ones(size(xd_priv)),xd_priv);
     xn_priv=max(zeros(size(xn)),xn+(D/epsi)*sign(randn(size(xn))).*log(rand(size(xn))));
     xn_priv=min(N*ones(size(xn_priv)),xn_priv);
    % optimize with private suff stats
    w_priv=fminunc(@(w) rasch_negloglik_grad_priv_suff(w,xd_priv,xn_priv,lam),w0,options);
    
    del_priv=w_priv((N+1):(N+D))';
    %now reoptimize beta
    betn_arr=zeros(N,1);
    for q=1:N
        betn0=0;
        student=X(q,:);
          )
        betn_arr(q)=fminunc(@(beta) single_beta_TD(beta,del_priv,student,lam),betn0,options);
    end
    delta=del_priv;
    beta=betn_arr;
end

