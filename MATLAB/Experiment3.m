%%% epsilon-differential Private Rasch Model
% Authors:
% MSc Student Teresa Anna Steiner (s170063@student.dtu.dk) 
% MSc Student David Enslev Nyrnberg (s123997@student.dtu.dk) 
% Professor Lars Kai Hansen (lkai@dtu.dk) 
% Input:
% Dichomotous data matrix of 0's and 1's
% Output:

% load real data
loadData = xlsread(gradeDir, 'Exam (Second time)');
data = round(loadData(:,2:end-1)./max(loadData(:,2:end-1)));

[N,I] = size(data);
% initialize result array corrMatrix_true_opPriv
error_OP = 0;
error_suffstat = 0;
error_non_priv = 0;
corrMatrix_nonPriv_opPriv = zeros(1,repetition);
corrMatrix_nonpriv_suffPriv = zeros(1,repetition);
corrMatrix_suffPriv_opPriv=zeros(1,repetition);

for rep = 1:repetition
    data_new = datasample(data,N);
    
    % non private rasch estimation
    w_ini = 0.001*randn(N+I, 1);
    % global section
    w_nonPriv = fminunc(@(w) rasch_neglog_likelihood(w, data_new, lam), w_ini, options);
    nonPriv_beta_est_global = w_nonPriv(1:N);
    nonPriv_delta_est_global = w_nonPriv(N+1:end)';
    % reopt section
    nonPriv_beta_est_reopt = zeros(N,1);
    for n = 1:N
        beta_n = 0;
        student = simTrainData(n,:);
        nonPriv_beta_est_reopt(n) = fminunc(@(beta) single_beta_TD(beta, nonPriv_delta_est_global, student, lam), beta_n, options);
    end
    nonPriv_raschModel = raschModel(nonPriv_beta_est_reopt, nonPriv_delta_est_global);

    % global estimation - PRIVATE - OP
    % generate noise vector
    b_norm = gamrnd(I, sqrt(I) / EX3_epsilon);
    bx = normrnd(0,1,[1,I]);
    b = bx / norm(bx) * b_norm;
    wpriv=fminunc(@(w) rasch_neglog_likelihood_private_TD(w,data_new,lam,b),w_ini,options);
    beta_est_priv = wpriv(1:N);
    delta_est_priv = wpriv(N+1:end)';
    nonPriv_beta_est_reopt=zeros(N,1);
    probRaschModel_OP=zeros(N,I);
    probRaschModel_suffstat=zeros(N,I);
    [delta_suffstat,beta_suffstat]=suff_stat_parameter_est(data_new,lam,EX3_epsilon,N,I);
    for n = 1:N
        beta_n=0;
        student=data_new(n,:);
        nonPriv_beta_est_reopt(n)=fminunc(@(beta) single_beta_TD(beta,delta_est_priv,student,lam),beta_n,options);
        for i = 1:I
            probRaschModel_suffstat(n,i)=exp(beta_suffstat(n)-delta_suffstat(i))/(1+exp(beta_suffstat(n)-delta_suffstat(i)));
            probRaschModel_OP(n,i)=exp(nonPriv_beta_est_reopt(n)-delta_est_priv(i)) / (1+exp(nonPriv_beta_est_reopt(n)-delta_est_priv(i)));
            error_OP=error_OP+abs(round(probRaschModel_OP(n,i))-data_new(n,i));
            error_suffstat=error_suffstat+abs(round(probRaschModel_suffstat(n,i))-data_new(n,i));
        end
    end
    corr_nonpriv_vs_OP=corrcoef(nonPriv_raschModel,probRaschModel_OP);
    corrMatrix_nonPriv_opPriv(rep) = corr_nonpriv_vs_OP(1,2);
    corr_nonpriv_vs_suffstat=corrcoef(nonPriv_raschModel,probRaschModel_suffstat);
    corrMatrix_nonpriv_suffPriv(rep) = corr_nonpriv_vs_suffstat(1,2);
    corr_OP_vs_suffstat=corrcoef(probRaschModel_OP,probRaschModel_suffstat);
    corrMatrix_suffPriv_opPriv(rep) = corr_OP_vs_suffstat(1,2);
end
error_OP=error_OP/(repetition*I*N);
error_suffstat=error_suffstat/(repetition*I*N);
error_non_priv=error_non_priv/(repetition*I*N);
