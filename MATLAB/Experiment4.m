%%% epsilon-differential Private Rasch Model
% Authors:
% MSc Student Teresa Anna Steiner (s170063@student.dtu.dk) 
% MSc Student David Enslev Nyrnberg (s123997@student.dtu.dk) 
% Professor Lars Kai Hansen (lkai@dtu.dk) 
% Input:
% Dichomotous data matrix of 0's and 1's
% Output:

% initialize result array
EX4.corrMatrix_nonPriv_opPriv = zeros(length(nStudent), bootstrap);
EX4.corrMatrix_nonPriv_suffPriv = zeros(length(nStudent), bootstrap);
EX4.corrMatrix_opPriv_suffPriv = zeros(length(nStudent), bootstrap);
EX4.error_nonPriv = zeros(length(nStudent), 1);
EX4.error_opPriv = zeros(length(nStudent), 1);
EX4.error_suffPriv = zeros(length(nStudent), 1);

I = iTest;
for k = 1:length(nStudent)
    N = nStudent(k);
    for boots = 1:bootstrap
        data_new = datasample(data, N);
        w_ini = 0.001*randn(N+I, 1);
        % Estimate - non private
        w_nonPriv = fminunc(@(w) rasch_neglog_likelihood(w,data_new,lam), w_ini, options);
        nonPriv_beta_est_global = w_nonPriv(1:N);
        nonPriv_delta_est_global = w_nonPriv(N+1:end)';
        % reopt section
        nonPriv_beta_est_reopt = zeros(N,1);
        for n = 1:N
            beta_n = 0;
            student = data_new(n,:);
            nonPriv_beta_est_reopt(n) = fminunc(@(beta) single_beta(beta, nonPriv_delta_est_global, student, lam), beta_n, options);
        end
        
        % private rasch estimation
        b_norm = gamrnd(I, sqrt(I)/EX4_epsilon);
        bx = normrnd(0,1,[1,I]);
        b = bx/norm(bx)*b_norm;
        w_opPriv = fminunc(@(w) rasch_neglog_likelihood_private(w,data_new,lam,b),w_ini,options);
        opPriv_est_beta_global = w_opPriv(1:N);
        opPriv_est_delta = w_opPriv(N+1:end)';
        
        % re-optimized est non-priv
        opPriv_est_beta = zeros(N, 1);
        for n = 1:N
            beta_n = 0;
            student = data_new(n,:);
            opPriv_est_beta(n) = fminunc(@(beta) single_beta(beta, opPriv_est_delta, student, lam), beta_n, options);
        end
        % Sufficient statistic
        [suffStatPriv_est_delta, suffStatPriv_est_beta] = suff_stat_parameter_est(data_new, lam, EX4_epsilon, N, I, options);
    
        % estimate Rasch models
        nonPriv_raschModel = raschModel(nonPriv_beta_est_reopt, nonPriv_delta_est_global);
        opPriv_raschModel = raschModel(opPriv_est_beta, opPriv_est_delta);
        suffStatPriv_raschModel = raschModel(suffStatPriv_est_beta, suffStatPriv_est_delta);
            
        % correlation matrixes corrMatrix_true_opPriv
        corr_nonPriv_vs_opPriv = corrcoef(nonPriv_raschModel, opPriv_raschModel);
        EX4.corrMatrix_nonPriv_opPriv(k,boots) = corr_nonPriv_vs_opPriv(1, 2);
        corr_nonPriv_vs_suffStatPriv = corrcoef(nonPriv_raschModel, suffStatPriv_raschModel);
        EX4.corrMatrix_nonPriv_suffPriv(k,boots) = corr_nonPriv_vs_suffStatPriv(1, 2);
        corr_opPriv_vs_suffStatPriv = corrcoef(opPriv_raschModel, suffStatPriv_raschModel);
        EX4.corrMatrix_opPriv_suffPriv(k,boots) = corr_opPriv_vs_suffStatPriv(1, 2);
        
        % nonPriv, OP and SuffStat error to test set
        EX4.error_nonPriv(k) = EX4.error_nonPriv(k)+sum(sum(abs(round(nonPriv_raschModel)-data_new)));
        EX4.error_opPriv(k) = EX4.error_opPriv(k)+sum(sum(abs(round(opPriv_raschModel)-data_new)));
        EX4.error_suffPriv(k) = EX4.error_suffPriv(k)+sum(sum(abs(round(suffStatPriv_raschModel)-data_new)));
    end
    % error normalization
    EX4.error_nonPriv(k) = EX4.error_nonPriv(k)/(bootstrap*I*N);
    EX4.error_opPriv(k) = EX4.error_opPriv(k)/(bootstrap*I*N);
    EX4.error_suffPriv(k) = EX4.error_suffPriv(k)/(bootstrap*I*N);
end