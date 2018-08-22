% initialize result arrays
EX1.corrMatrix_trueVSglobal = zeros(length(nStudent),repetition);
EX1.corrMatrix_trueVSreopt = zeros(length(nStudent),repetition);
EX1.corrMatrix_globalVSreopt = zeros(length(nStudent),repetition);
EX1.error_true = zeros(1,length(nStudent));
EX1.error_global = zeros(1,length(nStudent));
EX1.error_reopt= zeros(1,length(nStudent));

I = iTest;
for k = 1:length(nStudent)
    N = nStudent(k);
    for rep = 1:repetition
        % simulate beta and delta values
        betaTrue = meanStu+randn(N,1)*sdStu;
        deltaTrue = meanTest+sdTest*randn(1,I);
        % model the true model
        raschModel_true = raschModel(betaTrue, deltaTrue);
        % simulate data
        simRaschData = binornd(1, raschModel_true);
        simTestData = binornd(1, raschModel_true);
        
        w_ini = 0.001*randn(N+I, 1);
        % non private global rasch estimation
        w = fminunc(@(w) rasch_neglog_likelihood(w, simRaschData, lam), w_ini, options);
        beta_est_global = w(1:N);
        delta_est = w(N+1:end)';
        
        % re-optimized est non-priv
        beta_est_reopt = zeros(N, 1);
        for n = 1:N
            beta_n = 0;
            student = simRaschData(n,:);
            beta_est_reopt(n) = fminunc(@(beta) single_beta_TD(beta, delta_est, student, lam), beta_n, options);            
        end
        % estimate global and reopt Rasch
        raschModel_global = raschModel(beta_est_global, delta_est);
        raschModel_reopt = raschModel(beta_est_reopt, delta_est);
        % true, global and reopt error to test set
        EX1.error_true(k) = EX1.error_true(k)+sum(sum(abs(round(raschModel_true)-simTestData))); %is this correct? the round part
        EX1.error_global(k) = EX1.error_global(k)+sum(sum(abs(round(raschModel_global)-simTestData)));
        EX1.error_reopt(k) = EX1.error_reopt(k)+sum(sum(abs(round(raschModel_reopt)-simTestData)));
        % correlation coeficcients for results
        corr_true_vs_global = corrcoef(raschModel_true, raschModel_global);
        EX1.corrMatrix_trueVSglobal(k,rep) = corr_true_vs_global(1, 2);
        corr_true_vs_reopt = corrcoef(raschModel_true, raschModel_reopt);
        EX1.corrMatrix_trueVSreopt(k,rep) = corr_true_vs_reopt(1, 2);
        corr_global_vs_reopt = corrcoef(raschModel_global, raschModel_reopt);
        EX1.corrMatrix_globalVSreopt(k,rep) = corr_global_vs_reopt(1, 2);
    end
    % error normalization
    EX1.error_global(k) = EX1.error_global(k)/(N*I*repetition);
    EX1.error_reopt(k) = EX1.error_reopt(k)/(N*I*repetition);
    EX1.error_true(k) = EX1.error_true(k)/(N*I*repetition);
end
