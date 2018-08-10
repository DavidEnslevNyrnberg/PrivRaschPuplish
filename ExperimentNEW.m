% initialize result arrays
exNew.corrMatrix_true_nonPrivGlobal = zeros(length(nStudent),repetition);
exNew.corrMatrix_true_opPrivGlobal = zeros(length(nStudent),repetition);
exNew.corrMatrix_true_opPrivReopt = zeros(length(nStudent),repetition);
exNew.error_true = zeros(1,length(nStudent));
exNew.error_nonPriv_global = zeros(1,length(nStudent));
exNew.error_opPriv_global = zeros(1,length(nStudent));
exNew.error_opPriv_reopt = zeros(1,length(nStudent));

I = iTest;
for k = 1:length(nStudent)
    N = nStudent(k);
    for rep = 1:repetition
        % simulate beta and delta values
        betaTrue = meanStu+randn(N,1)*sdStu;
        deltaTrue = meanTest+sdTest*randn(1,I);
        % model true Rasch model
        raschModel_true = raschModel(betaTrue, deltaTrue);
        % simulate train and test data
        simTrainData = binornd(1, raschModel_true);
        simTestData = binornd(1, raschModel_true);
        
        % non private global rasch estimation
        w_ini = 0.001*randn(N+I, 1);
        
        w = fminunc(@(w) rasch_neglog_likelihood(w, simTrainData, lam), w_ini, options);
        nonPriv_beta_est_global = w(1:N);
        nonPriv_delta_est_global = w(N+1:end)';
        
        % private global est optimization
        wpriv_OP = fminunc(@(w) rasch_neglog_likelihood_private_TD(w, simTrainData, lam, epsi), w_ini, options);
        privOP_est_beta_global = wpriv_OP(1:N);
        privOP_est_delta_global = wpriv_OP(N+1:end)';
        
        % re-optimized est non-priv
        privOP_est_beta_reopt = zeros(N, 1);
        for n = 1:N
            beta_n = 0;
            student = simTrainData(n,:);
            privOP_est_beta_reopt(n) = fminunc(@(beta) single_beta_TD(beta, privOP_est_delta_global, student, lam), beta_n, options);            
        end
        
        % estimate global and reopt Rasch models
        nonPriv_raschModel_global = raschModel(nonPriv_beta_est_global, nonPriv_delta_est_global);
        privOP_raschModel_global = raschModel(privOP_est_beta_global, privOP_est_delta_global);
        privOP_raschModel_reopt = raschModel(privOP_est_beta_reopt, privOP_est_delta_global);
        
        % true, global and reopt error to test set
        exNew.error_true(k) = exNew.error_true(k)+sum(sum(abs(round(raschModel_true)-simTestData))); %is this correct? the round part
        exNew.error_nonPriv_global(k) = exNew.error_nonPriv_global(k)+sum(sum(abs(round(nonPriv_raschModel_global)-simTestData)));
        exNew.error_opPriv_global(k) = exNew.error_opPriv_global(k)+sum(sum(abs(round(privOP_raschModel_global)-simTestData)));
        exNew.error_opPriv_reopt(k) = exNew.error_opPriv_reopt(k)+sum(sum(abs(round(privOP_raschModel_reopt)-simTestData)));
        % correlation coeficcients for results
        corr_true_vs_nonPriv_global = corrcoef(raschModel_true, nonPriv_raschModel_global);
        exNew.corrMatrix_true_nonPrivGlobal(k,rep) = corr_true_vs_nonPriv_global(1, 2);
        corr_true_vs_opPriv_global = corrcoef(raschModel_true, privOP_raschModel_global);
        exNew.corrMatrix_true_opPrivGlobal(k,rep) = corr_true_vs_opPriv_global(1, 2);
        corr_true_vs_opPriv_reopt = corrcoef(raschModel_true, privOP_raschModel_reopt);
        exNew.corrMatrix_true_opPrivReopt(k,rep) = corr_true_vs_opPriv_reopt(1, 2);
    end
    % error normalization
    exNew.error_true(k) = exNew.error_true(k)/(N*I*repetition);
    exNew.error_nonPriv_global(k) = exNew.error_nonPriv_global(k)/(N*I*repetition);
    exNew.error_opPriv_global(k) = exNew.error_opPriv_global(k)/(N*I*repetition);
    exNew.error_opPriv_reopt(k) = exNew.error_opPriv_reopt(k)/(N*I*repetition);
end