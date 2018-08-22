% initialize result arrays
EX2.corrMatrix_true_opPriv = zeros(length(nStudent),repetition,length(epsi));
EX2.corrMatrix_nonPriv_opPriv= zeros(length(nStudent),repetition,length(epsi));
EX2.corrMatrix_true_suffPriv = zeros(length(nStudent), repetition, length(epsi));
EX2.corrMatrix_nonPriv_suffPriv = zeros(length(nStudent), repetition, length(epsi));
EX2.error_true = zeros(1,length(nStudent));
EX2.error_opPriv = zeros(length(nStudent), length(epsi));
EX2.error_suffPriv = zeros(length(nStudent), length(epsi));

I = iTest;
for k = 1:length(nStudent)
    N = nStudent(k);
    % initialize rasch size
    opPriv_raschModel = zeros(N, I, length(epsi));
    suffStatPriv_raschModel = zeros(N, I, length(epsi));
    for rep = 1:repetition
        % simulate beta and delta values
        betaTrue = meanStu+randn(N,1)*sdStu;
        deltaTrue = meanTest+sdTest*randn(1,I);
        % model true Rasch model
        raschModel_true = raschModel(betaTrue, deltaTrue);
        % simulate train and test data
        simTrainData = binornd(1, raschModel_true);
        simTestData = binornd(1, raschModel_true);
        
        % non private rasch estimation
        w_ini = 0.001*randn(N+I, 1);
        % global section
        w_nonPriv = fminunc(@(w) rasch_neglog_likelihood(w, simTrainData, lam), w_ini, options);
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
        
        % private rasch estimation
        for l = 1:length(epsi)
            % OP private rasch estimation
            b_norm = gamrnd(I, sqrt(I) / epsi(l));
            bx = normrnd(0,1,[1,I]);
            b = bx / norm(bx) * b_norm;
            w_opPriv=fminunc(@(w) rasch_neglog_likelihood_private_TD(w,simTrainData,lam,b),w_ini,options);
            
            opPriv_est_beta_global = w_opPriv(1:N);
            opPriv_est_delta = w_opPriv(N+1:end)';
            % re-optimized est non-priv
            opPriv_est_beta = zeros(N, 1);
            for n = 1:N
                beta_n = 0;
                student = simTrainData(n,:);
                opPriv_est_beta(n) = fminunc(@(beta) single_beta_TD(beta, opPriv_est_delta, student, lam), beta_n, options);
            end
            % Sufficient statistic
            [suffStatPriv_est_delta, suffStatPriv_est_beta] = suff_stat_parameter_est(simTrainData, lam, epsi(l), N, I, options);
            opPriv_raschModel(:,:,l) = raschModel(opPriv_est_beta, opPriv_est_delta);
            suffStatPriv_raschModel(:,:,l) = raschModel(suffStatPriv_est_beta, suffStatPriv_est_delta);
            % correlation matrixes corrMatrix_true_opPriv
            corr_true_vs_opPriv = corrcoef(raschModel_true, opPriv_raschModel(:,:,l));
            EX2.corrMatrix_true_opPriv(k,rep,l) = corr_true_vs_opPriv(1, 2);
            corr_nonPriv_vs_opPriv = corrcoef(nonPriv_raschModel, opPriv_raschModel(:,:,l));
            EX2.corrMatrix_nonPriv_opPriv(k,rep,l) = corr_nonPriv_vs_opPriv(1, 2);
            corr_true_vs_suffStatPriv = corrcoef(raschModel_true, suffStatPriv_raschModel(:,:,l));        
            EX2.corrMatrix_true_suffPriv(k,rep,l) = corr_true_vs_suffStatPriv(1, 2); 
            corr_nonPriv_vs_suffStatPriv = corrcoef(nonPriv_raschModel, suffStatPriv_raschModel(:,:,l));        
            EX2.corrMatrix_nonPriv_suffPriv(k,rep,l) = corr_nonPriv_vs_suffStatPriv(1, 2); 
        end
        % true, global and reopt error to test set
        EX2.error_true(k) = EX2.error_true(k)+sum(sum(abs(round(raschModel_true)-simTestData)));
        EX2.error_opPriv(k,:) = EX2.error_opPriv(k,:)+reshape(sum(sum(abs(round(opPriv_raschModel)-simTestData))), size(epsi));
        EX2.error_suffPriv(k,:) = EX2.error_suffPriv(k,:)+reshape(sum(sum(abs(round(suffStatPriv_raschModel)-simTestData))), size(epsi));
    end
    % error normalization
    EX2.error_true(k) = EX2.error_true(k)/(N*I*repetition);
    EX2.error_opPriv(k,:) = EX2.error_opPriv(k,:)/(N*I*repetition);
    EX2.error_suffPriv(k,:) = EX2.error_suffPriv(k,:)/(N*I*repetition);
end