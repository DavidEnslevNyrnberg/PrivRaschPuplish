%%% epsilon-differential Private Rasch Model
% Authors:
% MSc Student Teresa Anna Steiner (s170063@student.dtu.dk) 
% MSc Student David Enslev Nyrnberg (s123997@student.dtu.dk) 
% Professor Lars Kai Hansen (lkai@dtu.dk) 
% Input:
% Dichomotous data matrix of 0's and 1's
% Output:

clear; clc; close all
% general code parameters
doPLOT = 1;
figSaver = 1;
if figSaver == 1
    mkdir('plots')
    mkdir('fig')
end
repetition = 50;
bootstrap = 1000; bootstrap = 100;
alpha = 0.05;
options = optimoptions('fminunc','Algorithm','quasi-newton','Display','off','SpecifyObjectiveGradient',true,'MaxIter',10^5,'MaxFunEvals',10^5,'TolX',10^-5);

% privacy - model: parameters
epsi = [1,5,10]; % privacy parameter
lam = 0.01; % regularization parameter

% distribution simulation [students and test] parameters:
nStudent = 40:40:200;
meanStu = 0;
sdStu = 1;

iTest = 20;
meanTest = 0;
sdTest = 2;

%% experiment 1 - Retrain vs Global beta est
% if statement of experiments to run
Experiment1

if doPLOT == 1
    xOff = 10;
    alpha = .05;

    figure(1)
    hold on
    Lcolor = {'g','b','r'};
    Mstyle = {'d','s','o'};
    optionName = {'Color','LineStyle','LineWidth','Marker','MarkerSize','MarkerFaceColor','CapSize'};
    for l = 1:3
        optionValue{l} = {Lcolor{l},'-',1,Mstyle{l},6,Lcolor{l},15};
    end
    plt1 = errorbar(nStudent, mean(EX1.corrMatrix_trueVSglobal,2), std(EX1.corrMatrix_trueVSglobal,0,2)/sqrt(size(EX1.corrMatrix_trueVSglobal,2))*norminv(1-alpha/2));
    set(plt1, optionName,optionValue{1})
    plt2 = errorbar(nStudent, mean(EX1.corrMatrix_trueVSreopt,2), std(EX1.corrMatrix_trueVSreopt,0,2)/sqrt(size(EX1.corrMatrix_trueVSreopt,2))*norminv(1-alpha/2));
    set(plt2, optionName,optionValue{2})
    plt3 = errorbar(nStudent, mean(EX1.corrMatrix_globalVSreopt,2), std(EX1.corrMatrix_globalVSreopt,0,2)/sqrt(size(EX1.corrMatrix_globalVSreopt,2))*norminv(1-alpha/2));
    set(plt3, optionName,optionValue{3})
    hold off

    xlim([nStudent(1)-xOff,nStudent(end)+xOff])
    ylim([0.8,1])
    title('Correlation coefficients')
    xlabel('class size [N]')
    ylabel('Correlation coefficients')
    legend('True vs Global','True vs Re-est','Global vs Re-est','Location','southeast')
end
%% experiment 2 - OP and SuffStat on sim data
Experiment2

if doPLOT == 1
    xOff = 10;
    alpha = .05;

    figure(90)
    hold on
    Lcolor = {'g','b','r','g','b','r'};
    Mstyle = {'d','s','o','d','s','o'};
    Lstyle = {'-','-','-',':',':',':'};
    optionName = {'Color','LineStyle','LineWidth','Marker','MarkerSize','MarkerFaceColor','CapSize'};
    for l = 1:6
        optionValue{l} = {Lcolor{l},Lstyle{l},1,Mstyle{l},6,Lcolor{l},15};
    end
    X1 = reshape(EX2.corrMatrix_nonPriv_opPriv(:,:,1),[5,50]);
    X2 = reshape(EX2.corrMatrix_nonPriv_opPriv(:,:,2),[5,50]);
    X3 = reshape(EX2.corrMatrix_nonPriv_opPriv(:,:,3),[5,50]);
    X4 = reshape(EX2.corrMatrix_nonPriv_suffPriv(:,:,1),[5,50]);
    X5 = reshape(EX2.corrMatrix_nonPriv_suffPriv(:,:,2),[5,50]);
    X6 = reshape(EX2.corrMatrix_nonPriv_suffPriv(:,:,3),[5,50]);
    plt1 = errorbar(nStudent, mean(X1,2), std(X1,0,2)/sqrt(size(X1,2))*norminv(1-alpha/2));
    set(plt1, optionName,optionValue{1})
    plt2 = errorbar(nStudent, mean(X2,2), std(X2,0,2)/sqrt(size(X2,2))*norminv(1-alpha/2));
    set(plt2, optionName,optionValue{2})
    plt3 = errorbar(nStudent, mean(X3,2), std(X3,0,2)/sqrt(size(X3,2))*norminv(1-alpha/2));
    set(plt3, optionName,optionValue{3})
    plt4 = errorbar(nStudent, mean(X4,2), std(X4,0,2)/sqrt(size(X4,2))*norminv(1-alpha/2));
    set(plt4, optionName,optionValue{4})
    plt5 = errorbar(nStudent, mean(X5,2), std(X5,0,2)/sqrt(size(X5,2))*norminv(1-alpha/2));
    set(plt5, optionName,optionValue{5})
    plt6 = errorbar(nStudent, mean(X6,2), std(X6,0,2)/sqrt(size(X6,2))*norminv(1-alpha/2));
    set(plt6, optionName,optionValue{6})
    hold off
    
    xlim([nStudent(1)-xOff, nStudent(end)+xOff])
    ylim([0,1])
    title('non-Private to OP or suffStat - correlation coefficients')
    xlabel('class size [N]')
    ylabel('Correlation coefficients')
    legend('OP\_epsilon=1','OP\_epsilon=5','OP\_epsilon=10','suffStat\_epsilon=1','suffStat\_epsilon=5','suffStat\_epsilon=10','Location','southeast')
    if figSaver == 1
        saveas(gcf,'plots/ex2_CorrMatrix_compareNONP','epsc');
        saveas(gcf,'fig/ex2_CorrMatrix_compareNONP.png');
    end
    
    figure(91)
    hold on
    Lcolor = {'g','b','r','g','b','r'};
    Mstyle = {'d','s','o','d','s','o'};
    Lstyle = {'-','-','-',':',':',':'};
    optionName = {'Color','LineStyle','LineWidth','Marker','MarkerSize','MarkerFaceColor','CapSize'};
    for l = 1:6
        optionValue{l} = {Lcolor{l},Lstyle{l},1,Mstyle{l},6,Lcolor{l},15};
    end
    X1 = reshape(EX2.corrMatrix_true_opPriv(:,:,1),[5,50]);
    X2 = reshape(EX2.corrMatrix_true_opPriv(:,:,2),[5,50]);
    X3 = reshape(EX2.corrMatrix_true_opPriv(:,:,3),[5,50]);
    X4 = reshape(EX2.corrMatrix_true_suffPriv(:,:,1),[5,50]);
    X5 = reshape(EX2.corrMatrix_true_suffPriv(:,:,2),[5,50]);
    X6 = reshape(EX2.corrMatrix_true_suffPriv(:,:,3),[5,50]);
    plt1 = errorbar(nStudent, mean(X1,2), std(X1,0,2)/sqrt(size(X1,2))*norminv(1-alpha/2));
    set(plt1, optionName,optionValue{1})
    plt2 = errorbar(nStudent, mean(X2,2), std(X2,0,2)/sqrt(size(X2,2))*norminv(1-alpha/2));
    set(plt2, optionName,optionValue{2})
    plt3 = errorbar(nStudent, mean(X3,2), std(X3,0,2)/sqrt(size(X3,2))*norminv(1-alpha/2));
    set(plt3, optionName,optionValue{3})
    plt4 = errorbar(nStudent, mean(X4,2), std(X4,0,2)/sqrt(size(X4,2))*norminv(1-alpha/2));
    set(plt4, optionName,optionValue{4})
    plt5 = errorbar(nStudent, mean(X5,2), std(X5,0,2)/sqrt(size(X5,2))*norminv(1-alpha/2));
    set(plt5, optionName,optionValue{5})
    plt6 = errorbar(nStudent, mean(X6,2), std(X6,0,2)/sqrt(size(X6,2))*norminv(1-alpha/2));
    set(plt6, optionName,optionValue{6})
    hold off
    
    xlim([nStudent(1)-xOff,nStudent(end)+xOff])
    ylim([0,1])
    title('True to OP or suffStat - correlation coefficients')
    xlabel('class size [N]')
    ylabel('Correlation coefficients')
    legend('OP\_epsilon=1','OP\_epsilon=5','OP\_epsilon=10','suffStat\_epsilon=1','suffStat\_epsilon=5','suffStat\_epsilon=10','Location','southeast')
    if figSaver == 1
        saveas(gcf,'plots/ex2_CorrMatrix_compareTRUE','epsc')
        saveas(gcf,'fig/ex2_CorrMatrix_compareTRUE.png');
    end
    
    figure(92)
    hold on
    Lcolor = {'k','g','b','r'};
    Mstyle = {'p','d','s','o'};
    optionName = {'Color','LineStyle','LineWidth','Marker','MarkerSize','MarkerFaceColor'};
    for l = 1:4
        optionValue{l} = {Lcolor{l},'-',1,Mstyle{l},6,Lcolor{l}};
    end
    plt1 = plot(nStudent,EX2.error_true);
    set(plt1, optionName,optionValue{1})
    plt2 = plot(nStudent,EX2.error_opPriv(:,1)');
    set(plt2, optionName,optionValue{2})
    plt3 = plot(nStudent,EX2.error_opPriv(:,2)');
    set(plt3, optionName,optionValue{3})
    plt4 = plot(nStudent,EX2.error_opPriv(:,3)');
    set(plt4, optionName,optionValue{4})
    
    hold off
    xlim([nStudent(1)-xOff,nStudent(end)+xOff])
    ylim([0,.5])
    title('OP-Private Misclassification rates')
    xlabel('class size [N]')
    ylabel('Misclassification rate')
    legend('True','epsilon=1','epsilon=5','epsilon=10','Location','southeast')
    if figSaver == 1
        saveas(gcf,'plots/ex23_OP_Misclass','epsc')
        saveas(gcf,'fig/ex23_OP_Misclass.png')
    end
 
    figure(93)
    hold on
    Lcolor = {'k','g','b','r'};
    Mstyle = {'p','d','s','o'};
    optionName = {'Color','LineStyle','LineWidth','Marker','MarkerSize','MarkerFaceColor'};
    for l = 1:4
        optionValue{l} = {Lcolor{l},'-',1,Mstyle{l},6,Lcolor{l}};
    end
    plt1 = plot(nStudent,EX2.error_true);
    set(plt1, optionName,optionValue{1})
    plt2 = plot(nStudent,EX2.error_suffPriv(:,1)');
    set(plt2, optionName,optionValue{2})
    plt3 = plot(nStudent,EX2.error_suffPriv(:,2)');
    set(plt3, optionName,optionValue{3})
    plt4 = plot(nStudent,EX2.error_suffPriv(:,3)');
    set(plt4, optionName,optionValue{4})
    hold off
    
    xlim([nStudent(1)-xOff,nStudent(end)+xOff])
    ylim([0,.5])
    title('suffStat-Private Misclassification rates')
    xlabel('class size [N]')
    ylabel('Misclassification rate')
    legend('True','epsilon=1','epsilon=5','epsilon=10','Location','southeast')
    if figSaver == 1
        saveas(gcf,'plots/ex23_suffStat_Misclass','epsc')
        saveas(gcf,'fig/ex23_suffStat_Misclass.png')
    end
end
%% experiment 3 - Italian-Netherland data
% set path to data set for rasch estimation
dir = fullfile('.\ItalianNetherlandData');
gradeName = 'final_grades.xlsx';
gradeDir = fullfile(dir,gradeName);

EX3_epsilon = epsi(2);

% Experiment3
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
        student = data_new(n,:);
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

%% experiment 4 - DTU data
% Experiment4

%% experiment NEW - simmed priv global vs priv reopt
clear epsilon
EXNEW_epsilon = epsi(2);
ExperimentNEW

if doPLOT == 1
    xOff = 10;
    alpha = .05;
    
    figure(1)
    hold on
    Lcolor = {'g','b','r'};
    Mstyle = {'d','s','o'};
    optionName = {'Color', 'LineStyle', 'LineWidth', 'Marker', 'MarkerSize', 'MarkerFaceColor', 'CapSize'};
    for l = 1:3
        optionValue{l} = {Lcolor{l}, '-', 1, Mstyle{l}, 6, Lcolor{l}, 15};
    end
    plt1 = errorbar(nStudent, mean(EX2.corrMatrix_true_nonPrivGlobal,2), std(EX2.corrMatrix_true_nonPrivGlobal,0,2)/sqrt(size(EX2.corrMatrix_true_nonPrivGlobal,2))*norminv(1-alpha/2));
    set(plt1, optionName,optionValue{1})
    plt2 = errorbar(nStudent, mean(EX2.corrMatrix_true_opPrivGlobal,2), std(EX2.corrMatrix_true_opPrivGlobal,0,2)/sqrt(size(EX2.corrMatrix_true_opPrivGlobal,2))*norminv(1-alpha/2));
    set(plt2, optionName,optionValue{2})
    plt3 = errorbar(nStudent, mean(EX2.corrMatrix_true_opPrivReopt,2), std(EX2.corrMatrix_true_opPrivReopt,0,2)/sqrt(size(EX2.corrMatrix_true_opPrivReopt,2))*norminv(1-alpha/2));
    set(plt3, optionName,optionValue{3})
    hold off

    xlim([nStudent(1)-xOff, nStudent(end)+xOff])
    ylim([0.7,1])
    title('Correlation coefficients')
    xlabel('class size [N]')
    ylabel('Correlation coefficients')
    legend('True vs nonPriv global','True vs opPriv global','True vs opPriv reopt','Location','southeast')

    figure(2)
    hold on
    Lcolor = {'k','g','b','r'};
    Mstyle = {'p','d','s','o'};
    Lstyle = {'-','-','--',':'};
    optionName = {'Color','LineStyle','LineWidth','Marker','MarkerSize','MarkerFaceColor'};
    for l = 1:4
        optionValue{l} = {Lcolor{l},Lstyle{l},1,Mstyle{l},6,Lcolor{l}};
    end
    plt1 = plot(nStudent,EX2.error_true);
    set(plt1, optionName,optionValue{1})
    plt2 = plot(nStudent,EX2.error_nonPriv_global);
    set(plt2, optionName,optionValue{2})
    plt3 = plot(nStudent,EX2.error_opPriv_global);
    set(plt3, optionName,optionValue{3})
    plt4 = plot(nStudent,EX2.error_opPriv_reopt);
    set(plt4, optionName,optionValue{4})
    hold off

    xlim([nStudent(1)-xOff, nStudent(end)+xOff])
    ylim([0, .5])
    title('Misclassification rates')
    xlabel('class size [N]')
    ylabel('Misclassification rate')
    legend('True','nonPrivGlobal','opGlobal','opReopt','Location','southeast')
    
end