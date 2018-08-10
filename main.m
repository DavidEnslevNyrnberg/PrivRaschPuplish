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
repetition = 50;
bootstrap = 1000;
count = 1;
alpha = 0.05;
options = optimoptions('fminunc','Algorithm','quasi-newton','Display','off','SpecifyObjectiveGradient',true,'MaxIter',10^5,'MaxFunEvals',10^5,'TolX',10^-5);

% privacy - model: parameters
epsi = 5; % privacy parameter
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

%% experiment 3 - Italian-Netherland data

%% experiment 4 - DTU data

%% experiment NEW - simmed priv global vs priv reopt
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
    plt1 = errorbar(nStudent, mean(exNew.corrMatrix_true_nonPrivGlobal,2), std(exNew.corrMatrix_true_nonPrivGlobal,0,2)/sqrt(size(exNew.corrMatrix_true_nonPrivGlobal,2))*norminv(1-alpha/2));
    set(plt1, optionName,optionValue{1})
    plt2 = errorbar(nStudent, mean(exNew.corrMatrix_true_opPrivGlobal,2), std(exNew.corrMatrix_true_opPrivGlobal,0,2)/sqrt(size(exNew.corrMatrix_true_opPrivGlobal,2))*norminv(1-alpha/2));
    set(plt2, optionName,optionValue{2})
    plt3 = errorbar(nStudent, mean(exNew.corrMatrix_true_opPrivReopt,2), std(exNew.corrMatrix_true_opPrivReopt,0,2)/sqrt(size(exNew.corrMatrix_true_opPrivReopt,2))*norminv(1-alpha/2));
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
    plt1 = plot(nStudent,exNew.error_true);
    set(plt1, optionName,optionValue{1})
    plt2 = plot(nStudent,exNew.error_nonPriv_global);
    set(plt2, optionName,optionValue{2})
    plt3 = plot(nStudent,exNew.error_opPriv_global);
    set(plt3, optionName,optionValue{3})
    plt4 = plot(nStudent,exNew.error_opPriv_reopt);
    set(plt4, optionName,optionValue{4})
    hold off

    xlim([nStudent(1)-xOff, nStudent(end)+xOff])
    ylim([0, .5])
    title('Misclassification rates')
    xlabel('class size [N]')
    ylabel('Misclassification rate')
    legend('True','nonPrivGlobal','opGlobal','opReopt','Location','southeast')
    
end