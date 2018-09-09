%%% epsilon-differential Private Rasch Model
% Authors:
% MSc Student Teresa Anna Steiner (s170063@student.dtu.dk) 
% MSc Student David Enslev Nyrnberg (s123997@student.dtu.dk) 
% Professor Lars Kai Hansen (lkai@dtu.dk) 

clear; clc; close all
% general code parameters
doPLOT = 1;
figSaver = 0;
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

I = 20;
meanTest = 0;
sdTest = 2;

%% experiment 1 - Retrain vs Global beta est
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

% load real data
loadData = xlsread(gradeDir, 'Exam (Second time)');
data = round(loadData(:,2:end-1)./max(loadData(:,2:end-1)));
[classSize,iTest] = size(data);
nStudent = ceil(linspace(classSize/3, classSize, 3));

% select privacy strengh
EX3_epsilon = epsi(2);

Experiment3

%%
if doPLOT == 1
    xOff = 2;
    
    figure(9)
    hold on
    Lcolor = {'g','b','r'};
    Mstyle = {'d','s','o'};
    optionName = {'Color','LineStyle','LineWidth','Marker','MarkerSize','MarkerFaceColor','CapSize'};
    for l = 1:3
        optionValue{l} = {Lcolor{l},'-',1,Mstyle{l},6,Lcolor{l},15};
    end
    plt1 = errorbar(nStudent, mean(EX3.corrMatrix_nonPriv_opPriv,2), std(EX3.corrMatrix_nonPriv_opPriv,0,2)/sqrt(size(EX3.corrMatrix_nonPriv_opPriv,2))*norminv(1-alpha/2));
    set(plt1, optionName,optionValue{1})
    plt2 = errorbar(nStudent, mean(EX3.corrMatrix_nonPriv_suffPriv,2), std(EX3.corrMatrix_nonPriv_suffPriv,0,2)/sqrt(size(EX3.corrMatrix_nonPriv_suffPriv,2))*norminv(1-alpha/2));
    set(plt2, optionName,optionValue{2})
    plt3 = errorbar(nStudent, mean(EX3.corrMatrix_opPriv_suffPriv,2), std(EX3.corrMatrix_opPriv_suffPriv,0,2)/sqrt(size(EX3.corrMatrix_opPriv_suffPriv,2))*norminv(1-alpha/2));
    set(plt3, optionName,optionValue{3})
    hold off
    
    xlim([nStudent(1)-xOff, nStudent(end)+xOff])
    ylim([0.7,1])
    title(sprintf('Correlation coefficients - epsilon=%d',EX3_epsilon))
    xlabel('class size [N]')
    ylabel('Correlation coefficients')
    legend('nonPriv vs OP','nonPriv vs SuffStat','OP vs SuffStat','Location','southeast')
    if figSaver==1
        saveas(gcf,'plots/ex3_CorrMatrix_real','epsc')
        saveas(gcf,'fig/ex3_CorrMatrix_real.png')
    end
    
    figure(10)
    hold on
    Lcolor = {'g','b','r'};
    Mstyle = {'d','s','o'};
    optionName = {'Color','LineStyle','LineWidth','Marker','MarkerSize','MarkerFaceColor'};
    for l = 1:3
        optionValue{l} = {Lcolor{l},'-',1,Mstyle{l},6,Lcolor{l}};
    end
    plt1 = plot(nStudent,EX3.error_opPriv);
    set(plt1, optionName,optionValue{1})
    plt2 = plot(nStudent,EX3.error_suffPriv);
    set(plt2, optionName,optionValue{2})
    plt3 = plot(nStudent,EX3.error_nonPriv);
    set(plt3, optionName,optionValue{3})
    hold off
    
    xlim([nStudent(1)-xOff,nStudent(end)+xOff])
    ylim([0,.5])
    title('Misclassification rates')
    xlabel('class size [N]')
    ylabel('Misclassification rate')
    legend('OP','SuffStat','non-private','Location','southeast')
    if figSaver == 1
        saveas(gcf,'plots/ex3_Misclass','epsc')
        saveas(gcf,'fig/ex3_Misclass.png')
    end

    figure(11)
    sizeK = 1;
    hold on
    plt11 = plot(EX3.nonPriv_raschModel_vec{sizeK},EX3.opPriv_raschModel_vec{sizeK},'go','MarkerFaceColor','g');
    plt12 = plot(EX3.nonPriv_raschModel_vec{sizeK},EX3.suffStatPriv_raschModel_vec{sizeK},'rs','MarkerFaceColor','r');
    plot([0,1],[0,1],'b-','LineWidth',2);
    hold off
    
    xlim([0,1])
    ylim([0,1])
    title(sprintf('N=%d Rasch estimation: non-Private vs. OP or suffStat',nStudent(sizeK)))
    xlabel('non-Private estimation')
    ylabel('Private estimation')
    legend({'nonPriv vs OP','nonPriv vs suffStat','correlation-line'},'Location','southeast')
    if figSaver == 1
        saveas(gcf,sprintf('plots/ex3_scatter_%d',nStudent(sizeK)),'epsc')
        saveas(gcf,sprintf('fig/ex3_scatter_%d.png',nStudent(sizeK)))
    end

    figure(12)
    sizeK = 2;
    hold on
    plt21 = plot(EX3.nonPriv_raschModel_vec{sizeK},EX3.opPriv_raschModel_vec{sizeK},'go','MarkerFaceColor','g');
    plt22 = plot(EX3.nonPriv_raschModel_vec{sizeK},EX3.suffStatPriv_raschModel_vec{sizeK},'rs','MarkerFaceColor','r');
    plot([0,1],[0,1],'b-','LineWidth',2);
    hold off
    
    xlim([0,1])
    ylim([0,1])
    title(sprintf('N=%d Rasch estimation: non-Private vs. OP or suffStat',nStudent(2)))
    xlabel('non-Private estimation')
    ylabel('Private estimation')
    legend({'nonPriv vs OP','nonPriv vs suffStat','correlation-line'},'Location','southeast')
    if figSaver == 1
        saveas(gcf,sprintf('plots/ex3_scatter_%d',nStudent(sizeK)),'epsc')
        saveas(gcf,sprintf('fig/ex3_scatter_%d.png',nStudent(sizeK)))
    end

    figure(13)
    sizeK = 3;
    hold on
    plt31 = plot(EX3.nonPriv_raschModel_vec{sizeK},EX3.opPriv_raschModel_vec{sizeK},'go','MarkerFaceColor','g');
    plt32 = plot(EX3.nonPriv_raschModel_vec{sizeK},EX3.suffStatPriv_raschModel_vec{sizeK},'rs','MarkerFaceColor','r');
    plot([0,1],[0,1],'b-','LineWidth',2);
    hold off
    
    xlim([0,1])
    ylim([0,1])
    title(sprintf('N=%d Rasch estimation: non-Private vs. OP & SuffStat',nStudent(sizeK)))
    xlabel('non-Private estimation')
    ylabel('Private estimation')
    legend({'nonPriv vs OP','nonPriv vs suffStat','correlation-line'},'Location','southeast')
    if figSaver == 1
        saveas(gcf,sprintf('plots/ex3_scatter_%d',nStudent(sizeK)),'epsc')
        saveas(gcf,sprintf('fig/ex3_scatter_%d.png',nStudent(sizeK)))
    end
end

%% experiment 4 - DTU data
% load real data
loadData = load('DTU_Anonymized.mat');
data = loadData.BinDataPerm;
[classSize,iTest] = size(data);
nStudent = ceil(linspace(classSize/5, classSize, 5));

EX4_epsilon = epsi(2);

Experiment4

if doPLOT == 1
    xOff = 10;
    
    figure(14)
    hold on
    Lcolor = {'g','b','r'};
    Mstyle = {'d','s','o'};
    optionName = {'Color','LineStyle','LineWidth','Marker','MarkerSize','MarkerFaceColor','CapSize'};
    for l = 1:3
        optionValue{l} = {Lcolor{l},'-',1,Mstyle{l},6,Lcolor{l},15};
    end
    plt1 = errorbar(nStudent, mean(EX4.corrMatrix_nonPriv_opPriv,2), std(EX4.corrMatrix_nonPriv_opPriv,0,2)/sqrt(size(EX4.corrMatrix_nonPriv_opPriv,2))*norminv(1-alpha/2));
    set(plt1, optionName,optionValue{1})
    plt2 = errorbar(nStudent, mean(EX4.corrMatrix_nonPriv_suffPriv,2), std(EX4.corrMatrix_nonPriv_suffPriv,0,2)/sqrt(size(EX4.corrMatrix_nonPriv_suffPriv,2))*norminv(1-alpha/2));
    set(plt2, optionName,optionValue{2})
    plt3 = errorbar(nStudent, mean(EX4.corrMatrix_opPriv_suffPriv,2), std(EX4.corrMatrix_opPriv_suffPriv,0,2)/sqrt(size(EX4.corrMatrix_opPriv_suffPriv,2))*norminv(1-alpha/2));
    set(plt3, optionName,optionValue{3})
    hold off
    
    xlim([nStudent(1)-xOff,nStudent(end)+xOff])
    ylim([0.5,1])
    title(sprintf('Correlation coefficients - epsilon=%d',EX4_epsilon))
    xlabel('class size [N]')
    ylabel('Correlation coefficients')
    legend('nonPriv vs OP','nonPriv vs SuffStat','OP vs SuffStat','Location','southeast')
    if figSaver == 1
        saveas(gcf,'plots/ex4_CorrMatrix_DTU','epsc')
        saveas(gcf,'fig/ex4_CorrMatrix_DTU.png')
    end
    
    figure(15)
    hold on
    Lcolor = {'g','b','r'};
    Mstyle = {'d','s','o'};
    optionName = {'Color','LineStyle','LineWidth','Marker','MarkerSize','MarkerFaceColor'};
    for l = 1:3
        optionValue{l} = {Lcolor{l},'-',1,Mstyle{l},6,Lcolor{l}};
    end
    plt1 = plot(nStudent,EX4.error_nonPriv);
    set(plt1, optionName,optionValue{1})
    plt2 = plot(nStudent,EX4.error_opPriv);
    set(plt2, optionName,optionValue{2})
    plt3 = plot(nStudent,EX4.error_suffPriv);
    set(plt3, optionName,optionValue{3})
    hold off
    
    xlim([nStudent(1)-xOff,nStudent(end)+xOff])
    ylim([0,.5])
    title('Misclassification rates')
    xlabel('class size [N]')
    ylabel('Misclassification rate')
    legend('OP','SuffStat','non-private','Location','southeast')
    if figSaver == 1
        saveas(gcf,'plots/ex4_Misclass_DTU','epsc')
        saveas(gcf,'fig/ex4_Misclass_DTU.png')
    end
end
%% experiment NEW - simmed priv global vs priv reopt
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