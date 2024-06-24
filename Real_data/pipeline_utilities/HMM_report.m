function [] = HMM_report(folder_path, show)
% folder_path: folder in which threre are .calibration. 3 classes files (remove or 
% shift in an other folder the .calibration. files used for the classifier
% training), .avaluation. 3 classes files, .smr.mat binary classifier file
% if show==true some images are printed, all the valriables producted are
% saved in the same falder path under the name of HMM_report.mat

CODE.Trial_start = 1;
CODE.Fixation_cross = 786;
CODE.Both_Hand = 773;
CODE.Both_Feet = 771;
CODE.Rest = 783;
CODE.Continuous_feedback = 781;
CODE.Target_hit = 897;
CODE.Target_miss = 898;

%% load classifier (binary classifier)
root = [folder_path '/'];
file_info = dir(root);
file_name = {file_info.name};

for k = 1:length(file_name)
    if isempty(strfind(file_name{k},'.smr.mat'))
        %pass
    else
        disp(file_name{k})
        load(file_name{k}) %settings
    end
end

task = settings.bci.smr.taskset.classes;
task_name = {settings.bci.smr.taskset.modality(1:2), settings.bci.smr.taskset.modality(3:4)};
    

%% load control files (3 classes)
type = '.calibration.';

[data_cal, trial_cal] = load_and_preproc(settings,folder_path,type,CODE);

%% Gaussian classifier 

[raw_prob_cal] = gaussian_classifier(settings, data_cal.data); %??????????

%% gaussian classifier matlab

task1.data = data_cal.data(data_cal.label==task(1),:);
task2.data = data_cal.data(data_cal.label==task(2),:);
rest.data = data_cal.data(data_cal.label==CODE.Rest,:);

n_gaus = 1; %gaussian per dimension

option = statset('MaxIter',1000,'TolFun',1e-6);
task1.model = fitgmdist(task1.data,n_gaus,"Options",option);

option = statset('MaxIter',1000,'TolFun',1e-6);
task2.model = fitgmdist(task2.data,n_gaus,"Options",option);

disp('Multi dimensional mixture model trained')

raw_prob_cal = [];

for k = 1:data_cal.n_sample
    sample = data_cal.data(k,:);

    task1_likelihood = pdf(task1.model,sample);
    task2_likelihood = pdf(task2.model,sample);

    raw_likelihood = [task1_likelihood, task2_likelihood];

    raw_prob_cal(k,:) = raw_likelihood./sum(raw_likelihood);
end
clear raw_likelihood task1_likelihood task2_likelihood

disp('Raw output generated')

%% Distribution and visual

time_base = [0:length(data_cal.data)-1]/data_cal.f;
label_plot(data_cal.label==task(1)) = 0.9;
label_plot(data_cal.label==CODE.Rest) = 0.5;
label_plot(data_cal.label==task(2)) = 0.1;
if show==true
    figure(1)
    plot(time_base, raw_prob_cal(:,1),'ko','MarkerFaceColor','k','MarkerSize',0.5)
    hold on
    plot(time_base(data_cal.label==task(1)), label_plot(data_cal.label==task(1)), 'b.','LineWidth',2)
    plot(time_base(data_cal.label==task(2)), label_plot(data_cal.label==task(2)), 'r.','LineWidth',2)
    plot(time_base(data_cal.label==CODE.Rest), label_plot(data_cal.label==CODE.Rest), 'g.','LineWidth',2)
    hold off
    xlim([time_base(1), time_base(end)])
    xlabel('sec')
    ylabel('prob')
    title('calibration output')
    legend('raw trn output',task_name{1},task_name{2},'rest')
end


%% HMM setting

base = [0:0.01:1]';
% A =
% B = 
% A_1 = 
% B_1 = 
%param = [A, B, A_1, B_1];
task1.hmm_state = hmm_state(base,'task_1');%,'param',param);
task2.hmm_state = hmm_state(base,'task_2');
rest.hmm_state = hmm_state(base,'rest');


state_name = ["task_1" "task_2" "rest"];
%state_name = ["both feet", "both hand", "rest"];
state_code = [task(1), task(2), CODE.Rest];

N_state = length(state_name);

% Transition matrix
%for the moment is static, then it will be dynamic based on the
%TRAVERSABILITY output
T = ones(N_state,N_state)*1/N_state;

% Static probability
%the class has the same chance of occurrence
static_prob = ones(N_state,1)*1/N_state;

if show==true
    y_lim = [0 12];
    figure(3)
    sgtitle('Data distribution - HMM state')
    subplot(131)
    histogram(raw_prob_cal(data_cal.label==task(1),1),100,'Normalization',"pdf",'FaceColor',"#0072BD")
    hold on
    plot(base,task1.hmm_state,'b-','LineWidth',2)
    ylim(y_lim)
    title(task_name{1})
    
    subplot(132)
    histogram(raw_prob_cal(data_cal.label==task(2),1),100,'Normalization',"pdf",'FaceColor',"#D95319")
    hold on
    plot(base,task2.hmm_state,'r-','LineWidth',2)
    ylim(y_lim)
    title(task_name{2})
    
    subplot(133)
    histogram(raw_prob_cal(data_cal.label==CODE.Rest,1),100,'Normalization',"pdf",'FaceColor',"#77AC30")
    hold on
    plot(base,rest.hmm_state,'g-','LineWidth',2)
    ylim(y_lim)
    title('rest')
end

%% evaluation files
type = '.evaluation.';

[data_eval, trial_eval] = load_and_preproc(settings,folder_path,type,CODE);

%% Gaussian classifier 

[raw_prob_eval] = gaussian_classifier(settings, data_eval.data); %??????????

%% Matlab gaussian classifier

raw_prob_eval = [];

for k = 1:data_eval.n_sample
    sample = data_cal.data(k,:);

    task1_likelihood = pdf(task1.model,sample);
    task2_likelihood = pdf(task2.model,sample);

    raw_likelihood = [task1_likelihood, task2_likelihood];

    raw_prob_eval(k,:) = raw_likelihood./sum(raw_likelihood);
end
clear raw_likelihood task1_likelihood task2_likelihood



%% FORWARD ALGORITHM between buffer FIFO VERSION + FRAMEWORK

% hmm parameters and initialization
hmm.dt_buffer = 1; %sec
hmm.buffer_len = hmm.dt_buffer*data_eval.f;
fifo = [];
initial_posterior = ones(N_state,1).*1/N_state;
hmm.pred.raw_output = initial_posterior.*ones(N_state,hmm.buffer_len-1); %array of posterior
%hard classification --> start with rest
hmm.pred.hard = CODE.Rest*ones(hmm.buffer_len-1,1); %label point wise

% Exponential framework parameters (applied to each class)
y_stable_exp = 0.3*ones(N_state,1); %were the classifier has to to stai during INC (intentional non control)
exp_frw = 0.3*ones(N_state,hmm.buffer_len-1);
alpha = 0.95; %weigth on previous value of the framework


for k = 1:data_eval.n_sample
    %new raw output
    new_data = raw_prob_eval(k);
    
    %fifo loading
    fifo = [fifo; new_data];

    %once the fifo is fully loaded --> hmm inference
    if length(fifo) == hmm.buffer_len
   
        %likelihood on the entire buffer
        likelihood = zeros(N_state,1);

        for m = 1:N_state
            likelihood(m) = prod(hmm_state(fifo,state_name(m)));
        end
        likelihood = likelihood./sum(likelihood); %normalization

        %posterior
        posterior = likelihood.*(hmm.pred.raw_output(:,end)'*T)';
     
        %normalization    
        posterior = posterior./sum(posterior);

        hmm.pred.raw_output = [hmm.pred.raw_output, posterior];

        % fifo update
        fifo(1) = [];
        
        % hard classification
        [max_pp,max_idx] = max(posterior);

        hmm.pred.hard = [hmm.pred.hard; state_code(max_idx)];

        % exponential framework
        if trial_eval.start(k)==1 || k == data_eval.n_sample %new tiral
            y = y_stable_exp;
        else
            y = exp_frw(:,end).*alpha + hmm.pred.raw_output(:,end).*(1-alpha);
        end
        exp_frw = [exp_frw, y];
        
    end
end



% plot
if show==true
    %--------------------------------------------------------------------------
    clear state_plot
    state_plot(hmm.pred.hard==task(1)) = 1;
    state_plot(hmm.pred.hard==task(2)) = 0;
    state_plot(hmm.pred.hard==CODE.Rest) = 0.5;
    
    clear label_plot
    label_plot(data_eval.label==task(1)) = 1;
    label_plot(data_eval.label==CODE.Rest) = 0.5;
    label_plot(data_eval.label==task(2)) = 0;
    
    time_base = [0:data_eval.n_sample-1]/data_eval.f;
    y_lim = [-0.1 1.1];
    figure(4)
    subplot(311)
    plot(time_base, raw_prob_eval(:,1),'ko','MarkerFaceColor','k','MarkerSize',0.5)
    xlim([time_base(1), time_base(end)])
    hold on
    plot(time_base(data_eval.label==task(1)), label_plot(data_eval.label==task(1)), 'b.','LineWidth',2)
    plot(time_base(data_eval.label==task(2)), label_plot(data_eval.label==task(2)), 'r.','LineWidth',2)
    plot(time_base(data_eval.label==CODE.Rest), label_plot(data_eval.label==CODE.Rest), 'g.','LineWidth',2)
    hold off
    xlabel('sec')
    ylabel('prob')
    title('Binary classifier output probabilities')
    legend('raw trn output',task_name{1},task_name{2},'rest')
    
    subplot(312)
    plot(time_base,label_plot,'k-','LineWidth',1.5)
    xlim([time_base(1), time_base(end)])
    ylim(y_lim)
    xlabel('t[sec]')
    hold on
    plot(time_base,state_plot,'g--','LineWidth',0.5)
    hold off
    legend('True lable','Predicted label')
    title('HMM hard prediction')
    
    subplot(313)
    plot(time_base,label_plot,'k-','LineWidth',1.5)
    xlim([time_base(1), time_base(end)])
    ylim(y_lim)
    xlabel('t[sec]')
    hold on
    plot(time_base,exp_frw(1,:),'b-','LineWidth',0.5)
    plot(time_base,exp_frw(2,:),'r-','LineWidth',0.5)
    plot(time_base,exp_frw(3,:),'g-','LineWidth',0.5)
    stem(time_base(trial_eval.start==1),ones(trial_eval.n,1),'m-','LineWidth',2,'Marker','none')
    hold off
    legend('True lable',['exp ',task_name{1}], ['exp ',task_name{2}],'exp rest', 'trial')
    title('Exponential framework output')
end

%% Performances
hit_miss = trial_eval.TYP(trial_eval.TYP==CODE.Target_hit | trial_eval.TYP==CODE.Target_miss);

accuracy.averall = 100*sum(hit_miss==CODE.Target_hit)/trial_eval.n;
accuracy.task1 = 100*sum(hit_miss(trial_eval.label==task(1))==CODE.Target_hit)/sum(trial_eval.label==task(1));
accuracy.task2 = 100*sum(hit_miss(trial_eval.label==task(2))==CODE.Target_hit)/sum(trial_eval.label==task(2));
accuracy.rest = 100*sum(hit_miss(trial_eval.label==CODE.Rest)==CODE.Target_hit)/sum(trial_eval.label==CODE.Rest);

if show==true
    figure(5)
    bar([1:4], [accuracy.averall, accuracy.task1, accuracy.task2, accuracy.rest])
    xticklabels(["Overall" task_name{1} task_name{2} "Rest"])
    ylabel('Accuracy %')
    ylim([0 100])
    title('Evaluation Accuracy')
    grid on
end

save([folder_path '/HMM_report.mat'], "accuracy","trial_cal","trial_eval","data_eval","data_cal","raw_prob_eval","raw_prob_cal");

end


















