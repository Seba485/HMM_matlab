function [] = HMM_report(folder_path, show)
    % HMM_report(folder_path, show)
    % folder_path: folder in which threre are 3 classes .calibration., 
    % .smr.mat binary classifier file and all the evaluationn files with the
    % relative rosbag files.
    % if show==true some images are shown, accuracy and info structure are
    % saved in the same folder path under the name of HMM_report.mat
    
    CODE.Trial_start = 1;
    CODE.Fixation_cross = 786;
    CODE.Both_Hand = 773;
    CODE.Both_Feet = 771;
    CODE.Rest = 783;
    CODE.Continuous_feedback = 781;
    CODE.Target_hit = 897;
    CODE.Target_miss = 898;
    
    % key_words
    keyWords.classifier = ".smr.mat";
    keyWords.calibration = ".calibration.";
    keyWords.eval_bi = ".binary.";
    keyWords.eval_T = [".no_T.", ".T_1.", ".T_2."];
    
    % accuracy
    accuracy.binary.info = "binary (task_1, task_2) evaluation average accuracy between runs";
    accuracy.binary.t1 = 0;
    accuracy.binary.t2 = 0;
    accuracy.binary.overall = 0;
    
    no_T.info = "hmm evaluation with a constant traversability matrix at 1/3 [run 1, ..., run n, avg]";
    no_T.t1 = [];
    no_T.t2 = [];
    no_T.rest = [];
    no_T.overall = [];
    
    T_1.info = "hmm evaluation only the cue direction free [run 1, ..., run n, avg]";
    T_1.t1 = [];
    T_1.t2 = [];
    T_1.rest = [];
    T_1.overall = [];
    
    T_2.info = "hmm evaluation 2 direction free, the cue one and randomically an other one [run 1, ..., run n, avg]";
    T_2.t1 = [];
    T_2.t2 = [];
    T_2.rest = [];
    T_2.overall = [];
    
    acc_vect = {no_T, T_1, T_2}; %same order of keyWords.eval_T
    
    hmm_buffer_len = 16; 
    f = 16;%Hz
    
    %% load classifier (binary classifier)
    root = [folder_path '/'];
    file_info = dir(root);
    file_name = {file_info.name};
    
    for k = 1:length(file_name)
        if isempty(strfind(file_name{k},keyWords.classifier))
            %pass
        else
            disp(file_name{k});
            load([root file_name{k}]) %settings
        end
    end
    
    task = settings.bci.smr.taskset.classes;
    task_name = {settings.bci.smr.taskset.modality(1:2), settings.bci.smr.taskset.modality(3:4)};
    
    ch_feature = settings.bci.smr.channels;
    freq_feature = settings.bci.smr.bands;
    ch_names = settings.acq.channel_lbl; %cell array
    
    features_string = '';
    for run = 1:length(ch_feature)
        features_string = [features_string, string(ch_names{ch_feature(run)}), '-[', string(freq_feature{ch_feature(run)}), '] '];
    end
    features_string = join(features_string);
    
    info.task_code = task;
    info.task_name = task_name;
    info.info = join(["the hmm framework try to understand the rest phase given the other 2 task: ", string(task(1)),"(",task_name{1},") ",string(task(2)),"(",task_name{2},")"]);
    info.features = features_string;
    
    %% load control files (binary or tri classes)
    type = keyWords.calibration;
    
    [data_cal, trial_cal] = load_and_preproc(settings,folder_path,type,CODE);
    
    % Gaussian classifier 
    if isstruct(data_cal)
        [raw_prob_cal] = gaussian_classifier(settings, data_cal.data);
    
    
    % gaussian classifier matlab
    
    % task1.data = data_cal.data(data_cal.label==task(1),:);
    % task2.data = data_cal.data(data_cal.label==task(2),:);
    % rest.data = data_cal.data(data_cal.label==CODE.Rest,:);
    % 
    % n_gaus = 1; %gaussian per dimension
    % 
    % option = statset('MaxIter',1000,'TolFun',1e-6);
    % task1.model = fitgmdist(task1.data,n_gaus,"Options",option);
    % 
    % option = statset('MaxIter',1000,'TolFun',1e-6);
    % task2.model = fitgmdist(task2.data,n_gaus,"Options",option);
    % 
    % disp('Multi dimensional mixture model trained')
    % 
    % raw_prob_cal = [];
    % 
    % for k = 1:data_cal.n_sample
    %     sample = data_cal.data(k,:);
    % 
    %     task1_likelihood = pdf(task1.model,sample);
    %     task2_likelihood = pdf(task2.model,sample);
    % 
    %     raw_likelihood = [task1_likelihood, task2_likelihood];
    % 
    %     raw_prob_cal(k,:) = raw_likelihood./sum(raw_likelihood);
    % end
    % clear raw_likelihood task1_likelihood task2_likelihood
    
        %% Distribution and visual
        time_base = [0:length(data_cal.data)-1]/data_cal.f;
        label_plot(data_cal.label==task(1)) = 0.9;
        label_plot(data_cal.label==CODE.Rest) = 0.5;
        label_plot(data_cal.label==task(2)) = 0.1;
        
        base = [0:0.01:1]';
        % A =
        % B = 
        % A_1 = 
        % B_1 = 
        %param = [A, B, A_1, B_1];
        task1.hmm_state = hmm_state(base,'task_1');%,'param',param);
        task2.hmm_state = hmm_state(base,'task_2');
        rest.hmm_state = hmm_state(base,'rest');
        
        if show==true
            y_lim = [0 1];
            figure(1)
            plot(time_base, raw_prob_cal(:,1),'ko','MarkerFaceColor','k','MarkerSize',0.5)
            hold on
            plot(time_base(data_cal.label==task(1)), label_plot(data_cal.label==task(1)), 'b.','LineWidth',2)
            plot(time_base(data_cal.label==task(2)), label_plot(data_cal.label==task(2)), 'r.','LineWidth',2)
            plot(time_base(data_cal.label==CODE.Rest), label_plot(data_cal.label==CODE.Rest), 'g.','LineWidth',2)
            hold off
            xlim([time_base(1), time_base(end)])
            ylim(y_lim);
            xlabel('sec')
            ylabel('prob')
            title('Calibration output '+features_string)
            legend('raw trn output',task_name{1},task_name{2},'rest')
            
            y_lim = [0 12];
            figure(2)
            sgtitle('Data distribution Calibration - HMM state')
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
    end
    
    %% evaluation files (2 classes for classifier check)
    type = keyWords.eval_bi;
    
    [data_eval, trial_eval] = load_and_preproc(settings,folder_path,type,CODE);
    
    %% Gaussian classifier 
    if isstruct(data_eval)
        [raw_prob_eval] = gaussian_classifier(settings, data_eval.data);
        
        %% Accuracy
        hit_miss = trial_eval.TYP(trial_eval.TYP==CODE.Target_hit | trial_eval.TYP==CODE.Target_miss);
        
        accuracy.binary.overall = 100*sum(hit_miss==CODE.Target_hit)/trial_eval.n;
        accuracy.binary.t1 = 100*sum(hit_miss(trial_eval.label==task(1))==CODE.Target_hit)/sum(trial_eval.label==task(1));
        accuracy.binary.t2 = 100*sum(hit_miss(trial_eval.label==task(2))==CODE.Target_hit)/sum(trial_eval.label==task(2));
        
        
        %% Distribution and Visual
        
        time_base = [0:length(data_eval.data)-1]/data_eval.f;
        label_plot(data_eval.label==task(1)) = 0.9;
        label_plot(data_eval.label==task(2)) = 0.1;

        base = [0:0.01:1]';
        % A =
        % B = 
        % A_1 = 
        % B_1 = 
        %param = [A, B, A_1, B_1];
        task1.hmm_state = hmm_state(base,'task_1');%,'param',param);
        task2.hmm_state = hmm_state(base,'task_2');
        rest.hmm_state = hmm_state(base,'rest');
        
        if show==true
            y_lim = [0 1];
            figure(3)
            plot(time_base, raw_prob_eval(:,1),'ko','MarkerFaceColor','k','MarkerSize',0.5)
            hold on
            plot(time_base(data_eval.label==task(1)), label_plot(data_eval.label==task(1)), 'b.','LineWidth',2)
            plot(time_base(data_eval.label==task(2)), label_plot(data_eval.label==task(2)), 'r.','LineWidth',2)
            %plot(time_base(data_eval.label==CODE.Rest), label_plot(data_eval.label==CODE.Rest), 'g.','LineWidth',2)
            hold off
            xlim([time_base(1), time_base(end)])
            ylim(y_lim)
            xlabel('sec')
            ylabel('prob')
            title('Evaluation output')
            legend('raw tst output',task_name{1},task_name{2})
            
            y_lim = [0 12];
            figure(4)
            sgtitle('Data distribution Evaluation')
            subplot(121)
            histogram(raw_prob_eval(data_eval.label==task(1),1),100,'Normalization',"pdf",'FaceColor',"#0072BD")
            hold on
            plot(base,task1.hmm_state,'b-','LineWidth',2)
            ylim(y_lim)
            title(task_name{1})
        
            subplot(122)
            histogram(raw_prob_eval(data_eval.label==task(2),1),100,'Normalization',"pdf",'FaceColor',"#D95319")
            hold on
            plot(base,task2.hmm_state,'r-','LineWidth',2)
            ylim(y_lim)
            title(task_name{2})
        
            figure(5)
            bar([1:3], [accuracy.binary.overall, accuracy.binary.t1, accuracy.binary.t2])
            xticklabels(["Overall" task_name{1} task_name{2}])
            ylabel('Accuracy %')
            ylim([0 100])
            title('Binary evaluation accuracy')
            grid on
        
        end
    end
  
    %% evaluation with hmm
    
    root = [folder_path '/'];
    file_info = dir(root);
    file_name = {file_info.name};
    
    for mode = 1:length(keyWords.eval_T)
        
        %load csv and gdf (the gdf file is usefull only for the header event)
        h = {}; i = 0;
        j = 0;
        smrbci = {};
        hmm = {};
        exp = {};
        T = {};
        for k = 1:length(file_name)
            if isempty(strfind(file_name{k},join([keyWords.eval_T(mode),"bag"],""))) ...
                    && isempty(strfind(file_name{k},join([keyWords.eval_T(mode),"gdf"],"")))
                %pass
            else
                disp(file_name{k});
                if isempty(strfind(file_name{k},join([keyWords.eval_T(mode),"bag"],""))) %its a gdf file
                    i = i+1;
                    [~, h{i}] = sload([root file_name{k}]); %signal and header
                else %its a bag file
                    j = j+1;
                    bag = rosbag([root file_name{k}]);

                    %read the bag topics
                    smrbci{j} = readMessages(select(bag,"Topic", '/smrbci/neuroprediction'),'DataFormat','struct');
                    hmm{j} = readMessages(select(bag,"Topic", '/hmm/neuroprediction'),'DataFormat','struct');
                    exp{j} = readMessages(select(bag,"Topic", '/integrator/neuroprediction'),'DataFormat','struct');
                    T{j} = readMessages(select(bag,"Topic", '/traversability_output_topic'),'DataFormat','struct'); %useless for the moment
                end
            end
        end
        
        n_run = length(h);
        if n_run~=0
    
            %recompute the position of the event related to the PSD windows
            win_shift =  settings.modules.smr.win.shift;
            internal_win_size = settings.modules.smr.psd.win;
            Fs = h{1}.SampleRate;
            h_PSD = {};
            for run = 1:n_run
                h_PSD{run}.EVENT.POS = proc_pos2win(h{run}.EVENT.POS, win_shift*Fs, 'backward', internal_win_size*Fs);
                h_PSD{run}.EVENT.DUR = round(h{run}.EVENT.DUR./(win_shift*Fs));
                h_PSD{run}.EVENT.TYP = h{run}.EVENT.TYP;
            end
           
        
            %for each run compute the accuracy
            %acc_vect(mode).t1, .t2, .rest, .overall
            for run = 1:n_run
                hit_miss = h_PSD{run}.EVENT.TYP(h_PSD{run}.EVENT.TYP==CODE.Target_hit | h_PSD{run}.EVENT.TYP==CODE.Target_miss);
                trial_class = h_PSD{run}.EVENT.TYP(h_PSD{run}.EVENT.TYP==task(1) | h_PSD{run}.EVENT.TYP==task(2) | h_PSD{run}.EVENT.TYP==CODE.Rest);
                
                acc_vect{mode}.overall(run) = 100*sum(hit_miss==CODE.Target_hit)/length(trial_class);
                acc_vect{mode}.t1(run) = 100*sum(hit_miss(trial_class==task(1))==CODE.Target_hit)/sum(trial_class==task(1));
                acc_vect{mode}.t2(run) = 100*sum(hit_miss(trial_class==task(2))==CODE.Target_hit)/sum(trial_class==task(2));
                acc_vect{mode}.rest(run) = 100*sum(hit_miss(trial_class==CODE.Rest)==CODE.Target_hit)/sum(trial_class==CODE.Rest);
            end
            %compute the average accuracy
            k = run+1;
            acc_vect{mode}.overall(k) = mean(acc_vect{mode}.overall);
            acc_vect{mode}.t1(k) = mean(acc_vect{mode}.t1);
            acc_vect{mode}.t2(k) = mean(acc_vect{mode}.t2);
            acc_vect{mode}.rest(k) = mean(acc_vect{mode}.rest);
            
            if ~isempty(hmm) %if there are also the rosbag
                %merge the run for visual
                smr_out = [];
                hmm_out =  [];
                exp_out = [];
    
                trial.POS = [0];
                trial.TYP = [];
                trial.DUR = [];
                 
                classes = hmm{1}{1}.Decoder.Classes; %should be class_1, rest, class_2

                for run = 1:n_run
                    for k = 1:length(hmm{run})
                        smr_out = [smr_out; (smrbci{run}{k}.Softpredict.Data)'];
                        hmm_out = [hmm_out; (hmm{run}{k}.Softpredict.Data)'];
                        exp_out = [exp_out; (exp{run}{k}.Softpredict.Data)'];
                    end
                    trial.POS = [trial.POS; trial.POS(end)+h_PSD{run}.EVENT.POS]; 
                    trial.TYP = [trial.TYP; h_PSD{run}.EVENT.TYP];
                    trial.DUR = [trial.DUR; h_PSD{run}.EVENT.DUR];
                end
                trial.POS(1) = [];
                %------------------------------------------------------------------------------
                trial_start = trial.POS(trial.TYP==classes(1) | trial.TYP==classes(2) | trial.TYP==classes(3)); %cue
                trial_end = trial.POS(trial.TYP==CODE.Continuous_feedback) + trial.DUR(trial.TYP==CODE.Continuous_feedback); %end of continuous feedback
                trial.len = trial_end - trial_start; %length of each trial
    
                n_trial = length(trial.len);
                data.label = [];
                trial.label = zeros(n_trial,1);
                trial.start = [];
                trial.idx = []; %indexes of the actual trials
                for k = 1:n_trial
                    trial.start = [trial.start; 1; zeros(trial.len(k)-1,1)];
                    trial.label(k) = trial.TYP(find(trial.POS==trial_start(k)));
    
                    trial.idx = [trial.idx; (trial_start(k):trial_end(k)-1)'];
                    data.label = [data.label; repelem(trial.label(k),trial.len(k))'];
                end
                trial.n = length(trial.label);
                length(data.label)
                %---------------------------------------------------------------------------
                %data.label = array with all the lable for ach time point
                %trial = struct with .label, .idx (subset of idx for the trial),
                %.start(flag of trial start), .pos, .typ, .dur, .len(lenght per trial)

                %devo prendere i solo id dati dei trial e non tutti......
    
                time_base = [1:length(hmm_out)]/f;
                label_plot(data.label==classes(1)) = 0.9;
                label_plot(data.label==classes(3)) = 0.1;
                label_plot(data.label==classes(2)) = 0.5;
    
                if show==true
                    y_lim = [0 1];
                    figure()
                    subplot(311)
                    plot(time_base, smr_out(:,1),'ko','MarkerFaceColor','k','MarkerSize',0.5)
                    hold on
                    plot(time_base(data.label==classes(1)), label_plot(data.label==classes(1)), 'b.','LineWidth',2)
                    plot(time_base(data.label==classes(3)), label_plot(data.label==classes(3)), 'r.','LineWidth',2)
                    plot(time_base(data.label==classes(2)), label_plot(data.label==classes(2)), 'g.','LineWidth',2)
                    hold off
                    xlim([time_base(1), time_base(end)])
                    ylim(y_lim);
                    xlabel('sec')
                    ylabel('prob')
                    title(join(['Evaluation mode: ',keyWords.eval_T(mode)]))
                    legend('raw smr output',task_name{1},'rest',task_name{2})
    
                    subplot(312)
                    plot(time_base, hmm_out(:,1), 'b.','LineWidth',0.5)
                    hold on
                    plot(time_base, hmm_out(:,2), 'g.','LineWidth',0.5)
                    plot(time_base, hmm_out(:,3), 'r.','LineWidth',0.5)
                    hold off       
                    xlim([time_base(1), time_base(end)])
                    ylim(y_lim);
                    xlabel('sec')
                    ylabel('prob')
                    title('HMM raw probability output per state')
                    legend(task_name{1},'rest',task_name{2})
    
                    subplot(313)
                    plot(time_base(data.label==classes(1)), label_plot(data.label==classes(1)), 'b.','LineWidth',2)
                    hold on
                    plot(time_base(data.label==classes(3)), label_plot(data.label==classes(3)), 'r.','LineWidth',2)
                    plot(time_base(data.label==classes(2)), label_plot(data.label==classes(2)), 'g.','LineWidth',2)
                    plot(time_base, exp_out(:,1),'b-','LineWidth',1)
                    plot(time_base, exp_out(:,2),'g-','LineWidth',1)
                    plot(time_base, exp_out(:,3),'r-','LineWidth',1)
                    hold off       
                    xlim([time_base(1), time_base(end)])
                    ylim(y_lim);
                    xlabel('sec')
                    ylabel('prob')
                    title('Exponential framework output')
                    legend('raw smr output',task_name{1},'rest',task_name{2})
    
                    figure()
                    sgtitle(join(['Accuracy mode: '+keyWords.eval_T(mode)]))
                    for run = 1:n_run
                        subplot(1, n_run+1, run)
                        bar([1:4], [acc_vect{mode}.overall(run), acc_vect{mode}.t1(run), acc_vect{mode}.rest(run), acc_vect{mode}.t2(run)])
                        xticklabels(["Overall" task_name{1} "rest" task_name{2}])
                        ylabel('Accuracy %')
                        ylim([0 100])
                        title('Run: '+string(run))
                        grid on
                    end
                    k = run+1;
                    subplot(1, n_run+1, k)
                    bar([1:4], [acc_vect{mode}.overall(k), acc_vect{mode}.t1(k), acc_vect{mode}.rest(k), acc_vect{mode}.t2(k)])
                    xticklabels(["Overall" task_name{1} "rest" task_name{2}])
                    ylabel('Accuracy %')
                    ylim([0 100])
                    title('Average')
                    grid on
    
                end
            end
        end
    end

    accuracy.no_T = acc_vect{1};
    accuracy.T_1 = acc_vect{2};
    accuracy.T_2 = acc_vect{3};

    save("HMM_report.mat","info","accuracy");
end






















