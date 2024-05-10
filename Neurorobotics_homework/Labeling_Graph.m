function [EL] = Labeling_Graph(h,s,t,ch)
%[EL] = Labeling_Graph(h,s,t,ch,CODE)
%function that return: EL -> vector with the labels for each instant
%h -> structure of the format GDF with all the experiment information
%s -> matrix of the signal each colum is a channel
%t -> time vector
%ch -> channel to graph, if -1 no graph is shown

    N = size(s,1);
    if (ch>size(s,2) || ch<=0) && ch~=-1
        disp('ATTENTION: channel selected not valid')
        EL = zeros(N,1); %labels vector
    else
        fix_event = unique(h.EVENT.TYP); %labels event
        EL = zeros(N,1); %labels vector
        for k = 1:length(fix_event)
            event_pos = h.EVENT.POS(h.EVENT.TYP == fix_event(k));
            event_dur = h.EVENT.DUR(h.EVENT.TYP == fix_event(k));
            for j = 1:length(event_pos)
                EL(event_pos(j):event_pos(j)+event_dur(j)-1) = fix_event(k);
            end
        end
    
        colors = ['r','g','b','c','m','y','w','k'];
        flag = 0;
        if length(fix_event)>length(colors)
            disp('ATTENTION: Events type are more than available matlab colors')
            flag = 1;
        end
        
        if ch==-1 || flag==1
            %pass
        else
            figure('Name',join([h.ch_names(ch) 'labeled']))
            plot(t,s(:,ch))
            hold on

            max_s = max(s(:,ch));
            min_s = min(s(:,ch));

            for k = 1:length(h.EVENT.TYP)
                event = h.EVENT.TYP(k);
                duration = h.EVENT.DUR(k);
                idx = h.EVENT.POS(k);
                color_str = colors(find(fix_event==event));
                patch([t(idx), t(idx+duration-1), t(idx+duration-1), t(idx)], [max_s max_s min_s min_s], color_str, 'FaceAlpha',0.2);
                hold on
            end

            title_str = join(['Event labels' h.ch_names(ch) ':']);
            for k = 1:length(fix_event)
                title_str = title_str+' '+string(colors(k))+'='+string(fix_event(k));
            end

            xlabel('t[s]')
            ylabel('eeg [\muV]')
            title(title_str)
            xlim([0 t(end)])
            ylim([min_s max_s])
            grid on
        end

    end
end
