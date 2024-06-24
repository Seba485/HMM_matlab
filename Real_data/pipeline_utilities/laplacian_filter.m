function [lap] = laplacian_filter(n_ch, ch_names, electrode_map)
%[lap] = laplacian_filter(n_ch, ch_names, electrode_map)
%n_ch: number of channel
%ch_names: string array with channel names
%electrode_map: string matrix with the placement of channels
%lap = laplacina filter
%the function save automatically the laplacian filter in the working
%directory

    %% Laplacian filter
    lap = zeros(size(electrode_map));
    for k = 1:n_ch %we need to ecìxlude the "Status"
    
        %look for the channel in the map
        [ch_row, ch_col] = find(electrode_map==ch_names(k));
        
        lap(find(ch_names==ch_names(k)),k) = 1;
    
        non_zero_entry = [];
        %look up
        if ch_row>1
            non_zero_entry = [non_zero_entry, find(ch_names==electrode_map(ch_row-1,ch_col))];
        end
        %look sx
        if ch_col>1
            non_zero_entry = [non_zero_entry, find(ch_names==electrode_map(ch_row,ch_col-1))];
        end
        %look dx
        if ch_col<size(electrode_map,2)
            non_zero_entry = [non_zero_entry, find(ch_names==electrode_map(ch_row,ch_col+1))];
        end    
        %look down
        if ch_row<size(electrode_map,1)
            non_zero_entry = [non_zero_entry, find(ch_names==electrode_map(ch_row+1,ch_col))];
        end
        
        if isempty(non_zero_entry)
            %pass
        else
            lap(non_zero_entry,k) = -1/length(non_zero_entry);
        end
    end
    
    name = 'laplacian'+string(n_ch)+'.mat';
    
    save(name,'lap')

end