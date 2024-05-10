function [PSD_signal, h_PSD] = PSD_computation(s, h, selected_freq, laplacian_mask)
%[PSD_struct] = PSD_computation(signal, h, selected_freq, laplacian_mask)
    
    ch = size(laplacian_mask,1);

    %applicazione del filtro 
    s_laplacian = s(:,1:ch) * laplacian_mask;


    %power spectral density
    wlength = 0.5; % seconds. Length of the internal window
    pshift = 0.25; % seconds. Shift of the internal windows
    wshift = 0.0625; % seconds. Shift of the external window
    samplerate = h.SampleRate; %512Hz
    
    mlength = 1; % seconds
    
    [PSD, f] = proc_spectrogram(s_laplacian, wlength, wshift, pshift, samplerate, mlength);
    
    %select meaningfull frequences 
    f_selected = f(find(f==selected_freq(1)):find(f==selected_freq(2)));
    PSD_selected = PSD(:,find(f==selected_freq(1)):find(f==selected_freq(2)),:);
    
    %recompute the Position of the event related to the PSD windows
    winconv = 'backward';
    window_POS = proc_pos2win(h.EVENT.POS, wshift*h.SampleRate, winconv, wlength*h.SampleRate);
    
    window_DUR = round(h.EVENT.DUR./(wshift*samplerate));

    window_TYP = h.EVENT.TYP;

    PSD_signal = PSD_selected;
    h_PSD.EVENT.POS = window_POS;
    h_PSD.EVENT.DUR = window_DUR;
    h_PSD.EVENT.TYP = window_TYP;
    h_PSD.f = f_selected;
end
    


