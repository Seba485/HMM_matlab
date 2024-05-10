function [norm_posterior] = HMM_forward_step(data, T, start_posterior,k)
% [alpha] = HMM_forward_step(data, states, T, static_prob)
% Forward step following the HMM forward algorithm
% INPUT: data --> series of data
%        states --> cell array, in each cell theres the gmm state
%                   --> update: the pdf are defined in the function hmm_state.m
%        T --> Transition matrix
%        start_posterior --> posteriro probability for the starting step
% OUTPUT: norm_posterior --> posterior probability for the last step of
%                            inference inside the buffer
    

    N_step = length(data);
    %N_state = length(state_gmm);
    N_state = 3;
    
    %likelihood and emission matrix computation
    likelihood = zeros(N_state,N_step);
    % for m = 1:N_state
    %     likelihood(m,:) = pdf(state_gmm{m},data);
    % end

    likelihood(1,:) = hmm_state(data,'both hand');
    likelihood(2,:) = hmm_state(data,'both feet');
    likelihood(3,:) = hmm_state(data,'rest');
  
    norm_likelihood = likelihood./sum(likelihood,1); %so it becomes a probability
    %colums: step -1, current step
    %rows: states

    posterior = zeros(N_state, N_step);
    
    %first step
    posterior(:,1) = start_posterior.*norm_likelihood(:,1); %P(Si)*P(O1|Si)=P(Si|O1)*P(O1)
    posterior(:,1) = posterior(:,1)./sum(posterior(:,1));
    
    %inductive step --> probability for state j to be reached by other states
    for t = 1:N_step-1
        for j = 1:N_state
            posterior(j,t+1) = norm_likelihood(j,t+1)*sum(posterior(:,t).*T(:,j));
        end
        posterior(:,t+1) = posterior(:,t+1)./sum(posterior(:,t+1));
    end
    
    % probability of being in state Si at time t after a sequence of
    % observation P(o1,o2,....,ot, St = Si | model)
    norm_posterior = posterior(:,end);

end