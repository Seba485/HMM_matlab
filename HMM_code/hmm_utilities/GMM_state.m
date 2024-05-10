function [data_model, base, pdf_gmm] = GMM_state(data,n,iter,varargin)
% [data_model, base, pdf_gmm] = GMM_state(data,n,options)
% train the gaussian mixture model state
% INPUT: data --> data stream that we want to model
%        n --> number of gaussian (it has to be a vector of integer if we 
%              want to performe cross validation)
%        iter --> number of iteration in the gaussian fit
% Addictional input: 'CrossValidation' if we want to perform a cross validation
%                                       on the n gaussian
%                    k_fold --> specify the number of folds in cros
%                               validation (dafault value is 10)
%                    'Display' --> if present the function will
%                                   aoutomatically show the pdf in a 
%                                   figure and print the parameters tabel
%                    'Start', start --> starting condition: structure
%                    s.mean and s.sigma with corresponding initial mean and
%                    sigma (standard deviation)
% OUTPUT: data_model --> gaussian mixture model
%         base --> array on which the pdf is computed
%         pdf_gmm --> model pdf computed in 0:1

    %default values:
    k_fold = 10;
    method = 'None';
    disp_flag = false;
    start_flag = false;
    
    if nargin > 3 %if the number of imputs is greater than 2
        for k = 1:numel(varargin)
            if strcmpi(varargin{k}, 'CrossValidation')
                method = 'CV';
                if nargin>4
                    if isnumeric(varargin{k+1}) & varargin{k+1}>0
                        k_fold = varargin{k+1};
                    end
                end
            end
            if strcmpi(varargin{k}, 'Display')
                disp_flag = true;
            end
            if strcmpi(varargin{k}, 'Start')
                if nargin>4
                    if isstruct(varargin{k+1})
                        start_flag = true;
                        start = varargin{k+1};
                    end
                end
            end
        end
    end

    % initial gaussina point
    if start_flag
        S.mu = start.mean;
        for k = 1:n
            S.Sigma(:,:,k) = (start.sigma(k))^2;
        end
        S.ComponentProportion = ones(1,n)./n;
    end

    switch method
        case 'None'
            if start_flag
                option = statset('MaxIter',iter,'TolFun',1e-6);
                data_model = fitgmdist(data,n,"Options",option,"CovarianceType","diagonal","Start",S);
            else
                option = statset('MaxIter',iter,'TolFun',1e-6);
                data_model = fitgmdist(data,n,"Options",option,"CovarianceType","diagonal");
            end
        
        case 'CV'
            fold_size = floor(length(data)/k_fold);
            n_run = length(n);
    
            cv_log_likelihood = zeros(1,n_run);
            for n_idx = 1:n_run
                n_temp = n(n_idx);
                test_likelihood = zeros(1,k_fold);
                for fold = 1:k_fold
                    %yest and train set division
                    tst_set = data((fold_size*(fold-1))+1:fold_size*fold);
                    trn_set = data; trn_set((fold_size*(fold-1))+1:fold_size*fold) = [];
                    
                    %model for the fold
                    if start_flag
                        option = statset('MaxIter',iter,'TolFun',1e-6);
                        data_model = fitgmdist(trn_set,n_temp,"Options",option,"CovarianceType","diagonal","Start",S);
                    else
                        option = statset('MaxIter',iter,'TolFun',1e-6);
                        data_model = fitgmdist(trn_set,n_temp,"Options",option,"CovarianceType","diagonal");
                    end
                    
                    %log overall likelihood of the test set
                    test_likelihood(fold) = log(prod(pdf(data_model,tst_set)));
                end
                cv_log_likelihood(n_idx) = mean(test_likelihood);
            end
            
            %best model selection
            [~,best_model_idx] = max(cv_log_likelihood);
            n = n(best_model_idx);
            if start_flag
                option = statset('MaxIter',iter,'TolFun',1e-6);
                data_model = fitgmdist(trn_set,n,"Options",option,"CovarianceType","diagonal","Start",S);
            else
                option = statset('MaxIter',iter,'TolFun',1e-6);
                data_model = fitgmdist(trn_set,n,"Options",option,"CovarianceType","diagonal");
            end
    end

    if disp_flag
        model_param = [];
        name = strings(1,n);
        for k = 1:n
            model_param(1,k) = data_model.mu(k); %mu
            model_param(2,k) = sqrt(data_model.Sigma(:,:,k)); %sigma
            model_param(3,k) = data_model.ComponentProportion(k); %weight
        
            name(k) = string(k);
        end
        
        disp('Mixture model parameter')
        disp('Log-Likelihood: ' + string(-data_model.NegativeLogLikelihood))
        disp('AIC: ' + string(data_model.AIC))
        disp(array2table(model_param,"RowNames",["mu","sigma","weigth"],"VariableNames",name));
    end

    % % pdf of estimated model 3
    % base = [0:0.01:1];
    % pdf_gmm = zeros(1,length(base));
    % for k = 1:n
    %     pd = makedist('Normal','mu',model_param(1,k),'sigma',model_param(2,k));
    %     pdf_temp = pdf(pd,base);
    %     pdf_gmm = pdf_gmm + model_param(3,k)*(pdf_temp);
    % end
    % % in order to have probabilities
    % pdf_gmm = pdf_gmm./sum(pdf_gmm); 
    
    % pdf
    base = [0:0.01:1]';
    pdf_gmm = pdf(data_model,base);
    %pdf_gmm = pdf_gmm./sum(pdf_gmm); %this give probability

end
