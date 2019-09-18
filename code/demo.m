%% Matlab Demo file For Nyström Basis Transfer
% Script for demonstrating sampling scheme. For 5x2 sampling repeat every 
% iteration and use second index of crossvalind
% Scheme suggested in 
% Chen, Huanhuan ; Ti?o, Peter ; Yao, Xin: Probabilistic classification vector machines. 
% In: IEEE Transactions on Neural Networks Bd. 20 (2009), Nr. 6, S. 901–914

clear all;

addpath(genpath('/libsvm'));
addpath(genpath('../data'));
addpath(genpath('../code'));



%% Reuters Dataset
options.ker = 'rbf';        % kernel: 'linear' | 'rbf' | 'lap'
options.eta = 2.0;           % TKL: eigenspectrum damping factor
options.gamma = 1.0;         % TKL: width of gaussian kernel
options.k = 100;              % JDA: subspaces bases
options.lambda = 1.0;        % JDA: regularization parameter
options.svmc = 10.0;         % SVM: complexity regularizer in LibSVM
options.g = 40;              % GFK: subspace dimension
options.tcaNv = 60;          % TCA: numbers of Vectors after reduction
options.theta = 1;           %PCVM: Width of gaussian kernel
options.subspace_dim_d = 5;  %SA: Subspace Dimensions
options.landmarks = 400;     %NTVM: Number of Landmarks

for strData = {'org_vs_people','org_vs_place', 'people_vs_place'} %

    for iData = 1:2
        data = char(strData);
        data = strcat(data, '_', num2str(iData));
        load(strcat('../data/Reuters/', data));

        fprintf('data=%s\n', data);

        Xs=full(bsxfun(@rdivide, bsxfun(@minus,Xs,mean(Xs)), std(Xs)));
        Xt=full(bsxfun(@rdivide, bsxfun(@minus,Xt,mean(Xt)), std(Xt)));

        m = size(Xs, 2);
        n = size(Xt, 2);
        Xs = Xs';Xt = Xt';
        soureIndx = crossvalind('Kfold', Ys, 2);
        targetIndx = crossvalind('Kfold', Yt,2);
        Xs = Xs(find(soureIndx==1),:)';
        Ys = Ys(find(soureIndx==1),:);


        Xt = Xt(find(targetIndx==1),:)';
        Yt = Yt(find(targetIndx==1),:);
        m = size(Xs, 2);
        n = size(Xt, 2);

        %% SVM
        K = kernel(options.ker, [Xs, Xt], [],options.gamma);
        model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
        [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
        fprintf('SVM = %.2f%%\n', acc(1));

        %% TCA
        nt = length(Ys);
        mt = length(Yt);
        K = tca(Xs',Xt',options.tcaNv,options.gamma,options.ker);
        model = svmtrain(full(Ys),[(1:nt)',K(1:nt,1:nt)],['-s 0 -c ', num2str(options.svmc), ' -t 4 -q 1']);
        [label,acc,scores] = svmpredict(full(Yt),[(1:mt)',K(nt+1:end,1:nt)],model);

        fprintf('\nTCA %.2f%% \n',acc(1));

        %% JDA

        Cls = [];
        [Z,A] = JDA(Xs,Xt,Ys,Cls,options);
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(Xs,2));
        Zt = Z(:,size(Xs,2)+1:end);
        K = kernel(options.ker, Z, [],options.gamma);
        model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
        [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
        fprintf('\nJDA %.2f%% \n',acc(1));


        %% TKL SVM
        tic;
        K = TKL(Xs, Xt, options);
        model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-s 0 -c ', num2str(options.svmc), ' -t 4 -q 1']);
        [labels, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
        fprintf('\nTKL %.2f%%\n',acc(1));

        %% GFK
        xs = full(Xs');
        xt = full(Xt');
        Ps = pca(xs);
        Pt = pca(xt);
        nullP = null(Ps');
        G = GFK([Ps,nullP], Pt(:,1:options.g));
        [label, acc] = my_kernel_knn(G, xs, Ys, xt, Yt);
        acc = full(acc)*100;
        fprintf('\n GFK %.2f%%\n',acc);
        %% SA
        [Xss,~,~] = pca(Xs');
        [Xtt,~,~] = pca(Xt');
        Xss = Xss(:,1:options.subspace_dim_d);
        Xss = Xtt(:,1:options.subspace_dim_d);
        [predicted_label, acc, decision_values,model] =  Subspace_Alignment(Xs',Xt',Ys,Yt,Xss,Xtt);
        fprintf('\nSA %.2f%%\n',acc(1));

%         %% CGCA
%         acc= gca123(Xs,Ys,Xt,Yt,0.9, 0.2, 0.1)
%         fprintf('\nCGCA %.2f%% \n',acc);

        %% Coral
        acc = coral(Xs',Ys,Xt',Yt);
        fprintf('\n Coral %.2f%% \n', acc(1));

        %% NBT
        [model,K,m,n] = pd_cl_nbt(Xs',Xt',Ys,options);
        [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
        fprintf('\n PD_NBT %.2f%% \n', acc(1));
    end
end

clear all;
%% OFFICE vs CALLTECH-256 Dataset
options.ker = 'rbf';         % TKL: kernel: 'linear' | 'rbf' | 'lap'
options.eta = 1.1;           % TKL: eigenspectrum damping factor
options.gamma = 1.0;         % TKL: width of gaussian kernel
options.g = 65;              % GFK: subspace dimension
options.k = 100;              % JDA: subspaces bases
options.lambda = 1.0;        % JDA: regularization parameter
options.svmc = 10.0;         % SVM: complexity regularizer in LibSVM
options.tcaNv = 60;          % TCA: numbers of Vectors after reduction
options.theta = 1;          %PCVM: Width of gaussian kernel
options.landmarks = 200;     %NTVM: Number of Landmark
options.subspace_dim_d = 5;  %SA: Subspace Dimensions
srcStr = {'Caltech10', 'Caltech10', 'Caltech10', 'amazon', 'amazon', 'amazon', 'webcam', 'webcam', 'webcam', 'dslr', 'dslr', 'dslr'};
tgtStr = {'amazon', 'webcam', 'dslr', 'Caltech10', 'webcam', 'dslr', 'Caltech10', 'amazon', 'dslr', 'Caltech10', 'amazon', 'webcam'};

for iData = 1:12
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    data = strcat(src, '_vs_', tgt);
    fprintf('data=%s\n', data);
    load(['../data/OfficeCaltech/' src '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));
    Xs = zscore(fts, 1);
    Ys = labels;
    
    load(['../data/OfficeCaltech/' tgt '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts, 2), 1, size(fts,2));
    Xt = zscore(fts, 1);
    Yt = labels;
    
    
    soureIndx = crossvalind('Kfold', Ys, 2);
    targetIndx = crossvalind('Kfold', Yt,2);
    Xs = Xs(find(soureIndx==1),:)';
    Ys = Ys(find(soureIndx==1),:);
    
    
    Xt = Xt(find(targetIndx==1),:)';
    Yt = Yt(find(targetIndx==1),:);
    m = size(Xs, 2);
    n = size(Xt, 2);
    
    %% SVM
    K = kernel(options.ker, [Xs, Xt], [],options.gamma);
    model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
    [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
    fprintf('SVM = %.2f%%\n', acc(1));
    
    %% TCA
    nt = length(Ys);
    mt = length(Yt);
    K = tca(Xs',Xt',options.tcaNv,options.gamma,options.ker);
    model = svmtrain(full(Ys),[(1:nt)',K(1:nt,1:nt)],['-s 0 -c ', num2str(options.svmc), ' -t 4 -q 1']);
    [label,acc,scores] = svmpredict(full(Yt),[(1:mt)',K(nt+1:end,1:nt)],model);
    
    fprintf('\nTCA %.2f%% \n',acc(1));
    
    %% JDA
    Cls = [];
    [Z,A] = JDA(Xs,Xt,Ys,Cls,options);
    Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
    Zs = Z(:,1:size(Xs,2));
    Zt = Z(:,size(Xs,2)+1:end);
    K = kernel(options.ker, Z, [],options.gamma);
    model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
    [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
    fprintf('\nJDA %.2f%% \n',acc(1));
    
    
    %% TKL SVM
    tic;
    K = TKL(Xs, Xt, options);
    model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-s 0 -c ', num2str(options.svmc), ' -t 4 -q 1']);
    [labels, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
    fprintf('\nTKL %.2f%%\n',acc(1));
    
    %% GFK
    xs = full(Xs');
    xt = full(Xt');
    Ps = pca(xs);
    Pt = pca(xt);
    nullP = null(Ps');
    G = GFK([Ps,nullP], Pt(:,1:options.g));
    [label, acc] = my_kernel_knn(G, xs, Ys, xt, Yt);
    acc = full(acc)*100;
    fprintf('\n GFK %.2f%%\n',acc);
    %% SA
    [Xss,~,~] = pca(Xs');
    [Xtt,~,~] = pca(Xt');
    Xss = Xss(:,1:options.subspace_dim_d);
    Xss = Xtt(:,1:options.subspace_dim_d);
    [predicted_label, acc, decision_values,model] =  Subspace_Alignment(Xs',Xt',Ys,Yt,Xss,Xtt);
    fprintf('\nSA %.2f%%\n',acc(1));
    
    %% CGCA
    %acc= gca123(Xs,Ys,Xt,Yt,0.9, 0.2, 0.1)
    % fprintf('\nCGCA %.2f%% \n',acc);
    
    %% Coral
    acc = coral(Xs',Ys,Xt',Yt);
    fprintf('\n Coral %.2f%% \n', acc(1));
    
    %% NBT
    [model,K,m,n] = pd_cl_nbt(Xs',Xt',Ys,options);
    [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
    fprintf('\n PD_NBT %.2f%% \n', acc(1));
    
end

%% 20 Newsgroup
options.ker = 'rbf';        % kernel: 'linear' | 'rbf' | 'lap'
options.eta = 2.0;           % TKL: eigenspectrum damping factor
options.gamma = 1.0;         % TKL: width of gaussian kernel
options.k = 100;              % JDA: subspaces bases
options.lambda = 1.0;        % JDA: regularization parameter
options.svmc = 10.0;         % SVM: complexity regularizer in LibSVM
options.g = 40;              % GFK: subspace dimension
options.tcaNv = 60;          % TCA: numbers of Vectors after reduction
options.theta = 1;           %PCVM: Width of gaussian kernel
options.subspace_dim_d = 5;  %SA: Subspace Dimensions
options.landmarks = 1500;    %NTVM: Number of Landmarks
for name = {'comp_vs_rec','comp_vs_sci','comp_vs_talk','rec_vs_sci','rec_vs_talk','sci_vs_talk'}%
    for j=1:36
        data = char(name);
        data = strcat(data, '_', num2str(j));
        load(strcat('../data/20Newsgroup/', data));
        fprintf('data=%s\n', data);
        
        Xs=full(bsxfun(@rdivide, bsxfun(@minus,Xs,mean(Xs)), std(Xs)));
        Xt=full(bsxfun(@rdivide, bsxfun(@minus,Xt,mean(Xt)), std(Xt)));

        Xs = Xs';Xt = Xt';
        
        soureIndx = crossvalind('Kfold', Ys, 2);
        targetIndx = crossvalind('Kfold', Yt,2);
        Xs = Xs(find(soureIndx==1),:)';
        Ys = Ys(find(soureIndx==1),:);
        
        
        Xt = Xt(find(targetIndx==1),:)';
        Yt = Yt(find(targetIndx==1),:);
        m = size(Xs, 2);
        n = size(Xt, 2);
        %% SVM
        K = kernel(options.ker, [Xs, Xt], [],options.gamma);
        model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
        [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
        fprintf('SVM = %.2f%%\n', acc(1));
        
        %% TCA
        nt = length(Ys);
        mt = length(Yt);
        K = tca(Xs',Xt',options.tcaNv,options.gamma,options.ker);
        model = svmtrain(full(Ys),[(1:nt)',K(1:nt,1:nt)],['-s 0 -c ', num2str(options.svmc), ' -t 4 -q 1']);
        [label,acc,scores] = svmpredict(full(Yt),[(1:mt)',K(nt+1:end,1:nt)],model);
        
        fprintf('\nTCA %.2f%% \n',acc(1));
        
        %% JDA
        
        Cls = [];
        [Z,A] = JDA(Xs,Xt,Ys,Cls,options);
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(Xs,2));
        Zt = Z(:,size(Xs,2)+1:end);
        K = kernel(options.ker, Z, [],options.gamma);
        model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
        [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
        fprintf('\nJDA %.2f%% \n',acc(1));
        
        
        %% TKL SVM
        tic;
        K = TKL(Xs, Xt, options);
        model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-s 0 -c ', num2str(options.svmc), ' -t 4 -q 1']);
        [labels, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
        fprintf('\nTKL %.2f%%\n',acc(1));
        
        %% GFK
        xs = full(Xs');
        xt = full(Xt');
        Ps = pca(xs);
        Pt = pca(xt);
        nullP = null(Ps');
        G = GFK([Ps,nullP], Pt(:,1:options.g));
        [label, acc] = my_kernel_knn(G, xs, Ys, xt, Yt);
        acc = full(acc)*100;
        fprintf('\n GFK %.2f%%\n',acc);
        %% SA
        [Xss,~,~] = pca(Xs');
        [Xtt,~,~] = pca(Xt');
        Xss = Xss(:,1:options.subspace_dim_d);
        Xss = Xtt(:,1:options.subspace_dim_d);
        [predicted_label, acc, decision_values,model] =  Subspace_Alignment(Xs',Xt',Ys,Yt,Xss,Xtt);
        fprintf('\nSA %.2f%%\n',acc(1));
        
        %% CGCA
        %acc= gca123(Xs,Ys,Xt,Yt,0.9, 0.2, 0.1)
        % fprintf('\nCGCA %.2f%% \n',acc);
        
        
        %% Coral
        acc = coral(Xs',Ys,Xt',Yt);
        fprintf('\n Coral %.2f%% \n', acc(1));
        
        %% NBT
        [model,K,m,n] = pd_cl_nbt(Xs',Xt',Ys,options);
        [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
        fprintf('\n PD_NBT %.2f%% \n', acc(1));
    end
end