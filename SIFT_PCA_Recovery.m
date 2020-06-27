%%%%%%%%%%%%%%%%%%%% SIFT PCA recovery based on Stiefel center of mass calculation %%%%%%%%%%%%%%%%%%%%

%author: Wenqing Hu (Missouri S&T)

clearvars;
clear classes;

%rng(1, 'twister');

%the PCA embedding dimension = kd_siftStiefel
kd_siftStiefel = 16;
%train_size = the SIFT training data size
train_size = 200*2^10;
%ht = the partition tree height
ht = 8;
%test_size = the SIFT test data size
test_size = 20;
%interpolation_number = number of frames used for interpolation between cluster PCA frames
interpolation_number = 3;

%First sample a training dataset sift_train from the SIFT data set
%Set the partition tree depth = ht
%Tree partition sift_train into clusters C_1, ..., C_{2^{ht}} with centers m_1, ..., m_{2^{ht}}
%Generate from sift_train the local frames A_1, ..., A_{2^{ht}} in St(kd_siftStiefel, kd_sift)

%Then sample a test dataset sift_test from the SIFT data set
%sift_train = the sift training data set , size is traing_size
%leafs = leafs{k}, the cluster indexes in sift_train
%Seq = the Stiefel frames corresponding to each cluster, A_1, ..., A_{2^{ht}}
%sift_test = the sift test data set, size is test_size
[sift_train, Seq, leafs, sift_test] = SIFT_PCA_train(kd_siftStiefel, train_size, ht, test_size);

%sift original dimension
kd_sift = size(sift_test, 2);

m = zeros(kd_sift, 2^ht);
%find m_1, ..., m_{2^{ht}}
for k=1:2^ht 
    m(:, k) = mean(sift_train(leafs{k}, :), 1);
end

%For each point x in sift_test, 
%(1) Find the nearest cluster center m_k from m_1, ..., m_{2^{ht}}
%    Recover x from its PCA projection y = A_k x by considering x_hat = A_k^- y where A_k^- is the pseudo-inverse of A_k
%(2) Find the nearest 3 cluster centers m_k1, m_k2, m_k3 from m_1, ..., m_{2^{ht}}
%    Find the weights w_i = exp(-|x-m_i|^2) for i = 1, 2, 3 and calculate the Stiefel mean A^c_k from f_F(A)=\sum_{i=1}^3 w_i |A-A_ki|_F^2
%    Recover x from its PCA projection y^c = A^c_k x by considering x_hat^c = (A^c_k)^- y where (A^c_k)^- is the pseudo-inverse of A^c_k
%Compare the two recover errors |x-x_hat| and |x-x_hat^c| over sift_test
%Geometric insights suggest that the latter error may be smaller, i.e., the recover is more efficient

counter_success = 0; %counting the number of times when recover via interpolation is more efficient
K = 1e-8; %the scaling coefficient for calculating the weights w = e^{-K distance^2}
error_bm = zeros(test_size, 1); %recovery errors for the nearest frame, benchmark
error_c = zeros(test_size, 1);  %recovery errors using the Stiefel center method

for test_index=1:test_size
    x = sift_test(test_index, :);
    %Sort the cluster centers m_1, ..., m_{2^{ht}} by ascending distances to x 
    dist = zeros(2^ht, 1);
    for k=1:2^ht
        dist(k) = norm(x-m(:, k));
    end
    [dist_sort, indexes] = sort(dist, 1, 'ascend');
    %find the Stiefel projection frames A_k1, ..., A_k{interpolation_number} for the first (interpolation_number) closest clusters to x
    frames = zeros(kd_sift, kd_siftStiefel, interpolation_number);
    for i=1:interpolation_number
        frames(:, :, i) = Seq(:, :, indexes(i));
    end
    %find the weights w_1, ..., w_{interpolation_number} for the first (interpolation_number) closest clusters to x
    w = zeros(interpolation_number, 1);
    for i=1:interpolation_number
        w(i) = exp(-K*(dist_sort(i))^2);
    end
    %obtain the projection y = A_k1 x and the recovery x_hat = (A_k1)^- y, calculate |x-x_hat|
    y = x * frames(:, :, 1);
    x_hat = y * pinv(frames(:, :, 1));
    error_bm(test_index) = norm(x-x_hat);
    %obtain the Euclidean center of mass A_c on St(p,n) for A_k1, ..., A_k{interpolation_number} under weights w_1, ..., w_{interpolation_number}
    %choose an initial frame to start the GD, randomly selected from A_k1, ..., A_k{interpolation_number}
    init_label = randi(interpolation_number);
    A = frames(:, :, init_label);
    %all these frames are on St(n, p), actually n=128 and p=kd_siftStiefel
    n = size(A, 1);
    p = size(A, 2);
    %Set the parameters for GD on Stiefel St(p, n)
    iteration = 1000;
    lr = 0.01;
    lrdecayrate = 1;
    gradnormthreshold = 1e-4;
    checkonStiefelthreshold = 1e-10;
    %bulid the Stiefel Optimization Object
    StiefelOpt = Stiefel_Optimization(w, frames, iteration, lr, lrdecayrate, gradnormthreshold, checkonStiefelthreshold);
    %find the Euclidean center of mass A_c
    [fseq, gradfnormseq, distanceseq, A_c] = StiefelOpt.GD_Stiefel_Euclid(A);
    %obtain the projection y = A_c x and the recovery x_hatc = (A_c)^- y, calculate |x-x_hatc|
    y = x * A_c;
    x_hatc = y * pinv(A_c);
    error_c(test_index) = norm(x-x_hatc);
    %count if the recovery by Stiefel center method is better
    if error_c(test_index) < error_bm(test_index)
        counter_success = counter_success + 1;
    end
end


fprintf("rate that interpolated mean projection recovery efficiency is better than nearest neighbor = %f %% \n", counter_success/test_size*100);





function [sift_train, Seq, leafs, sift_test] = SIFT_PCA_train(kd_siftStiefel, train_size, ht, test_size)

%Sample a training dataset sift_train from the SIFT data set
%Set the partition tree depth = ht
%Tree partition sift_train into clusters C_1, ..., C_{2^{ht}} with centers m_1, ..., m_{2^{ht}}
%Generate from sift_train the local frames A_1, ..., A_{2^{ht}} in St(kd_siftStiefel, kd_sift)
%Sample a test dataset sift_test from the SIFT data set for testing purposes

%Input
%   kd_siftStiefel = the PCA embedding dimension 
%   train_size = the SIFT training data size
%   ht = the partition tree height
%Output
%   sift_train = the sift training data set , size is traing_size
%   leafs = leafs{k}, the cluster indexes in sift_train
%   Seq = the Stiefel frames corresponding to each cluster

%load the sift dataset
%structure: 
%         n_mp: [3000 3000 3000 364 400 4005 2550]
%        n_nmp: [29904 29904 29904 3640 4000 48675 25500]
%       mp_fid: [16319×2 double]
%      nmp_fid: [171527×2 double]
%        sifts: [10068850×128 uint8]
%    sift_offs: [1×33590 double]

load ~/文档/work_SubspaceIndexingStiefleGrassmann/Code_Subspace_indexing_Stiefel_Grassman/cdvs-sift300-dataset.mat;

%n_sift=10068850 is the number of samples in sifts dataset, kd_sift is the original dimension of each sample
%kd_sift=128
[n_sift, kd_sift] = size(sifts);

indexes = randperm(n_sift); 
%randomly pick the training sample of size train_size from sift dataset
train_indexes = indexes(1: train_size);
%form the sift_train dataset
sift_train = double(sifts(train_indexes, :));

%randomly pick the test sample of size test_size from sift dataset, must be disjoint from sift_train
test_indexes = indexes(train_size + 1: train_size + test_size);
%form the sift_test dataset
sift_test = double(sifts(test_indexes, :));


%do an initial PCA on sift_train
[A0, s0, lat0] = pca(sift_train);

%bulid a kd_siftStiefel-dimensional embedding of sift_train in x0
x0 = sift_train * A0(:, 1:kd_siftStiefel);

%from x0, partition into 2^ht leaf nodes, each leaf node can give samples for a local PCA
[indx, leafs]=buildVisualWordList(x0, ht);

% build PCA Model for each leaf
doBuildSiftModel = 1;
% input: sift, indx, leafs
if doBuildSiftModel
    for k=1:length(leafs)
        %form the sift subsample for the k-th cluster
        sift_train_k = sift_train(leafs{k}, :); 
        [A{k}, s, lat] = pca(sift_train_k);
        Seq(:, :, k) = A{k}(:, 1:kd_siftStiefel); 
    end
end    

end