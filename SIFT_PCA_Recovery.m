%%%%%%%%%%%%%%%%%%%% SIFT PCA recovery based on Stiefel center of mass calculation %%%%%%%%%%%%%%%%%%%%

%author: Wenqing Hu (Missouri S&T)

clearvars;
clear classes;

%rng(1, 'twister');

%the PCA embedding dimension = kd_siftStiefel
kd_siftStiefel = 16;
%train_size = the SIFT training data size
train_size = 100*2^13;
%ht = the partition tree height
ht = 13;
%test_size = the SIFT test data size
test_size = 5;

%set the sequence of interpolation numbers and the threshold ratio for determining the interpolation number
interpolation_number_seq = ones(test_size, 1);
ratio_threshold = 1.001;

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

%all these Stiefel frames are on St(n, p), actually n=128 and p=kd_siftStiefel
n = size(Seq, 1);
p = size(Seq, 2);

%sift original dimension
kd_sift = size(sift_test, 2);

m = zeros(kd_sift, 2^ht);
%find m_1, ..., m_{2^{ht}}, the means of the chosen clusters
for k=1:2^ht 
    m(:, k) = mean(sift_train(leafs{k}, :), 1);
end

%For each point x in sift_test, 
%(1) Find the nearest cluster center m_k from m_1, ..., m_{2^{ht}}
%    Recover x from its PCA projection y = A_k x by considering x_hat = A_k^- y where A_k^- is the pseudo-inverse of A_k
%(2) Find the nearest (interpolation_number) cluster centers m_k1, m_k2, m_k(interpolation number) from m_1, ..., m_{2^{ht}}
%    interpolation_number is determined by |x-m_k(interpolation_number+1)| > ratio_threshold * |x-m_k1|
%    Find the weights w_i = exp(-|x-m_i|^2) for i = 1, ..., interpolation_number and calculate the Stiefel mean A^c_k 
%    Stiefel mean A^c_k are chosen 
%        (1) from Euclidean norm: f_F(A)=\sum_{i=1}^interpolation_number w_i |A-A_ki|_F^2, using GD/direct method
%        (2) from retraction method: 
%    Recover x from its PCA projection y^c = A^c_k x by considering x_hat^c = (A^c_k)^- y where (A^c_k)^- is the pseudo-inverse of A^c_k
%Compare the two recover errors error_bm=|x-x_hat| and error_c=|x-x_hat^c| over sift_test
%Geometric insights suggest that the latter error may be smaller, i.e., the recover is more efficient

counter_success = 0; %counting the number of times when recover via interpolation is more efficient
K = 1e-8; %the scaling coefficient for calculating the weights w = e^{-K distance^2}
error_bm = zeros(test_size, 1); %recovery errors for the nearest frame, benchmark
error_c = zeros(test_size, 1);  %recovery errors using the Stiefel center method

doEuclideanCenter = 1; %do or do not do Euclidean center of mass
doGD = 1; %do or do not do GD for finding Euclidean center of mass

tic;
for test_index=1:test_size
    fprintf("\ntest point %d -----------------------------------------------------------\n", test_index);
    x = sift_test(test_index, :);
    %sort the cluster centers m_1, ..., m_{2^{ht}} by ascending distances to x 
    dist = zeros(2^ht, 1);
    for k=1:2^ht
        dist(k) = norm(x-m(:, k));
    end
    [dist_sort, indexes] = sort(dist, 1, 'ascend');
    %count the number of St(p, n) interpolation clusters for current test point x
    %interpolation_number = number of frames used for interpolation between cluster PCA frames
    interpolation_number = 1;
    for k=2:2^ht
        if dist_sort(k) <= ratio_threshold * dist_sort(1)
            interpolation_number = interpolation_number + 1;
        else
            break;
        end    
    end
    fprintf("interpolation number = %d\n", interpolation_number);
    %record the sequence of all interpolation numbers for each test point x
    interpolation_number_seq(test_index) = interpolation_number;
    %find the Stiefel projection frames A_k1, ..., A_k{interpolation_number} for the first (interpolation_number) closest clusters to x
    frames = zeros(kd_sift, kd_siftStiefel, interpolation_number);
    for i=1:interpolation_number
        frames(:, :, i) = Seq(:, :, indexes(i));
    end
    %find the weights w_1, ..., w_{interpolation_number} for the first (interpolation_number) closest clusters to x
    w = zeros(interpolation_number, 1);
    for i=1:interpolation_number
        w(i) = exp(- K * (dist_sort(i))^2);
    end
    %obtain the projection y = A_k1 x and the recovery x_hat = (A_k1)^- y, calculate error_bm=|x-x_hat|
    y = x * frames(:, :, 1);
    x_hat = y * pinv(frames(:, :, 1));
    error_bm(test_index) = norm(x-x_hat);
    %Obtain the Euclidean center of mass A_c on St(p,n) for A_k1, ..., A_k{interpolation_number} under weights w_1, ..., w_{interpolation_number}
    %set the Stiefel Optimization object with given threshold parameters
    threshold_gradnorm = 1e-4;
    threshold_fixedpoint = 1e-4;
    threshold_checkonStiefel = 1e-10;
    threshold_logStiefel = 1e-4;
    %bulid the Stiefel Optimization Object
    StiefelOpt = Stiefel_Optimization(w, frames, threshold_gradnorm, threshold_fixedpoint, threshold_checkonStiefel, threshold_logStiefel);
    %find the center of mass A_c
    %compare several methods: Euclidean center of mass, direct or via GD, or Retraction-based center of mass
    if doEuclideanCenter
        if doGD
            %choose an initial frame to start the GD, randomly selected from A_k1, ..., A_k{interpolation_number}
            init_label = randi(interpolation_number);
            A = frames(:, :, init_label);
            %Set the parameters for GD on Stiefel St(p, n)
            iteration = 1000;
            lr = 0.001;
            lrdecayrate = 1;
            [A_c, gradfnormseq, distanceseq] = StiefelOpt.Center_Mass_GD_Euclid(A, iteration, lr, lrdecayrate);
        else
            [A_c, minvalue, gradminfnorm] = StiefelOpt.Center_Mass_Euclid;
        end
    else
        break;   
    end
    %obtain the projection y = A_c x and the recovery x_hatc = (A_c)^- y, calculate error_c=|x-x_hatc|
    y = x * A_c;
    x_hatc = y * pinv(A_c);
    error_c(test_index) = norm(x-x_hatc);
    %count if the recovery by Stiefel center method is better
    fprintf("error for mean projection recovery = %f, error for benchmark nearest neighbor = %f\n", error_c(test_index), error_bm(test_index));
    if error_c(test_index) < error_bm(test_index) || interpolation_number == 1
        counter_success = counter_success + 1;
        fprintf("efficient! :)\n");
    else
        fprintf("not efficient :(\n");
    end
end
toc;

%output the recovery efficiency percentage and related data
fprintf("rate that interpolated mean projection recovery efficiency is better than nearest neighbor = %f %% \n", counter_success/test_size*100);
fprintf("mean of Error-c = %f, mean of Error-bm = %f \n", mean(error_c), mean(error_bm));

%plot the PCA recovery errors
figure;
plot(error_c, '-*', 'Color', [1 0 0], 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:test_size);
hold on;
plot(error_bm, '-*', 'Color', [0 0 1], 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:test_size);
xlabel('Index of test sample');
ylabel('Recovery error');
legend('Stiefel Euclidean center recovery', 'benchmark nearest neighbor recovery');
title('PCA Recovery Errors');
hold off;

%sort the benchmark errors in ascending order, and reorder the Stiefel center errors correspondingly
[error_bm_sort, indexes] = sort(error_bm, 1, 'ascend');
error_c_sort = error_c(indexes);
%plot the PCA recovery errors with ascending benchmark error
figure;
plot(error_c_sort, '-*', 'Color', [1 0 0], 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:test_size);
hold on;
plot(error_bm_sort, '-*', 'Color', [0 0 1], 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:test_size);
xlabel('Reordered Index of test sample');
ylabel('Recovery error');
legend('Stiefel Euclidean center recovery', 'benchmark nearest neighbor recovery');
title('PCA Recovery Errors');
hold off;

%plot the differences of PCA recovery errors
figure;
plot(sort(error_c_sort - error_bm_sort, 'descend'), '-*', 'Color', [0 0 0], 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:test_size);
hold on;
xlabel('Reordered index of test sample');
ylabel('Error\_c - Error\_bm');
title('Difference in PCA Recovery Errors');
hold off;




function [sift_train, Seq, leafs, sift_test] = SIFT_PCA_train(kd_siftStiefel, train_size, ht, test_size)

%Sample a training dataset sift_train from the SIFT data set
%Set the partition tree depth = ht
%Tree partition sift_train into clusters C_1, ..., C_{2^{ht}} with centers m_1, ..., m_{2^{ht}}
%Generate from sift_train the local frames A_1, ..., A_{2^{ht}} in St(kd_siftStiefel, kd_sift)
%Sample a test dataset sift_test from the SIFT data set for testing purposes

%Input
%   kd_siftStiefel = the PCA embedding dimension 
%   train_size, test_size = the SIFT training/testing data set size
%   ht = the partition tree height
%Output
%   sift_train, sift_test = the sift training/testing data set , size is traing_size/test_size
%   leafs = leafs{k}, the cluster indexes in sift_train
%   Seq = the Stiefel frames corresponding to each cluster in sift_train

%load the sift dataset
%structure: 
%         n_mp: [3000 3000 3000 364 400 4005 2550]
%        n_nmp: [29904 29904 29904 3640 4000 48675 25500]
%       mp_fid: [16319×2 double]
%      nmp_fid: [171527×2 double]
%        sifts: [10068850×128 uint8]
%    sift_offs: [1×33590 double]

load ~/文档/work_SubspaceIndexingStiefleGrassmann/Code_Subspace_indexing_Stiefel_Grassman/DATA_cdvs-sift300-dataset.mat sifts

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


