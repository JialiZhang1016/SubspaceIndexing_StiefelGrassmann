%%%%%%%%%%%%%%%%%%%% LPP analysis based on Grassmann center of mass calculation %%%%%%%%%%%%%%%%%%%%

% author: Wenqing Hu (Missouri S&T)

clearvars;
clear classes;

%rng(1, 'twister');


doNWPU = 0;
if doNWPU
% load the nwpu-aerial-images dataset
% structure: 
%   x: [31500×4096 double]
%   y: [31500×1 double]
data = load('~/Documents/Code_Subspace_indexing_Stiefel_Grassman/DATA_nwpu-aerial-images.mat');
% load the size of data and shuffle the indexes to divide it into original training and testing data sets
[n_data, kd_data] = size(data.x);
indexes = randperm(n_data); 
% randomly pick the original train and test data set, must be disjoint
train_indexes = indexes(1: 30000);
test_indexes = indexes(30001: 31500);
% form the data_original_train and data_original_text dataset
data_original_train.x = double(data.x(train_indexes, :));
data_original_test.x = double(data.x(test_indexes, :));
data_original_train.y = double(data.y(train_indexes, :));
data_original_test.y = double(data.y(test_indexes, :));
end


doMNIST = 0;
if doMNIST
% load the MNIST dataset
% structure: 
%    testX: [10000×784 uint8]
%    testY: [1×10000 uint8]
%    trainY: [1×60000 uint8]
%    trainX: [60000×784 uint8]
mnist = load('~/Documents/Code_Subspace_indexing_Stiefel_Grassman/DATA_mnist.mat');
% preprocess the dataset to fit the format we use
data_original_train.x = double(mnist.trainX);
data_original_test.x = double(mnist.testX);
data_original_train.y = double(mnist.trainY');
data_original_test.y = double(mnist.testY');
end


doCIFAR10 = 1;
if doCIFAR10
% load the CIFAR-10 dataset, data from https://www.cs.toronto.edu/~kriz/cifar.html
% structure: 
% each cifar10_k, k=1,...,5
%          data: [10000×3072 uint8]
%        labels: [10000×1 uint8]
%   batch_label: 'training batch k of 5'
% cifar10_test
%          data: [10000×3072 uint8]
%        labels: [10000×1 uint8]
%   batch_label: 'testing batch 1 of 1'    
cifar10(1) = load('~/Documents/Code_Subspace_indexing_Stiefel_Grassman/DATA_cifar10/data_batch_1.mat');
cifar10(2) = load('~/Documents/Code_Subspace_indexing_Stiefel_Grassman/DATA_cifar10/data_batch_2.mat');
cifar10(3) = load('~/Documents/Code_Subspace_indexing_Stiefel_Grassman/DATA_cifar10/data_batch_3.mat');
cifar10(4) = load('~/Documents/Code_Subspace_indexing_Stiefel_Grassman/DATA_cifar10/data_batch_4.mat');
cifar10(5) = load('~/Documents/Code_Subspace_indexing_Stiefel_Grassman/DATA_cifar10/data_batch_5.mat');
cifar10(6) = load('~/Documents/Code_Subspace_indexing_Stiefel_Grassman/DATA_cifar10/test_batch.mat');
% preprocess the dataset to fit the format we use
d_prepre = 784;
for i = 1:6
    cifar10(i).data = double(cifar10(i).data);
    % do an initial PCA on data
    [A0, s0, lat0] = pca(cifar10(i).data);
    % bulid a given dimensional d_prepre embedding of cifar10 into new cifar10, for computational memory constraint only
    cifar10(i).data = cifar10(i).data * A0(:, 1:d_prepre);
end
data_original_train.x = double(cifar10(1).data);
data_original_train.y = double(cifar10(1).labels);
for i = 2:5
    data_original_train.x = double(vertcat(data_original_train.x, cifar10(i).data));
    data_original_train.y = double(vertcat(data_original_train.y, cifar10(i).labels));
end
data_original_test.x = double(cifar10(6).data);
data_original_test.y = double(cifar10(6).labels);
end




% the data preprocessing projection dimension
d_pre = 784;
% the PCA embedding dimension = kd_PCA
kd_PCA = 64;
% the LPP embedding dimension = kd_LPP
kd_LPP = 16;
% train_size = the training data size
train_size = 6*2^13;
% ht = the partition tree height
ht = 13;
% test_size = the test data size
test_size = 1000;

% obtain the train, test sets and the LPP frames Seq(:,:,k) for each cluster with indexes in leafs
[data_train, leafs, data_test] = LPP_ObtainData(data_original_train, data_original_test, d_pre, kd_LPP, kd_PCA, train_size, test_size, ht);
[Seq] = LPP_BuildDataModel(data_train, leafs, kd_PCA, kd_LPP);

% all these LPP Stiefel frames are on St(n, p)
n = size(Seq, 1);
p = size(Seq, 2);

% data original dimension kd_data
kd_data = size(data_train.x, 2);

% find m_1, ..., m_{2^{ht}}, the means of the chosen clusters
m = zeros(kd_data, 2^ht);
for k=1:2^ht 
    m(:, k) = mean(data_train.x(leafs{k}, :), 1);
end

% set the sequence of interpolation numbers and the threshold ratio for determining the interpolation number
interpolation_number_seq = ones(test_size, 1);
ratio_threshold = 1.1; % the ratio for determinining the interpolation_number, serve as a tuning parameter
ratio_seq = zeros(test_size, 2); % the sequence of second smallest (or largest) to-center distance over smallest to-center distance, for tuning ratio_threshold


K = 1e-8; % the scaling coefficient for calculating the weights w = e^{-K distance^2}
k_nearest_neighbor = 1; % the parameter k for k-nearest-neighbor classification

classified_o = zeros(test_size, 1); % list of classified/not classified projections for using the original data point and nearest cluster
classified_agg_o = zeros(test_size, 1); % list of classified/not classified projections for using the original data point and nearest (interpolation_number) clusters
classified_bm = zeros(test_size, 1); % list of classified/not classified projections for using the nearest frame, benchmark
classified_c = zeros(test_size, 1);  % list of classified/not classified projections for using the Grassmann center method

doGrassmannpFCenter = 0; % do or do not do projected Frobenius center of mass for Grassmannian frame
doStiefelEuclidCenter = 1; % do or do not do Euclid center of mass for Stiefel frame 
doGD = 0; % do or do not do GD for finding projected Frobenius center of mass


tic;
for test_index=1:test_size
    fprintf("\ntest point %d -----------------------------------------------------------\n", test_index);
    x = data_test.x(test_index, :);
    y = data_test.y(test_index);
    % sort the cluster centers m_1, ..., m_{2^{ht}} by ascending distances to x 
    dist = zeros(2^ht, 1);
    for k=1:2^ht
        dist(k) = norm(x'-m(:, k));
    end
    [dist_sort, indexes] = sort(dist, 1, 'ascend');
    fprintf("ratio between [%f, %f]\n", dist_sort(2)/dist_sort(1), dist_sort(2^ht)/dist_sort(1));
    ratio_seq(test_index, 1) = dist_sort(2)/dist_sort(1);
    ratio_seq(test_index, 2) = dist_sort(2^ht)/dist_sort(1);
    % count the number of St(p, n) interpolation clusters for current test point x
    % interpolation_number = number of frames used for interpolation between cluster LDA frames
    interpolation_number = 1;
    for k=2:2^ht
        if dist_sort(k) <= ratio_threshold * dist_sort(1)
            interpolation_number = interpolation_number + 1;
        else
            break;
        end    
    end
    fprintf("interpolation number = %d\n", interpolation_number);
    % record the sequence of all interpolation numbers for each test point x
    interpolation_number_seq(test_index) = interpolation_number;
    % find the LPP Stiefel projection frames A_k1, ..., A_k{interpolation_number} for the first (interpolation_number) closest clusters to x
    frames = zeros(kd_data, kd_LPP, interpolation_number);
    for i=1:interpolation_number
        frames(:, :, i) = Seq(:, :, indexes(i));
    end
    % find the weights w_1, ..., w_{interpolation_number} for the first (interpolation_number) closest clusters to x
    w = zeros(interpolation_number, 1);
    for i=1:interpolation_number
        w(i) = exp(- K * (dist_sort(i))^2);
    end
    % collect all indexes in clusters corresponding to the first (interpolation_number) closest clusters to x
    aggregate_cluster = [];
    for i=1:interpolation_number
        aggregate_cluster = union(aggregate_cluster, leafs{indexes(i)});
    end
    % do k-nearest-neighbor classification based on the closest cluster to x, in original space
    x_test = x;
    y_test = y;
    X_train = data_train.x(leafs{indexes(1)}, :);
    Y_train = data_train.y(leafs{indexes(1)});
    isclassified_o = knn(x_test, y_test, X_train, Y_train, k_nearest_neighbor);
    classified_o(test_index) = isclassified_o;
    % do k-nearest-neighbor classification based on the (interpolation_number) nearest clusters to x, in oroginal space
    x_test = x;
    y_test = y;
    X_train = data_train.x(aggregate_cluster, :);
    Y_train = data_train.y(aggregate_cluster);
    isclassified_agg_o = knn(x_test, y_test, X_train, Y_train, k_nearest_neighbor);
    classified_agg_o(test_index) = isclassified_agg_o;
    % project x to A1 x and classify it using k-nearest-neighbor on the projection via A1 of the closest cluster
    x_test = x * frames(:,:,1);
    y_test = y;
    X_train = data_train.x(leafs{indexes(1)}, :) * frames(:,:,1);
    Y_train = data_train.y(leafs{indexes(1)});
    isclassified_bm = knn(x_test, y_test, X_train, Y_train, k_nearest_neighbor);
    classified_bm(test_index) = isclassified_bm;
    % calculate the center of mass for the (interpolation_number) nearest cluster LPP frames with respect to weights w 
    threshold_gradnorm = 1e-4;
    threshold_fixedpoint = 1e-4;
    threshold_checkonGrassmann = 1e-10;
    threshold_checkonStiefel = 1e-10;
    threshold_logStiefel = 1e-4;
    if doGrassmannpFCenter
        % do Grassmann center of mass method
        GrassmannOpt = Grassmann_Optimization(w, frames, threshold_gradnorm, threshold_fixedpoint, threshold_checkonGrassmann);
        if doGD
            break;
        else
            [center, value, grad] = GrassmannOpt.Center_Mass_pFrobenius;
        end
    else
        % do Stiefel center of mass method
        StiefelOpt = Stiefel_Optimization(w, frames, threshold_gradnorm, threshold_fixedpoint, threshold_checkonStiefel, threshold_logStiefel);
        if doStiefelEuclidCenter
            if doGD
                break;
            else
                [center, value, gradnorm] = StiefelOpt.Center_Mass_Euclid;
            end
        else
            break;
        end
    end
    % project x to center x and classify it using k-nearest-neighbor on the projection via center of all (interpolation number) clusters
    x_test = x * center;
    y_test = y;
    X_train = data_train.x(aggregate_cluster, :) * center;
    Y_train = data_train.y(aggregate_cluster);    
    isclassified_c = knn(x_test, y_test, X_train, Y_train, k_nearest_neighbor);
    classified_c(test_index) = isclassified_c;
    % output the result
    fprintf("original dimension classified = %d, original dimension aggregate classified = %d, benchmark classified = %d, center mass classfied = %d\n", isclassified_o, isclassified_agg_o, isclassified_bm, isclassified_c);
end
toc;

fprintf("original dimension classification rate = %f %%, original dimension aggregate classification rate = %f %%, benchmark correct classification rate = %f %%, center mass correct classification rate = %f %%\n", (sum(classified_o)/test_size)*100, (sum(classified_agg_o)/test_size)*100, (sum(classified_bm)/test_size)*100, (sum(classified_c)/test_size)*100);



function [data_train, leafs, data_test] = LPP_ObtainData(data_original_train, data_original_test, d_pre, kd_LPP, kd_PCA, train_size, test_size, ht)

% Sample a training dataset data_train from the original data_original_train set, data_train = (data_train.x, data_train.y)
% Set the partition tree depth = ht
% Tree partition data_train into clusters C_1, ..., C_{2^{ht}} with centers m_1, ..., m_{2^{ht}}
% first project each C_i to local PCA with dimension kd_PCA  
% then continue to construct the local LPP frames A_1, ..., A_{2^{ht}} in G(kd_data, kd_LPP) using supervised affinity
% Sample a test dataset data_test from the original data_original_test set for testing purposes, data_test = (data_test.x, data_test.y)

% Input:
%   data_original_train, data_original_test = the original training and testing data set
%   d_pre = the data preprocessing projection dimension
%   kd_PCA = the initial PCA embedding dimension
%   kd_LPP = the LPP embedding dimension 
%   train_size, test_size = the training/testing data set size
%   ht = the partition tree height
% Output:
%   data_train, data_test = the training/testing data set , size is traing_size/test_size
%   leafs = leafs{k}, the cluster indexes in data_train

%compute the sizes of the original training and testing dataset
[n_data_original_train, kd_data_original_train] = size(data_original_train.x);
[n_data_original_test, kd_data_original_test] = size(data_original_test.x);

% first concatnate data_train.x and data_test.x together
data_x = double(vertcat(data_original_train.x, data_original_test.x));
[n_data_x, kd_data_x] = size(data_x);
% do an initial PCA on data
[A0, s0, lat0] = pca(data_x);
% bulid a given dimensional d_pre embedding of data_orginal_train(test).x into new data_original_train(test).x, for faster computation only
data_original_train.x = data_original_train.x * A0(:, 1:d_pre);
data_original_test.x = data_original_test.x * A0(:, 1:d_pre);

% build the training and testing data sets
indexes = randperm(n_data_original_train); 
% randomly pick the training sample of size train_size from data.x dataset
train_indexes = indexes(1: train_size);
% form the data_train dataset
data_train.x = double(data_original_train.x(train_indexes, :));
data_train.y = double(data_original_train.y(train_indexes, :));

indexes = randperm(n_data_original_test); 
% randomly pick the test sample of size test_size from data dataset, must be disjoint from data_train
test_indexes = indexes(1: test_size);
% form the data_test dataset
data_test.x = double(data_original_test.x(test_indexes, :));
data_test.y = double(data_original_test.y(test_indexes, :));

% do an initial PCA on data_train
[A0, s0, lat0] = pca(data_train.x);
% bulid a kd_PCA dimensional embedding of data_train in x0
x0 = data_train.x * A0(:, 1:kd_PCA);
% from x0, partition into 2^ht leaf nodes, each leaf node can give samples for a local LPP, done in LPP_BuildDataModel
[indx, leafs] = buildVisualWordList(x0, ht);

end




function [Seq] = LPP_BuildDataModel(data_train, leafs, kd_PCA, kd_LPP)
% build LPP Model for each leaf in data_train
% Assume the tree partition indexes of data_train into clusters C_1, ..., C_{2^{ht}} with centers m_1, ..., m_{2^{ht}} is given in leafs
% first project each C_i to local PCA with dimension kd_PCA  
% then continue to construct the local LPP frames A_1, ..., A_{2^{ht}} in G(kd_data, kd_LPP) using supervised affinity

% Input:
%   data_train = the training data set
%   kd_PCA = the initial PCA embedding dimension
%   kd_LPP = the LPP embedding dimension 
% Output:
%   Seq = the LPP frames corresponding to each cluster in data_train, labeling the correponding Grassmann equivalence class

[n_data_train, kd_data_train] = size(data_train.x);
% initialize the LPP frames A_1,...,A_{2^{ht}}
Seq = zeros(kd_data_train, kd_LPP, length(leafs));

% input: data, indx, leafs
for k=1:length(leafs)
    % form the data_train subsample for the k-th cluster
    data_train_x_k = data_train.x(leafs{k}, :);
    data_train_y_k = data_train.y(leafs{k});
    % augment the data_train_x_k and data_train_y_k by GMM sampling and pre-trained network prediction
    % ...to be filled...
    % do an initial PCA first, for the k-th cluster, so data_train_x_k dimension is reduced to kd_PCA
    [PCA_k, lat] = pca(data_train_x_k);
    PCA_k = Complete_SpecialOrthogonal(PCA_k);
    data_train_x_k = data_train_x_k * PCA_k(:, 1:kd_PCA);
    % then do LPP for the PCA embedded data_train_x_k and reduce the dimension to kd_LPP
    % construct the supervise affinity matrix S
    between_class_affinity = 0;
    S_k = affinity_supervised(data_train_x_k, data_train_y_k, between_class_affinity);
    % construct the graph Laplacian L and degree matrix D
    [L_k, D_k] = graph_laplacian(S_k);
    % do LPP
    [A_k, lambda] = LPP(data_train_x_k, L_k, D_k);
    [LPP_k, R] = qr(A_k);        
    % obtain the frame Seq(:,:,k)
    Seq(:, :, k) = PCA_k(:, 1:kd_PCA) * LPP_k(:, 1:kd_LPP);
    fprintf("frame %d, size = (%d, %d), Stiefel = %f \n", k, size(Seq(:,:,k), 1), size(Seq(:,:,k), 2), norm(Seq(:,:,k)'*Seq(:,:,k)-eye(kd_LPP), 'fro'));
end

end





function [isclassified] = knn(x_test, y_test, X_train, Y_train, k)
% k-nearest neighbor classfication
% given test data x and label y, find in a training set (X, Y) the k-nearest points x1,...,xk to x, and classify x as majority vote on y1,...,yk
% if the classification is correct, return 1, otherwise return 0
    m = length(Y_train);
    if k>m
        k=m;
    end
    % find the first k-nearest neighbor
    dist = zeros(m, 1);
    for i=1:m
        dist(i) = norm(x_test-X_train(i,:));
    end
    [dist_sort, indexes] = sort(dist, 1, 'ascend');
    % do a majority vote on the first k-nearest neighbor
    label = Y_train(indexes(1:k));
    vote = tabulate(label);
    [max_percent, max_vote_index] = max(vote(:, 3));
    % class is the predicted label based on majority vote
    class = vote(max_vote_index, 1);
    if class == y_test
        isclassified = 1;
    else
        isclassified = 0;
    end
end


function [Q] = Complete_SpecialOrthogonal(A)
%given the matrix A in St(p, n), complete it into Q = [A B] in SO(n)
   n = size(A, 1);
   p = size(A, 2);
   if n > p
       [O1, D, O2] = svd(A);
       O2_ext = [O2 zeros(p, n-p); zeros(n-p, p) eye(n-p)]; 
       Q = O1 * O2_ext';
       if det(Q) < 0
           Q(:, p+1) = -Q(:, p+1);
       end
   else
       Q = A;
   end
end    


function [W, lambda] = LPP(X, L, D)
% solve the laplacian embedding, given data set X={x1,...,xm}, the graph laplacian L and degree matrix D    
    mtx_L = X' * L * X;
    mtx_D = X' * D * X;
    [W, LAMBDA] = eig(mtx_L, mtx_D);
    lambda = diag(LAMBDA);
    [lambda, SortOrder] = sort(lambda, 'descend');
    W = W(:,SortOrder);
end


function [L, D] = graph_laplacian(S)
% construct the graph laplacian L and the degress matrix D from the given affinity matrix S, 
    D = diag(sum(S, 1));
    L = D - S;
end


function [S] = affinity_supervised(X, Y, between_class_affinity)
% given a set of data points X={x1,...,xm} with label Y={y1,...,ym}, construct their spuervised affinity matrix S for LPP
    % original distances squares between xi and xj
    f_dist1 = pdist2(X, X);
    % heat kernel size
    mdist = mean(f_dist1(:)); 
    h = -log(0.15)/mdist;
    S1 = exp(-h*f_dist1);
    % utilize supervised info
    id_dist = pdist2(Y, Y);
    S2 = S1; 
    S2(find(id_dist~=0)) = between_class_affinity;
    % obtain the spuervised affinity S
    S = S2;
end



