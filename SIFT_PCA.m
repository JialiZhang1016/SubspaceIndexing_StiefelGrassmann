%from the SIFT data set train a sequence of frames A_1, ..., A_m on St(p, n)
%first train a global SIFT PCA embedding A_0
%then the sequence of frames A_1, ..., A_m correspond to a partition of the SIFT dataset
%from this partition, calculate the weights w_k=exp(-d_k) where d_k is the distance on SIFT dataset from the SIFT mean to the mean at each cluster
%return [A_1, ..., A_m] and [w_1, ..., w_m] and send to GD_Stiefel_Euclid.m

%author: Wenqing Hu (Missouri S&T)

function [Seq, omega, sift_sample] = SIFT_PCA(kd_siftStiefel)
%the PCA embedding dimension = kd_siftStiefel

%load the sift dataset
%structure: 
%         n_mp: [3000 3000 3000 364 400 4005 2550]
%        n_nmp: [29904 29904 29904 3640 4000 48675 25500]
%       mp_fid: [16319×2 double]
%      nmp_fid: [171527×2 double]
%        sifts: [10068850×128 uint8]
%    sift_offs: [1×33590 double]

load ~/文档/work_SubspaceIndexingStiefleGrassmann/Code_Subspace_indexing_Stiefel_Grassman/cdvs-sift300-dataset.mat;

%n_sift is the number of samples in sifts dataset, kd_sift is the original dimension of each sample
%kd_sift=128
[n_sift, kd_sift] = size(sifts);

%randomly pick 204800 samples from sift dataset, for partition purposes
offs = randperm(n_sift); 
offs = offs(1: 200*2^10);
%form the sift_sample dataset
sift_sample = double(sifts(offs, :));

%find the total mean of sift_sample, set as the ceter point for calculating distances
mean_sift_0 = mean(sift_sample);

%do an initial PCA on sift_samples dataset
[A0, s0, lat0] = pca(sift_sample);

%plot the whole PCA spectrum for sift_samples
doplotPCAspectrum = 0;
if doplotPCAspectrum
    figure;
    hold on; grid on;
    stem(lat0, '.'); 
    title('sift pca eigenvalues');
end

%set the kd-partition tree height = ht
ht = 8; 

%bulid a 12-dimensional embedding of sift_samples in x0
x0 = sift_sample * A0(:, 1:kd_siftStiefel);

%from x0, partition into 2^ht leaf nodes, each leaf node can give samples for a local PCA
[indx, leafs]=buildVisualWordList(x0, ht);


%the weight sequence
omega = zeros(length(leafs), 1);

% build PCA Model for each leaf
doBuildSiftModel = 1;
% input: sift, indx, leafs
if doBuildSiftModel
    for k=1:length(leafs)
        %form the sift subsample for the k-th cluster
        sift_k = sift_sample(leafs{k}, :); 
        %find the mean of sift_k, set as the ceter point for the k-th cluster
        mean_sift_k = mean(sift_k);
        %calculate the distance d_k = dist(mean_sift_0, mean_sift_k)
        d_k = norm(mean_sift_0 - mean_sift_k);
        %let omega_k = exp(-d_k) (too close to 0), so need to modify to exp(-0.01*d_k)
        omega(k) = exp(-0.01 * d_k);
        %do PCA analysis of sift_k and form the k-th frame A_k
        [A{k}, s, lat] = pca(sift_k);
        Seq(:, :, k) = A{k}(:, 1:kd_siftStiefel); 
    end
end    


end
