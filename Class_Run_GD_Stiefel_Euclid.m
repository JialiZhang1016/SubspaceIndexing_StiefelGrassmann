%Finding the Euclidean Center of Mass via Gradient Descent on Stiefel Manifolds
%Given objective function f_F(A)=\sum_{k=1}^m \omega_k \|A-A_k\|_F^2 where A, A_k\in St(p, n)
%Use Gradient Descent to find min_A f_F(A) 

%Runfile for the class Class_GD_Stiefel_Euclid

%author: Wenqing Hu (Missouri S&T)

%Generate from SIFT data the local frames A_1, ..., A_{256}
%Find their center of mass A in Euclidean norm under weight w_k = exp(-d_k) where d_k is the SIFT distance to each cluster's centroid  
%Test the PCA energy spectrum for SIFT projection on A

clearvars;
clear classes;

%set the A_1,...,A_m on St(p, n) and the weight sequence 
%the initial point A on St(p, n) is chosen as one of the A_k's

%the PCA embedding dimension = kd_siftStiefel
kd_siftStiefel = 12;
%select the sift_sample in SIFT dataset that we will be working on
%generate A_1,...,A_m and omega_1,...,omega_m
[Seq, omega, sift_sample] = SIFT_PCA(kd_siftStiefel);

%choose an initial frame to start the GD, randomly selected from A_1,...,A_m
rng(1);
m = length(Seq);
init_label = randi(m);
A = Seq(:, :, init_label);

%all these frames are on St(n, p), actually n=128 and p=kd_siftStiefel
n = size(A, 1);
p = size(A, 2);

%Set the parameters for GD on Stiefel St(p, n)
iteration = 1000;
lr = 0.01;
lrdecayrate = 1;
gradnormthreshold = 1e-4;
checkonStiefelthreshold = 1e-10;

StiefelOpt = Class_GD_Stiefel_Euclid(omega, Seq, iteration, lr, lrdecayrate, gradnormthreshold, checkonStiefelthreshold);

[fseq, gradfnormseq, distanceseq, minf] = StiefelOpt.GD_Stiefel(A);


%output the center of mass and check if it is still on Stiefel manifold
disp(minf);
fprintf("the center is given by the above matrix of size %d times %d\n", n, p);
[ifStiefel, distance] = StiefelOpt.CheckOnStiefel(minf);
fprintf("if still on Stiefel= %d, distance to Stiefel= %f\n", ifStiefel, distance);



%plot the objective value, gradient norm and distance to St(p, n)
figure;
plot(fseq, '-.', 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:iteration);
xlabel('iteration');
ylabel('Objective Value');
legend('objective value');
title('Gradient Descent on Stiefel Manifold');

figure;
plot(gradfnormseq, '-*', 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:iteration);
xlabel('iteration');
ylabel('Gradient Norm');
legend('gradient norm');
title('Gradient Descent on Stiefel Manifold');

figure;
plot(distanceseq, '--','Color', [0.6350 0.0780 0.1840],  'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:iteration);
xlabel('iteration');
ylabel('Distance to Stiefel');
title('Gradient Descent on Stiefel Manifold');
legend('distance to Stiefel');



%test the PCA spectrum of SIFT projection onto the eigenspace spanned by the center on St(p, n) that we found
%do an initial PCA on sift_samples dataset
[A0, s0, lat0] = pca(sift_sample);

figure;
plot(lat0, '-.', 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:n);

%project sift_sample onto the center frame on St(p, n)
x_mean = sift_sample * minf;
%analyze the PCA spectrum of the low-dimensional projection
[A_mean, s_mean, lat_mean] = pca(x_mean);
%plot the PCA spectrum for the projection of sift_sample onto x_mean
%figure;
hold on; 
grid on;
%stem(lat_mean, '.'); 
plot(lat_mean, '--','Color', [0.6350 0.0780 0.1840],  'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:kd_siftStiefel);
%title('sift projected onto mean eigenspaces pca eigenvalues');

%to compare, randomly pick one element in Seq and do projection and PCA spectrum
init_label = 1;
x_bm = sift_sample * Seq(:, :, init_label);
%analyze the PCA spectrum of the low-dimensional projection
[A_bm, s_bm, lat_bm] = pca(x_bm);
%plot the PCA spectrum for the projection of sift_sample onto x_mean
%figure;
%hold on; 
%grid on;
plot(lat_bm, '-*', 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:kd_siftStiefel);
%stem(lat_bm, '.'); 
title('sift projected onto frames pca eigenvalues');
%title('sift projected onto randomly selected cluster frames pca eigenvalues');
legend('sift original', 'total center', 'random center');
xlabel('dimension');
ylabel('eigenvalues');
hold off;


figure;
hold on; 
grid on;
stem(lat_mean, '.'); 
xlabel('dimension');
ylabel('eigenvalues');
title('sift projected onto the center of mass Stiefel matrix pca eigenvalues');



