clearvars;
clear classes;

Seq(:, :, 1) = [0 1 0; 1 0 0; 0 0 0; 0 0 1];
Seq(:, :, 2) = [1 0 0; 0 0 0; 0 1 0; 0 0 1];
Seq(:, :, 3) = [1 0 0; 0 1 0; 0 0 0; 0 0 1];

omega = [1 1 1];

%all these frames are on St(n, p), actually n=128 and p=kd_siftStiefel
n = size(Seq, 1);
p = size(Seq, 2);

%Set the Stiefel Optimization object with given threshold parameters
threshold_gradnorm = 1e-7;
threshold_fixedpoint = 1e-4;
threshold_checkonStiefel = 1e-10;
threshold_logStiefel = 1e-10;

StiefelOpt = Stiefel_Optimization(omega, Seq, threshold_gradnorm, threshold_fixedpoint, threshold_checkonStiefel, threshold_logStiefel);


%control variables
doQR_Retraction = 0;
doQR_Lifting = 0;
doQR_Retraction_Center = 0;
doLogStiefel = 0;
doEuclidGD = 0;
doEuclidCenterDirect = 0;
doCompleteSpecialOrthogonal = 0;
doSOLiftingCenter = 1;



if doQR_Retraction
    %compute the QR decomposition type retraction at X with tangent vector V
    X = Seq(:, :, 2);
    [f, gradf] = StiefelOpt.Center_Mass_function_gradient_Euclid(X);
    V = gradf;
    [Q, R] = StiefelOpt.Retraction_QR(X, V);
    disp(Q);
    fprintf("the retraction is given by the above matrix in St(%d, %d)\n", p, n);
    fprintf("ifStiefel = %d\n", StiefelOpt.CheckOnStiefel(Q));
end



if doQR_Lifting
    %compute the QR decomposition type lifting at X with respect to Q
    X = Seq(:, :, 3);
    disp(X);
    fprintf("at the above matrix in St(%d, %d)\n", p, n);    
    [f, gradf] = StiefelOpt.Center_Mass_function_gradient_Euclid(X);
    V_original = gradf;
    disp(V_original);
    fprintf("along the above matrix in T_X(St(%d, %d))\n", p, n); 
    %first get the Q from V, the recover V from lifting
    [Q, R_retraction] = StiefelOpt.Retraction_QR(X, V_original);
    disp(Q);
    fprintf("the retraction Q = P_X(V) is given by the above matrix in St(%d, %d)\n", p, n);
    fprintf("ifStiefel = %d\n", StiefelOpt.CheckOnStiefel(Q));
    [V, R_lifting] = StiefelOpt.Lifting_QR(X, Q);
    disp(V);
    fprintf("the lifting V = P_X^{-1}(Q) is given by the above matrix in T_X(St(%d, %d))\n", p, n);
    fprintf("ifTangentStiefel = %d\n", StiefelOpt.CheckTangentStiefel(X, V));
end



if doQR_Retraction_Center
    %choose the Euclidean center of mass as initial frame to start the fixed point iteration
    [A, initvalue, initgradnorm] = StiefelOpt.Center_Mass_Euclid;
    %Set the iteration number
    iteration = 1000;    
    A_c = StiefelOpt.Center_Mass_QR_Retraction(A, iteration);
    disp(A_c);
    fprintf("the QR-retraction center is given by the above matrix in St(%d, %d)\n", p, n);
    fprintf("value is %f, distance of GD minimizer to St(p, n) is %f20, if still on Stiefel is %d\n", StiefelOpt.gradientStiefel_Euclid(A_c), norm(A_c'*A_c-eye(p), 'fro'), StiefelOpt.CheckOnStiefel(A_c));
end


if doLogStiefel
    %compute exp_X(V_original)
    X = Seq(:, :, 2);
    disp(X);
    fprintf("at the above matrix in St(%d, %d)\n", p, n);    
    [f, gradf] = StiefelOpt.Center_Mass_function_gradient_Euclid(X);
    V_original = gradf;
    disp(V_original);
    fprintf("along the above matrix in T_X(St(%d, %d))\n", p, n); 
    [M, N, Q, exp] = StiefelOpt.ExpStiefel(X, V_original);
    disp(exp);
    fprintf("exp_X(V) is given by the above matrix in St(%d, %d)\n", p, n);
    fprintf("ifStiefel = %d\n", StiefelOpt.CheckOnStiefel(exp));
    iteration = 1000;
    [A, B, Q, log] = StiefelOpt.LogStiefel(X, exp, iteration);
    disp(log);
    fprintf("log_X(exp_X(V)) is given by the above matrix in T_X(St(%d, %d))\n", p, n);
    fprintf("ifTangentStiefel = %d\n", StiefelOpt.CheckTangentStiefel(X, log));   
end    


if doEuclidGD
    %use GD to find Euclidean Stiefel center of mass
    tic;
    %choose an initial frame to start the GD, randomly selected from A_1,...,A_m
    %rng(1, 'twister');
    m = length(Seq);
    init_label = 1;
    A = Seq(:, :, init_label);
    %A = [0.9856 -0.1691; 0.1196 0.6969; 0.1196 0.6969];
    %Set the parameters for GD on Stiefel St(p, n)
    iteration = 100;
    lr = 0.01;
    lrdecayrate = 1;
    [minf, fseq, gradfnormseq, distanceseq] = StiefelOpt.Center_Mass_GD_Euclid(A, iteration, lr, lrdecayrate);
    fprintf("GD min value is %f, distance of GD minimizer to St(p, n) is %f20, if still on Stiefel is %d\n", StiefelOpt.Center_Mass_function_gradient_Euclid(minf), norm(minf'*minf-eye(p), 'fro'), StiefelOpt.CheckOnStiefel(minf));
    toc;
end


if doEuclidCenterDirect
    %use direct method to find Euclidean Stiefel center of mass
    tic;
    [minf2, minfvalue2, gradminfnorm2] = StiefelOpt.Center_Mass_Euclid;
    fprintf("SVD min value is %f, distance of SVD minimizer to St(p, n) is %f20, if still on Stiefel is %d\n", minfvalue2, norm(minf2'*minf2-eye(p), 'fro'), StiefelOpt.CheckOnStiefel(minf2));
    toc;
end

if doEuclidGD && doEuclidCenterDirect
    %compare
    fprintf("GD minimizer to St(p, n)/SVD minimizer to St(p, n)=%f\n", norm(minf'*minf-eye(p), 'fro')/norm(minf2'*minf2-eye(p), 'fro'));
end

if doCompleteSpecialOrthogonal
    %complete a given Stiefel matrix to special orthogonal
    %X = Seq(:, :, 2);
    [minf2, minfvalue2, gradminfnorm2] = StiefelOpt.Center_Mass_Euclid;
    X = minf2;
    Q = StiefelOpt.Complete_SpecialOrthogonal(X);
    disp(Q);
    fprintf("the orthogonal matrix is given by the above matrix in SO(%d)\n", n);
    fprintf("ifStiefel = %d\n", StiefelOpt.CheckOnStiefel(Q));
end

if doSOLiftingCenter
    [minf2, minfvalue2, gradminfnorm2] = StiefelOpt.Center_Mass_Euclid;
    Q = StiefelOpt.Complete_SpecialOrthogonal(minf2);
    iteration = 1000;
    SOCenter = StiefelOpt.Center_Mass_SO_Lifting(Q, iteration);
    disp(SOCenter);
    fprintf("SO(n) lifting center of mass is given by the above matrix\n")
end    
