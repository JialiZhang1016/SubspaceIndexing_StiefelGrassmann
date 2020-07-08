clearvars;
clear classes;

Seq(:, :, 1) = [0 1 0 0; 1 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 1];
Seq(:, :, 2) = [1 0 0 0; 0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
Seq(:, :, 3) = [1 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 1];

omega = [1 1 1];

%all these frames are on St(n, p), actually n=128 and p=kd_siftStiefel
n = size(Seq, 1);
p = size(Seq, 2);

%Set the Stiefel and Grassmann Optimizations object with given threshold parameters
threshold_gradnorm = 1e-7;
threshold_fixedpoint = 1e-4;
threshold_checkonStiefel = 1e-10;
threshold_logStiefel = 1e-10;
threshold_checkonGrassmann = 1e-10;


%build the Stiefel and Grassmann Optimization objects
StiefelOpt = Stiefel_Optimization(omega, Seq, threshold_gradnorm, threshold_fixedpoint, threshold_checkonStiefel, threshold_logStiefel);
GrassmannOpt = Grassmann_Optimization(omega, Seq, threshold_gradnorm, threshold_fixedpoint, threshold_checkonGrassmann);




doStiefel = 1;

if doStiefel

doQR_Retraction = 0;
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


doQR_Lifting = 0;
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


doQR_Retraction_Center = 0;
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


doLogStiefel = 0;
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


doEuclidGD = 0;
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


doEuclidCenterDirect = 0;
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


doCompleteSpecialOrthogonal = 0;
if doCompleteSpecialOrthogonal
    %complete a given Stiefel matrix to special orthogonal
    X = Seq(:, :, 2);
    %[minf2, minfvalue2, gradminfnorm2] = StiefelOpt.Center_Mass_Euclid;
    %X = minf2;
    Q = StiefelOpt.Complete_SpecialOrthogonal(X);
    disp(Q);
    fprintf("the orthogonal matrix is given by the above matrix in SO(%d)\n", n);
    fprintf("ifStiefel = %d\n", StiefelOpt.CheckOnStiefel(Q));
end


doSOLiftingCenter = 0;
if doSOLiftingCenter
    [minf2, minfvalue2, gradminfnorm2] = StiefelOpt.Center_Mass_Euclid;
    iteration = 1000;
    SOCenter = StiefelOpt.Center_Mass_SO_Lifting(minf2, iteration);
    disp(SOCenter);
    fprintf("SO(n) lifting center of mass is given by the above matrix\n\n")
    disp(SOCenter' * SOCenter);
end    


doSOLiftingGD = 0;
if doSOLiftingGD
    [minf2, minfvalue2, gradminfnorm2] = StiefelOpt.Center_Mass_Euclid;
    iteration = 10000;
    lr = 0.001;
    lrdecayrate = 1;
    [GD_SOCenter, gradnormseq] = StiefelOpt.Center_Mass_GD_SO_Lifting(minf2, iteration, lr, lrdecayrate);
end    

end %if doStiefel


doGrassmann = 1;

if doGrassmann

doExpLog = 0;
if doExpLog
    Y = Seq(:, :, 2);
    %Y_tilde = Seq(:, :, 1);
    [Y_tilde, minfvalue2, gradminfnorm2] = StiefelOpt.Center_Mass_Euclid;
    H = GrassmannOpt.projection_tangent(Y, Y_tilde);
    fprintf("Y = \n"); disp(Y);    
    fprintf("H = \n"); disp(H);
    iftangentGrassmann = GrassmannOpt.CheckTangentGrassmann(Y, H);
    fprintf("H is tangent to Grassmann at Y? %d\n", iftangentGrassmann);
    exp = GrassmannOpt.ExpGrassmann(Y, H);
    fprintf("exp_Y(H) = \n"); disp(exp);
    log = GrassmannOpt.LogGrassmann(Y, exp);
    fprintf("log_Y(exp_Y(H)) = \n"); disp(log);
    
    Y = Seq(:, :, 2);
    [Y_tilde, minfvalue2, gradminfnorm2] = StiefelOpt.Center_Mass_Euclid;
    fprintf("Y = \n"); disp(Y);    
    fprintf("Y_tilde = \n"); disp(Y_tilde);
    ifGrassmann = GrassmannOpt.CheckOnGrassmann(Y_tilde);
    fprintf("Y_tilde is on Grassmann at Y? %d\n", ifGrassmann);
    log = GrassmannOpt.LogGrassmann(Y, Y_tilde);
    fprintf("log_Y(Y_tilde) = \n"); disp(log);
    iftangentGrassmann = GrassmannOpt.CheckTangentGrassmann(Y, log);
    fprintf("log_Y(Y_tilde) is tangent to Grassmann at Y? %d\n", iftangentGrassmann);
    exp = GrassmannOpt.ExpGrassmann(Y, log);
    fprintf("exp_Y(log_Y(Y_tilde)) = \n"); disp(exp);    
    log = GrassmannOpt.LogGrassmann(Y, exp);
    fprintf("log_Y(exp_Y(log_Y(Y_tilde))) = \n"); disp(log);
    iftangentGrassmann = GrassmannOpt.CheckTangentGrassmann(Y, log);
end


doArcGD = 0;
if doArcGD
    %use GD to find Arc-distance Grassmann center of mass
    tic;
    A = StiefelOpt.Center_Mass_Euclid;
    %Set the parameters for arc-GD on G_{n,p}
    iteration = 1000;
    lr = 0.01;
    lrdecayrate = 1;
    [center, gradnormseq, distanceseq] = GrassmannOpt.Center_Mass_GD_Arc(A, iteration, lr, lrdecayrate);
    disp(center);
    %fprintf("GD min value is %f, distance of GD minimizer to St(p, n) is %f20, if still on Stiefel is %d\n", StiefelOpt.Center_Mass_function_gradient_Euclid(minf), norm(minf'*minf-eye(p), 'fro'), StiefelOpt.CheckOnStiefel(minf));
    toc;
end

doArcCenter = 0;
if doArcCenter
    %use GD to find Arc-distance Grassmann center of mass
    tic;
    %A = Seq(:, :, 2);
    A = StiefelOpt.Center_Mass_Euclid;
    %Set the parameters for arc-GD on G_{n,p}
    iteration = 1000;
    [center1, gradnormseq, errornormseq, valueseq, distanceseq] = GrassmannOpt.Center_Mass_Arc(A, iteration);
    disp(center1);
    disp(center*center');
    disp(center1*center1');
    %fprintf("GD min value is %f, distance of GD minimizer to St(p, n) is %f20, if still on Stiefel is %d\n", StiefelOpt.Center_Mass_function_gradient_Euclid(minf), norm(minf'*minf-eye(p), 'fro'), StiefelOpt.CheckOnStiefel(minf));
    toc;
end

dopFCenter = 1;
if dopFCenter
    %directly find the projection Frobenius center of mass
    tic;
    [pfCenter, value, grad] = GrassmannOpt.Center_Mass_pFrobenius;
    disp(pfCenter);
end


end %if doGrassmann