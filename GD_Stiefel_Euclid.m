%Finding the Euclidean Center of Mass via Gradient Descent on Stiefel Manifolds
%Given objective function f_F(A)=\sum_{k=1}^m \omega_k \|A-A_k\|_F^2 where A, A_k\in St(p, n)
%Use Gradient Descent to find min_A f_F(A)

%author: Wenqing Hu (Missouri S&T)

clearvars;

%set the A_1,...,A_m on St(p, n) and the weight sequence 
%the initial point A on St(p, n) is chosen as one of the A_k's

[Seq, omega] = SIFT_PCA;

A = Seq(:, :, 1);

n = size(A, 1);
p = size(A, 2);

%run the GD on Stiefel St(p, n)
iteration = 10000;
lr = 0.001;
lrdecayrate = 1;
gradnormthreshold = 1e-4;
checkonStiefelthreshold = 1e-10;

[fseq, gradfnormseq, distanceseq, minf] = GD_Stiefel(A, omega, Seq, iteration, lr, lrdecayrate, gradnormthreshold, checkonStiefelthreshold);


%output the center of mass and check if it is still on Stiefel manifold
fprintf("the center is given by the following matrix of size %d times %d\n", n, p);
disp(minf);
[ifStiefel, distance] = CheckOnStiefel(minf, 1);
fprintf("if still on Stiefel= %d, distance to Stiefel= %f\n", ifStiefel, distance);



%plot the objective value, gradient norm and distance to St(p, n)
figure;
plot(fseq, '-.', 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:iteration);
hold on;
plot(gradfnormseq, '-*', 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:iteration);
plot(distanceseq, '--','Color', [0.6350 0.0780 0.1840],  'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:iteration);
legend('objective value', 'gradient norm', 'distance to Stiefel');
xlabel('iteration');
ylabel('Objective Value');
hold off;
title('Gradient Descent on Stiefel Manifold');



%gradient descent on Stiefel Manifolds
%Given objective function f_F(A)=\sum_{k=1}^m \omega_k \|A-A_k\|_F^2 where A, A_k\in St(p, n)
%Use Gradient Descent to find min_A f_F(A)
function [fseq, gradfnormseq, distanceseq, minf] = GD_Stiefel(Y, omega, Seq, iteration, lr, lrdecayrate, gradnormthreshold, checkonStiefelthreshold)
    fseq = zeros(iteration, 1);
    gradfnormseq = zeros(iteration, 1);
    distanceseq = zeros(iteration, 1);
    A = Y;
    for i = 1:iteration
        %record the previous step
        A_previous = A;
        %calculate the function value and gradient on Stiefel
        [f, gradf] = gradientStiefel(A, omega, Seq);
        %record the function value and gradient norm
        fseq(i) = f;
        gradfnormseq(i) = norm(gradf, 'fro');
        %if the gradient norm is small than the threshold value, then decay the stepsize exponentially
        %we are able to tune the decay rate, and so far due to convexity it seems not decay is the best option
        if norm(gradf, 'fro') < gradnormthreshold
            lr = lr * lrdecayrate;
        end
        %gradient descent on Stiefel, obtain the new step A
        H = lr * (-1) * gradf;
        [M, N, Q] = ExpStiefel(A, H);
        A = A * M + Q * N;        
        %check if this A is still on Stiefel manifold
        [ifStiefel, distanceseq(i)] = CheckOnStiefel(A, checkonStiefelthreshold);
        %if not, pull it back to Stiefel manifold using the projection and another exponential map
        if ~ifStiefel
            Z = A - A_previous;
            prj_tg = projection_tangent(A_previous, Z);
            [M, N, Q] = ExpStiefel(A_previous, prj_tg);
            A = A_previous * M + Q * N;
        end
        %print the iteration value and gradient norm
        fprintf("iteration %d, value= %f, gradnorm= %f\n", i, f, norm(gradf, 'fro'));
    end
    %obtain the center of mass
    minf = A;
end


%test if the given matrix Y is on the Stiefel manifold St(p, n)
%Y is the matrix to be tested, threshold is a threshold value, if \|Y^TY-I_p\|_F < threshold then return true
function [ifStiefel, distance] = CheckOnStiefel(Y, threshold)
    n = size(Y, 1);
    p = size(Y, 2);
    Mtx = Y'*Y - eye(p);
    distance = norm(Mtx, 'fro');
    if distance <= threshold
        ifStiefel = true;
    else
        ifStiefel = false;
    end
end


%test if the given matrix H is on the tangent space of Stiefel manifold T_Y St(p, n)
%H is the matrix to be tested, threshold is a threshold value, if \|Y^TH+H^TY\| < threshold then return true
function [ifTangentStiefel] = CheckTangentStiefel(Y, H, threshold)
    n = size(Y, 1);
    p = size(Y, 2);
    n_H = size(H, 1);
    p_H = size(H, 2);
    if (n == n_H) && (p == p_H)
        Mtx = Y' * H + H' * Y;
        distance = norm(Mtx + Mtx', 'fro');
        if distance <= threshold
            ifTangentStiefel = true;
        else
            ifTangentStiefel = false;
        end
    else
        ifTangentStiefel = false;
    end
end


%Exponential Map on Stiefel manifold St(p, n)
%Y is the matrix on St(p, n) and H is the tangent vector
%returns M, N, Q and based on them one can calculate exp_Y(H)=YM+QN
function [M, N, Q] = ExpStiefel(Y, H)
    n = size(Y, 1);
    p = size(Y, 2);
    W = (eye(n) - Y*Y') * H;
    [Q, R] = qr(W);
    Q = Q(:, 1:p);
    R = R(1:p, :);
    O = zeros(p, p);
    Mtx = [Y'*H -R'; R O];
    Exponential = expm(Mtx);
    i = [eye(p); zeros(p, p)];
    Multiply = Exponential*i;
    M = Multiply(1:p, :);
    N = Multiply(p+1:2*p, :);
end


%calculate the function value and the gradient on Stiefel manifold St(p, n) of the Euclidean center of mass function 
%f_F(A)=\sum_{k=1}^m w_k \|A-A_k\|_F^2
function [f, gradf] = gradientStiefel(Y, omega, Seq)
    m = length(omega);
    f = 0;
    for i = 1:m
        f = f + omega(i)*(norm(Y-Seq(:,:,i), 'fro')^2);
    end
    gradf = 0;
    for i = 1:m
        gradf = gradf + 2*omega(i)*((Y-Seq(:,:,i))-Y*(Y-Seq(:,:,i))'*Y);
    end
end


%calculate the projection onto tangent space of Stiefel manifold St(p, n)
%Pi_{T, Y}(Z) projects matrix Z of size n by p onto the tangent space of St(p, n) at point Y\in St(p, n)
%returns the tangent vector prj_tg on T_Y(St(p, n))
function [prj_tg] = projection_tangent(Y, Z)
    n = size(Y, 1);
    p = size(Y, 2);
    skew = (Y' * Z - Z' * Y)/2;
    prj_tg = Y * skew + (eye(n) - Y * Y') * Z;
end
