%Optimization On Stiefel Manifolds
%contains various functions for operating optimization calculus and related geometries on Stiefel Manifold St(p, n)

%author: Wenqing Hu (Missouri S&T)

classdef Stiefel_Optimization
   
%class open variables 
properties  
    omega %the weight sequence
    Seq   %the sequence of pointes on St(p, n)
    threshold_gradnorm   %the threshold for gradient norm when using GD
    threshold_fixedpoint %the threshold for fixed-point iteration for average
    threshold_checkonStiefel  %the threshold for checking if iteration is still on St(p, n)
    threshold_logStiefel %the threshold for calculating the Stiefel logarithmic map via iterative method
end  

   
%functions in the class
methods

    
function self = Stiefel_Optimization(omega, Seq, threshold_gradnorm, threshold_fixedpoint, threshold_checkonStiefel, threshold_logStiefel)           
%class constructor function
    if nargin > 0  
        self.omega = omega;  
        self.Seq = Seq;  
        self.threshold_gradnorm = threshold_gradnorm;
        self.threshold_fixedpoint = threshold_fixedpoint;
        self.threshold_checkonStiefel = threshold_checkonStiefel;
        self.threshold_logStiefel = threshold_logStiefel;
    end  
end


function [Q] = Complete_SpecialOrthogonal(self, A)
%given the matrix A in St(p, n), complete it into Q = [A B] in SO(n)
   n = size(A, 1);
   p = size(A, 2);
   [O1, D, O2] = svd(A);
   O2_ext = [O2 zeros(p, n-p); zeros(n-p, p) eye(n-p)]; 
   Q = O1 * O2_ext';
   if det(Q) < 0
       Q(:, p+1) = -Q(:, p+1);
   end    
end    


function [SO_Lifting_Center] = Center_Mass_SO_Lifting(self, Y, iteration)
%complete every St(p, n) matrix in Seq to SO(n) matrix, using fixed point iteration to average them on SO(n) with respect to weights w 
   n = size(Y, 1); 
   p = size(Y, 2);
   m = length(self.omega);
   SO_Seq = zeros(n, n, m);
   for k = 1:m
       SO_Seq(:, :, k) = self.Complete_SpecialOrthogonal(self.Seq(:, :, k));
   end
   A = self.Complete_SpecialOrthogonal(Y);
   for i = 1:iteration
       Mtx = zeros(n, n);
       for k = 1:m
           Q = A' * SO_Seq(:,:,k);
           M = logm(Q);
           Mtx = Mtx + self.omega(k) * M;
       end
       A_previous = A;
       A = A * expm(Mtx);
       error = norm(A - A_previous, 'fro');
       fprintf("iteration = %d, fixed point error = %f", i, error);
       if error < self.threshold_fixedpoint
           break;
       end
       %check if this A is still on SO(n)
       [ifStiefel, distanceseq(i)] = self.CheckOnStiefel(A);
       fprintf(", distance to SO = %f, ifStiefel=%d \n", distanceseq(i), ifStiefel);
       %if not, pull it back to SO(n) using svd decomposition
       if ~ifStiefel
           [O1, D, O2] = svd(A);
           A = O1 * O2';
       end
   end
   SO_Lifting_Center = A;
end


function [GD_SO_Lifting_Center, gradnormseq, distanceseq] = Center_Mass_GD_SO_Lifting(self, Y, iteration, lr, lrdecayrate)
%complete every St(p, n) matrix in Seq to SO(n) matrix, using gradient descent on SO(n) to average them on SO(n) with respect to weights w 
   n = size(Y, 1); 
   p = size(Y, 2);
   learning_rate = lr;
   m = length(self.omega);
   gradnormseq = zeros(iteration, 1);
   distanceseq = zeros(iteration, 1);
   SO_Seq = zeros(n, n, m);
   for k = 1:m
       SO_Seq(:, :, k) = self.Complete_SpecialOrthogonal(self.Seq(:, :, k));
   end
   A = self.Complete_SpecialOrthogonal(Y);
   for i = 1:iteration
       Mtx = zeros(n, n);
       for k = 1:m
           Q = A' * SO_Seq(:, :, k);
           M = logm(Q);
           Mtx = Mtx + self.omega(k) * M;
       end
       gradnorm = norm(Mtx, 'fro');
       gradnormseq(i) = gradnorm;
       if gradnorm < 0.1 * self.threshold_gradnorm   
           break;
       else
           if gradnorm < self.threshold_gradnorm   
               learning_rate = learning_rate * lrdecayrate;
           end
       end
       A = A * expm(- learning_rate * Mtx);
       %check if this A is still on SO(n)
       [ifStiefel, distanceseq(i)] = self.CheckOnStiefel(A);
       fprintf("iteration = %d, gradient norm = %f, distance to SO = %f, ifStiefel=%d \n", i, gradnorm, distanceseq(i), ifStiefel);
       %if not, pull it back to SO(n) using svd decomposition
       if ~ifStiefel
           [O1, D, O2] = svd(A);
           A = O1 * O2';
       end
   end
   GD_SO_Lifting_Center = A;
end


function [Q, R] = Retraction_QR(self, X, V)
%calculate the QR-decomposition type retraction Q = P_X(V) where X in St(p, n), V in T_X(St(p, n)) and Q in St(p, n)
    n_X = size(X, 1);
    p_X = size(X, 2);
    n_V = size(V, 1);
    p_V = size(V, 2);
    if (n_X == n_V) && (p_X == p_V)
        Mtx = X + V;
        [Q, R] = qr(Mtx);
        D = diag(sign(diag(R)));
        zeroD = zeros(n_X-size(D, 1), n_X-size(D, 1));
        zero1 = zeros(size(D, 1), n_X-size(D, 1));
        zero2 = zeros(n_X-size(D, 1), size(D, 1));
        D = [D zero1; zero2 zeroD];
        Q = Q * D; R = D * R;
        Q = Q(:, 1:p_X);
        R = R(1:p_X, :);
    else
        fprintf("QR Retraction Q=P_X(V): size Error!\n");
        fprintf("size of X is (%d, %d), size of V is (%d, %d)\n", size(X, 1), size(X, 2), size(V, 1), size(V, 2));
    end
end


function [V, R] = Lifting_QR(self, X, Q)
%calculate the QR-decomposition type lifting V = P_X^{-1}(Q) where X in St(p, n), Q in St(p, n) and V in T_X(St(p, n))
    n_X = size(X, 1);
    p_X = size(X, 2);
    n_Q = size(Q, 1);
    p_Q = size(Q, 2);
    if (n_X == n_Q) && (p_X == p_Q) 
        Mtx = X' * Q;
        R = zeros(p_X, p_X);
        if Mtx(1, 1) > 0
            R(1, 1) = 1/Mtx(1, 1);
        else
            fprintf("QR Lifting P_X^{-1}(Q): Minor(1,1)<=0 Error!\n");
            fprintf("X=\n"); disp(X);
            fprintf("Q=\n"); disp(Q);
            fprintf("M=\n"); disp(Mtx);
            return;
        end
        for i = 2:p_X
           M_tilde_i = Mtx(1:i, 1:i);
           if det(M_tilde_i) ~= 0
              b_i = zeros(i, 1);
              for j = 1:i-1
                  b_i(j) = - Mtx(i, 1:j) * R(1:j, j);
              end
              b_i(i) = 1;
              r_tilde_i = linsolve(M_tilde_i, b_i);
              R(:, i) = [r_tilde_i; zeros(p_X-i, 1)];
           else
              fprintf("QR Lifting P_X^{-1}(Q): det(M_tilde_i) = 0 Error!\n");
              fprintf("X=\n"); disp(X);
              fprintf("Q=\n"); disp(Q);
              fprintf("M=\n"); disp(M);
              fprintf("M_tilde_i=\n"); disp(M_tilde_i);
              return;
           end
           if R(i, i) <= 0
              fprintf("QR Lifting P_X^{-1}(Q): R(i, i) <=0 Error!\n");
              fprintf("X=\n"); disp(X);
              fprintf("Q=\n"); disp(Q);
              fprintf("R=\n"); disp(R);
              return;
           end
        end
        V = Q * R - X;
    else
        fprintf("QR Lifting P_X^{-1}(Q): size Error!\n");
        fprintf("size of X is (%d, %d) and size of Q is (%d, %d)\n", size(X, 1), size(X, 2), size(Q, 1), size(Q, 2));
    end
end


function [QR_Retraction_Center] = Center_Mass_QR_Retraction(self, Y, iteration)
%using fixed-point iteration, calculate the QR-decomposition type retraction-based center of mass of A_k with weights w_k
    A = Y;
    m = length(self.omega);
    n = size(Y, 1);
    p = size(Y, 2);
    V_new = zeros(n, p);
    for k = 1:m
        [V, R] = self.Lifting_QR(self.Seq(:, :, k), A);
        V_new = V_new + self.omega(k) * V;
    end
    for i=1:iteration
        [A_new, R] = self.Retraction_QR(A, V_new);
        error = norm(A_new-A, 'fro');
        fprintf("iteration = %d, fixed point iteration error = %f\n", i, error);
        %disp(A_new);
        if error < self.threshold_fixedpoint
            break;
        end
        A = A_new;
        V_new = zeros(n, p);
        for k = 1:m
            [V, R] = self.Lifting_QR(self.Seq(:, :, k), A);
            V_new = V_new + self.omega(k) * V;
        end
    end
    QR_Retraction_Center = A;
end


function [f, gradf] = Center_Mass_function_gradient_Euclid(self, Y)
%calculate the function value and the gradient on Stiefel manifold St(p, n) of the Euclidean center of mass function f_F(A)=\sum_{k=1}^m w_k \|A-A_k\|_F^2
    m = length(self.omega);
    f = 0;
    for i = 1:m
        f = f + self.omega(i)*(norm(Y-self.Seq(:,:,i), 'fro')^2);
    end
    gradf = 0;
    for i = 1:m
        gradf = gradf + 2*self.omega(i)*((Y-self.Seq(:,:,i))-Y*(Y-self.Seq(:,:,i))'*Y);
    end
end


function [Euclid_Center, value, gradnorm] = Center_Mass_Euclid(self)
%directly calculate the Euclidean center of mass that is the St(p, n) minimizer of f_F(A)=\sum_{k=1}^m w_k\|A-A_k\|_F^2, according to our elegant lemma based on SVD
    m = length(self.omega);
    n = size(self.Seq, 1);
    p = size(self.Seq, 2);
    B = zeros(n, p);
    for i=1:m
        B = B + self.omega(i) * self.Seq(:, :, i);
    end
    [O1, D, O2] = svd(B);
    O = zeros(p, n-p);
    Mtx = [eye(p) O];
    Mtx = Mtx';
    Euclid_Center = O1 * Mtx * O2';
    [value, grad] = self.Center_Mass_function_gradient_Euclid(Euclid_Center);
    gradnorm = norm(grad, 'fro');
end


function [GD_Euclid_Center, valueseq, gradnormseq, distanceseq] = Center_Mass_GD_Euclid(self, Y, iteration, lr, lrdecayrate)
%find the Euclidean Center of Mass via Gradient Descent on Stiefel Manifolds
%Given objective function f_F(A)=\sum_{k=1}^m \omega_k \|A-A_k\|_F^2 where A, A_k\in St(p, n)
%Use Gradient Descent to find min_A f_F(A) 
    learning_rate = lr; 
    valueseq = zeros(iteration, 1);
    gradnormseq = zeros(iteration, 1);
    distanceseq = zeros(iteration, 1);
    A = Y;
    for i = 1:iteration
        %record the previous step
        A_previous = A;
        %calculate the function value and gradient on Stiefel
        [f, gradf] = self.Center_Mass_function_gradient_Euclid(A);
        %print the iteration value and gradient norm
        fprintf("iteration %d, value= %f, gradnorm= %f\n", i, f, norm(gradf, 'fro'));
        %record the function value and gradient norm
        valueseq(i) = f;
        gradnormseq(i) = norm(gradf, 'fro');
        %if the gradient norm is smaller than 0.1 times the threshold value, stop iteration 
        if norm(gradf, 'fro') < 0.1 * self.threshold_gradnorm
            break;
        elseif norm(gradf, 'fro') < self.threshold_gradnorm
            %if the gradient norm is smaller than the threshold value, then decay the stepsize exponentially
            %we are able to tune the decay rate, and so far due to convexity it seems not decay is the best option
            learning_rate = learning_rate * lrdecayrate;
        end
        %gradient descent on Stiefel, obtain the new step A
        H = learning_rate * (-1) * gradf;
        [M, N, Q, A] = self.ExpStiefel(A, H);      
        %check if this A is still on Stiefel manifold
        [ifStiefel, distanceseq(i)] = self.CheckOnStiefel(A);
        %if not, pull it back to Stiefel manifold using the projection and another exponential map
        if ~ifStiefel
            Z = A - A_previous;
            prj_tg = self.projection_tangent(A_previous, Z);
            [M, N, Q, A] = self.ExpStiefel(A_previous, prj_tg);
        end
    end
    %obtain the center of mass
    GD_Euclid_Center = A;
end

     
function [ifStiefel, distance] = CheckOnStiefel(self, Y)
%test if the given matrix Y is on the Stiefel manifold St(p, n)
%Y is the matrix to be tested, threshold is a threshold value, if \|Y^TY-I_p\|_F < threshold then return true
    n = size(Y, 1);
    p = size(Y, 2);
    Mtx = Y'*Y - eye(p);
    distance = norm(Mtx, 'fro');
    if distance <= self.threshold_checkonStiefel
        ifStiefel = true;
    else
        ifStiefel = false;
    end
end
        

function [ifTangentStiefel] = CheckTangentStiefel(self, Y, H)
%test if the given matrix H is on the tangent space of Stiefel manifold T_Y St(p, n)
%H is the matrix to be tested, threshold is a threshold value, if \|Y^TH+H^TY\| < threshold then return true
    n = size(Y, 1);
    p = size(Y, 2);
    n_H = size(H, 1);
    p_H = size(H, 2);
    if (n == n_H) && (p == p_H)
        Mtx = Y' * H + H' * Y;
        distance = norm(Mtx + Mtx', 'fro');
        if distance <= self.threshold_checkonStiefel
            ifTangentStiefel = true;
        else
            ifTangentStiefel = false;
        end
    else
        ifTangentStiefel = false;
    end
end


function [M, N, Q, exp] = ExpStiefel(self, Y, H)
%Exponential Map on Stiefel manifold St(p, n)
%Y is the matrix on St(p, n) and H is the tangent vector
%returns M, N, Q and based on them one can calculate exp_Y(H) = YM+QN
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
    exp = Y * M + Q * N;
end


function [A, B, Q, log] = LogStiefel(self, Y, Y_tilde, iteration)
%Logarithmic Map on Stiefel manifold St(p, n)
%Y is the matrix on St(p, n) and Y_tilde is another matrix on St(p, n) close to Y
%returns A, B, Q such that one can calculate log_Y(Y_tilde) = H = YA+QB
%guarentees exp_Y(H) = Y_tilde with desired precision
    n = size(Y, 1);
    p = size(Y, 2);
    M = Y' * Y_tilde;
    [Q, N] = qr(Y_tilde - Y * M);
    Q = Q(:, 1:p); N = N(1:p, :);
    Mtx = [M; N];
    [O1, D, O2] = svd(Mtx);
    O2_ext = [O2 zeros(p, p); zeros(p, p) eye(p)]; 
    V = O1 * O2_ext';
    disp(V);
    disp(Mtx);
    disp(V'*V);
    disp(D);
    for k = 1:iteration
        Log_Matrix = logm(V);
        A = Log_Matrix(1:p, 1:p);
        B = Log_Matrix(p+1:2*p, 1:p);
        C = Log_Matrix(p+1:2*p, p+1:2*p);
        if norm(C, 'fro') < self.threshold_logStiefel
            break;
        end
        Phi = expm(-C);
        W = [eye(p) zeros(p, p); zeros(p, p) Phi];
        V = V * W;
    end
    log = Y * A + Q * B;
end    


function [prj_tg] = projection_tangent(self, Y, Z)
%calculate the projection onto tangent space of Stiefel manifold St(p, n)
%Pi_{T, Y}(Z) projects matrix Z of size n by p onto the tangent space of St(p, n) at point Y\in St(p, n)
%returns the tangent vector prj_tg on T_Y(St(p, n))
    n = size(Y, 1);
    p = size(Y, 2);
    skew = (Y' * Z - Z' * Y)/2;
    prj_tg = Y * skew + (eye(n) - Y * Y') * Z;
end
       

end %end of class methods
  
end %end of class