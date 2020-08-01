%Optimization On Grassmann Manifolds
%contains various functions for operating optimization calculus and related geometries on Grassmann manifold G_{n,p}

%author: Wenqing Hu (Missouri S&T)

classdef Grassmann_Optimization
   
%class open variables 
properties  
    omega %the weight sequence
    Seq   %the sequence of points on St(p, n) that are identified as points on G_{n,p} 
    threshold_gradnorm   %the threshold for gradient norm when using GD
    threshold_fixedpoint %the threshold for fixed-point iteration for average
    threshold_checkonGrassmann  %the threshold for checking if iteration is still on the Grassmann manifold (actually St(p,n))
end  

   
%functions in the class
methods

    
function self = Grassmann_Optimization(omega, Seq, threshold_gradnorm, threshold_fixedpoint, threshold_checkonGrassmann)           
%class constructor function
    if nargin > 0  
        self.omega = omega;  
        self.Seq = Seq;  
        self.threshold_gradnorm = threshold_gradnorm;
        self.threshold_fixedpoint = threshold_fixedpoint;
        self.threshold_checkonGrassmann = threshold_checkonGrassmann;
    end  
end


function [value, grad] = Center_Mass_function_gradient_pFrobenius(self, Y)
%find the value and grad of the projected Frobenius distance center of mass function f(A)=\sum_{k=1}^m w_k |AA^T-A_kA_k^T|_F^2 on G_{n,p}
    A = Y;
    m = length(self.omega);
    n = size(A, 1);
    p = size(A, 2);
    value = 0;
    for k = 1:m
        Mtx = A * A' - self.Seq(:,:,k) * self.Seq(:,:,k)';
        value = value + self.omega(k) * (norm(Mtx, 'fro')^2);
    end
    grad = zeros(n, p);
    for k = 1:m
        M1 = A .* (2 * self.omega(k));
        M2 = self.Seq(:,:,k) * self.Seq(:,:,k)' * A .* (4 * self.omega(k));
        grad = grad + M1 - M2;
    end
    grad = grad - A * A' * grad;
end    


function [pF_Center, value, grad] = Center_Mass_pFrobenius(self)
%directly calculate the center of mass on G_{n,p} with respect to projected Frobenius norm
    m  = length(self.omega);
    n = size(self.Seq, 1);
    p = size(self.Seq, 2);
    total_weight = sum(self.omega);
    Mtx = zeros(n, n);
    for k = 1:m
        Mtx = Mtx + (self.Seq(:,:,k) * self.Seq(:,:,k)').*(self.omega(k)/total_weight);
    end
    [Q, D, Q1] = svd(Mtx);
    I = [diag(ones(p, 1)); zeros(n-p, p)];
    pF_Center = Q * I;
    [value, grad] = self.Center_Mass_function_gradient_pFrobenius(pF_Center);
end


function [GD_pF_Center, valueseq, gradnormseq, distanceseq] = Center_Mass_GD_pFrobenius(self, Y, iteration, lr, lrdecayrate)
%find the projected Frobenius distance Center of Mass via Gradient Descent on Grassmann Manifolds
%Given objective function f(A)=\sum_{k=1}^m \omega_k |AA^T-A_kA_k^T|_F^2 where A, A_k\in St(p, n)
%Use Gradient Descent to find min_A f_F(A) 
    learning_rate = lr; 
    distanceseq = zeros(iteration, 1);
    gradnormseq = zeros(iteration, 1);
    valueseq = zeros(iteration, 1);
    A = Y;
    n = size(A, 1);
    p = size(A, 2);
    m = length(self.omega);
    for i = 1:iteration
        [value, grad] = self.Center_Mass_function_gradient_pFrobenius(A);
        gradnormseq(i) = norm(grad, 'fro');
        valueseq(i) = value;
        if norm(grad, 'fro') < 0.1 * self.threshold_gradnorm
            break;
        elseif norm(grad, 'fro') < self.threshold_gradnorm
            learning_rate = learning_rate * lrdecayrate;
        end
        A_previous = A;
        A = self.ExpGrassmann(A, -learning_rate * grad);
        %check if this A is still on Grassmann manifold
        [ifGrassmann, distanceseq(i)] = self.CheckOnGrassmann(A);
        fprintf("iteration %d, value= %f, gradnorm= %f, ifGrassmann= %d\n", i, value, norm(grad, 'fro'), ifGrassmann);
        %if not, pull it back to Grassmann manifold using the projection and another exponential map
        if ~ifGrassmann
            Z = A - A_previous;
            prj_tg = self.projection_tangent(A_previous, Z);
            A = self.ExpGrassmann(A_previous, prj_tg);
        end
    end
    GD_pF_Center = A;
end


function [value, grad] = Center_Mass_function_gradient_Arc(self, Y)
%find the value and grad of the arc-distance center of mass function f(A)=\sum_{k=1}^m w_kd^2(A, A_k) where d is the arc-distance on G_{n,p}
    A = Y;
    m = length(self.omega);
    n = size(A, 1);
    p = size(A, 2);
    value = 0;
    for k = 1:m
        %H = self.LogGrassmann(A, self.Seq(:,:,k));
        %H = self.projection_tangent(A, H);
        Mtx = Y' * self.Seq(:,:,k);
        [O1, D, O2] = svd(Mtx);
        add = sum(acos(diag(D)).^2);
        value = value + self.omega(k) * add;
        %value = value + self.omega(k) * (norm(H, 'fro')^2);
    end
    grad = zeros(n, p);
    for k = 1:m
        grad = grad + self.omega(k) * self.LogGrassmann(A, self.Seq(:,:,k));
    end 
    grad = self.projection_tangent(A, grad);
end


function [Arc_Center, valueseq, gradnormseq, distanceseq, errornormseq] = Center_Mass_Arc(self, Y, iteration)
%find the Arc-distance (natural geodesic distance) Center of Mass via Fixed-Point Iteration on Grassmann Manifolds
    valueseq = zeros(iteration, 1);
    gradnormseq = zeros(iteration, 1);
    distanceseq = zeros(iteration, 1);
    errornormseq = zeros(iteration, 1);
    A = Y;
    n = size(A, 1);
    p = size(A, 2);
    m = length(self.omega);
    for i = 1:iteration
        value = self.Center_Mass_function_gradient_Arc(A);
        valueseq(i) = value;
        A_previous = A;
        grad = zeros(n, p);
        for k = 1:m
            grad = grad + self.omega(k) * self.LogGrassmann(A, self.Seq(:,:,k));
        end
        %grad_previous = grad;
        %grad = self.projection_tangent(A, grad);
        %disp(grad-grad_previous);
        fprintf("iteration %d, value= %f, gradnorm= %f, ", i, value, norm(grad, 'fro'));
        gradnormseq(i) = norm(grad, 'fro');
        if norm(grad, 'fro') < 0.1 * self.threshold_gradnorm
            break;
        end
        A = self.ExpGrassmann(A, grad);
        %[O1, D, O2] = svd(A); O2 = [O2 zeros(p,n-p); zeros(n-p,p) eye(n-p)]; A = O1 * O2'; A = A(:, 1:p);
        error = A*A'-A_previous*A_previous';
        fprintf("A difference= %f, ", norm(error,'fro'));
        errornormseq(i) = norm(error, 'fro');
        %check if this A is still on Grassmann manifold
        [ifGrassmann, distanceseq(i)] = self.CheckOnGrassmann(A);
        fprintf("ifGrassmann= %d, distance= %d \n", ifGrassmann, distanceseq(i));
        %if not, pull it back to Grassmann manifold using the projection and another exponential map
        %if ~ifGrassmann
        %    Z = A - A_previous;
        %    prj_tg = self.projection_tangent(A_previous, Z);
        %    A = self.ExpGrassmann(A_previous, prj_tg);
        %end
    end
    Arc_Center = A;
end


function [GD_Arc_Center, valueseq, gradnormseq, distanceseq] = Center_Mass_GD_Arc(self, Y, iteration, lr, lrdecayrate)
%find the Arc-distance (natural geodesic distance) Center of Mass via Gradient Descent on Grassmann Manifolds
%Given objective function f(A)=\sum_{k=1}^m \omega_k d^2_{G_{n,p}}(A, A_k) where A, A_k\in St(p, n)
%Use Gradient Descent to find min_A f_F(A) 
    learning_rate = lr; 
    valueseq = zeros(iteration, 1);
    gradnormseq = zeros(iteration, 1);
    distanceseq = zeros(iteration, 1);
    A = Y;
    n = size(A, 1);
    p = size(A, 2);
    m = length(self.omega);
    for i = 1:iteration
        A_previous = A;
        grad = zeros(n, p);
        value = self.Center_Mass_function_gradient_Arc(A);
        valueseq(i) = value;
        for k = 1:m
            grad = grad + self.omega(k) * self.LogGrassmann(A, self.Seq(:,:,k));
        end
        grad = self.projection_tangent(A, grad);
        fprintf("iteration %d, value= %f, gradnorm= %f, ", i, value, norm(grad, 'fro'));
        gradnormseq(i) = norm(grad, 'fro');
        if norm(grad, 'fro') < 0.1 * self.threshold_gradnorm
            break;
        elseif norm(grad, 'fro') < self.threshold_gradnorm
            learning_rate = learning_rate * lrdecayrate;
        end
        A = self.ExpGrassmann(A, -learning_rate * grad);
        %[O1, D, O2] = svd(A); O2 = [O2 zeros(p,n-p); zeros(n-p,p) eye(n-p)]; A = O1 * O2'; A = A(:, 1:p);
        fprintf("A difference= %f, ", norm(A*A'-A_previous*A_previous','fro'));
        %check if this A is still on Grassmann manifold
        [ifGrassmann, distanceseq(i)] = self.CheckOnGrassmann(A);
        fprintf("ifGrassmann= %d \n", ifGrassmann);
        %if not, pull it back to Grassmann manifold using the projection and another exponential map
        if ~ifGrassmann
            Z = A - A_previous;
            prj_tg = self.projection_tangent(A_previous, Z);
            A = self.ExpGrassmann(A_previous, prj_tg);
        end
    end
    GD_Arc_Center = A;
end

     
function [ifGrassmann, distance] = CheckOnGrassmann(self, Y)
%test if the given matrix Y is on the Grassmann manifold G_{n,p}
%same as tesing that Y is on St(p, n)
%Y is the matrix to be tested, threshold is a threshold value for returning true
    n = size(Y, 1);
    p = size(Y, 2);
    Mtx = Y'*Y - eye(p);
    distance = norm(Mtx, 'fro');
    if distance <= self.threshold_checkonGrassmann
        ifGrassmann = true;
    else
        ifGrassmann = false;
    end
end
        

function [ifTangentGrassmann, distance] = CheckTangentGrassmann(self, Y, H)
%test if the given matrix H is on the tangent space of Grassmann manifold T_Y G_{n,p}
%H is the matrix to be tested, threshold is a threshold value for returning true
    n = size(Y, 1);
    p = size(Y, 2);
    n_H = size(H, 1);
    p_H = size(H, 2);
    if (n == n_H) && (p == p_H)
        Mtx = Y' * H;
        distance = norm(Mtx, 'fro');
        if distance <= self.threshold_checkonGrassmann
            ifTangentGrassmann = true;
        else
            ifTangentGrassmann = false;
        end
    else
        ifTangentGrassmann = false;
    end
end


function [exp, U, S, V] = ExpGrassmann(self, Y, H)
%Exponential Map on Grassmann manifold G_{n,p}
%Y is the matrix on St(p,n) and H is the tangent vector to G_{n,p}, which is an n times p matrix
%returns the svd decomposition H = U S V^T and based on them one can calculate exp_Y(H) = YVcos(S)+Usin(S)
    n = size(Y, 1);
    p = size(Y, 2);
    [U, S, V] = svd(H);
    U = U(:, 1:p);
    S = S(1:p, :);
    cosS = diag(cos(diag(S)));
    sinS = diag(sin(diag(S)));
    exp = Y * V * cosS + U * sinS;
end


function [log, U, G, V] = LogGrassmann(self, Y, Y_tilde)
%Logarithmic Map on Grassmann manifold G_{n,p}
%Y is the matrix on St(p,n) and Y_tilde is another matrix on St(p,n), both correspond to some points on G_{n,p} 
%returns U, G, V such that one can calculate log_Y(Y_tilde) = H = U (tan^{-1}(G)) V^T
    n = size(Y, 1);
    p = size(Y, 2);
    M = Y' * Y_tilde;
    if det(M)~=0 
        Mtx = (eye(n) - Y * Y') * Y_tilde * inv(M);
        [U, G, V] = svd(Mtx);
        U = U(:, 1:p);
        G = G(1:p, :);
        taninvG = diag(atan(diag(G)));
        log = U * taninvG * V';
        log = self.projection_tangent(Y, log);
        %exp = self.ExpGrassmann(Y, log);
        %if norm(Y_tilde*Y_tilde'-exp*exp', 'fro') ~= 0
        %    fprintf("log has error %f in accuracy!\n", norm(Y_tilde*Y_tilde'-exp*exp', 'fro'));
        %end    
    else
        fprintf("Error LogGrassmann: Y'*Y_tilde not invertible!\n")
        fprintf("Y = \n"); disp(Y);
        fprintf("Y_tilde= \n"); disp(Y_tilde);
        fprintf("Y' * Y_tilde = \n"); disp(Y' * Y_tilde);
    end
end    


function [prj_tg] = projection_tangent(self, Y, Z)
%calculate the projection onto tangent space of Grassmann manifold G_{n,p}
%Pi_{T, Y}(Z) projects matrix Z of size n times p onto the tangent space of G_{n,p} at point Y\in St(p, n)
%returns the tangent vector prj_tg on T_Y(G(n,p))
    n = size(Y, 1);
    p = size(Y, 2);
    prj_tg = (eye(n) - Y * Y') * Z;
end
       


end %end of class methods

end %end of class Grassmann_Optimization