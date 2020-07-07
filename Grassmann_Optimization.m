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

function [GD_Arc_Center, gradnormseq, distanceseq] = Center_Mass_GD_Arc(self, Y, iteration, lr, lrdecayrate)
%find the Arc-distance (natural geodesic distance) Center of Mass via Gradient Descent on Grassmann Manifolds
%Given objective function f(A)=\sum_{k=1}^m \omega_k d^2_{G_{n,p}}(A, A_k) where A, A_k\in St(p, n)
%Use Gradient Descent to find min_A f_F(A) 
    learning_rate = lr; 
    gradnormseq = zeros(iteration, 1);
    distanceseq = zeros(iteration, 1);
    A = Y;
    for i = 1:iteration
        %record the previous step
        A_previous = A;
        %calculate the function value and gradient on Stiefel
        gradf = self.Center_Mass_function_gradient_Euclid(A);
        %print the iteration value and gradient norm
        fprintf("iteration %d, gradnorm= %f\n", i, norm(gradf, 'fro'));
        %record the function value and gradient norm
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


function [U, S, V, exp] = ExpGrassmann(self, Y, H)
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


function [U, G, V, log] = LogGrassmann(self, Y, Y_tilde)
%Logarithmic Map on Grassmann manifold G_{n,p}
%Y is the matrix on St(p,n) and Y_tilde is another matrix on St(p,n), both correspond to some points on G_{n,p} 
%returns U, G, V such that one can calculate log_Y(Y_tilde) = H = U (tan^{-1}(G)) V^T
    n = size(Y, 1);
    p = size(Y, 2);
    M = Y' * Y_tilde;
    if det(M) > 0
        Mtx = (eye(n) - Y * Y') * Y_tilde * inv(M);
        [U, G, V] = svd(Mtx);
        U = U(:, 1:p);
        G = G(1:p, :);
        taninvG = diag(atan(diag(G)));
        log = U * taninvG * V';
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