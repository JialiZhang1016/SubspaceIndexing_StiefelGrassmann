%Optimization On Stiefel Manifolds
%contains various functions for operating optimization calculus and related geometries on Stiefel Manifold St(p, n)

%author: Wenqing Hu (Missouri S&T)

classdef Stiefel_Optimization
   
%class open variables 
properties  
    omega %the weight sequence
    Seq   %the sequence of pointes on St(p, n)
    iteration %number of iterations
    lr        %learning rate
    lrdecayrate  %learning rate decay rate
    gradnormthreshold   %the threshold for gradient norm
    checkonStiefelthreshold  %the threshold for checking if iteration is still on St(p, n)
end  

   
%functions in the class
methods
    
%class constructive function
function self = Stiefel_Optimization(omega, Seq, iteration, lr, lrdecayrate, gradnormthreshold, checkonStiefelthreshold)           
    if nargin > 0  
        self.omega = omega;  
        self.Seq = Seq;  
        self.iteration = iteration;  
        self.lr = lr;
        self.lrdecayrate = lrdecayrate;
        self.gradnormthreshold = gradnormthreshold;
        self.checkonStiefelthreshold = checkonStiefelthreshold;
    end  
end 


%calculate the function value and the gradient on Stiefel manifold St(p, n) of the Euclidean center of mass function 
%f_F(A)=\sum_{k=1}^m w_k \|A-A_k\|_F^2
function [f, gradf] = gradientStiefel_Euclid(self, Y)
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

%directly calculate the Euclidean center of mass that is the St(p, n) minimizer of f_F(A)=\sum_{k=1}^m w_k\|A-A_k\|_F^2
%according to Professor Tiefeng Jiang's elegant lemma based on SVD
function [minvalue, gradminfnorm, minf] = CenterMass_Stiefel_Euclid(self, Y)
    A = Y;
    m = length(self.omega);
    n = size(Y, 1);
    p = size(Y, 2);
    B = zeros(n, p);
    for i=1:m
        B = B + self.omega(i) * self.Seq(i);
    end
    [O1, D, O2] = svd(B);
    O = zeros(p, n-p);
    Mtx = [eye(p) O];
    Mtx = Mtx';
    minf = O1 * Mtx * O2';
    [minvalue, gradminf] = self.gradientStiefel_Euclid(minf);
    gradminfnorm = norm(gradminf, 'fro');
end

%gradient descent on Stiefel Manifolds
%GD_Stiefel_Euclid can find the Euclidean Center of Mass via Gradient Descent on Stiefel Manifolds
%Given objective function f_F(A)=\sum_{k=1}^m \omega_k \|A-A_k\|_F^2 where A, A_k\in St(p, n)
%Use Gradient Descent to find min_A f_F(A) 
function [fseq, gradfnormseq, distanceseq, minf] = GD_Stiefel_Euclid(self, Y)
    learning_rate = self.lr; %when doing learning rate decay we do not want to change the lr in the class, just local
    fseq = zeros(self.iteration, 1);
    gradfnormseq = zeros(self.iteration, 1);
    distanceseq = zeros(self.iteration, 1);
    A = Y;
    for i = 1:self.iteration
        %record the previous step
        A_previous = A;
        %calculate the function value and gradient on Stiefel
        [f, gradf] = self.gradientStiefel_Euclid(A);
        %record the function value and gradient norm
        fseq(i) = f;
        gradfnormseq(i) = norm(gradf, 'fro');
        %if the gradient norm is small than the threshold value, then decay the stepsize exponentially
        %we are able to tune the decay rate, and so far due to convexity it seems not decay is the best option
        if norm(gradf, 'fro') < self.gradnormthreshold
            learning_rate = learning_rate * self.lrdecayrate;
        end
        %gradient descent on Stiefel, obtain the new step A
        H = learning_rate * (-1) * gradf;
        [M, N, Q] = self.ExpStiefel(A, H);
        A = A * M + Q * N;        
        %check if this A is still on Stiefel manifold
        [ifStiefel, distanceseq(i)] = self.CheckOnStiefel(A);
        %if not, pull it back to Stiefel manifold using the projection and another exponential map
        if ~ifStiefel
            Z = A - A_previous;
            prj_tg = self.projection_tangent(A_previous, Z);
            [M, N, Q] = self.ExpStiefel(A_previous, prj_tg);
            A = A_previous * M + Q * N;
        end
        %print the iteration value and gradient norm
        %fprintf("iteration %d, value= %f, gradnorm= %f\n", i, f, norm(gradf, 'fro'));
    end
    %obtain the center of mass
    minf = A;
end

      
%test if the given matrix Y is on the Stiefel manifold St(p, n)
%Y is the matrix to be tested, threshold is a threshold value, if \|Y^TY-I_p\|_F < threshold then return true
function [ifStiefel, distance] = CheckOnStiefel(self, Y)
    n = size(Y, 1);
    p = size(Y, 2);
    Mtx = Y'*Y - eye(p);
    distance = norm(Mtx, 'fro');
    if distance <= self.checkonStiefelthreshold
        ifStiefel = true;
    else
        ifStiefel = false;
    end
end
        

%test if the given matrix H is on the tangent space of Stiefel manifold T_Y St(p, n)
%H is the matrix to be tested, threshold is a threshold value, if \|Y^TH+H^TY\| < threshold then return true
function [ifTangentStiefel] = CheckTangentStiefel(self, Y, H)
    n = size(Y, 1);
    p = size(Y, 2);
    n_H = size(H, 1);
    p_H = size(H, 2);
    if (n == n_H) && (p == p_H)
        Mtx = Y' * H + H' * Y;
        distance = norm(Mtx + Mtx', 'fro');
        if distance <= self.checkonStiefelthreshold
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
function [M, N, Q] = ExpStiefel(self, Y, H)
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


%calculate the projection onto tangent space of Stiefel manifold St(p, n)
%Pi_{T, Y}(Z) projects matrix Z of size n by p onto the tangent space of St(p, n) at point Y\in St(p, n)
%returns the tangent vector prj_tg on T_Y(St(p, n))
function [prj_tg] = projection_tangent(self, Y, Z)
    n = size(Y, 1);
    p = size(Y, 2);
    skew = (Y' * Z - Z' * Y)/2;
    prj_tg = Y * skew + (eye(n) - Y * Y') * Z;
end
       
end %end of class methods
  
end %end of class