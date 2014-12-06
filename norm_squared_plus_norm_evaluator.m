%[f,g]=norm_squared_plus_norm_evaluator(x,A,b,lambda2)
% creates a function evaluator the smooth part of an elastic-net regularizer
%   f= |  A x - B |_2^2   + | lambda2 .* x | _2

function [fx,gx]=norm_squared_plus_norm_evaluator(x,A,b,lambda2)
    res=A*x-b;
%     lambda2=sqrt(lambda2);
    f1=norm(res,2).^2;
    f2=norm(x.*lambda2,2);
    fx=f1+f2;

    if nargout > 1
        g1=2*A'*res;
        g2=x.*lambda2;
        g2=g2.*lambda2/max(norm(g2,2),1e-6);
        gx=g1+g2;
    end

end