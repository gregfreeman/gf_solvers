function obj=modified_lasso_evaluator(eval,lambda1,lambda2,Phi,y,subset)
% obj=modified_lasso_evaluator(eval,lambda1,lambda2,Phi,y,subset)
%
% x2=x(subset)
% f(x)=norm(y-Phi*x2,2)^2 + lambda1*norm(x2,1) + lambda2*g(x)
%
%  g(x) = eval.f
%  gradient g(x) = eval.df
%
%   obj.f=@f;  f(X)
%   obj.df=@df; gradient f(x)
%
%

    obj.f=@f;
    obj.df=@df;

    function [fx,f_components]=f(x)
        x1=x(subset);
        f0=norm(y-Phi*x1,2).^2;
        f1=norm(x1,1);
        if(lambda2>0)            
            f2=eval.f(x);
        else 
            f2=0;
        end
        fx=f0+lambda1*f1+lambda2*f2;
        if(nargout>1)
            f_components={f0,f1,f2};
        end                        
    end
    function [g,g_components]=df(x)
        x1=x(subset);
        g0=zeros(size(x));
        g0(subset)=2.*(Phi'*(Phi*x1-y));
        g1=sign(x);
        if(lambda2>0)            
            tic
            g2=eval.df(x);
            toc
        else 
            g2=0;
        end
        g=g0+lambda1*g1+lambda2*g2;
        g(~subset)=0;
        if(nargout>1)
            g_components={g0,g1,g2};
        end                        
    end

end