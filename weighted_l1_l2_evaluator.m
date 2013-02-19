%obj=weighted_l1_l2_evaluator(Phi,fine_samples,vfine)
% creates a function evaluator for itertively reweighting the LASSO l1-l2 
%  elastic-net regularizer
%  
%    obj.f=@f;
%    obj.df=@df;
%    obj.setLambda=@setLambda;
%      setLambda(lambda1,lambda2)
%          lambda1(Nx1)
%          lambda2(Nx1)

function obj=weighted_l1_l2_evaluator(Phi,fine_samples,vfine)

    obj.f=@f;
    obj.df=@df;
    obj.setLambda=@setLambda;
    lambda1=zeros(size(vfine));
    lambda2=zeros(size(vfine));

    function setLambda(x1,x2)
        lambda1=x1;
        lambda2=sqrt(x2);
    end

    function [fx,f_components]=f(x)
        x1=x(vfine);
        f0=norm(fine_samples-Phi*x1,2).^2;
        f1=norm(x.*lambda1,1);
        f2=norm(x.*lambda2,2);
        fx=f0+f1+f2;
        if(nargout>1)
            f_components={f0,f1,f2};
        end                        
    end
    function [g,g_components]=df(x)
        x1=x(vfine);
        g0=zeros(size(x));
        g0(vfine)=2.*(Phi'*(Phi*x1-fine_samples));
        g1=sign(x).*lambda1;
        g1(abs(x) < 1e-5)=0; % truncate small x;
        % f2= (sum (lambda2_i*x_i)^2 )^(1/2) 
        % g2_i(lambda2_i*x_i)*(sum (lambda2_i*x_i)^2 )^(-1/2)   
        %g2=lambda2.*x./max(norm(x,2),1e-6);
        g2=x.*lambda2;
        g2=g2.*lambda2./max(norm(g2,2),1e-6);
        g=g0+g1+g2;
        g(~vfine)=0;
        if(nargout>1)
            g_components={g0,g1,g2};
        end                        
    end

end