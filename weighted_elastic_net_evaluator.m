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

function obj=weighted_elastic_net_evaluator(A,b,lambda1,lambda2)

    obj.f=@f;
    obj.df=@df;
%     lambda2=sqrt(lambda2);
    
    function [fx,f_components]=f(x)
       
        f0=norm(b-A*x,2).^2;
        f1=norm(x.*lambda1,1);
        f2=norm(x.*lambda2,2);
        fx=f0+f1+f2;
        if(nargout>1)
            f_components={f0,f1,f2};
        end                        
    end
    function [g,g_components]=df(x)
        g0=2.*(A'*(A*x-b));
        g1=sign(x).*lambda1;
        g1(abs(x) < 1e-5)=0; % truncate small x;
        g2=x.*lambda2;
        g2=g2.*lambda2./max(norm(g2,2),1e-6);
        g=g0+g1+g2;        
        if(nargout>1)
            g_components={g0,g1,g2};
        end                        
    end

end