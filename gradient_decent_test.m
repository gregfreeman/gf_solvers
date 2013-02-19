function [fs,ts,fx]=gradient_decent_test

     randn('state',1);
     n = 100; 
     m = 200;
     A = randn(m,n);
     
    options=struct();
    options.f=@f;
    options.df=@df;
    options.x_0=ones(n,1)*0.01;
    options.threshold=1e-2;
    options.exact_linesearch=1;
    options.plot_linesearch=0;
    options.save_linesearch=1;
    options.bt_linesearch=0;


    
    [fs,ts]=gradient_decent(options);

    fx=fs(end);
    
    figure(1), semilogy(fs-fx)
    ylabel 'Error from optimal' 
    xlabel 'Iteration'
    title 'Gradient Decent'
    figure(2), plot(ts)
    ylabel 'Line search coefficient t' 
    xlabel 'Iteration'
    title 'Gradient Decent'
    figure(3), plot(fs)
    ylabel 'f' 
    xlabel 'Iteration'


    function y=f(x)
        LL=log(1-A*x);
        LL(imag(LL)~=0)=-inf;
        y= 0 - sum(LL) -sum(log(1-x)) -sum(log(1+ x));
        y2= 0 - sum(log(1-A*x)) -sum(log(1-x.^2)) ;
        if imag(y)
            y=inf;
        end
        if imag(y2)
            y2=inf;
        end
        d=norm(y-y2);
        if(d>0.001)
            disp('Error')
        end
    end
    function y=df(x)
         y=zeros(n,1);
         for j2=1:n
             y(j2)= sum( A(:,j2)./ (1- A*x ) ) + 1./(1-x(j2)) - 1./(1+x(j2)) ;
         end
%          d=norm(-y + A'*(1./(1-A*x)) + 1./(1-x) - 1./(1+x))
    end
end