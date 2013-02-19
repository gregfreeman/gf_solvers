function [fs,ts,xs]=gradient_decent(options)


    
    if ~isfield(options,'max_iter')
        options.max_iter=1000;
    end   
    max_iter=options.max_iter;
    
    if ~isfield(options,'alpha')
        options.alpha = 0.01;  
    end    
    alpha=options.alpha;
    
    if ~isfield(options,'beta')
        options.beta = 0.5;
    end
    beta=options.beta;
    
    if ~isfield(options,'threshold')
        options.threshold=1e-4;
    end
    threshold=options.threshold;

    if ~isfield(options,'save_linesearch')
        options.save_linesearch=0;
    end
    if ~isfield(options,'save_bt_linesearch')
        options.save_bt_linesearch=0;
    end
    if ~isfield(options,'plot_linesearch')
        options.plot_linesearch=0;
    end
    if ~isfield(options,'plot_bt_linesearch')
        options.plot_bt_linesearch=0;
    end
    if ~isfield(options,'exact_linesearch')
        options.exact_linesearch=0;
    end
    if ~isfield(options,'bt_linesearch')
        options.bt_linesearch=1;
    end

    if ~isfield(options,'save_gradient_components')
        options.save_gradient_components=0;
    end
    if ~isfield(options,'output_prefix')
        options.output_prefix='';
    end
    if ~isfield(options,'override_convexity_check')
        options.override_convexity_check=0;
    end
    if ~isfield(options,'constant_step')
        options.constant_step=0;
    else
        options.bt_linesearch=0;
        options.exact_linesearch=0;
    end    
    if ~isfield(options,'x_0') 
        error('must define x_0 in options');
    end
    
    if ~isfield(options,'f') 
        error('must define f in options');
    end
    
    if ~isfield(options,'df') 
        error('must define df in options');
    end
    
    x=options.x_0;
    f=options.f;
    df=options.df;

    n=numel(x);
    
    % gradient decent
    % line search
    fx=f(x);
    if imag(fx) || isinf(fx)
        error('x not in domain')
    end
    notDone=true;

    ts=zeros(max_iter,1);
    fs=zeros(max_iter,1);
    xs=zeros(max_iter,n);
    j=1;
    ts(j)=0;
    fs(j)=fx;
    xs(j,:)=x;
    t=1;
    
    bt_iter_max=30;
    while notDone
        disp('*** Computing Gradient ****')        
        if options.save_gradient_components==1
            [g,g_components]=df(x);            
            fname=sprintf('%sgradient_comp_%03d',options.output_prefix,j);
            save(fname,'g','g_components','options');
        else
            g=df(x);
        end
        dx=-g;
        j=j+1;
        if options.bt_linesearch
           
            t=t*2;
            disp('|-----------|------------------|--------|--|----|')
            disp('| f(x+t*dx) |fx + alpha*t*gt*dx|    t   | k|  j |')
            disp('|-----------|------------------|--------|--|----|')
            
            f_btls=zeros(bt_iter_max,1); % f for backtracking linesearch
            t_btls=zeros(bt_iter_max,1); % t for backtracking linesearch
            f_btls(1)=fx;
            t_btls(1)=0;
            fx2=f(x+t*dx);
            fx3=fx + alpha*t*g'*dx;
            f_btls(2)=fx2;
            t_btls(2)=t;
            f3_btls=fx3;
            t3_btls=t;
            fx2_last=fx2;
            t_last=t;

            if(g'*dx>0)
                error('gradient issue?')
            end

            k=2;
            
            % backtracking line search
            str=sprintf('| %10.4f|        %10.4f|%8.2e|%2d|%4d|',fx2,  fx3, t, k ,j);
            

            % increase t until btls condition fails
            while (fx2 < fx3 && ~isnan(fx2) && ~isinf(fx2) ) && k<bt_iter_max
                t=t*2;
                fx2=f(x+t*dx);
                fx3=fx + alpha*t*g'*dx;
                str=sprintf('| %10.4f|        %10.4f|%8.2e|%2d|%4d|',fx2,  fx3, t, k ,j);
                disp(str)
            end
            
            disp(str)
            k=2;
            while (fx2 > fx3 || isnan(fx2) || isinf(fx2) ) && k<bt_iter_max
                k=k+1;
                fx2_last=fx2;
                t_last=t;
                t=t*beta;      
                fx2=f(x+t*dx);
                fx3=fx + alpha*t*g'*dx;
                f_btls(k)=fx2;
                t_btls(k)=t;
                str=sprintf('| %10.4f|        %10.4f|%8.2e|%2d|%4d|',fx2,  fx3, t, k ,j);
                disp(str)
            end
            if options.plot_bt_linesearch || options.save_bt_linesearch
                t_btls=t_btls(1:k);
                f_btls=f_btls(1:k);
                [t_btls,idx]=sort(t_btls);
                f_btls=f_btls(idx);                
            end
            if options.plot_bt_linesearch
                figure(options.plot_bt_linesearch)
                plot(t_btls,f_btls,'b',[0 t3_btls],[fx,f3_btls],'g');
                %compare linear bound with function evaluation.
            end     
            if options.save_bt_linesearch
                fname=sprintf('%sbt_line_search_%03d',options.output_prefix,j);
                save(fname,'t_btls','f_btls','t3_btls','f3_btls','fx','x','dx');
            end   
            if k==bt_iter_max
                % convexity violated or at minimum
                if -g'*dx > threshold 
                   warning('convexity violated')
                else
%                     t=0;
                end

            end
        elseif options.exact_linesearch
            % initialize state for exact line search only
            fx2=f(x+t*dx);
            t_last=t*2;
            fx2_last=f(x+t_last*dx);
            % make sure fx2 is in the domain
            while  isinf(fx2)
                t_last=t;
                fx2_last=fx2;
                t=t*beta;
                fx2=f(x+t*dx);
            end
        end
        % exact line search
        if options.exact_linesearch
            [fx,t]=line_search(options,x,dx,fx,t,fx2,t_last,fx2_last,j);
            str=sprintf('| %10.4f|        %10.4f|%8.2e|LS|%4d|',fx,  0, t, j);
            disp(str)
        end
        
        if options.constant_step
            t=options.constant_step;
            
            disp('|-----------|------------------|--------|--|----|')
            disp('| f(x)      |                  |    t   | k|  j |')      
            str=sprintf('| %10.4f|        %10.4f|%8.2e|  |%4d|',fx,  0, t, j);
            disp(str)
        end
        
        x=x+t*dx;
        fx=f(x);
        if -g'*dx < threshold || j==max_iter
            notDone=false;
        end
        ts(j)=t;
        fs(j)=fx;
        xs(j,:)=x;
    end
    ts=ts(1:j);
    fs=fs(1:j);
    xs=xs(1:j,:);

end


function [f,t]=line_search(options,x,dx,fx,t2,fx2,t3,fx3,j)
% alternating bounded line search
% x current x
% dx step direction
% fx = f(x)
% fx,t = low bound of minimum  fx=f(x+t*dx)
% fx2,t2 = closest to minimum  fx2=f(x+t2*dx)
% fx3,t3 = highest bound of minimum fx3=f(x+t3*dx)
% iteration number
t=0;
f=options.f;
% find high bound of minimum since backtracking does not guarantee
k=1;
while(fx3<=fx2 && k<10)
    k=k+1;
    fx=fx2;
    fx2=fx3;
    t=t2;
    t2=t3;
    t3=t3*2;
    fx3=f(x+t3*dx);
end
if k==10 
    if options.override_convexity_check~=1
        save line_search_error fx3 fx2 fx t2 t3
        error('convexity is not strong enough, t=%f',t)
    else
        warning('convexity weak, t=%f',t)
    end  
end

if options.plot_linesearch || options.save_linesearch %save debug info
    f_ls=zeros(15,1);
    t_ls=zeros(15,1);
    f_ls(1)=fx;
    f_ls(2)=fx2;
    f_ls(3)=fx3;
    t_ls(1)=t;
    t_ls(2)=t2;
    t_ls(3)=t3;
end
alternate=false;
k=3;
while(k<15)
    k=k+1;
    alternate=~alternate;
    if alternate  % test  [t,t2]
        tn=(t+t2)/2;
        fxn=f(x+tn*dx);
        if fxn<fx2 % better
            fx3=fx2;
            t3=t2;
            fx2=fxn;
            t2=tn;
        else % shrink bound
            fx=fxn;
            t=tn;
        end
    else  % test [t2,t3]
        tn=(t3+t2)/2;
        fxn=f(x+tn*dx);
        if fxn<fx2 % better
            fx=fx2;
            t=t2;
            fx2=fxn;
            t2=tn;
        else % shrink bound
            fx3=fxn;
            t3=tn;
        end
    end
    if options.plot_linesearch || options.save_linesearch %save debug info
        f_ls(k)=fxn;
        t_ls(k)=tn;
    end
end
if options.plot_linesearch || options.save_linesearch %sort debug info
    [t_ls,idx]=sort(t_ls);
    f_ls=f_ls(idx);
end
if options.plot_linesearch %plot debug info
    figure(options.plot_linesearch)
    plot(t_ls(1:k),f_ls(1:k),'-x')
end
if options.save_linesearch %save debug info
    [t_ls,idx]=sort(t_ls);
    f_ls=f_ls(idx);   
    fname=sprintf('%sexact_line_search_%03d',options.output_prefix,j);
    save(fname,'t_ls','f_ls','x','dx');
end
% check that not already optimal (t=0)
if(fx2<fx)
    f=fx2;
    t=t2;
else
    warning('already optimal t=%f',t);
    f=fx;
%     t=t;
end

end
