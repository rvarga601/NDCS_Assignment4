% Initialization
clear all
aircraft;

if ~exist('./optimal_centralized.mat', 'file')
    disp('Running centralized optimization first...')
    centralized;
end

% Loading the centralized solution for later comparison
load('optimal_centralized.mat')

% Folder for figures
if ~exist('./Figures', 'dir')
    mkdir('./Figures')
end

[n,m] = size(B1);      % Number of states and inputs

% Construct prediction models
predM1 = createPredictionModel(A1,B1,Tfinal);
predM2 = createPredictionModel(A2,B2,Tfinal);
predM3 = createPredictionModel(A3,B3,Tfinal);
predM4 = createPredictionModel(A4,B4,Tfinal);


%% Distributed solution with fixed step-sizes
steps = 10000;
mu = zeros(1,n*4*3);

thetav = zeros(n,4,steps);

alpha = [2e-1 2e-2 2e-3]*2;

dvalv = zeros(steps,length(alpha));
muv = zeros(4*3*n,steps,length(alpha));
xev = zeros(n,steps+1,length(alpha));

% Looping through the different fixed step-sizes
for j=1:length(alpha)
    
    mu = zeros(1,n*4*3);
    xev(1,1,j) = norm(-theta_opt)/norm(theta_opt);
    xev(1,2,j) = norm(-theta_opt)/norm(theta_opt);
    xev(1,3,j) = norm(-theta_opt)/norm(theta_opt);
    xev(1,4,j) = norm(-theta_opt)/norm(theta_opt);
    
    for i=1:steps

        temp = num2cell(reshape(mu,[n,12])',2);
        [mu12,mu13,mu14,mu21,mu23,mu24,mu31,mu32,mu34,mu41,mu42,mu43]=temp{:};

        mu1 = mu12+mu13+mu14-mu21-mu31-mu41;
        mu2 = mu21+mu23+mu24-mu12-mu32-mu42;
        mu3 = mu31+mu32+mu34-mu13-mu23-mu43;
        mu4 = mu41+mu42+mu43-mu14-mu24-mu34;
        [theta1,u1] = computeDistributedMin(predM1,A1,x01,mu1,n,m,Tfinal,umax);
        [theta2,u2] = computeDistributedMin(predM2,A2,x02,mu2,n,m,Tfinal,umax);
        [theta3,u3] = computeDistributedMin(predM3,A3,x03,mu3,n,m,Tfinal,umax);
        [theta4,u4] = computeDistributedMin(predM4,A4,x04,mu4,n,m,Tfinal,umax);

        states_all = blkdiag(predM1.u,predM2.u,predM3.u,predM4.u)*[u1;u2;u3;u4]+...
                blkdiag(predM1.x0,predM2.x0,predM3.x0,predM4.x0)*[x01;x02;x03;x04];


        theta_star = [theta1-theta2; theta1-theta3; theta1-theta4;
                      theta2-theta1; theta2-theta3; theta2-theta4;
                      theta3-theta1; theta3-theta2; theta3-theta4;
                      theta4-theta1; theta4-theta2; theta4-theta3];

        mu = [mu12 mu13 mu14 mu21 mu23 mu24 mu31 mu32 mu34 mu41 mu42 mu43];

        dval = states_all'*states_all+[u1;u2;u3;u4]'*[u1;u2;u3;u4]+...
            mu*theta_star;

        dvalv(i,j) = dval;
        muv(:,i,j) = mu';
        
        xev(1,i+1,j) = norm(theta1-theta_opt)/norm(theta_opt);
        xev(2,i+1,j) = norm(theta2-theta_opt)/norm(theta_opt);
        xev(3,i+1,j) = norm(theta3-theta_opt)/norm(theta_opt);
        xev(4,i+1,j) = norm(theta4-theta_opt)/norm(theta_opt);

        thetav(:,:,i) = [theta1,theta2,theta3,theta4];

        mu = mu + alpha(j)*theta_star';
    end
end


%% Plot results

ylabels = {'$\frac{||\theta_1^*-\theta_{\mathrm{opt}}||}{||\theta_{\mathrm{opt}}||}$',...
    '$\frac{||\theta_2^*-\theta_{\mathrm{opt}}||}{||\theta_{\mathrm{opt}}||}$',...
    '$\frac{||\theta_3^*-\theta_{\mathrm{opt}}||}{||\theta_{\mathrm{opt}}||}$',...
    '$\frac{||\theta_4^*-\theta_{\mathrm{opt}}||}{||\theta_{\mathrm{opt}}||}$'};

titles = {'1. aircraft','2. aircraft','3. aircraft','4. aircraft'};

figure(1)
for i=1:4
    subplot(2,2,i)
    semilogy(0:steps,xev(i,:,1)','LineWidth',1.5);
    hold on
    semilogy(0:steps,xev(i,:,2)','LineWidth',1.5);
    semilogy(0:steps,xev(i,:,3)','LineWidth',1.5);
    hold off
    grid on
    ylabel(ylabels(i),'Interpreter','latex','FontSize',18)
    xlabel('Optimization step')
    title(titles(i))
    if i==1
       legend({'$\alpha=0.4$',...
           '$\alpha=0.04$',...
           '$\alpha=0.004$'},'Interpreter','latex') 
    end
    if i==4
        axis([0 steps 1e-6 1e-3])
    else
        % axis([0 steps 1e-5 1e-4])
        axis([0 steps 1e-6 1e-3])
    end
end

% Save figure
saveas(gcf,'Figures\fixed_stepsize_error','epsc')

%% Distributed solution with different update sequences
steps = 5000;
mu = zeros(1,n*4*3);

thetav = zeros(n,4,steps);

dvalv = zeros(steps,3);
muv = zeros(4*3*n,steps,3);
xev = zeros(n,steps+1,3);
alpha0 = 2e0;

alpha_pow = [0.5 1 2];

for j=1:3
    mu = zeros(1,n*4*3);
    xev(1,1,j) = norm(-theta_opt)/norm(theta_opt);
    xev(1,2,j) = norm(-theta_opt)/norm(theta_opt);
    xev(1,3,j) = norm(-theta_opt)/norm(theta_opt);
    xev(1,4,j) = norm(-theta_opt)/norm(theta_opt);
    for i=1:steps

        temp = num2cell(reshape(mu,[n,12])',2);
        [mu12,mu13,mu14,mu21,mu23,mu24,mu31,mu32,mu34,mu41,mu42,mu43]=temp{:};

        mu1 = mu12+mu13+mu14-mu21-mu31-mu41;
        mu2 = mu21+mu23+mu24-mu12-mu32-mu42;
        mu3 = mu31+mu32+mu34-mu13-mu23-mu43;
        mu4 = mu41+mu42+mu43-mu14-mu24-mu34;
        [theta1,u1] = computeDistributedMin(predM1,A1,x01,mu1,n,m,Tfinal,umax);
        [theta2,u2] = computeDistributedMin(predM2,A2,x02,mu2,n,m,Tfinal,umax);
        [theta3,u3] = computeDistributedMin(predM3,A3,x03,mu3,n,m,Tfinal,umax);
        [theta4,u4] = computeDistributedMin(predM4,A4,x04,mu4,n,m,Tfinal,umax);

        states_all = blkdiag(predM1.u,predM2.u,predM3.u,predM4.u)*[u1;u2;u3;u4]+...
                blkdiag(predM1.x0,predM2.x0,predM3.x0,predM4.x0)*[x01;x02;x03;x04];


        theta_star = [theta1-theta2; theta1-theta3; theta1-theta4;
                      theta2-theta1; theta2-theta3; theta2-theta4;
                      theta3-theta1; theta3-theta2; theta3-theta4;
                      theta4-theta1; theta4-theta2; theta4-theta3];

        mu = [mu12 mu13 mu14 mu21 mu23 mu24 mu31 mu32 mu34 mu41 mu42 mu43];

        dval = states_all'*states_all+[u1;u2;u3;u4]'*[u1;u2;u3;u4]+...
            mu*theta_star;

        dvalv(i,j) = dval;
        muv(:,i,j) = mu';
   
        xev(1,i+1,j) = norm(theta1-theta_opt)/norm(theta_opt);
        xev(2,i+1,j) = norm(theta2-theta_opt)/norm(theta_opt);
        xev(3,i+1,j) = norm(theta3-theta_opt)/norm(theta_opt);
        xev(4,i+1,j) = norm(theta4-theta_opt)/norm(theta_opt);

        thetav(:,:,i) = [theta1,theta2,theta3,theta4];

        alpha = alpha0/i^alpha_pow(j);
        mu = mu + alpha*theta_star';
    end
end

%% Plot results

ylabels = {'$\frac{||\theta_1^*-\theta_{\mathrm{opt}}||}{||\theta_{\mathrm{opt}}||}$',...
    '$\frac{||\theta_2^*-\theta_{\mathrm{opt}}||}{||\theta_{\mathrm{opt}}||}$',...
    '$\frac{||\theta_3^*-\theta_{\mathrm{opt}}||}{||\theta_{\mathrm{opt}}||}$',...
    '$\frac{||\theta_4^*-\theta_{\mathrm{opt}}||}{||\theta_{\mathrm{opt}}||}$'};

titles = {'1. aircraft','2. aircraft','3. aircraft','4. aircraft'};

figure(2)
for i=1:4
    subplot(2,2,i)
    semilogy(0:steps,xev(i,:,1)','LineWidth',1.5);
    hold on
    semilogy(0:steps,xev(i,:,2)','LineWidth',1.5);
    semilogy(0:steps,xev(i,:,3)','LineWidth',1.5);
    hold off
    grid on
    ylabel(ylabels(i),'Interpreter','latex','FontSize',18)
    xlabel('Optimization step')
    title(titles(i))
    if i==1
       legend({'$\alpha=\alpha_0 /\sqrt{i}$',...
           '$\alpha=\alpha_0 /i$',...
           '$\alpha=\alpha_0 /i^2$'},'Interpreter','latex') 
    end
    axis([0 steps 1e-5 1e-0])
end

% Save figure
saveas(gcf,'Figures\sequences_error','epsc')



%% Functions

function predM = createPredictionModel(A,B,Tfinal)

[n,m] = size(B);      % Number of states and inputs

% Initialize the prediction matricies
predM.x0 = zeros(n*(Tfinal+1),n);
predM.u = zeros(n*(Tfinal+1),m*Tfinal);

% Constructing the prediction matrix for x0
offset = 0;
for i=1:Tfinal+1
    predM.x0(offset+1:offset+n,:) = A^(i-1);
    offset = offset + n;
end

% Constructing the prediction matrix for u
offset = n;
temp = [];
for i=1:Tfinal
    temp = [A^(i-1)*B, temp];
    predM.u(offset+1:offset+n,1:i*m) = temp;
    offset = offset + n;
end

predM.xT = predM.u(end-n+1:end,:);

end

function [xT,u] = computeDistributedMin(predM,A,x0,mu,n,m,Tfinal,umax)

Hx = predM.u'*predM.u;
Hu = eye(m*Tfinal);
H = blkdiag(2*(Hx+Hu),zeros(n));
f = [2*x0'*predM.x0'*predM.u, mu];

Aeq = [predM.xT, -eye(n)];
beq = -A^(Tfinal)*x0;

ub = [ones(m*Tfinal,1)*umax/Tfinal; ones(n,1)*Inf];
lb = -ub;

options = optimoptions('quadprog','Display','off');

% [theta,fval,exitflag,output] = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);
[theta,~,~,~] = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);

xT = theta(end-n+1:end);
u = theta(1:end-n);

end
