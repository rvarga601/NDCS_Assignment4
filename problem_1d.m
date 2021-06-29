clear all
aircraft;
load('optimal_centralized.mat')
load('distributed.mat')

% Given consensus matrix
W = [0.75 0.25 0 0; 0.25 0.5 0.25 0; 0 0.25 0.5 0.25; 0 0 0.25 0.75];

% Set of consensus iteration numbers
phi = [0, 1, 10];

[n,m] = size(B1);      % Number of states and inputs

% Construct prediction models
predM1 = createPredictionModel(A1,B1,Tfinal);
predM2 = createPredictionModel(A2,B2,Tfinal);
predM3 = createPredictionModel(A3,B3,Tfinal);
predM4 = createPredictionModel(A4,B4,Tfinal);

% Optimization matrices
Hx1 = (predM1.u)'*predM1.u;
Hx2 = (predM2.u)'*predM2.u;
Hx3 = (predM3.u)'*predM3.u;
Hx4 = (predM4.u)'*predM4.u;
Hu = eye(m*Tfinal);
H1 = [2*(Hx1+Hu), predM1.xT'; predM1.xT, zeros(n)];
H2 = [2*(Hx2+Hu), predM2.xT'; predM2.xT, zeros(n)];
H3 = [2*(Hx3+Hu), predM3.xT'; predM3.xT, zeros(n)];
H4 = [2*(Hx4+Hu), predM4.xT'; predM4.xT, zeros(n)];

ub = [ones(m*Tfinal,1)*umax/Tfinal; Inf*ones(n,1)];
lb = -ub;

steps = 100;

alpha0 = 5e-1;

thetav = zeros(n,4);

xev = zeros(n,steps+1,3);

options = optimoptions('quadprog','Display','off');

for i=1:3
    
    xev(1,1,i) = norm(-theta_opt)/norm(theta_opt);
    xev(2,1,i) = norm(-theta_opt)/norm(theta_opt);
    xev(3,1,i) = norm(-theta_opt)/norm(theta_opt);
    xev(4,1,i) = norm(-theta_opt)/norm(theta_opt);
    
    theta1 = zeros(n,1);
    theta2 = zeros(n,1);
    theta3 = zeros(n,1);
    theta4 = zeros(n,1);
    
    % xe1v(1,i) = norm(theta1-theta_opt)/norm(theta_opt);
    W_phi = W^phi(i);

    for k=1:steps

        f1 = [2*x01'*predM1.x0'*predM1.u, (A1^(Tfinal)*x01-theta1)'];
        f2 = [2*x02'*predM2.x0'*predM2.u, (A2^(Tfinal)*x02-theta2)'];
        f3 = [2*x03'*predM3.x0'*predM3.u, (A3^(Tfinal)*x03-theta3)'];
        f4 = [2*x04'*predM4.x0'*predM4.u, (A4^(Tfinal)*x04-theta4)'];

        [y1,~,~,~] = quadprog(H1,f1,[],[],[],[],lb,ub,[],options);
        [y2,~,~,~] = quadprog(H2,f2,[],[],[],[],lb,ub,[],options);
        [y3,~,~,~] = quadprog(H3,f3,[],[],[],[],lb,ub,[],options);
        [y4,~,~,~] = quadprog(H4,f4,[],[],[],[],lb,ub,[],options);

        mu1 = y1(end-n+1:end);
        mu2 = y2(end-n+1:end);
        mu3 = y3(end-n+1:end);
        mu4 = y4(end-n+1:end);
        
        alpha = alpha0/k;

        theta1 = theta1 + alpha*mu1;
        theta2 = theta2 + alpha*mu2;
        theta3 = theta3 + alpha*mu3;
        theta4 = theta4 + alpha*mu4;

        for j=1:4
            thetav(:,j) = W_phi(j,1)*theta1+W_phi(j,2)*theta2+...
                            W_phi(j,3)*theta3+W_phi(j,4)*theta4;    
        end

        theta1 = thetav(:,1);
        theta2 = thetav(:,2);
        theta3 = thetav(:,3);
        theta4 = thetav(:,4);

        % xe1v(k+1,i) = norm(theta1-theta_opt)/norm(theta_opt);
        xev(1,k+1,i) = norm(theta1-theta_opt)/norm(theta_opt);
        xev(2,k+1,i) = norm(theta2-theta_opt)/norm(theta_opt);
        xev(3,k+1,i) = norm(theta3-theta_opt)/norm(theta_opt);
        xev(4,k+1,i) = norm(theta4-theta_opt)/norm(theta_opt);

    end

end

u1 = y1(1:end-n);
u2 = y2(1:end-n);
u3 = y3(1:end-n);
u4 = y4(1:end-n);

% See results
states_all = blkdiag(predM1.u,predM2.u,predM3.u,predM4.u)*[u1;u2;u3;u4]+...
    blkdiag(predM1.x0,predM2.x0,predM3.x0,predM4.x0)*[x01;x02;x03;x04];

states = zeros(n,Tfinal+1,4);

offset_n = 0;
for i=1:4
    states(:,:,i) = reshape(states_all(offset_n+1:offset_n+(Tfinal+1)*n),[n,(Tfinal+1)]);
    offset_n = offset_n + n*(Tfinal+1);
end

dval = states_all'*states_all+[u1;u2;u3;u4]'*[u1;u2;u3;u4];

%% Plot results

ylabels = {'$\frac{||\theta_1^*-\theta_{\mathrm{opt}}||}{||\theta_{\mathrm{opt}}||}$',...
    '$\frac{||\theta_2^*-\theta_{\mathrm{opt}}||}{||\theta_{\mathrm{opt}}||}$',...
    '$\frac{||\theta_3^*-\theta_{\mathrm{opt}}||}{||\theta_{\mathrm{opt}}||}$',...
    '$\frac{||\theta_4^*-\theta_{\mathrm{opt}}||}{||\theta_{\mathrm{opt}}||}$'};

titles = {'1. aircraft','2. aircraft','3. aircraft','4. aircraft'};

figure(2)
for i=1:4
    subplot(2,2,i)
    semilogy(0:steps,reshape(xev(i,:,1),[1 steps+1]),'-x');
    hold on
    semilogy(0:steps,reshape(xev(i,:,2),[1 steps+1]),'-x');
    semilogy(0:steps,reshape(xev(i,:,3),[1 steps+1]),'-x');
    semilogy(0:steps,reshape(xev_dist(i,:),[1 steps+1]),'-x');
    hold off
    grid on
    ylabel(ylabels(i),'Interpreter','latex','FontSize',18)
    xlabel('Optimization step')
    title(titles(i))
    if i==1
        legend({'$\varphi=0$','$\varphi=1$','$\varphi=10$','subg.'},'Interpreter','latex')
    end
    axis([0 steps -Inf Inf])
end

% Save figure
saveas(gcf,'Figures\consensus_error','epsc')

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