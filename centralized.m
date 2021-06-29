clear all
aircraft;

[n,m] = size(B1);      % Number of states and inputs

% Construct prediction models
predM1 = createPredictionModel(A1,B1,Tfinal);
predM2 = createPredictionModel(A2,B2,Tfinal);
predM3 = createPredictionModel(A3,B3,Tfinal);
predM4 = createPredictionModel(A4,B4,Tfinal);

% Matrix construction for quadratic terms
Hx = blkdiag(predM1.u,predM2.u,predM3.u,predM4.u)'*...
                blkdiag(predM1.u,predM2.u,predM3.u,predM4.u);
Hu = eye(4*m*Tfinal);
H = blkdiag(2*(Hx+Hu),zeros(n));

% Vector corresponding to the linear term
f = 2*[x01'*predM1.x0'*predM1.u,...
    x02'*predM2.x0'*predM2.u,...
    x03'*predM3.x0'*predM3.u,...
    x04'*predM4.x0'*predM4.u,...
    zeros(1,n)];

% Equality constraint elements
Aeq = [blkdiag(predM1.xT,predM2.xT,predM3.xT,predM4.xT),...
    -kron(ones(4,1),eye(n))];
beq = -[A1^(Tfinal)*x01;...
       A2^(Tfinal)*x02;...
       A3^(Tfinal)*x03;...
       A4^(Tfinal)*x04];

% Setting limit for the actuator saturation
ub = [ones(4*m*Tfinal,1)*umax/Tfinal; ones(n,1)*Inf];
lb = -ub;

% Solution of the problem
options = optimoptions('quadprog','Display','off');
[theta,fval,exitflag,output] = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);

% Computing optimal states
states_all = blkdiag(predM1.u,predM2.u,predM3.u,predM4.u)*theta(1:4*m*Tfinal)+...
    blkdiag(predM1.x0,predM2.x0,predM3.x0,predM4.x0)*[x01;x02;x03;x04];

% Reshape of the state vector
states = zeros(n,Tfinal+1,4);
offset_n = 0;
for i=1:4
    states(:,:,i) = reshape(states_all(offset_n+1:offset_n+(Tfinal+1)*n),...
                [n,(Tfinal+1)]);
    offset_n = offset_n + n*(Tfinal+1);
end

% Calculate optimal objective function value
dval_opt = states_all'*states_all+theta(1:4*m*Tfinal)'*theta(1:4*m*Tfinal);

% Calculate optimal meeting point
theta_opt = theta(end-n+1:end);

states_opt = states;

% Save results to compare with other algorithms
save('optimal_centralized.mat','dval_opt','theta_opt','states_opt');

%% Plot results

ylabels = {'$x$ state','$y$ state','$\dot{x}$ state','$\dot{y}$ state'};

figure()
for i=1:4
    subplot(2,2,i)
    plot(0:Tfinal, reshape(states(i,:,:),[Tfinal+1,4]),'LineWidth',1.5);
    grid on
    ylabel(ylabels(i),'Interpreter','latex')
    xlabel('Time step')
    xticks(0:Tfinal)
    axis([0 Tfinal -Inf Inf])
    if i==1
        legend('sys1','sys2','sys3','sys4')
    end
end

% Folder for figures
if ~exist('./Figures', 'dir')
    mkdir('./Figures')
end

% Save figure
saveas(gcf,'Figures\centralized_states','epsc')


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







