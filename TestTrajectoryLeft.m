clear all
clc

%Generate sample reference input

TestTrack = load('TestTrack.mat')
bl = TestTrack.TestTrack.bl;
br = TestTrack.TestTrack.br;
cline = TestTrack.TestTrack.cline;
thetaCline = TestTrack.TestTrack.theta;

% for i = 1 : length(thetaCline) - 1
%     deltaTheta(i) = thetaCline(i + 1) - thetaCline(i);
% end
% 
% interp_size = 100;
% cline = [interp(cline(1,:),interp_size) ; interp(cline(2,:),interp_size)];
% deltaTheta = [interp(deltaTheta(:),interp_size)];

% figure(1)
% plot(cline(1,:) , cline(2,:) , 'k')
% hold on
% plot(1:length(thetaCline) , thetaCline , 'r')
% plot(1:length(deltaTheta) , deltaTheta , 'b')

dt = 0.01;

%U = vector of constant steering angle, acceleration for given time

U_left = [
    repmat([0.01, 100],50,1);
    repmat([-0.01, 200],200,1)
    repmat([-0.0, 300],550,1);
    repmat([-0.01, 300],100,1);
    repmat([0.0, 400],550,1);
    repmat([-0.01, 300],500,1);
    repmat([-0.02, 200],50,1);
    repmat([-0.03, 200],100,1);
    repmat([-0.03, 300],400,1);
    repmat([-0.02, 200],300,1);
    repmat([-0.01, 200],400,1);
    repmat([0.0, 300],700,1);
    repmat([0.01, 200],50,1);
    repmat([0.02, 100],200,1);
    repmat([0.03, 0],150,1);
    repmat([0.02, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.0, 0],50,1);
    repmat([-0.01, 0],150,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.04, 0],50,1);
    repmat([-0.05, 0],50,1);
    repmat([-0.04, 0],50,1);
    repmat([-0.03, -100],50,1);
    repmat([-0.02, -200],50,1);
    repmat([-0.01, -300],50,1);
    repmat([0.0, -200],50,1);
    repmat([0.01, -100],50,1);
    repmat([0.02, -200],50,1);
    repmat([0.03, -300],50,1);
    repmat([0.04, -200],50,1);
    repmat([0.05, 0],200,1);
    repmat([0.04, 0],100,1);
    repmat([0.03, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.00, 0],50,1);
    repmat([-0.01, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.03, 0],100,1);
    repmat([-0.04, 0],50,1);
    repmat([-0.03, 0],300,1);
    repmat([-0.02, 0],600,1);
    repmat([-0.03, 0],350,1);
    repmat([-0.02, 0],150,1);
    repmat([-0.01, -100],350,1);
    repmat([0.0, 0],350,1);
    repmat([0.01, -100],50,1);
    repmat([0.02, -100],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.05, -100],50,1);
    repmat([0.06, -100],50,1);
    repmat([0.07, -100],50,1);
    repmat([0.08, -100],50,1);
    repmat([0.08, 0],50,1);
    repmat([0.07, 0],50,1);
    repmat([0.06, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.0, 0],350,1);
    repmat([0.01, 0],50,1);
    repmat([0.02, 0],200,1);
    repmat([0.01, 0],50,1);
    repmat([0.00, 0],150,1);
    repmat([-0.01, 0],50,1);
    repmat([-0.02, 0],150,1);
    repmat([-0.03, 100],200,1);
    repmat([-0.04, 0],50,1);
    repmat([-0.05, 0],50,1);
    repmat([-0.06, 100],50,1);
    repmat([-0.07, 0],200,1);
    repmat([-0.06, 0],450,1);
    repmat([-0.05, 0],50,1);
    repmat([-0.04, 0],50,1);
    repmat([-0.03, 0],100,1);
    repmat([-0.02, 0],100,1);
    repmat([-0.01, 0],50,1);
    repmat([0.0, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.06, 0],50,1);
    repmat([0.07, 0],50,1);
    repmat([0.08, 0],50,1);
    repmat([0.09, 0],200,1);
    repmat([0.08, 0],50,1);
    repmat([0.07, 0],50,1);
    repmat([0.06, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.01, 100],50,1);
    repmat([0.00, 200],500,1);
    repmat([-0.01, 300],300,1);
    repmat([0.00, 400],700,1);
    repmat([0.01, 400],100,1);
    repmat([0.02, 400],50,1);
    repmat([0.03, 400],300,1);
    repmat([0.02, 400],200,1);
    repmat([0.01, 500],50,1);
    repmat([0.0, 600],1350,1);
    repmat([0.01, 700],350,1);
]
   
time = size(U_left,1) * 0.01;

U_ref_left = U_left;
save('ROB535_ControlProject_part1_Team19','U_left')

% deltafSmooth = smooth(U(:,1));
% accSmooth = smooth(U(:,2)); 
% U_ref = [deltafSmooth , accSmooth];

x0 = [287 , 5 , -176 , 0 , 2 , 0];
endpoint = [1470 , 810];

[Y,T] = forwardIntegrateControlInput(U_ref_left,x0);
Y_ref = Y;

% x0 = Y(end,:);
% 
% while Y(1,end) < endpoint(1) && Y(3,end) < endpoint(2)
%     
%     [value , idx] = min(
%     deltaTheta = 
%     U = [U , [deltaTheta,0];
%     [Y,T] = forwardIntegrateControlInput(U,x0);
%     
%     x0 = Y(end,:);
% 
% end

figure(2)
plot(bl(1,:),bl(2,:),'k')
hold on
plot(br(1,:),br(2,:),'k')
plot(cline(1,:),cline(2,:),'--k')
plot(Y(:,1),Y(:,3),'r')
%xlim([1100 1600])
%ylim([400 900])
hold off

figure(3)
yyaxis left
plot(T,U_ref_left(:,1),'r')
ylabel('Steeting Angle')
hold on
yyaxis right
plot(T,U_ref_left(:,2),'r')
ylabel('Acceleration')
xlabel('Time')
hold off

