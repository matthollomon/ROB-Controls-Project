clear
curr_state = [287 , 5 , -176 , 0 , 2 , 0];
load('TestTrack.mat')
Xobs = generateRandomObstacles(10, TestTrack);
% Xobs{1} = 1000 * [   1.3922    0.7086;
%     1.3976    0.7042;
%     1.3998    0.7070;
%     1.3945    0.7113];
Ycheck = [];

Umanual = [repmat([-0.04, 2400],100,1);
    repmat([0, 2400],200,1);
    repmat([0.0, 2400],30,1);
    repmat([0.0, 2400],100,1);
    repmat([0.0, 2400],180,1);
    repmat([-0.0, 1800],130,1);
    repmat([-0.03,0],350,1);
    repmat([+0.05, 0],100,1);
    repmat([0.025, 0],100,1);
    repmat([0, 0],100,1);
    repmat([0, 0],100,1);
    repmat([0, 0],100,1);
    repmat([0, 0],100,1);
    repmat([0, 0],100,1);
    repmat([0, 0],100,1);
    repmat([0, 0],100,1);
    repmat([0.03, 0],100,1);
    repmat([0.03, 0],100,1);
    repmat([0.01, 0],100,1);
    repmat([0, 0],100,1);
    repmat([-0.02, 0],100,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.04, 0],50,1);
    repmat([-0.05, 0],50,1);
    repmat([-0.04, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0, 0],50,1);
    repmat([0, 0],50,1);
    repmat([-0.01, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.04, 0],50,1);
    repmat([-0.05, 0],50,1);
    repmat([-0.04, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([0.0, 0],50,1);
    repmat([-0.01, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.04, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.01, 0],50,1);
    repmat([-0.01, 0],50,1);
    repmat([0.00, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.0, 0],50,1);
    repmat([0.0, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.06, 0],50,1);
    repmat([0.06, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0 , 0],50,1);
    repmat([0 , 0],50,1);
    repmat([0 , 0],50,1);
    repmat([0 , 0],50,1);
    repmat([0 , 0],50,1);
    repmat([0 , 0],50,1);
    repmat([0 , 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.04 , 0],50,1);
    repmat([-0.06 , 0],50,1);
    repmat([-0.04 , 0],50,1);
    repmat([-0.06 , 0],50,1);
    repmat([-0.06 , 0],50,1);
    repmat([-0.04 , 0],50,1);
    repmat([-0.04 , 0],50,1);
    repmat([-0.04 , 0],50,1);
    repmat([-0.04 , 0],50,1);
    repmat([-0.04 , 0],50,1);
    repmat([-0.05 , 0],50,1);
    repmat([-0.05 , 0],50,1);
    repmat([-0.03 , 0],50,1);
    repmat([-0.01 , 0],50,1);
    repmat([0.0 , 0],50,1);
    repmat([0.01 , 0],50,1);
    repmat([0.02 , 0],50,1);
    repmat([0.1 , 0],50,1);
    repmat([0.1 , 0],50,1);
    repmat([0.20 , 0],50,1);
    repmat([0.05 , 0],50,1);
    repmat([0.03 , 0],50,1);
    repmat([0.02 , 0],50,1);
    repmat([0.01 , 0],50,1);
    repmat([0.0 , 0],50,1);
    repmat([0.0 , 0],50,1);
    repmat([0.0 , 0],50,1);
    repmat([0.0 , 0],50,1);
    repmat([0.0 , 0],50,1);
    repmat([0.0 , 0],50,1);
    repmat([0 , 0],50,1);
    repmat([0.0 , 0],50,1);
    repmat([0.0 , 0],100,1);
    repmat([0.0 , 0],100,1);
    repmat([0.0 , 0],100,1);
    repmat([0.0 , 0],100,1);
    repmat([0.0 , 0],100,1);
    repmat([0.0 , 0],100,1);
    repmat([0.02 , 0],100,1);
    repmat([0.03 , 0],100,1);
    repmat([0.03 , 0],100,1);
    repmat([0.02 , 0],100,1);
    repmat([0.02, 0],100,1);
    repmat([0.01 , 0],100,1);
    repmat([0.01 , 0],100,1);
    repmat([0.012 , 0],200,1);
    repmat([0 , 5000],850,1);
    
    ];
% x0 = curr_state;
% [Ymanual,Tmanual] = forwardIntegrateControlInput(Umanual,curr_state);

for i = 1:10
    curr_pos = [curr_state(1), curr_state(3)];
    Xobs_seen = senseObstacles(curr_pos, Xobs);

    [sol_2, FLAG_terminate] = ROB535_ControlProject_part2_Team19(TestTrack, Xobs_seen, curr_state);
    disp("Forward integrate")
    i
    [Y,T] = forwardIntegrateControlInput(sol_2, curr_state);
    disp("forward done")

    curr_state = Y(end, :);

    Ycheck = [Ycheck; Y];
    
end
bl = TestTrack.bl;
br = TestTrack.br;
cline = TestTrack.cline;
thetaCline = TestTrack.theta;

figure(1)
%plot(Y(1,1:testLength),Y(3,1:testLength),'c')
plot(Ycheck(:,1),Ycheck(:,3),'r')
hold on
% plot(Ymanual(:,1),Ymanual(:,3),'b')
plot(cline(1,:),cline(2), '--k')
plot(bl(1,:),bl(2,:),'k')
plot(br(1,:),br(2,:),'k')
plot(Xobs{1}(:,1), Xobs{1}(:,2), 'o')
%plot(cline(1,:),cline(2,:),'--k')
legend('MPC trajectory','Manual Trajectory')
% xlim([200 400])
% ylim([-200 200])
hold off