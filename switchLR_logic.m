curr_pos = [287,-176];
Xobs = generateRandomObstacles(10);
Xobs_seen = senseObstacles(curr_pos,Xobs);    % the final sim works like this
Xobs_seen = Xobs;           % only for test

%% call these once before forloop controller
dist_along_cline = cumsum([0, sqrt(diff(cline(1,:)).^2 + diff(cline(2,:)).^2)]);
Xobs_seen = cell2mat(Xobs_seen');       % final sim takes Xobs_seen as input every .5s
idx_obs = 1;
buffer_after_obs = 1;
% if we travel this distance passing the current obstacle, switch to the next obstacle


%% call these every time inside forloop controller

curr_pos = ???;
[~, dist_curr] = distAlongCline(curr_pos);

curr_Xobs = Xobs_seen{idx_obs};     % 4 by 2 coordinates
midpt_Xobs = [mean([curr_Xobs(1,1) curr_Xobs(3,1)]) mean([curr_Xobs(1,2) curr_Xobs(3,2)])];
[k1, dist_obs] = distAlongCline(midpt_Xobs);

if dist_curr > dist_obs + buffer_after_obs)
    idx_obs = idx_obs + 1;      % +1 after passing an obstacle

end

dist_to_left = norm(bl(:,k1),midpt_Xobs);
dist_to_right = norm(br(:,k1),midpt_Xobs);

if dist_to_left > dist_to_right
    keep_left = true;
else
    keep_left = false;
end




%% put this function at the end
function [k1 dist] = distAlongCline(pos, cline, dist_along_cline)
k1 = dsearchn(cline',pos);     % nearest
k2 = dsearchn(cline(:,(k1-1):2:(k1+1))',pos);   % 2nd nearest

curr_pos_proj_on_cline = zeros(1,2);
curr_pos_proj_on_cline(1) = ( cline(2,k1) - cline(1,k1)*tan(theta(k1)) - pos(2) - pos(1)/tan(theta(k1)) )...
    /(-1/tan(theta(k1)) - tan(theta(k1)));
curr_pos_proj_on_cline(2) = tan(theta(k1))*curr_pos_proj_on_cline(1) + cline(2,k1) - cline(1,k1)*tan(theta(k1));

if k2 == 1
    idx_past = k1-1;
else
    idx_past = k1;
end
dist = dist_along_cline(idx_past) + norm(curr_pos_proj_on_cline - cline(:,idx_past));
end