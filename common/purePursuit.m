function [w_radps, gp] = purePursuit(currentPose, lineSegBeginPoint, lineSegEndPoint, v_mps)
%PUREPURSUIT Compute steering command to follow a line segment
%   Uses the pure pursuit algorithm to steer a robot at pose currentPose
%   [x,y,theta] toward a lookahead goal point along the line segment from
%   lineSegBeginPoint to lineSegEndPoint. Returns the angular velocity
%   w_radps needed to reach the goal point gp while traveling at v_mps.

    lookaheadDist_m = 1;
    
    % Compute cross track error
    err = getCrossTrackError(lineSegBeginPoint, lineSegEndPoint, currentPose);
    
    % Get goal point along line segment based on lookahead distance
    gp = getGoalPoint(currentPose, lineSegBeginPoint, lineSegEndPoint, lookaheadDist_m);
    
    % Compute angle between vector from robot to goal point and current
    % segment to goalpoint
    vec2goal = gp - currentPose(1:2);
    alpha = (atan2(vec2goal(2), vec2goal(1)) - currentPose(3));
    
    % Compute curavature
    kappa = 2*sin(alpha)/lookaheadDist_m;
    
    w_radps = kappa * v_mps;
end

function [point] = getGoalPoint(currentPose, lineSegBeginPoint, lineSegEndPoint, r)
    if(distToGoalPoint(currentPose(1:2), lineSegEndPoint) < r)
        point = lineSegEndPoint;
    else
        % https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm/1084899#1084899
        d = lineSegEndPoint - lineSegBeginPoint;
        f = lineSegBeginPoint - currentPose(1:2);

        a = dot(d,d);
        b = 2*dot(f,d);
        c = dot(f,f) - r*r;
        discriminant = b*b - 4*a*c;
        
        point = [];
        point1 = [];
        point2 = [];
        if (discriminant < 0)
            % no intersection
        else
            discriminant = sqrt(discriminant);
            t1 = (-b - discriminant)/(2*a);
            t2 = (-b + discriminant)/(2*a);
            
            if (t1 >= 0 && t1 <=1)
                % return t1 intersection
                point1 = lineSegBeginPoint + t1*d;
                % assign in case t2 doesn't exist
                point = point1;
            end
            
            if (t2 >= 0 && t2 <=1)
                % return t2 intersection
                point2 = lineSegBeginPoint + t2*d;
                % assign in case t1 doesn't exist
                point = point2;
            end
        end

        if(~isempty(point1) && ~isempty(point2))
            if(distToGoalPoint(point1, lineSegEndPoint) < distToGoalPoint(point2, lineSegEndPoint))
                point = point1;
            else
                point = point2;
            end
        end

        if(isempty(point))
            % The lookahead circle doesn't intersect the segment at all
            % (e.g. the robot has drifted further than the lookahead
            % distance off the path). Fall back to steering toward the
            % end of the segment so the robot can recover.
            point = lineSegEndPoint;
        end
    end
    
end

function [error] = getCrossTrackError(s1, s2, pose)
    p = projectOntoSegment(s1, s2, pose(1:2));
    
    error = sqrt((p(1) - pose(1))^2 + (p(2) - pose(2))^2);
end

function [projected_p] = projectOntoSegment(s1, s2, p1)
    % get vector that describes line segment
    s = s2 - s1;
    
    % get vector that describes p relative to line segment
    p = p1 - s1;
    
    % project p onto s
    projected_p = (dot(p,s)/dot(s,s)) * s;
    
    % add projected vector onto start of line segment
    projected_p = projected_p + s1;
end

function [dist] = distToGoalPoint(cp, gp)
    dist = sqrt((cp(1) - gp(1))^2 + (cp(2) - gp(2))^2);
end
