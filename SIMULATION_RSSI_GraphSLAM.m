clc;
clear all;
close all;
addpath(fullfile(fileparts(mfilename('fullpath')), 'common'));

% ------------------------------------------------------------------
% Environment: ground-truth RSSI anchor positions and path waypoints
%
% Anchors are fixed radio beacons placed around the environment (e.g.
% BLE/WiFi/UWB beacons mounted on walls). The robot carries a "tag" that
% measures the received signal strength (RSSI) from any anchor within
% radio range. Unlike a range-bearing sensor (e.g. a camera or lidar),
% RSSI gives *no* bearing information at all -- only a noisy estimate of
% distance -- so this is a range-only SLAM problem.
% ------------------------------------------------------------------
anchors = [0, 0;
           8, 0;
           8, 6;
           0, 6;
           4, 0;
           8, 3;
           4, 6;
           0, 3];
numAnchors = size(anchors, 1);

goalPoints = [1,1; 7,1; 7,5; 1,5; 4,3; 1,1];

% ------------------------------------------------------------------
% Robot/controller parameters
% ------------------------------------------------------------------
v_mps = 0.5;
robotRadius_m = 0.3;

% ------------------------------------------------------------------
% Simulation parameters shared with getControlAndMeasurement
% ------------------------------------------------------------------
simParams.anchors    = anchors;
simParams.model      = 'icr';          % kinematic model: 'icr' or 'simple'
simParams.dt_s       = 0.1;            % time between iterations (s)
simParams.sigma_v    = 0.02;           % linear velocity noise std dev (m/s)
simParams.sigma_w    = deg2rad(1);     % angular velocity noise std dev (rad/s)
simParams.maxRange_m = 7;              % radio range beyond which an anchor can't be heard (m)

% Log-distance path loss model: RSSI(d) = A - 10*n*log10(d/d0)
simParams.rssi.A          = -45;       % reference RSSI at d0 (dBm)
simParams.rssi.d0_m       = 1;         % reference distance for A (m)
simParams.rssi.n          = 2.5;       % path loss exponent (typical indoor value)
simParams.rssi.sigma_rssi = 2;         % RSSI measurement noise std dev (dB)

% The robot's own odometry (e.g. wheel encoders) is an imperfect *reading*
% of its actual motion, separate from the real-world disturbances already
% baked into the true motion above. This extra noise is what the robot
% dead-reckons with, and it's the main thing that makes dead reckoning
% drift away from the truth over time.
simParams.sigma_v_odom = 0.08;         % extra linear odometry noise std dev (m/s)
simParams.sigma_w_odom = deg2rad(0.5); % extra angular odometry noise std dev (rad/s)

% ------------------------------------------------------------------
% RSSI anchor initialization parameters
%
% A single range reading only constrains an anchor to lie somewhere on a
% circle -- it takes several readings of the *same* anchor from
% *different* poses before its 2D position can be pinned down (classic
% multilateration). So, unlike the range-bearing case (where a landmark
% can be triangulated from one observation), a new anchor's observations
% are buffered until enough geometrically diverse readings have come in,
% at which point a linear least-squares multilateration solve seeds its
% node in the graph and every buffered reading is released into the log.
% ------------------------------------------------------------------
minObservationsForInit = 6;

% ------------------------------------------------------------------
% SLAM parameters
%
% Unlike a filter (e.g. EKF SLAM), GraphSLAM doesn't correct a running
% estimate every iteration. Instead it collects the *entire* history of
% odometry and RSSI range readings, builds a "graph" of constraints
% between every robot pose and every anchor, and then solves for the
% most likely trajectory + map all at once (a batch optimization).
%
% So during the simulation loop below we only do dead reckoning
% (integrate noisy odometry with no correction) and log everything
% needed to build that graph afterward:
%   poseHistory     - dead-reckoned pose estimate at every time step
%   controlHistory  - noisy odometry reading [v, w, dt] used at every time step
%   measurementLog  - every RSSI range reading [poseIdx, anchorId, range, sigma_r]
%                      for anchors that have already been initialized
%   truePoseHistory - the robot's true pose at every time step (for plotting only)
% ------------------------------------------------------------------
poseHistory = [0, 0.2, 0];
controlHistory = zeros(0, 3);
measurementLog = zeros(0, 4);
truePoseHistory = poseHistory;

anchorEstimates = zeros(numAnchors, 2);
anchorInitialized = false(numAnchors, 1);
anchorBuffer = cell(numAnchors, 1);
for j = 1:numAnchors
    anchorBuffer{j} = zeros(0, 3);  % pending [poseIdx, range, sigma_r], awaiting init
end

% Odometry noise for the graph's motion constraints, as a covariance PER SECOND
% in pose space [x; y; theta]: independent x/y/theta noise. (The two noisy
% inputs, v and w, only span 2 of these 3 dimensions, so mapping noise through
% the motion model the way the EKF does would make the information matrix
% singular.) This reflects the uncertainty in the *odometry reading* logged in
% controlHistory, not the true motion's actuation noise. graphSLAM multiplies it
% by dt^2 to get the covariance of one step.
R = diag([simParams.sigma_v_odom^2, simParams.sigma_v_odom^2, simParams.sigma_w_odom^2]);

% ------------------------------------------------------------------
% Simulation loop
% ------------------------------------------------------------------
ii = 1;
lastGoal = poseHistory(1,1:2);
currentGoal = goalPoints(ii, :);
atFinalGoal = 0;

while(~atFinalGoal)
    estimatedPose = poseHistory(end, :);

    % Simulate one iteration of motion + sensing (known data association)
    [u, measurements, truePose] = getControlAndMeasurement(estimatedPose, v_mps, lastGoal, currentGoal, simParams);

    % ---------------- Dead reckoning ----------------
    % The robot doesn't have access to the "true" noisy control u above
    % (that reflects whatever actually happened, including real-world
    % disturbances); it only has its own further-imperfect odometry
    % *reading* of that control (e.g. wheel-encoder error). This extra
    % odometry noise is what GraphSLAM will need to reason about.
    v = u(1) + simParams.sigma_v_odom*randn;
    w = u(2) + simParams.sigma_w_odom*randn;
    dt = u(3); theta = estimatedPose(3);
    newPose = [estimatedPose(1) + v*cos(theta)*dt, ...
               estimatedPose(2) + v*sin(theta)*dt, ...
               wrapToPi(theta + w*dt)];
    poseHistory(end+1, :) = newPose;
    controlHistory(end+1, :) = [v, w, dt];
    truePoseHistory(end+1, :) = truePose;
    poseIdx = size(poseHistory, 1);

    % ---------------- Log RSSI range measurements ----------------
    for k = 1:size(measurements, 1)
        j = measurements(k, 1);
        range_m = measurements(k, 2);
        sigma_r = measurements(k, 3);

        if(anchorInitialized(j))
            measurementLog(end+1, :) = [poseIdx, j, range_m, sigma_r];
        else
            anchorBuffer{j}(end+1, :) = [poseIdx, range_m, sigma_r];
            if(size(anchorBuffer{j}, 1) >= minObservationsForInit)
                bufferedIdx = anchorBuffer{j}(:,1);
                [p, ok] = trilaterate(poseHistory(bufferedIdx, 1:2), anchorBuffer{j}(:,2));
                if(ok)
                    anchorEstimates(j, :) = p;
                    anchorInitialized(j) = true;
                    % Flush every buffered reading into the log now that
                    % this anchor finally has a node in the graph.
                    measurementLog = [measurementLog; anchorBuffer{j}(:,1), repmat(j, size(anchorBuffer{j},1), 1), anchorBuffer{j}(:,2:3)]; %#ok<AGROW>
                    anchorBuffer{j} = zeros(0, 3);
                end
            end
        end
    end

    % ---------------- Goal management ----------------
    if(atGoalPoint(truePose, currentGoal))
        if(ii < size(goalPoints, 1))
            lastGoal = currentGoal;
            ii = ii + 1;
            currentGoal = goalPoints(ii, :);
        else
            atFinalGoal = 1;
        end
    end

    % ---------------- Render environment ----------------
    %======================================================================
    clf; hold on;
    title(sprintf('Current Goal Point: %0.2f, %0.2f', currentGoal(1), currentGoal(2)));
    xlim([-1 9]); ylim([-1.5 7.5]);
    xlabel('meters'); ylabel('meters');

    % anchors: true (bright) vs. multilaterated estimate (gray), stars
    plot(anchors(:,1), anchors(:,2), 'r*', 'MarkerSize', 10, 'LineWidth', 1.5);
    seen = find(anchorInitialized);
    if(~isempty(seen))
        plot(anchorEstimates(seen,1), anchorEstimates(seen,2), '*', 'Color', [.7 .7 .7], 'MarkerSize', 10, 'LineWidth', 1.5);
    end

    % robot: true pose (default/bright) vs. dead-reckoned pose (gray)
    drawRobot(truePose(1), truePose(2), rad2deg(truePose(3)), robotRadius_m);
    drawRobot(newPose(1), newPose(2), rad2deg(newPose(3)), robotRadius_m, 'gray');

    pause(0.001);
    % End render environment
    %----------------------------------------------------------------------
end

fprintf('Drive complete: %d/%d anchors initialized via multilateration before the end of the run.\n', ...
    nnz(anchorInitialized), numAnchors);

% ------------------------------------------------------------------
% GraphSLAM back-end
% Now that the full trajectory has been driven, build the graph from
% every logged motion and RSSI range constraint and solve it.
% ------------------------------------------------------------------
[optimizedPoses, optimizedAnchors] = graphSLAM(poseHistory, controlHistory, measurementLog, anchorEstimates, anchorInitialized, R);

% ---------------- Render final result ----------------
clf; hold on;
title('RSSI Range-Only GraphSLAM result: true vs. dead-reckoned vs. optimized');
xlim([-1 9]); ylim([-1.5 7.5]);
xlabel('meters'); ylabel('meters');

plot(anchors(:,1), anchors(:,2), 'r*', 'MarkerSize', 10, 'LineWidth', 1.5);
plot(optimizedAnchors(anchorInitialized,1), optimizedAnchors(anchorInitialized,2), '*', 'Color', [0 .6 0], 'MarkerSize', 10, 'LineWidth', 1.5);

plot(truePoseHistory(:,1), truePoseHistory(:,2), 'b-', 'LineWidth', 1.5);
plot(poseHistory(:,1), poseHistory(:,2), '--', 'Color', [.7 .7 .7], 'LineWidth', 1.5);
plot(optimizedPoses(:,1), optimizedPoses(:,2), '-', 'Color', [0 .6 0], 'LineWidth', 1.5);

legend('true anchors', 'optimized anchors', 'true path', 'dead-reckoned path', 'optimized path', 'Location', 'northeastoutside');

function [atGoal] = atGoalPoint(robotPose, goalPoint)
    goalRadius_m = 0.25;
    dist = sqrt((robotPose(1) - goalPoint(1))^2 + (robotPose(2) - goalPoint(2))^2);
    atGoal = abs(dist) < goalRadius_m;
end

function [wrapped_rad] = wrapToPi(angle_rad)
    % wrap an angle in radians to the range [-pi, pi]
    wrapped_rad = mod(angle_rad + pi, 2*pi) - pi;
end

function [p, ok] = trilaterate(poses_xy, ranges)
%TRILATERATE Linear least-squares multilateration from range-only readings
%   Given K>=3 poses and the ranges measured from each of them to the
%   same unknown anchor, solves for the anchor's 2D position. Averaging
%   every circle equation (x_i-x)^2+(y_i-y)^2=r_i^2 together and
%   subtracting that average from each one cancels the quadratic
%   x^2+y^2 term in the unknown position, leaving an over-determined
%   linear system (every reading contributes one row, symmetrically,
%   rather than picking one reading as an arbitrary pivot) that's solved
%   by least squares.
%
%   Inputs:
%       poses_xy - Kx2 positions the readings were taken from
%       ranges   - Kx1 range reading from each position
%
%   Outputs:
%       p  - 1x2 estimated anchor position (NaN if the solve failed)
%       ok - true if the geometry was diverse enough to solve reliably

    xbar = mean(poses_xy(:,1));
    ybar = mean(poses_xy(:,2));
    A = 2*[poses_xy(:,1) - xbar, poses_xy(:,2) - ybar];
    rhs = (poses_xy(:,1).^2 - mean(poses_xy(:,1).^2)) + (poses_xy(:,2).^2 - mean(poses_xy(:,2).^2)) ...
        - (ranges.^2 - mean(ranges.^2));

    % Reject poorly conditioned geometry (e.g. the robot barely moved, or
    % moved in a straight line through the anchor) rather than seed the
    % graph with a wild initial guess; the caller simply keeps buffering
    % readings until the geometry is diverse enough. A loose threshold
    % here (e.g. cond ~ 1e6) still lets through geometry that's barely
    % non-degenerate, which blows the small amount of range noise up
    % into a huge position error -- so this is kept tight.
    if(rank(A) < 2 || cond(A) > 15)
        p = [NaN, NaN];
        ok = false;
        return;
    end

    p = (A \ rhs)';

    % Belt-and-suspenders plausibility check: even a well-conditioned
    % linear solve can land on a position that doesn't actually fit the
    % original (nonlinear) range readings well if the noise realization
    % was unlucky. Reject those too instead of handing GraphSLAM a bad
    % seed it may never fully recover from.
    predictedRanges = sqrt(sum((poses_xy - p).^2, 2));
    if(sqrt(mean((predictedRanges - ranges).^2)) > 1.0)
        p = [NaN, NaN];
        ok = false;
        return;
    end

    ok = true;
end

function [poses, anchors] = graphSLAM(poseHistory, controlHistory, measurementLog, anchorEstimates, anchorInitialized, R)
%GRAPHSLAM Batch-optimize a pose/anchor graph with Gauss-Newton
%   Builds the information matrix H and information vector b describing
%   every motion constraint (between consecutive poses) and every
%   RSSI range constraint (between a pose and the anchor it heard),
%   then solves H*dx = -b for the correction that makes the whole graph
%   most consistent. Because the motion/measurement models are
%   nonlinear, the graph is relinearized and re-solved a few times
%   (Gauss-Newton), each time starting from the previous solution.
%
%   Unlike range-bearing GraphSLAM, each RSSI observation constrains only
%   the *distance* between a pose and an anchor -- not the direction --
%   so every measurement Jacobian here has a single row (one scalar
%   residual) instead of two.
%
%   Inputs:
%       poseHistory        - Tx3 dead-reckoned poses [x,y,theta], used as
%                             the initial guess for every pose node
%       controlHistory     - (T-1)x3 odometry controls [v,w,dt] used to
%                             move from poseHistory(t,:) to poseHistory(t+1,:)
%       measurementLog      - Mx4 observations [poseIdx, anchorId, range, sigma_r]
%       anchorEstimates     - Nx2 multilaterated initial guess for each anchor
%       anchorInitialized   - Nx1 logical, true for anchors with enough
%                             diverse readings to have a node in the graph
%       R                   - 3x3 odometry noise covariance per second of
%                             motion, in pose space [x; y; theta]. The
%                             covariance of one step of length dt is R*dt^2.
%
%   Outputs:
%       poses   - Tx3 optimized poses [x,y,theta]
%       anchors - Nx2 optimized anchor positions (NaN for uninitialized ones)

    numPoses = size(poseHistory, 1);
    numAnchors = size(anchorEstimates, 1);

    % Only initialized anchors get a node in the graph; one without any
    % constraints would leave the graph singular.
    initializedIds = find(anchorInitialized)';
    numInitialized = numel(initializedIds);
    anchorNodeOf = zeros(numAnchors, 1);
    anchorNodeOf(initializedIds) = 1:numInitialized;

    poseDims = 3*numPoses;
    n = poseDims + 2*numInitialized;

    % Pack the initial guess (dead reckoning + multilaterated anchor
    % positions) into one big state vector: mu = [pose1; pose2; ...; a1; a2; ...]
    mu = zeros(n, 1);
    mu(1:poseDims) = reshape(poseHistory', [], 1);
    mu(poseDims+1:end) = reshape(anchorEstimates(initializedIds,:)', [], 1);

    numIterations = 15;
    for iter = 1:numIterations
        H = zeros(n);
        b = zeros(n, 1);

        % Anchor the first pose to its known starting value. Without this,
        % every constraint is *relative*, so the whole graph is free to
        % translate/rotate rigidly and the system is underdetermined.
        H(1:3,1:3) = H(1:3,1:3) + diag([1e6, 1e6, 1e6]);

        % ---------------- motion constraints ----------------
        for t = 1:numPoses-1
            idx_t  = 3*(t-1)+1 : 3*(t-1)+3;
            idx_t1 = 3*t+1     : 3*t+3;

            theta = mu(idx_t(3));
            v = controlHistory(t,1); w = controlHistory(t,2); dt = controlHistory(t,3);

            % error between where pose t+1 actually is and where the
            % motion model predicts it should be, given pose t and control t
            predicted = [mu(idx_t(1)) + v*cos(theta)*dt;
                         mu(idx_t(2)) + v*sin(theta)*dt;
                         wrapToPi(theta + w*dt)];
            e = mu(idx_t1) - predicted;
            e(3) = wrapToPi(e(3));

            G = [1, 0, -v*sin(theta)*dt;
                 0, 1,  v*cos(theta)*dt;
                 0, 0,  1];
            Ap = -G;      % d(e)/d(pose_t)
            Bp = eye(3);  % d(e)/d(pose_t+1)

            % The covariance of this step's odometry noise is R scaled by
            % dt^2, and its information matrix is the inverse of that.
            omega_motion = inv(R*dt^2);

            H(idx_t,  idx_t)  = H(idx_t,  idx_t)  + Ap'*omega_motion*Ap;
            H(idx_t,  idx_t1) = H(idx_t,  idx_t1) + Ap'*omega_motion*Bp;
            H(idx_t1, idx_t)  = H(idx_t1, idx_t)  + Bp'*omega_motion*Ap;
            H(idx_t1, idx_t1) = H(idx_t1, idx_t1) + Bp'*omega_motion*Bp;
            b(idx_t)  = b(idx_t)  + Ap'*omega_motion*e;
            b(idx_t1) = b(idx_t1) + Bp'*omega_motion*e;
        end

        % ---------------- RSSI range constraints ----------------
        for k = 1:size(measurementLog, 1)
            t = measurementLog(k,1);
            j = measurementLog(k,2);
            z = measurementLog(k,3);
            sigma_r = measurementLog(k,4);

            idx_pose = 3*(t-1)+1 : 3*(t-1)+3;
            node = anchorNodeOf(j);
            idx_anchor = poseDims + 2*(node-1)+1 : poseDims + 2*(node-1)+2;

            pose = mu(idx_pose);
            anchor = mu(idx_anchor);

            delta = anchor - pose(1:2);
            range_hat = sqrt(delta'*delta);
            e = z - range_hat;

            Ap = [delta(1)/range_hat, delta(2)/range_hat, 0];  % d(e)/d(pose)
            Bp = [-delta(1)/range_hat, -delta(2)/range_hat];   % d(e)/d(anchor)

            % Each RSSI reading carries its own propagated range variance
            % (far readings are noisier), so its information weight is
            % computed per-measurement rather than from one fixed Q.
            omega_meas = 1/sigma_r^2;

            H(idx_pose, idx_pose)     = H(idx_pose, idx_pose)     + Ap'*omega_meas*Ap;
            H(idx_pose, idx_anchor)   = H(idx_pose, idx_anchor)   + Ap'*omega_meas*Bp;
            H(idx_anchor, idx_pose)   = H(idx_anchor, idx_pose)   + Bp'*omega_meas*Ap;
            H(idx_anchor, idx_anchor) = H(idx_anchor, idx_anchor) + Bp'*omega_meas*Bp;
            b(idx_pose)   = b(idx_pose)   + Ap'*omega_meas*e;
            b(idx_anchor) = b(idx_anchor) + Bp'*omega_meas*e;
        end

        % Gauss-Newton update: solve H*dx = -b and apply the correction
        dx = -(H\b);
        mu = mu + dx;
        for t = 1:numPoses
            mu(3*t) = wrapToPi(mu(3*t));
        end
    end

    poses = reshape(mu(1:poseDims), 3, numPoses)';
    anchors = nan(numAnchors, 2);
    anchors(initializedIds,:) = reshape(mu(poseDims+1:end), 2, numInitialized)';
end
