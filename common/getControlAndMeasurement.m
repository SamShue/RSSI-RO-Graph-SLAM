function [u, measurements, truePose] = getControlAndMeasurement(estimatedPose, v_mps, lastGoal, currentGoal, simParams)
%GETCONTROLANDMEASUREMENT Simulates one iteration of robot motion and sensing
%   Computes a steering command via pure pursuit, advances the robot's true
%   pose with a noisy differential-drive motion model, and generates a
%   noisy RSSI-derived range reading to every anchor within radio range.
%   Unlike a range-bearing sensor, an RSSI receiver on the tag has no way
%   to tell what *direction* a signal came from -- only how strong it
%   was -- so every observation this function returns is range-only.
%
%   Inputs:
%       estimatedPose - filter's current pose estimate [x,y,theta]; used to
%                       command the robot and to seed the true pose on the
%                       first call
%       v_mps         - commanded linear velocity (m/s)
%       lastGoal      - [x,y] start point of the current path segment
%       currentGoal   - [x,y] end point of the current path segment
%       simParams     - struct of fixed simulation parameters with fields:
%                           anchors     - Nx2 matrix of anchor [x,y] positions
%                           model       - kinematic model: 'icr' or 'simple'
%                           dt_s        - time between iterations (s)
%                           sigma_v     - linear velocity noise std dev (m/s)
%                           sigma_w     - angular velocity noise std dev (rad/s)
%                           maxRange_m  - radio range beyond which an anchor
%                                         can't be heard (m)
%                           rssi        - struct of path-loss parameters,
%                                         see rssiToRange.m
%
%   Outputs:
%       u            - noisy odometry reading [v_mps, w_radps, dt_s]
%       measurements - Kx3 matrix of [anchorId, range_m, sigma_r], one row
%                      per anchor currently within maxRange_m
%       truePose     - robot's true pose [x,y,theta] after this iteration's motion

    persistent trueRobotPose;

    % initialize trueRobotPose if first call
    if(isempty(trueRobotPose))
        trueRobotPose = estimatedPose;
    end

    % Pure pursuit steering command toward the current path segment
    w_radps = purePursuit(estimatedPose, lastGoal, currentGoal, v_mps);

    % Corrupt the commanded velocities with odometry noise
    v_noisy = v_mps + simParams.sigma_v*randn;
    w_noisy = w_radps + simParams.sigma_w*randn;
    u = [v_noisy, w_noisy, simParams.dt_s];

    % Advance the true pose using the noisy control
    trueRobotPose = differentialDriveKinematics(trueRobotPose, v_noisy, w_noisy, simParams.dt_s, simParams.model);

    % Generate an RSSI-derived range reading to every anchor within radio range
    numAnchors = size(simParams.anchors, 1);
    measurements = zeros(0, 3);
    for j = 1:numAnchors
        dx = simParams.anchors(j,1) - trueRobotPose(1);
        dy = simParams.anchors(j,2) - trueRobotPose(2);
        trueRange_m = sqrt(dx^2 + dy^2);

        if(trueRange_m <= simParams.maxRange_m)
            % Log-distance path loss model, corrupted with noise in the
            % dB domain (where RSSI noise is actually roughly Gaussian),
            % then inverted back to a range reading.
            rssi_true_dBm = simParams.rssi.A - 10*simParams.rssi.n*log10(trueRange_m/simParams.rssi.d0_m);
            rssi_meas_dBm = rssi_true_dBm + simParams.rssi.sigma_rssi*randn;
            [range_m, sigma_r] = rssiToRange(rssi_meas_dBm, simParams.rssi);

            measurements(end+1, :) = [j, range_m, sigma_r]; %#ok<AGROW>
        end
    end

    truePose = trueRobotPose;
end
