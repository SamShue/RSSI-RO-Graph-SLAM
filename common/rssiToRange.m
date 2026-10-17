function [range_m, sigma_r] = rssiToRange(rssi_dBm, rssiParams)
%RSSITORANGE Convert a received RSSI reading into an estimated range
%   Inverts the log-distance path loss model
%       RSSI(d) = A - 10*n*log10(d / d0)
%   to turn a (noisy) RSSI reading in dBm into an estimated range in
%   meters, and propagates the RSSI noise into a range standard
%   deviation so the graph solver can weight each observation by how
%   trustworthy it is (RSSI noise is roughly constant in dB, which
%   translates into *larger* range uncertainty for anchors that are far
%   away, since the path loss curve flattens out with distance).
%
%   Inputs:
%       rssi_dBm   - received signal strength (dBm)
%       rssiParams - struct with fields:
%                       A          - reference RSSI at distance d0 (dBm)
%                       d0_m       - reference distance for A (m)
%                       n          - path loss exponent (unitless)
%                       sigma_rssi - RSSI measurement noise std dev (dB)
%
%   Outputs:
%       range_m - estimated range (m)
%       sigma_r - propagated range standard deviation (m), via
%                 first-order (linearized) error propagation of
%                 sigma_rssi through the inverted path loss model

    A = rssiParams.A;
    d0 = rssiParams.d0_m;
    n = rssiParams.n;
    sigma_rssi = rssiParams.sigma_rssi;

    range_m = d0 * 10^((A - rssi_dBm)/(10*n));

    % d(range)/d(rssi) = -range * ln(10) / (10*n)
    sigma_r = range_m * log(10)/(10*n) * sigma_rssi;
end
