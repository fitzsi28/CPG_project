function [x,y] = sa2xy(s,A,varargin)
%% function [x,y] = sa2xy(s,A,{N_nodes})
% -------------------------------------------------------------------------
% INPUT:
%   s - arc length
%   a - parabolic A coefficient
%   varargin:
%       N_nodes - number of nodes to use in approximation
% OUTPUT:
%   (x,y) - (x,y) coordinates of resulting segment
% -------------------------------------------------------------------------
%
% This function is included as an accessory function to elastica2D, and its
% use is mentioned in the elastica2D guide.
%
% Dr. Mitra Hartmann's SeNSE Group,
% a part of the Neuroscience and Robotics (NxR) Lab
%
% Brian Quist
% November 3, 2011

%% Handle inputs
N_nodes = 50;
if nargin >= 3 && ~isempty(varargin{1}), N_nodes = varargin{1}; end

%% Correct arc length
options = optimset('MaxFunEvals',1000);
x_max = fminsearch(@LOCAL_CorrectArclength,s,options,s,N_nodes,A);

%% Compute (x,y)
x = (0:1/(N_nodes-1):1).*x_max;
y = A.*(x.^2);

function e = LOCAL_CorrectArclength(x_max,s_ref,N_nodes,A)
x = (0:1/(N_nodes-1):1).*x_max;
y = A.*(x.^2);
s = sum(sqrt((x(2:end)-x(1:end-1)).^2+((y(2:end)-y(1:end-1)).^2)));
e = abs(s-s_ref);