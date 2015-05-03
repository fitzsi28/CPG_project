function [x,y,P] = elastica2D(x1,y1,C,varargin)
%% function [x,y,P] = elastica2D(x1,y1,C,{...})
% -------------------------------------------------------------------------
% Two-dimensional elastic beam model
%
% Associated publication:
% "Mechanical signals at the base of a rat vibrissa: The effect of
% intrinsic vibrissa curvature and implications for tactile exploration"
% Quist BW and Hartmann MJZ (2012) Journal of Neurophysiology
%
% Please visit our website for the most current code and helpful examples:
% http://nxr.northwestern.edu/digital-rat
%
% Dr. Mitra Hartmann's SeNSE Group,
% a part of the Neuroscience and Robotics (NxR) Lab
% -------------------------------------------------------------------------
% OVERVIEW:
% The beam model has two modes of operation:
%   (1) 'point' {default} - the model is deflected so that the contact 
%           point occurs at a (x,y) location of (a,b) by varying the
%           applied force (f) and arc length (s) where the force acts
%           e.g. [x,y,P] = elastica2D(x1,y1,[a,b]);
%   (2) 'force' - the model has a force (f) applied at a radial distance
%           along the beam length of (s)
%           e.g. [x,y,P] = elastica2D(x1,y1,[f,s]);
% -------------------------------------------------------------------------
% INPUT:
%   (x1,y1) - undeflected beam shape [m]
%   C - 2x1 vector constraint to add to the beam depending on the mode:
%       'point' {default}: [a,b] ... point location in 2D space [m]
%       'force': [f,s] ... apply a force (f)[in N] at arc length (s)[in m]
%   varargin:
%       + Structure these inputs as ... 'param_name',param, ...
%       // PLOTTING
%           'plot' - plot final result (true/false)             
%           'plot_steps' - plot iterative steps of solver (true/false)
%       // MODEL OPTIONS
%           'mode' - switch between modes
%               + 'point' {default}: apply [f,s] to reach a contact point (a,b)
%               + 'force': apply [f,s]
%           'BC' - change the boundary condition of the first node
%               + 'r': {default} rigid boundary condition ... 
%                       first segment remains aligned with the x-axis
%               + 'e': elastic boundary condition, use E(1) and I(1)
%               +  # : elastic boundary condition, with the input # as the
%                       stiffness of the first node
%           'E' - provide a Young's modulus [Pa]
%               + {default}: Constant of 3.3 GPa for each node
%               + # : constant for each node
%               + [1xN]: specify Young's modulus at each node
%           'I' - provide area moment of inertia [m^4]
%               + {default}: Tapered beam with base-to-tip radius ratio 
%                       of 15 and a base radius of 100 um
%               + # : constant for each node
%               + [1xN]: specify area moment of inertia at each node
%       // OPTIMIZATION SETTINGS
%           'opt_params' - optimization parameters for fminsearch
%               + Input as a cell array
%           'fs_guess' - provide an initial guess for [f,s] when in 'point' mode
%           'ds_thresh' - provide allowable error for 'point' mode [m]
% OUTPUT:
%   (x,y) - deflected beam shape [m]
%    P - struct of additional model outputs:
%       .E - Young's modulus along the whisker [Pa]
%       .I - area moment of inertia along the whisker [m^4]
%       .fx - axial force at the base [N]
%       .fy - transverse force at the base [N]
%       .m - moment along the whisker arc length [N-m]
%       .dk - change in curvature along the whisker arc length [1/m]
%       .k - total curvature along the whisker arc length [1/m]
%       .k_global - global curvature of the deflected whisker [1/m]
%       .fs - applied force and radial distance [(N) (m)]
%       .gamma - angle at contact point [rad]
%       .contact - location of the contact point [m]
% -------------------------------------------------------------------------
% MODEL ASSUMPTIONS:
% + The simulation runs in *whisker centered coordinates*, where:
%   {1} The whisker base is centered at (0,0)
%   {2} The first segment of the whisker base is co-linear with the x-axis
%   {3} The applied force is normal to the whisker surface
%   {4} There is no friction
% + The resulting forces and moments at the base have a sign that means:
%   - Compression for negative forces
%   - Tension for positive forces
%   - Counter-clock-wise rotation for positive moments
%   - Clock-wise rotation for negative moments
% + The resulting forces/moment at the base should be thought of as "what
%   the whisker would apply to the follicle". e.g. a negative Fx force
%   corresponds to compression of the whisker, pushing into the follicle
% -------------------------------------------------------------------------
% EXAMPLES:
%   [x,y,P] = elastica2D(x1,y1,[a b]);
%   [x,y,P] = elastica2D(x1,y1,[a b],'E',E,'I',I,'plot',1);
%   [x,y,P] = elastica2D(x1,y1,[f s],'mode','force','E',E,'I',I,'plot',1);
% -------------------------------------------------------------------------
% Joe Solomon and Brian Quist
% June 20, 2012
% 
% Lucie Huet
% Nov 7, 2012
global LOCAL

%% + PARAM: Defaults
% Plotting:
TGL_plotfinal = 0; % Plot final deflected shape
TGL_plotsteps = 0; % Plot each iteration of the fitting algorithm

% Mode of operation:
TGL_mode = 'point'; % either 'point' or 'force'.
TGL_BC = 'r'; % Toggle boundary condition: 'r','e',#

% Equidist on or off
run_equidist = 1;

% Optimization parameters:
opt_params = {'TolX',1e-17,'MaxFunEvals',5000};
fs_guess = [NaN NaN]; % [f s] ... units of:[N m]
ds_thresh = 1e-10; % [m]

% Stiffness parameters:
E = NaN; % [Pa]
I = NaN; % [m^4]

% Contact point setup:
a = NaN; % [m]
b = NaN; % [m]

%% + PARAM: User inputs
if ~isempty(varargin)
    for ii = 1:2:length(varargin)
        switch varargin{ii}
            case 'plot', TGL_plotfinal = varargin{ii+1};
            case 'plot_steps', TGL_plotsteps = varargin{ii+1};
            case 'mode', TGL_mode = varargin{ii+1};
            case 'BC', TGL_BC = varargin{ii+1};
            case 'equidist', run_equidist = varargin{ii+1};
            case 'E', E = varargin{ii+1};
            case 'I', I = varargin{ii+1};
            case 'opt_params', opt_params = varargin{ii+1};
            case 'fs_guess', fs_guess = varargin{ii+1};
            case 'ds_thresh', ds_thresh = varargin{ii+1};
            otherwise
                error('Param_name not supported.');
        end
    end
end

% Check for errors
switch TGL_mode
    case 'point',
%         if C(1) < 0, error('C(1) ''a'' must be positive'); end
        if x1(1) ~= 0, error('x1(1) must equal zero'); end
        if y1(1) ~= 0, error('y1(1) must equal zero'); end
    case 'force',
        if C(2) < 0, error('C(2) ''s'' must be positive'); end
    otherwise
end

% Make x1 and y1 equidistant nodes if called for
if run_equidist, [x1,y1] = equidist(x1,y1); end

% Correct shape of (x1,y1) to be rows
if size(x1,2) == min(size(x1)), x1 = x1'; end
if size(y1,2) == min(size(y1)), y1 = y1'; end

%% + Setup: E and I
% Setup: E
% E value selected from: Quist et al. (2012) "Variation in Young's modulus
%   along the length of a rat vibrissa." J. Biomechanics. 44:2775-2781
if isnan(E(1)),
    e = (3.3e9).*ones(1,length(x1));
elseif length(E) == 1,
    e = E.*ones(1,length(x1));
elseif length(E) == length(x1),
    e = E;
else
    error('length of E is not length(1) or length(x)');
end
if size(e,2) == 1, e = e'; end
P.E = e;
switch TGL_BC,
    case 'r'
    otherwise
        e = [e(1) e];
end

% Setup: I
if isnan(I(1)),
    % No I provided.
    % Assume: linear taper with a taper ratio of 15.
    % Assume: base radius of 100 um (typical for a large caudal vibrissa)
    % taper ratio = base diameter / tip diameter
    L_x1 = length(x1);
    rt = 1/15;
    r = (100e-6).*(1:-((1-rt)/(L_x1-1)):rt);
    ii = (0.25*pi).*(r.^4);
    clear L_x1 rt r
elseif length(I) == 1,
    % Assume cylinder
    ii = I.*ones(1,length(x1));
elseif length(I) == length(x1),
    ii = I;
else
    error('length of I is not 1 or length(x)');
end
if size(ii,2) == 1, ii = ii'; end
P.I = ii;
switch TGL_BC,
    case 'r'
    otherwise
        ii = [ii(1) ii];
end

% Setup: BC effect
switch TGL_BC
    case 'r', % Do nothing
    case 'e', % Do nothing
    otherwise
        % TGL_BC is a specified number, but when implemented k = e(2)*i(2)
        % Add this by dividing by i(2), algorithm calls e(2)*i(2)
        e(2) = TGL_BC/ii(2);
        P.E(1) = e(2);
end

% Setup: BC effect on (x1,y1)
switch TGL_BC,
    case 'r'
        % rigid boundary condition. Do nothing.
        dx = 0;
    otherwise
        % Variable boundary condition. Add extra node to front.
        dx = x1(2)-x1(1);
        x1 = [0 x1+dx]; %#ok<*NASGU>
        y1 = [0 y1];
end

%% + Setup: Geometric calculations
[ss,k,ds,dphi] = cart2arc(x1,y1);
phi0 = atan((y1(2)-y1(1))/(x1(2)-x1(1)));
        
%% + Run: elastica2D_solver
switch TGL_mode
    case 'point'
        
        % Geometric computations
        a = C(1)+dx;
        b = C(2);
        
        % Guess contact length and applied force (if necessary)
        if isnan(fs_guess(1)),
            
            % Guess contact length
            c_obj = sqrt(a^2+b^2);
            c = sqrt(x1.^2+y1.^2);
            [~,index] = min(abs(c-c_obj));
            s_guess = ss(index);
            
            % Guess applied force
            d = sqrt((x1(index)-a)^2+(y1(index)-b)^2);
            ei_guess = interp1(ss,e.*ii,s_guess,'linear','extrap');
            if length(I) == 1,
                % Assumes a cylinder
                f_guess = -3*(ei_guess)*d/c_obj^3;
            else
                % Assumes a tapered beam
                c_tip = sqrt(x1(end)^2+y1(end)^2);
                f_guess = -3*(e(1)*ii(1))*d*(c_tip-c_obj)/(c_tip*c_obj^3); 
                clear c_tip
            end
            
            % Apply
            fs_guess = [f_guess s_guess];
            clear f_guess s_guess
            clear c_obj c index d ei_guess
        end

        % Run elastica2D_solver
        fminsearch(@elastica2D_solver,[1,1], ...
            optimset(opt_params{:}), ...
            fs_guess, ...
            x1,y1,e,ii,dphi,phi0,ss,ds,...
            a,b,TGL_BC,TGL_plotsteps);
        
    case 'force'
        
        % Set variables
        f = C(1);
        s_force = C(2)+dx;

        % Run elastica2D_solver
        elastica2D_solver([1,1],[f s_force], ...
            x1,y1,e,ii,dphi,phi0,ss,ds, ...
            a,b,TGL_BC,TGL_plotsteps);
        
end

%% + Finalize: Output shape
x2 = fliplr(LOCAL.x2);
y2 = fliplr(LOCAL.y2);
dk = -fliplr(LOCAL.dk);

% Correct for added BC
switch TGL_BC,
    case 'r', % Do nothing.
    otherwise,
        x2 = x2(2:end)-dx;
        y2 = y2(2:end);
        x1 = x1(2:end)-dx;
        y1 = y1(2:end);
        a = a-dx;
        dk = dk(2:end);
        k = k(2:end);
        LOCAL.s = LOCAL.s-dx;
end

% Finish computing final shape
gamma = atan2(-(y2(end)-y2(end-1)),(x2(end)-x2(end-1))); % [rad]
if abs(arclength(x2,y2)-ss(end)) > ds_thresh % add part of beam after contact point
    i = length(x2);
    dphi_node_i = atan2(y2(i)-y2(i-1),x2(i)-x2(i-1));
    if LOCAL.new_node,
        % Remove extra (interpolated) node
        % Compute dphi_node_i before truncating, otherwise final shape is wrong
        x2 = x2(1:i-1); y2 = y2(1:i-1); i = i - 1;
    end
    if (i+1) <= length(y1),
        dphi_cp = dphi_node_i - atan2(y1(i+1)-y1(i),x1(i+1)-x1(i));
    else
        dphi_cp = dphi_node_i;
    end
    [x_afterCP,y_afterCP] = rotate2(x1(i+1:end),y1(i+1:end),dphi_cp,[x1(i),y1(i)]);
    x = [x2 x2(i)-x1(i)+x_afterCP]; 
    y = [y2 y2(i)-y1(i)+y_afterCP];
else
    x = x2; % [m]
    y = y2; % [m]
end
%% + Finalize: Output parameter struct


% Forces
P.fx = LOCAL.fx; % [N]
P.fy = LOCAL.fy; % [N]

% Moment
% P.m = -1*LOCAL.m;  % [N-m]

% Final computed applied force and radial distance
P.fs = [LOCAL.f LOCAL.s]; % f units:[N]  ... s units:[m]

% Compute curvature
% P.dk0 = k(1) - (atan((y(3)-y(2))/(x(3)-x(2)))-atan((y(2)-y(1))/(x(2)-x(1))))/ds(1); % [1/m]
DK = zeros(1,length(x)-2);
DK(1:length(dk)) = dk;
P.dk = DK; % [1/m]
P.k = P.dk + k; % [1/m]

% Angle at the contact point
P.gamma = -gamma; % [rad]

% Location of force contact
[~,s_final] = arclength(x,y);
index = find(s_final - P.fs(2) <= 0,1,'last');
if index == length(x)
    P.contact = [x(index) y(index)];
else
    pct_node = (P.fs(2)-s_final(index))/(s_final(index+1)-s_final(index));
    P.contact = [x(index)+pct_node*(x(index+1)-x(index)),...
        y(index)+pct_node*(y(index+1)-y(index))];
end
         
% Compute global curvature (based off three points)
px1 = x(1); py1 = y(1);
px3 = P.contact(1); py3 = P.contact(2);
[~,half_index] = min(abs(s_final/2 - P.fs(2)));
px2 = x(half_index); py2 = y(half_index);
M1 = det([px1 py1 1; px2 py2 1; px3 py3 1]);
M2 = det([px1^2+py1^2 py1 1; px2^2+py2^2 py2 1; px3^2+py3^2 py3 1]);
M3 = det([px1^2+py1^2 px1 1; px2^2+py2^2 px2 1; px3^2+py3^2 px3 1]);
M4 = det([px1^2+py1^2 px1 py1; px2^2+py2^2 px2 py2; px3^2+py3^2 px3 py3]);
xk = M2/(2*M1);
yk = -M3/(2*M1);
rk = sqrt(xk^2 + yk^2 + M4/M1);
P.k_global = 1/rk;

% Compute moment everywhere
f_vect = [P.fx;P.fy;0];
r_vect = [P.contact(1) - x(1:index); P.contact(2) - y(1:index)];
r_vect = [r_vect; zeros(1,size(r_vect,2))];
M_all = cross(r_vect,repmat(f_vect,1,size(r_vect,2)));
P.m = zeros(1,length(x));
P.m(1:index) = M_all(3,:);

% Reorder fields into more logical structure
P = orderfields(P,[1 2 3 4 11 6 7 10 5 8 9]);

%% + Plotting
if TGL_plotfinal,
    figure;
    plot(x1,y1,'k.-'); hold on;
    plot(x,y,'b.-');
    switch TGL_mode,
        case 'point',
            plot(a,b,'ro','LineWidth',2);
        case 'force',
            plot(P.contact(1),P.contact(2),'m*');
    end
end

function er = elastica2D_solver(q,FS_REF, ...
    x1,y1,e,ii,dphi,phi0,ss,ds,a,b,TGL_BC,TGL_plotsteps)
%% * function er = elastica2D_solver(q,FS_REF, ...
%       x1,y1,e,ii,dphi,phi0,ss,ds,a,b,TGL_BC,TGL_plotsteps)
% -------------------------------------------------------------------------
% Mode: 'point'
%   * Computes the distance error between desired deflection point (a,b)
%       and the deflection point resulting from force f acting at
%       arclength ss of beam defined by (x1,y1)
% Mode: 'force'
%   * Computes the shape of the beam for a force f 
%       acting at arclength s_force
% -------------------------------------------------------------------------
% INPUTS:
%   q = a vector containing [f,s], 
%       -> 'f' is a force acting at arclength 's' of the beam
%   FS_REF - reference [f_guess s_guess]
% OUTPUTS:
%   er = distance between (a,b) and the deflected point on the beam at s.
global LOCAL
LOCAL.new_node = [];

% Set f and s_force (our "exploratory" variables)
f = FS_REF(1)*q(1);
s_force = FS_REF(2)*q(2);

% Check s_force is valid
if s_force > ss(end)
    s_force = ss(end);
    warning('s_force > arclength(x,y). Forcing s_force to arclength(x,y).'); %#ok<WNTAG>
end

% Add a new node to correspond to s_force (if necessary)
[~,index] = findc(ss,s_force);
if index >= length(ds), index = length(ds); end 
if index < 2, index = 2; end
if ss(index) == s_force ||  ...
        abs(ss(index)-s_force)/ds(index) < 0.001 
    % Use an existing node
    LOCAL.new_node = 0;
    E = e(1:index);
    I = ii(1:index);
    DS = ds(1:index-1);
    DPHI = dphi(1:index-2);
else
    % Add a new node
    LOCAL.new_node = 1;
    if ss(index) > s_force, index = index - 1; end
    if index <= 1, index = 2; end
    
    pct_node = (s_force-ss(index))/ds(index);
    x1_new = x1(index) + pct_node*(x1(index+1)-x1(index));
    y1_new = y1(index) + pct_node*(y1(index+1)-y1(index));
    i_new  =  ii(index) + pct_node*( ii(index+1)- ii(index));
    e_new  =  e(index) + pct_node*( e(index+1)- e(index));
    ds_new = sqrt((x1_new-x1(index))^2+(y1_new-y1(index))^2);
    
    X1 = [x1(1:index) x1_new];
    Y1 = [y1(1:index) y1_new];
    I = [ii(1:index) i_new];
    E = [e(1:index) e_new];
    DS = [ds(1:index-1) ds_new];
    
    dphi_new = atan2(Y1(end)-Y1(end-1),X1(end)-X1(end-1))-atan2(Y1(end-1)-Y1(end-2),X1(end-1)-X1(end-2));
    DPHI = [dphi(1:index-2) dphi_new];
end

% Compute new shape based on f and s_force
x2 = [0 -DS(end)];
y2 = [0 0];
phi = 0;
th = pi;
DK = zeros(1,length(E)-2);
for j = 3:(length(E)+1)
    d = sqrt(x2(j-1)^2+y2(j-1)^2);
    dk = d*f*cos(th)/(E(end-j+2)*I(end-j+2));
    if j <= length(E)
        DK(j-2) = dk;
        phi = phi + dk*DS(end-j+2) - DPHI(end-j+3); % ADD: -ve to DPHI
        x2(j) = x2(j-1) - DS(end-j+2)*cos(phi);
        y2(j) = y2(j-1) - DS(end-j+2)*sin(phi);
        th = atan2(y2(j),x2(j)); % <-- "wrap around" version: th = atan(y2(j)/x2(j));
        switch TGL_BC
            case 'r', % Do nothing
            otherwise
            % Compute moment one node prior
            tau = dk*(E(end-j+2)*I(end-j+2));
        end
    end
end
switch TGL_BC,
    case 'r', 
        tau = dk*(E(end-j+2)*I(end-j+2));
    otherwise
        % Do nothing
end

% Translate 'base' back to origin
x2 = x2 - x2(end); 
y2 = y2 - y2(end);

% Rotate 'base' to be aligned with x-axis
[x2,y2] = rotate2(x2,y2,phi0-phi);

% Rotate 'force' with same angle as 'base'
[fx,fy] = rotate2(0,f,phi0-phi);

% Output: Forces
LOCAL.fx = fx; % [N]
LOCAL.fy = fy; % [N]
LOCAL.m = tau; % [N-m]
% Output: Final parameters
LOCAL.f = f;   % [N]
LOCAL.s = s_force; % [m]
% Output: Final shape
LOCAL.x2 = x2; % [m]
LOCAL.y2 = y2; % [m]
LOCAL.dk = DK; % [1/m]

% Compute error
er = sqrt((a-x2(1))^2 + (b-y2(1))^2);

% Plot
if TGL_plotsteps,
    figure(999); clf(999);
    plot(a,b,'ro','LineWidth',2); hold on;
    plot(x2,y2,'k.-');
    drawnow;
end

function [a,b,ds] = equidist(x,y,varargin)
%% function [a,b,ds] = equidist(x,y,nNodes)
% ---------------------------------------------------------------
% Resamples (x,y) using linear interpolation to make (x,y) points
% equidistant by ds
% ---------------------------------------------------------------
% INPUTS:
%   x = x-coordinates
%   y = y-coordinates
%   varargin:
%       nNodes - number of sample nodes
% OUTPUTS:
%   a = x-coordinates of resampled (x,y) of length nNodes
%   b = y-coordinates of resampled (x,y) of length nNodes
%   ds = equidistant link length 
% ---------------------------------------------------------------
% NOTES:
%   + Requires: arclength.m
% ---------------------------------------------------------------
% Brian Quist 
% July 1, 2010
%
% ed: Lucie Huet
% Oct. 29, 2012 - remove TGL_plot; have the function run twice in a more
% simple manner

%% Process inputs
nNodes = length(x);
if nargin >= 3, if ~isempty(varargin{1}), nNodes = varargin{1}; end; end

for jj = 1:2 % run through equidist twice
    %% Parameterize the function
    [s_total,s] = arclength(x,y);
    s_inc = s_total/(nNodes-1);
    s_target = 0:s_inc:s_inc*(nNodes-1);
    
    %% Interp ds
    a = zeros(nNodes,1);
    b = zeros(nNodes,1);
    % -----
    a(1) = x(1);
    b(1) = y(1);
    % -----
    for ii = 2:length(s_target)
        if ii < length(s_target),
            id = find(s >= s_target(ii),1);
        else
            id = length(s);
        end
        if isempty(id), disp('ERROR: equidist!'); return; end
        % -----
        a(ii) = interp1(s(id-1:id),x(id-1:id),s_target(ii),'linear','extrap');
        b(ii) = interp1(s(id-1:id),y(id-1:id),s_target(ii),'linear','extrap');
    end
    
    %% Replace variables to run a second time
    x = a;
    y = b;
end

%% Output
[~,~,ds_total] = arclength(a,b);
ds = ds_total(1);

function [ss,k,ds,dth] = cart2arc(x,y)
%% * function [s,k,ds,dth] = cart2arc(x,y)
% Converts a curve defined in the Cartesian domain to the arc length/curvature domain.
% INPUTS:
%   x = x-coordinates
%   y = y-coordinates
% OUTPUTS:
%   ss = monotonically increasing arc length vector, starting at 0 (length(s) = length(x))
%   k = curvature at each node (length(k) = length(x) - 2)
%   ds = diff(s)
%   dth = change in slope at each node

% To find curvature, concentrate on three nodes at a time. Positive curvature for positive dth.
[~,ss,ds] = arclength(x,y);
dth = atan2(y(3:end)-y(2:end-1),x(3:end)-x(2:end-1)) - atan2(y(2:end-1)-y(1:end-2),x(2:end-1)-x(1:end-2));
k = dth./ds(1:end-1);

function [s_total,s,ds] = arclength(x,y)
%% * function [s_total,s,ds] = arclength(x,y)
% Returns the arc length of curve defined by (x,y).
% INPUTS:
%   x = x-coordinates
%   y = y-coordinates
% OUTPUTS:
%   s_total = total arc length
%   s = arclength at each (x,y) (starts at zero, ends at s_tot)
%   ds = diff(s)

if size(x,2) == min(size(x)),
    x = x';
end
if size(y,2) == min(size(y)),
    y = y';
end

ds = sqrt((x(2:end)-x(1:end-1)).^2+((y(2:end)-y(1:end-1)).^2));
s = [0 cumsum(ds)];
s_total = s(end);

function [val_out,index] = findc(vect,val_in)
%% * function [val_out,index] = findc(vect,val_in)
% Returns the closest value in vector vect to val_in and its index.
% INPUTS:
%   vect = input vector
%   val_in = target value
% OUTPUTS:
%   val_out = closest value
%   index = index = valOut

vect2 = abs(vect) - val_in;
[~,index] = min(abs(vect2));
val_out = vect(index);

function [x2,y2] = rotate2(x1,y1,theta,origin)
%% * function [x2,y2] = rotate2(x1,y1,theta,origin)
% Rotates points (x1,y1) by angle theta about specified origin.
% INPUTS:
%   x1 = input x-coordinates
%   y1 = input y-coordinates
%   theta = angle to rotate by (counter-clockwise)
%   origin = origin about which to rotate (vector of length 2)
% OUTPUTS:
%   x2 = output x-coordinates
%   y2 = output y-coordinates

if nargin == 3
    origin = [0 0];
end

if size(x1,1) == 1, x1 = x1'; end
if size(y1,1) == 1, y1 = y1'; end

x1 = x1 - origin(1);
y1 = y1 - origin(2);
q = [x1 y1]*[cos(theta) sin(theta); -sin(theta) cos(theta)];
x2 = q(:,1)' + origin(1);
y2 = q(:,2)' + origin(2);