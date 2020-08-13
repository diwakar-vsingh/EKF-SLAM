function f = ekf_filter( varargin )

if nargin == 0
    
   f.type = [];
    
   f.x_k_k = [];
   f.p_k_k = [];
   
   f.std_a = [];
   f.std_alpha = [];
   f.std_z = [];
   
   f.x_k_km1 = [];
   f.p_k_km1 = [];
   f.predicted_measurements = [];
   f.H_predicted = [];
   f.R_predicted = [];
   f.S_predicted = [];
   f.S_matching = [];
   f.z = [];
   f.h = [];
   f.H_matching = [];
   f.measurements = [];
   f.R_matching = [];
   f.x_k_k_mixing_estimate = [];
   f.p_k_k_mixing_covariance = [];
   
   f = class(f,'ekf_filter');
   
elseif isa(varargin{1},'ekf_filter')
    
    f = varargin{1};
    
else
    
   f.type = varargin{6}; 
    
   f.x_k_k = varargin{1};
   f.p_k_k = varargin{2};
   
   f.std_a = varargin{3};
   f.std_alpha = varargin{4};
   f.std_z = varargin{5};
   
   f.x_k_km1 = [];
   f.p_k_km1 = [];
   f.predicted_measurements = [];
   f.H_predicted = [];
   f.R_predicted = [];
   f.S_predicted = [];
   f.S_matching = [];
   f.z = [];
   f.h = [];
   f.H_matching = [];
   f.measurements = [];
   f.R_matching = [];
   f.x_k_k_mixing_estimate = [];
   f.p_k_k_mixing_covariance = [];
   
%    f = class(f,'ekf_filter');
   
end