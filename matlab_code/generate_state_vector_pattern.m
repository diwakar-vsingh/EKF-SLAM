function [state_vector_pattern, z_id, z_euc] = generate_state_vector_pattern( features_info, x )

state_vector_pattern = zeros(length(x),4);
position = 14;
z_id = [];
z_euc = [];

for i=1:length( features_info )
    
    if strcmp(features_info(i).type, 'inversedepth')
        if ~isempty(features_info(i).z)
            state_vector_pattern(position:position+2,1) = ones(3,1);
            state_vector_pattern(position+3:position+4,2) = ones(2,1);
            state_vector_pattern(position+5,3) = 1;
            z_id = [z_id [features_info(i).z(1); features_info(i).z(2)]];
        end
        position = position + 6;
    end
    if strcmp(features_info(i).type, 'cartesian')
        if ~isempty(features_info(i).z)
            state_vector_pattern(position:position+2,4) = ones(3,1);
            z_euc = [z_euc [features_info(i).z(1); features_info(i).z(2)]];
        end
        position = position + 3;
    end
    
end