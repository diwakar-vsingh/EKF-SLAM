function [X_km1_km1_new,P_km1_km1_new] = delete_a_feature( X_km1_km1,P_km1_km1,featToDelete,features_info )


if strcmp(features_info(featToDelete).type,'cartesian')
    parToDelete = 3;
else
    parToDelete = 6;
end

indexFromWichDelete = 14;

for i=1:featToDelete-1
    if strcmp(features_info(i).type, 'inversedepth')
        indexFromWichDelete = indexFromWichDelete + 6;
    end
    if strcmp(features_info(i).type, 'cartesian')
        indexFromWichDelete = indexFromWichDelete + 3;
    end
end

X_km1_km1_new = [X_km1_km1(1:indexFromWichDelete-1); X_km1_km1(indexFromWichDelete+parToDelete:end)];
    

P_km1_km1_new = [P_km1_km1(:,1:indexFromWichDelete-1) P_km1_km1(:,indexFromWichDelete+parToDelete:end)];
P_km1_km1_new = [P_km1_km1_new(1:indexFromWichDelete-1,:); P_km1_km1_new(indexFromWichDelete+parToDelete:end,:)];