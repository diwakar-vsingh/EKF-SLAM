directory_storage_name = 'figures';
if( exist( directory_storage_name, 'dir' ) )
else
    mkdir( directory_storage_name );
end

figure_all = figure;
oldUnits = get( figure_all, 'Units' );
set( figure_all, 'Units', 'normalized' );
set( figure_all, 'Position', [0.01,0.2,0.7,0.4] );
set( figure_all, 'Units', oldUnits );

% models_fig = subplot('position',[0.05 0.1 0.05 0.8]);
im_fig = subplot('position',[0.05 0.1 0.425 0.8]);
near3D_fig = subplot('position',[0.575 0.1 0.375 0.8]);

view(-180,0);