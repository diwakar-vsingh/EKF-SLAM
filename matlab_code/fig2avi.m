movie = avifile( sprintf( '%s\\results.avi', 'figures' ), 'compression', 'None' );

for i=91:2169
    % if mod(i,2)==0
    fid = fopen(sprintf( '%s\\image%04d.fig', 'figures', i), 'r');
    if (fid~=-1)
        h = openfig(sprintf( '%s\\image%04d.fig', 'figures', i) );
        im = getframe(gcf);
        movie = addframe( movie, im );
        i
        fclose(fid);
        close(h);
    end 
    % end
end

movie = close( movie );