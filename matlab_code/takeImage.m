function im=takeImage(image_file_name_prefix,k)

imRGB=imread(sprintf('%s%04d.pgm',image_file_name_prefix,k));
im=imRGB(:,:,1);
