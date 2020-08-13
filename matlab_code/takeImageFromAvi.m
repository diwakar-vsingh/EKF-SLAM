function im_320x240_8bpp = takeImageFromAvi( fileName, frameNum )

im_640x480_RGB = aviread( fileName, frameNum );
im_640x480_8bpp = rgb2gray( im_640x480_RGB.cdata );
im_320x240_8bpp = imresize( im_640x480_8bpp, 0.5 );