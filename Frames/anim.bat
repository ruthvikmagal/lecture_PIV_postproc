ffmpeg -framerate 60 -r 60 -i convergence_%d.png -pix_fmt yuv420p -profile:v high -level:v 4.1 -crf:v 20 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -movflags +faststart conv60fps.mp4

ffmpeg -framerate 30 -r 30 -i convergence_%d.png -pix_fmt yuv420p -profile:v high -level:v 4.1 -crf:v 20 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -movflags +faststart conv30fps.mp4