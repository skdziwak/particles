@echo off
ffmpeg -y -r 30 -f image2 -s 1024x1024 -i tmp/f%%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p result.mp4