import os

images_path = '.'
video_name = './output.mp4'
fps = 15

os.system(
    f'ffmpeg -y -framerate {fps} -i "{images_path}/concat%d.png" -c:v libx264 -r {fps} -pix_fmt yuv420p "{video_name}"')