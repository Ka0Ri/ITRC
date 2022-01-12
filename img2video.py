import cv2
import os
from glob import glob

root_path = r'D:\Code_Torch\dataset\video'

# video2imgs(root_path, video_name)
imgs_folder = r'D:\Code_Torch\dataset\video\CCTV1'
# imgs_folder = r'D:\Code_Torch\dataset\video\swinir_color_dn_noise25'
out_video_name = 'cctv1_noise.mp4v'

pathOut = os.path.join(root_path, out_video_name)
print(pathOut)

# imgs_list = glob(imgs_folder + '/*png')
imgs_list = glob(imgs_folder + '/*jpg')
print(f'Total number of the images: {len(imgs_list)}')
# print(imgs_list)

fps = 15
frame_array = []
for idx, img_path in enumerate(imgs_list):
    img = cv2.imread(img_path)
    w, h, ch = img.shape
    # image_size_ratio = image_size_ratio
    # dst = cv2.resize(img, dsize=(0, 0), fx=image_size_ratio, fy=image_size_ratio, interpolation=cv2.INTER_AREA)
    # width, height, channel = dst.shape
    # print(img.shape, dst.shape)
    size = (w, h)
    frame_array.append(img)

print(size)
print(len(frame_array))

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(pathOut, fourcc, fps, size)

for i in range(len(frame_array)):
    # writing to a image array
    # print(frame_array[i].shape)
    out.write(frame_array[i])
out.release()
print("finish! convert frames to video")