import cv2
import os
from glob import glob


def video2imgs(root_path, video_name):
    video_path = os.path.join(root_path, video_name)
    output_name = video_name[:-4]
    output_folder_name = os.path.join(root_path, output_name)
    # print(output_folder_name)

    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    # Load video
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(f'{output_folder_name}\{"%07d"%count}.jpg', image)     # save frame as JPEG file
      success,image = vidcap.read()
      print(f'{count}, Read a new frame: ', success)
      count += 1

    print("finish! convert video to frame")


def imgs2video(img_folder, root_path, out_video_name, fps=25):
    pathOut = os.path.join(root_path, out_video_name)

    imgs_list = glob(img_folder + '/*jpg')                # .jpg
    # imgs_list = glob(img_folder + '/*png')                  # .png
    print(f'Total number of the images: {len(imgs_list)}')

    fps = fps
    frame_array = []
    for idx, img_path in enumerate(imgs_list):
        # if (idx % 2 == 0) | (idx % 5 == 0):
        #     continue
        img = cv2.imread(img_path)
        w, h, ch = img.shape
        img = cv2.resize(img, (640, 480))
        # image_size_ratio = image_size_ratio
        # dst = cv2.resize(img, dsize=(0, 0), fx=image_size_ratio, fy=image_size_ratio, interpolation=cv2.INTER_AREA)
        # width, height, channel = dst.shape
        # print(img.shape, dst.shape)
        size = (w, h)
        frame_array.append(img)
    size = (640, 480)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
   
    for i in range(len(frame_array), 10):
        # writing to a image array
        out.write(frame_array[i])
        # print(f'{i}, write a new frame')
    out.release()
    print("finish! convert frames to video")


import moviepy.video.io.ImageSequenceClip

if __name__ == '__main__':
    root_path = 'dataset/video'
    video_name = 'CCTV1.mp4'

    # video2imgs(root_path, video_name)

    imgs_path = 'dataset/video/CCTV1_noisy'
    out_video_name = 'CCTV1_noisy_video.mp4'
    pathOut = os.path.join(root_path, out_video_name)
    print(pathOut)
    
    # imgs2video(imgs_path, root_path, out_video_name, fps=10)


    image_files = [os.path.join(imgs_path,img)
               for img in os.listdir(imgs_path)
               if img.endswith(".jpg")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=25)
    clip.write_videofile(pathOut)
    # import os
    # print(os.getcwd())
    # imgs_list = glob(imgs_path + '/*png')
    # # print(imgs_path)
    # print(imgs_list)
    # print(os.listdir('/home/vips/share/Gwanghyun/ITRC/results/swinir_color_dn_noise25'))

