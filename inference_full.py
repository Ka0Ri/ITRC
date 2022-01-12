# -*- coding: utf-8 -*-

import torch
# from torch import nn
# from torch import functional
# from torchvision import datasets, transforms, models
# import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np
import subprocess
import os
from SwinIR.models.network_swinir import SwinIR
from yolov5.detect import run
import re

#For Depth estimation
from torchvision.transforms import Compose
import DPT.util.io as io
from DPT.dpt.models import DPTDepthModel
# from DPT.dpt.midas_net import MidasNet_large
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet

class Final_inference():

    def __init__(self, mode, ckpt_path=None, transform=None, device="cuda:0"):
        """
         input: mode = inference mode ["detection", "denoising", "sr"]
                ckpt_path = path to pretrained model
                transform = preprocessing (depend on each model)
        """

        if(mode == "denoising"):

            self.model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv').to(device)
            param_key_g = 'params'
            try:
                pretrained_model = torch.load(ckpt_path)
                self.model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
            except: print("Loading model failed")
            self.model.eval()

        elif(mode == "sr"):

            self.model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv').to(device)
            param_key_g = 'params_ema'
            pretrained_model = torch.load(ckpt_path)
            self.model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
            try:
                pretrained_model = torch.load(ckpt_path)
                self.model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
            except: print("Loading model failed")
            self.model.eval()

        elif(mode == "detection"):
            self.model_ckpt = ckpt_path

        elif(mode == "depth"):
            net_w = 384
            net_h = 384
            try:
                self.model = DPTDepthModel(
                    path=ckpt_path,
                    backbone="vitb_rn50_384",
                    non_negative=True,
                    enable_attention_hooks=False,
                ).to(device)
            except: print("Loading model failed")
            self.model.eval()
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            transform = Compose(
                [
                    Resize(
                        net_w,
                        net_h,
                        resize_target=None,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method="minimal",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    normalization,
                    PrepareForNet(),
                ]
            )
        
        else:
            print("mode is not supported")

        self.transform = transform
        self.device = device

    def detection_run(self, input_img):

        cv2.imwrite("./dataset/temp/test.jpg", input_img)

        run(weights=self.model_ckpt,  
            source="./dataset/temp",
            project="./results/yolo_detect",
            imgsz=[416, 416], 
            conf_thres=0.5,
            )
        
        exp_listdir = [s[:3] + "0" + s[3:]  for s in os.listdir("./results/yolo_detect")]
        exp_listdir.sort(key=lambda f: int(re.sub('\D', '', f)))
        final_dir = exp_listdir[-1]
        final_dir = final_dir[:2] + final_dir[3:]
      
        detected_image = cv2.imread(os.path.join("./results/yolo_detect", final_dir, "test.png"))
        
        return detected_image
        
    def detection(self, input):

        input.save("./temp/test.png")
        # cv2.imwrite("./temp/test.jpg", input)

        argument = ["--weights", self.model_ckpt, "--img", "416", "--source", "./temp", "--project", "./detect_results"]
        test_file = 'yolov5/detect.py'
        ###### environment problem --> need to resolve
        subprocess.call(['/home/dspserver/anaconda3/envs/condaenv/bin/python3', test_file]+ argument)

        exp_listdir = sorted(os.listdir("./detect_results"))
        # print(exp_listdir)
        final_dir = exp_listdir[-1]

        predicted_image = cv2.imread(os.path.join("./detect_results", final_dir, "test.png"))
        
        return predicted_image

    def sr(self, input_img):

        window_size = 8
         # read image
        img_lq = input_img.astype(np.float32) / 255.
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = self.model(img_lq)
            output = output[..., :h_old * 4, :w_old * 4]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

        return output


    def depth_estimate(self, input_img):
   
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) / 255.0
        img_input = self.transform({"image": img})["image"]

        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)

            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        depth = prediction
        depth_min = depth.min()
        depth_max = depth.max()

        max_val = (2 ** (8 * 2)) - 1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.dtype)

        return out
        

    
    def denoise(self, input_img):


        window_size = 8
        # read image
        img_lq = input_img.astype(np.float32) / 255.
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = self.model(img_lq)
            output = output[..., :h_old * 4, :w_old * 4]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
       
        return output
   

if __name__ == "__main__":

    # test_transform = transforms.Compose([
    #             transforms.Resize((224, 224)),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # test sr
    # enhancer = Final_inference(mode='sr', 
    #         ckpt_path="SwinIR/experiments/pretrained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth")
    
    # img = cv2.imread("dataset/Image_SR_test/005321_jpg.rf.149d4fd7de04bf4b2153cded33d18492.jpg")
    # out_img = enhancer.denoise(img)
    # cv2.imwrite("results/results_sr.jpg", out_img)

    #test denoise
    # enhancer = Final_inference(mode='denoising', 
    #         ckpt_path="SwinIR/experiments/pretrained_models/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth")
    
    # img = cv2.imread("dataset/Image_Denoise_test/005321_jpg.rf.149d4fd7de04bf4b2153cded33d18492.jpg")
    # out_img = enhancer.denoise(img)
    # cv2.imwrite("results/results_denoise.jpg", out_img)

    # test detection
    # detector = Final_inference(mode="detection", ckpt_path="yolov5/runs/train/Mask_yolov5s_results/weights/best.pt")
    # img = cv2.imread("dataset/Image_SR_test/005321_jpg.rf.149d4fd7de04bf4b2153cded33d18492.jpg")
    # out_img = detector.detection_run(img)

    #test depth
    estimator = Final_inference(mode="depth", ckpt_path="DPT/weights/dpt_hybrid-midas-501f0c75.pt")
    img = cv2.imread("dataset/1.png")
    out_img = estimator.depth_estimate(img)
    cv2.imwrite("results/results_depth.png", out_img.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
   



    
  
