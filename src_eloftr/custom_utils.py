import cv2
import torch 
import numpy as np
from PIL import Image


def extract_correspondences_original(img0_pth, img1_pth, matcher,  precision='fp16', model_type = 'full'):

    img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)

    img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))  # input size shuold be divisible by 32
    img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32*32, img1_raw.shape[0]//32*32))

    if precision == 'fp16':
        img0 = torch.from_numpy(img0_raw)[None][None].half().cuda() / 255.
        img1 = torch.from_numpy(img1_raw)[None][None].half().cuda() / 255.
        # img1 = torch.concat((img1, img1), 0)
        # img0 = torch.from_numpy(img0_raw)[None].half().permute(0,3,1,2).cuda() / 255. # self.half() is equivalent to self.to(torch.float16)
        # img1 = torch.from_numpy(img1_raw)[None].half().permute(0,3,1,2).cuda() / 255.

    batch = {'image0': img0, 'image1': img1}

    # Inference with EfficientLoFTR and get prediction
    with torch.no_grad():
        if precision == 'mp':
            with torch.autocast(enabled=True, device_type='cuda'):
                matcher(batch)
        else:
            matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()
    
    # if model_type == 'opt':
    #     print(mconf.max())
    #     mconf = (mconf - min(20.0, mconf.min())) / (max(30.0, mconf.max()) - min(20.0, mconf.min()))

    # color = cm.jet(mconf)
    # text = [
    #     'LoFTR',
    #     'Matches: {}'.format(len(mkpts0)),
    # ]

    # fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)

    return len(mkpts0) # number of corresponeces between 2 images


def extract_correspondences(img0_raw, img1_raw, matcher, precision='fp16', model_type = 'full'):
    '''
    input templates as rgb images ( not transform) and sam proposals as rgb image
    output: max number of corres outof 42 templates- jsut 42 templates to save time
    '''

    # img0_raw = torch.tensor(np.array(Image.open(img0_pth))/255.0).permute(2,0,1).cuda()
    # img1_raw = torch.tensor(np.array(Image.open(img1_pth))/255.0).permute(2,0,1).cuda()

    #convert to gray
    img0_raw = 0.2989 * img0_raw[0] + 0.5870 * img0_raw[1] + 0.114 * img0_raw[2]
    img1_raw = 0.2989 * img1_raw[0] + 0.5870 * img1_raw[1] + 0.114 * img1_raw[2]

    # img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
    # img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)

    # img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))  # input size shuold be divisible by 32
    # img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32*32, img1_raw.shape[0]//32*32))

    if precision == 'fp16':
        img0 = img0_raw[None][None].half() # self.half() is equivalent to self.to(torch.float16)
        # img0 = torch.concat((img0, img0), 0)
        img1 = img1_raw[None][None].half()
        # img1 = torch.concat((img1, img1), 0)
        # img0 = torch.from_numpy(img0_raw)[None].half().permute(0,3,1,2).cuda() / 255. # self.half() is equivalent to self.to(torch.float16)
        # img1 = torch.from_numpy(img1_raw)[None].half().permute(0,3,1,2).cuda() / 255.

    batch = {'image0': img0, 'image1': img1}

    # Inference with EfficientLoFTR and get prediction
    with torch.no_grad():
        if precision == 'mp':
            with torch.autocast(enabled=True, device_type='cuda'):
                matcher(batch)
        else:
            matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy() # match point for image 0 i believe
        mkpts1 = batch['mkpts1_f'].cpu().numpy() # match point for image 1 i believe
        mconf = batch['mconf'].cpu().numpy()
    
    # if model_type == 'opt':
    #     print(mconf.max())
    #     mconf = (mconf - min(20.0, mconf.min())) / (max(30.0, mconf.max()) - min(20.0, mconf.min()))

    # color = cm.jet(mconf)
    # text = [
    #     'LoFTR',
    #     'Matches: {}'.format(len(mkpts0)),
    # ]

    # fig = make_matching_figure(np.array(Image.open(img0_pth)), np.array(Image.open(img1_pth)), mkpts0, mkpts1, color, text=text)

    return len(mkpts0) # number of corresponeces between 2 images