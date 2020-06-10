from rgb_lab_formulation_pytorch import *
import cv2
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
import copy

def processImage(dataset_path,img_name):
    x = cv2.imread(dataset_path+img_name, 1)/255.0
    x = x[:, :, (2, 1, 0)]
    # C, W, H 순으로 바꿈.
    x = x.transpose(2, 0, 1)  
    x = torch.from_numpy(x).float()
    # Tnsor shape : [1,3,160,160]
    x.unsqueeze_(0)
    x = Variable(x.cuda())
    return x

# L의 디테일 강화
def detail_enhance_lab(img, smooth_img):
    # 디테일 강화 parameter
    val0 = 15
    val2 = 1
    exposure = 1.0
    gamma = 1.0
    
    # [1,C,W,H] -> [W,H,C]
    img = img.squeeze().permute(1,2,0)#(2,1,0)
    smooth_img = smooth_img.squeeze().permute(1,2,0)

    # 원래이미지, smooth 이미지 rgb -> lab로 변환
    img_lab=rgb_to_lab(img)    
    smooth_img_lab=rgb_to_lab(smooth_img)
    
    # detail 강화 알고리즘
    img_l, img_a, img_b =torch.unbind(img_lab,dim=2)
    smooth_l, smooth_a, smooth_b =torch.unbind(smooth_img_lab,dim=2)
    
    #원래 이미지 - 스무딩된 이미지 (img_l-smooth_l) : 둘의 차이가 detail이라고 볼 수 있음.
    diff = sig((img_l-smooth_l)/100.0,val0)*100.0
    
    # 스무딩 된 이미지 조절
    base = (sig( (exposure*smooth_l-56.0)/100.0,val2 ) *100.0)+56.0
    # ress :디테일 강화 이미지 = 디테일 강화 + 스무딩(structure 보존)
    res = base + diff
    
    img_l = res
    # a,b는 그대로 반환
    img_a = img_a
    img_b = img_b
    img_lab = torch.stack([img_l, img_a, img_b], dim=2)
    
    # Lab to RGB
    L_chan, a_chan, b_chan = preprocess_lab(img_lab)
    img_lab = deprocess_lab(L_chan, a_chan, b_chan)
    img_final = lab_to_rgb(img_lab)

    return img_final

def sig(x,a):
    # sigmoid 적용 => 이미지 [0-1] => rescale
    y = 1./(1+torch.exp(-a*x)) - 0.5

    y05 = 1./(1+torch.exp(-torch.tensor(a*0.5,dtype=torch.float32))) - 0.5
    y = y*(0.5/y05)

    return y 


def recreate_image(im_as_var):
    
    if im_as_var.shape[0] == 1:
        recreated_im = copy.copy(im_as_var.cpu().data.numpy()[0]).transpose(1,2,0)
    else:    
        recreated_im = copy.copy(im_as_var.cpu().data.numpy())
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)
    # RBG -> GBR
    recreated_im = recreated_im[..., ::-1]
    return recreated_im    


def PreidictLabel(x, classifier):
    new_x= x.contiguous() 
    # prob, logit
    h_x, logit_x = classifier.forward(new_x) # 원래 모델의 logit값도 반환받음.
    h_x = h_x.data.squeeze()
    probs_x, idx_x = h_x.sort(0, True)
    
    # 정답 idx
    class_x = idx_x[0]

    return class_x, logit_x
     
def AdvLoss(logits, target, num_classes=10, kappa=0):
    target_one_hot = torch.eye(num_classes).type(logits.type())[target.long()]

    # https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
    
    #정답 class logit
    real = torch.sum(target_one_hot*logits, 1)
    '''
        (1-target_one_hot) * logit : 정답 class 제외한 나머지 logit
        
        target_logit에 음수 큰값 주고, max 뽑으면 2번째로 큰 logit을 가지는 class
    '''
    other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)

    return torch.sum(torch.max(real-other, kappa))
