# -*- coding: utf-8 -*-
from detect_normal import *    

from load_model import combine_model


def combine_crop(path, device, half, model_1, model_2):
    img_tmp = cv2.imread(path) 
    imgsz = 640
    img = crop_plate(img_ = img_tmp, imgsz_ = imgsz, device_ = device, model_ = model_1, half_ = half)   
    txt = crop_char_1(img_ = img, device_ = device, model_ = model_2, half_ = half, imgsz_ = 640, iou_thres = 0.5, conf_thres = 0.4)
    if txt != '無車牌' and txt != '無字元':
        txt = crop_char_1_fix(x_ = txt)
    if len(txt) != 6:
        imgsz = 960
        img = crop_plate(img_ = img_tmp, imgsz_ = imgsz, device_ = device, model_ = model_1, half_ = half)   
        txt_ = crop_char_1(img_ = img, device_ = device, model_ = model_2, half_ = half, imgsz_ = 640, iou_thres = 0.5, conf_thres = 0.4)
        if txt_ != '無車牌' and txt_ != '無字元':
            txt_ = crop_char_1_fix(x_ = txt_)
        if len(txt_) != 6:
            imgsz = 320
            img = crop_plate(img_ = img_tmp, imgsz_ = imgsz, device_ = device, model_ = model_1, half_ = half)   
            txt_ = crop_char_1(img_ = img, device_ = device, model_ = model_2, half_ = half, imgsz_ = 640, iou_thres = 0.5, conf_thres = 0.4)
            if txt_ != '無車牌' and txt_ != '無字元':
                txt_ = crop_char_1_fix(x_ = txt_)       
            if len(txt_) != 6: 
                if txt == '無車牌' or txt == '無字元':
                    txt = txt
                else:
                    txt = txt[txt[:, 4].sort(descending=True)[1]][:6,:]
                    txt = crop_char_1_name(x_ = txt, model_ = model_2)
                    if len(txt) == 6:
                        txt = txt
                    else:
                        txt = "無法定位字元"                
            else:
                txt = crop_char_1_name(x_ = txt_, model_ = model_2)     
        else:
            txt = crop_char_1_name(x_ = txt_, model_ = model_2)    
    else:
        txt = crop_char_1_name(x_ = txt, model_ = model_2)    
    return txt


def crop_char_1_name(x_, model_):
    # Get names
    names = model_.module.names if hasattr(model_, 'module') else model_.names   
    
    s = ''
    for c in list(map(lambda x:int(x), x_[:,5].cpu().detach().numpy().tolist())):
        s += '%s' % (names[int(c)])
    return s


def crop_char_1_fix(x_):
    ind = x_[:,0].cpu().detach().numpy().tolist()
    unique, counts = np.unique(ind, return_counts=True)
    list_muti = unique[counts>1].tolist()
    for i in list_muti:
        tmp = x_[x_[:,0] != i]
        tmp_ = x_[x_[:,0] == i]
        tmp_ = tmp_[torch.argmax(tmp_[:,4])]
        x_ = torch.cat((tmp,tmp_.reshape(1, -1)),0)
    return x_[x_[:, 0].sort()[1]]


def crop_char_1(img_, device_ , model_, half_, imgsz_ = 640, iou_thres = 0.5, conf_thres = 0.4):
    if img_.size == 0:
        return "無車牌"
    
    else:
        imgsz_ = check_img_size(imgsz_, s = model_.stride.max())  # check img_size
        if half_:
            model_.half()  # to FP16    
    
        im0 = cv2.resize(img_, (79,33), interpolation = cv2.INTER_AREA)
    
        # Padded resize
        img = letterbox(im0, new_shape=imgsz_)[0]
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
    
        # start prediction
        img = torch.from_numpy(img).to(device_)
        img = img.half() if half_ else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        pred = model_(img, augment = False)[0]
    
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
    
        det = pred[0]
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            return det
        else:
            return "無字元"


def combine_crop_old(path, device, half, model_1, model_2):
    img_tmp = cv2.imread(path) 
    imgsz = 640
    img = crop_plate(img_ = img_tmp, imgsz_ = imgsz, device_ = device, model_ = model_1, half_ = half)   
    txt = crop_char(img_ = img, device_ = device, model_ = model_2, half_ = half)
    if len(txt)==6:
        return txt
    else:
        imgsz = 320
        img = crop_plate(img_ = img_tmp, imgsz_ = imgsz, device_ = device, model_ = model_1, half_ = half)
        txt = crop_char(img_ = img, device_ = device, model_ = model_2, half_ = half)
        if len(txt)==6:
            return txt
        else:
            imgsz = 960
            img = crop_plate(img_ = img_tmp, imgsz_ = imgsz, device_ = device, model_ = model_1, half_ = half)
            txt = crop_char(img_ = img, device_ = device, model_ = model_2, half_ = half)
            if len(txt)==6:
                return txt
            elif txt != "僅限用6碼車牌":
                return txt
            else:         
                return "查無車牌"


def crop_char(img_, device_ , model_, half_, imgsz_ = 640, iou_thres = 0.5, conf_thres = 0.4, max = True):
    if img_.size == 0:
        return "無法定位車牌"
    
    else:
        imgsz_ = check_img_size(imgsz_, s = model_.stride.max())  # check img_size
        if half_:
            model_.half()  # to FP16    
    
        im0 = cv2.resize(img_, (79,33), interpolation = cv2.INTER_AREA)
    
        # Get names
        names = model_.module.names if hasattr(model_, 'module') else model_.names
    
        # Padded resize
        img = letterbox(im0, new_shape=imgsz_)[0]
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
    
        # start prediction
        img = torch.from_numpy(img).to(device_)
        img = img.half() if half_ else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        pred = model_(img, augment = False)[0]
    
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
    
        det = pred[0]
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            if max:
                det = det[:6,:]
            det = det[det[:, 0].sort()[1]]
            s = ''
            for c in list(map(lambda x:int(x), det[:,5].cpu().detach().numpy().tolist())):
                s += '%s' % (names[int(c)])
            return s
        else:
            return "無法定位字元"


def crop_plate(img_, imgsz_, device_ , model_, half_, per_scale = False, iou_thres = 0.5, conf_thres = 0.4):
    imgsz_ = check_img_size(imgsz_, s = model_.stride.max())  # check img_size
    if half_:
        model_.half()  # to FP16    
    
    im0 = cv2.resize(img_, (imgsz_,int(imgsz_/img_.shape[1]*img_.shape[0])), interpolation = cv2.INTER_AREA)
        
    # Padded resize
    img = letterbox(im0, new_shape=imgsz_)[0]
        
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    
    # start prediction
    img = torch.from_numpy(img).to(device_)
    img = img.half() if half_ else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Inference
    pred = model_(img, augment = False)[0]
    
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    det = pred[0]
    
    if det is not None and len(det):        
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        det = det[list(map(lambda x:np.argmax(x[:,4].cpu().detach().numpy(), axis=0), pred))[0]]
        # 裁切區域的 x 與 y 座標（左上角）
        x = int(det[0].item())
        y = int(det[1].item())    
        # 裁切區域的長度與寬度
        w = int(det[2].item())-int(det[0].item())
        h = int(det[3].item())-int(det[1].item())
        # 裁切圖片
        if per_scale:
            crop_img = im0[int(y-h*per_scale):int(y+h*(1+per_scale)), int(x-w*per_scale):int(x+w*(1+per_scale))]
        else:
            crop_img = im0[y:y+h, x:x+w]
        return crop_img    
    else:
        return np.array([])

