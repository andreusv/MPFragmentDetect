import torch
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision.models.detection import maskrcnn_resnet50_fpn
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .utils import overlay_mask, draw_bbox

class MPInference:
    def __init__(self, model_type, weights_path, device, patch_size=640, stride=480):
        self.model_type = model_type
        self.device = device
        self.patch_size = patch_size
        self.stride = stride
        self.model = self._load(weights_path)

        self.unet_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    def _load(self, path):
        if self.model_type == "yolo":
            return YOLO(path).to(self.device)
        
        elif self.model_type == "maskrcnn":
            model = maskrcnn_resnet50_fpn(num_classes=2)
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
            return model.to(self.device).eval()
        
        elif self.model_type == "unet":
            ckpt = torch.load(path, map_location=self.device)
            cfg = ckpt.get('config', {})
            model = smp.Unet(
                encoder_name=cfg.get('encoder', 'timm-efficientnet-b3'),
                encoder_depth=cfg.get('depth', 5),
                decoder_channels=cfg.get('decoder_channels', [256, 128, 64, 32, 16]),
                classes=1
            )
            model.load_state_dict(ckpt['model'])
            return model.to(self.device).eval()

    def _infer_unet_patches(self, img_rgb):
        H, W = img_rgb.shape[:2]

        pad_h = max(0, self.patch_size - H)
        pad_w = max(0, self.patch_size - W)
        img_pad = np.pad(img_rgb, ((0, pad_h + self.patch_size), (0, pad_w + self.patch_size), (0, 0)), mode='constant')

        prob_map = np.zeros(img_pad.shape[:2], dtype=np.float32)
        count_map = np.zeros_like(prob_map)

        # Sliding Window
        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                tile = img_pad[y:y+self.patch_size, x:x+self.patch_size]
                t = self.unet_transform(image=tile)['image'].unsqueeze(0).to(self.device)

                with torch.no_grad():
                    p = torch.sigmoid(self.model(t)).squeeze().cpu().numpy()

                prob_map[y:y+self.patch_size, x:x+self.patch_size] += p
                count_map[y:y+self.patch_size, x:x+self.patch_size] += 1.0

        # Mean of overlapping zones and recover original shape
        prob_map /= np.maximum(count_map, 1e-6)
        return prob_map[:H, :W]

    def predict(self, img_path):
        img_orig = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        
        if self.model_type == "yolo":
            results = self.model(img_orig, conf=0.25)[0]
            return results.plot()

        elif self.model_type == "maskrcnn":
            input_tensor = torch.from_numpy(img_rgb / 255.).permute(2,0,1).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(input_tensor)[0]
            
            res_img = img_orig.copy()
            for i in range(len(out['masks'])):
                if out['scores'][i] > 0.5:
                    mask = out['masks'][i, 0].cpu().numpy()
                    res_img = overlay_mask(res_img, mask, color=(0, 255, 0))
                    res_img = draw_bbox(res_img, out['boxes'][i].cpu().numpy(), "Fragment")
            return res_img

        elif self.model_type == "unet":
            prob_mask = self._infer_unet_patches(img_rgb)
            binary_mask = (prob_mask > 0.467).astype(np.uint8)
            return overlay_mask(img_orig, binary_mask, color=(255, 0, 255))
