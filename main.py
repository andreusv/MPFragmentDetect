import argparse
import os
from glob import glob
from src.inference import MPInference
from src.utils import create_dir, get_device
import cv2

def main():
    parser = argparse.ArgumentParser(description="MP Fragment Detection Demo")
    parser.add_argument("--model", choices=["yolo", "maskrcnn", "unet"], required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--input_dir", type=str, default="examples")
    args = parser.parse_args()

    device = get_device()
    
    output_dir = os.path.join("predictions", args.model)
    create_dir(output_dir)

    engine = MPInference(args.model, args.weights, device)

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
    img_files = []
    for ext in extensions:
        img_files.extend(glob(os.path.join(args.input_dir, ext)))

    print(f"[*] Processing {len(img_files)} images with {args.model}...")

    for f in img_files:
        name = os.path.basename(f)
        prediction = engine.predict(f)
        
        save_path = os.path.join(output_dir, f"result_{name}")
        cv2.imwrite(save_path, prediction)
        print(f"[+] Saved in: {save_path}")

if __name__ == "__main__":
    main()
