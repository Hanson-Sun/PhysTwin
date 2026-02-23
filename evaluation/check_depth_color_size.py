import os, glob, numpy as np
from PIL import Image
import cv2

base = './data/different_types/double_stretch_sloth_da3'
depth_root = os.path.join(base, 'depth')
color_root = os.path.join(base, 'color')

cams = sorted([d for d in os.listdir(depth_root) if os.path.isdir(os.path.join(depth_root,d))])
if not cams:
    print('No camera subfolders found in', depth_root); raise SystemExit(1)

for cam in cams:
    depth_dir = os.path.join(depth_root, cam)
    color_dir = os.path.join(color_root, cam)
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.lower().endswith('.npy')])
    print(f'--- Camera {cam} ---')
    if not depth_files:
        print('  No depth files found in', depth_dir); continue

    # Determine canonical depth resolution from first depth file
    d0 = np.load(os.path.join(depth_dir, depth_files[0]))
    DH, DW = d0.shape[:2]
    print(f'  Depth resolution (H,W): {DH} x {DW}  |  Depth frames: {len(depth_files)}')

    matched = 0
    mismatched = 0
    missing = 0
    examples = []

    for df in depth_files:
        name = os.path.splitext(df)[0]
        img_path = os.path.join(color_dir, f'{name}.png')
        if not os.path.exists(img_path):
            missing += 1
            if len(examples) < 5: examples.append(('missing', img_path))
            continue
        try:
            im = Image.open(img_path)
            IW, IH = im.size
        except Exception as e:
            mismatched += 1
            if len(examples) < 5: examples.append(('read_error', img_path, str(e)))
            continue
        if (IH, IW) == (DH, DW):
            matched += 1
        else:
            mismatched += 1
            if len(examples) < 5: examples.append(('mismatch', img_path, (IH,IW)))
    print(f'  PNG frames: matched={matched}, mismatched={mismatched}, missing={missing}')
    if examples:
        print('  Examples:')
        for ex in examples:
            print('   -', ex)

    # Check video resolution if exists
    video_path = os.path.join(color_root, f'{cam}.mp4')
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if (h,w) == (DH,DW):
            print(f'  Video {video_path} resolution matches depth: (H,W) = {h} x {w}')

    print("Done for camera", cam)

