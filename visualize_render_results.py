import argparse
import csv
import json
import os

import cv2
import numpy as np


def _load_case_names(data_config_path):
    case_names = []
    with open(data_config_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # skip empty lines
                case_names.append(row[0])
    return case_names


def main():
    parser = argparse.ArgumentParser(
        description="Render overlay videos by compositing dynamic renderings on original videos."
    )
    parser.add_argument("--base_path", default="./data/different_types", type=str)
    parser.add_argument(
        "--prediction_dir", default="./gaussian_output_dynamic_white", type=str
    )
    parser.add_argument(
        "--human_mask_path", default="./data/different_types_human_mask", type=str
    )
    parser.add_argument(
        "--object_mask_path", default="./data/render_eval_data", type=str
    )
    parser.add_argument("--data_config", default="data_config.csv", type=str)
    parser.add_argument("--height", default=480, type=int)
    parser.add_argument("--width", default=848, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--alpha", default=0.7, type=float)
    parser.add_argument("--num_views", default=3, type=int)
    args = parser.parse_args()

    case_names = _load_case_names(args.data_config)

    for case_name in case_names:
        print(f"Processing {case_name}!!!!!!!!!!!!!!!")

        split_path = f"{args.base_path}/{case_name}/split.json"
        if not os.path.exists(split_path):
            print(f"[Skip] Missing split file: {split_path}")
            continue

        with open(split_path, "r") as f:
            split = json.load(f)
        frame_len = split["frame_len"]

        for i in range(args.num_views):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_video_path = f"{args.prediction_dir}/{case_name}/{i}_integrate.mp4"
            video_writer = cv2.VideoWriter(
                output_video_path,
                fourcc,
                args.fps,
                (args.width, args.height),
            )

            for frame_idx in range(frame_len):
                render_path = (
                    f"{args.prediction_dir}/{case_name}/{i}/{frame_idx:05d}.png"
                )
                origin_image_path = (
                    f"{args.base_path}/{case_name}/color/{i}/{frame_idx}.png"
                )
                human_mask_image_path = (
                    f"{args.human_mask_path}/{case_name}/mask/{i}/0/{frame_idx}.png"
                )
                object_image_path = (
                    f"{args.object_mask_path}/{case_name}/mask/{i}/{frame_idx}.png"
                )

                render_img = cv2.imread(render_path, cv2.IMREAD_UNCHANGED)
                origin_img = cv2.imread(origin_image_path)
                human_mask = cv2.imread(human_mask_image_path)
                object_mask = cv2.imread(object_image_path)

                if (
                    render_img is None
                    or origin_img is None
                    or human_mask is None
                    or object_mask is None
                ):
                    print(
                        f"[Skip frame] Missing data for {case_name}, view {i}, frame {frame_idx}."
                    )
                    continue

                if render_img.ndim == 2:
                    render_img = cv2.cvtColor(render_img, cv2.COLOR_GRAY2BGRA)
                elif render_img.shape[2] == 3:
                    alpha_channel = np.full(
                        (render_img.shape[0], render_img.shape[1], 1),
                        255,
                        dtype=render_img.dtype,
                    )
                    render_img = np.concatenate([render_img, alpha_channel], axis=2)

                human_mask = cv2.cvtColor(human_mask, cv2.COLOR_BGR2GRAY) > 0
                object_mask = cv2.cvtColor(object_mask, cv2.COLOR_BGR2GRAY) > 0

                final_image = origin_img.copy()
                render_mask = np.logical_and(
                    (render_img != 0).any(axis=2), render_img[:, :, 3] > 100
                )
                render_img[~render_mask, 3] = 0

                final_image[:, :, :] = args.alpha * final_image + (
                    1 - args.alpha
                ) * np.array([255, 255, 255], dtype=np.uint8)

                test_alpha = render_img[:, :, 3] / 255
                final_image[:, :, :] = render_img[:, :, :3] * test_alpha[
                    :, :, None
                ] + final_image * (1 - test_alpha[:, :, None])

                final_image[human_mask] = args.alpha * origin_img[human_mask] + (
                    1 - args.alpha
                ) * np.array([255, 255, 255], dtype=np.uint8)

                video_writer.write(final_image)

            video_writer.release()


if __name__ == "__main__":
    main()
