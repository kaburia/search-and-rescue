'''
Run the overhead scheme Megadetector + SAHI
- PytorchWildlife --- MDV6-yolov9-e
- SAHI (512,512) overlap 80%
gdown --id 1TfVarxmQgcN5-nIemN9XNzyf44wNDw7N --folder -O RPi-cam/aerial-survey-data-kws --remaining-ok
'''

import os
import json
from pathlib import Path
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, visualize_object_predictions
from PytorchWildlife.models import detection as pw_detection
from datetime import datetime
import time
import multiprocessing as mp


# Load the detection model
def load_megdet(version="MDV6-yolov9-e", device='cuda'):
    detection_model = pw_detection.MegaDetectorV6(device=device, 
                                              pretrained=True, 
                                              version=version)
    return detection_model

# Configure the Autodetection model for an image
def tiling_with_sahi(image_path,
                model_type='ultralytics',
                model_path="/home/conservacam/.cache/torch/hub/checkpoints/MDV6-yolov9-e-1280.pt",
                confidence_threshold=0.5,
                device="cuda:0",
                slice_height = 512,
                slice_width = 512,
                overlap_height_ratio = 0.7,
                overlap_width_ratio = 0.7
        ):
    

    # Set up the auto detection
    detection_model = AutoDetectionModel.from_pretrained(
                            model_type=model_type,
                            model_path=model_path,
                            confidence_threshold=confidence_threshold,
                            device=device
                        )
    # Get the results for a single image
    result = get_sliced_prediction(
                image_path,
                detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio
    )

    return result

def visualize_predictions(image_path, result, save_dir="outputs-aerial"):
    os.makedirs(save_dir, exist_ok=True)

    image = cv2.imread(image_path)
    vis_path = os.path.join(save_dir, Path(image_path).stem + "_detections.jpg")

    visualize_object_predictions(
        image=image,
        object_prediction_list=result.object_prediction_list,
        rect_thickness=2,
        text_size=1,
        output_path=vis_path
    )
    return vis_path


# Save predictions in JSON format
def save_predictions_json(image_path, result, save_dir="outputs-aerial"):
    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(save_dir, Path(image_path).stem + "_detections.json")
    with open(json_path, "w") as f:
        json.dump(result.to_coco_annotations(), f, indent=2)
    return json_path


# Run pipeline on a folder of images
def run_on_folder(
        folder_path="data/images",
        save_dir="outputs-aerial",
        device="cpu"
):
    

    os.makedirs(save_dir, exist_ok=True)
    image_files = [str(p) for ext in ("*.jpg", "*.JPG", "*.png") for p in Path(folder_path).glob(ext)]
    print(f"Processing {len(image_files)} images...")
    print(folder_path)

    for img in image_files:
        print(f"Processing {img}...")
        try:
            result = tiling_with_sahi(img, device=device)
            save_predictions_json(img, result, save_dir)
            visualize_predictions(img, result, save_dir)  # call only if needed
        except Exception as e:
            print(f"Error with {img}: {e}")

# Worker for processing a single image
def process_single_image(args):
    img, save_dir, device = args
    try:
        result = tiling_with_sahi(img, device=device)
        save_predictions_json(img, result, save_dir)
        visualize_predictions(img, result, save_dir)
        return f"Processed {img}"
    except Exception as e:
        return f"Error with {img}: {e}"


# Run pipeline on a folder of images in parallel
def run_on_folder_parallel(
        folder_path="data/images",
        save_dir="outputs",
        device="cpu"
):
    os.makedirs(save_dir, exist_ok=True)
    image_files = [str(p) for ext in ("*.jpg", "*.JPG", "*.png") for p in Path(folder_path).glob(ext)]
    print(f"Found {len(image_files)} images in {folder_path}")
    # Use ~80% of cores
    num_workers = max(1, int(mp.cpu_count() * 0.8))
    print(f"Processing {len(image_files)} images with {num_workers} workers...")

    args_list = [(img, save_dir, device) for img in image_files]

    with mp.Pool(num_workers) as pool:
        results = pool.map(process_single_image, args_list)

    # Print summary
    for r in results:
        print(r)


if __name__ == '__main__':
    # Starting time
    start_time = time.time()
    folder_path = "/home/conservacam/Desktop/RPi-cam/RPi-cam/aerial-survey-data-kws/EWB Tsavo survey Left observer/L 02-25-2014"
    output_dir = "output_aerial_megdet_sahi"

    # Get the time now
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    print("Current Time =", current_time)
    
    # get the megdet to get the path 
    # load_megdet()

    # # run tiling for one image
    # img_result = tiling_with_sahi('/home/conservacam/Desktop/RPi-cam/RPi-cam/aerial-survey-data-kws/EWB Tsavo survey Left observer/L 02-25-2014/IMG_8716.JPG')

    # # Sve the predictions
    # save_predictions_json('/home/conservacam/Desktop/RPi-cam/RPi-cam/aerial-survey-data-kws/EWB Tsavo survey Left observer/L 02-25-2014/IMG_8716.JPG', 
    #                       img_result, 
    #                       save_dir=f"{output_dir}/{current_time}")
    # visualize_predictions('/home/conservacam/Desktop/RPi-cam/RPi-cam/aerial-survey-data-kws/EWB Tsavo survey Left observer/L 02-25-2014/IMG_8716.JPG',
    #                         img_result, 
    #                         save_dir=f"{output_dir}/{current_time}")

    # # run tiling for a folder of images
    run_on_folder(folder_path=folder_path, 
                  save_dir=f"{output_dir}/{current_time}", 
                  device="cpu")
    
    # parallel
    # run_on_folder_parallel(folder_path=folder_path,
    #                        save_dir=f"{output_dir}/{current_time}_parallel",
    #                        device="cpu")
    
    print(f"Finished in {time.time() - start_time:.2f} seconds")





