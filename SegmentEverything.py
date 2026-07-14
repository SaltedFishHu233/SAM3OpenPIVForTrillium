import torch
from PIL import Image,ImageOps
import os
import cv2
import numpy as np

from transformers import AutoTokenizer, Sam3Model, Sam3ImageProcessor
from datetime import datetime,timedelta

THRESHOLD = 0.5  # Set the threshold for segmentation mask generation.
MASK_THRESHOLD = 0.5  # Set the threshold for mask generation.

# Segment an input image using a text prompt and the local SAM3 model.
def SegmentThis(image: Image.Image, FileName: str, prompt: str, OutputFolder: str = "OutputImages"):
    # Load the model and processor from the local checkpoint directory.

    print("Image Info:")
    depth, height, width = np.array(image).shape
    print(f"Depth: {depth}")
    print(f"Height: {height}") 
    print(f"Width: {width}")  

    img_with_border = ImageOps.expand(image,border=300,fill='white')

    model = Sam3Model.from_pretrained("./LocalSAM3", device_map="auto")     #auto detect "cuda" or "cpu"
    processor = Sam3ImageProcessor.from_pretrained("./LocalSAM3Processor")
    tokenizer = AutoTokenizer.from_pretrained("./LocalSAM3Tokenizer")
    # Prepare the image and prompt for model inference.
    image_inputs = processor(images=img_with_border, return_tensors="pt").to(model.device)
    text_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs = {**image_inputs, **text_inputs}

    # Run inference without tracking gradients.
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert the model output into segmentation results.
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=THRESHOLD, 
        mask_threshold=MASK_THRESHOLD,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    print(f"Results Ready, found {len(results['masks'])} masks" )
    
    # Report the type of segmentation output for debugging.
    masks = results["masks"]
    
    masks_np = [mask.squeeze().cpu().numpy() for mask in masks]    
    IndivMask = masks_np[0]  # Assuming you want to use the first mask for cropping

    print(f"Saving mask for {FileName}" )

    # Crop the mask to the original image dimensions
    CroppedMask=IndivMask[300:depth+300, 300:height+300]
    IntMask=CroppedMask.astype(np.uint8) * 255
    OutputName=os.path.join(OutputFolder, f"{FileName}_mask.png")
    cv2.imwrite(OutputName, IntMask)

if __name__ == "__main__":
    StartTime = datetime.now()
    # Determine the current working directory and input/output folders.
    CurrentDIR = os.getcwd()
    InputFolder = os.path.join(CurrentDIR, "InputImages")
    OutputFolder = os.path.join(CurrentDIR, "OutputImages")

    # Process each image in the input folder.
    for filename in os.listdir(InputFolder):
        ImagePath = os.path.join(InputFolder, filename)
        PILImage = Image.open(ImagePath).convert("RGB")
        SegmentThis(PILImage, filename.split(".") [0], "Black Airfoil", OutputFolder)

    # End of the main process.
    EndTime = datetime.now()
    print(f"Exit - Processing Time: {EndTime - StartTime}")
