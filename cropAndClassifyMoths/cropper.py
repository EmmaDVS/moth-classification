import torch
from PIL import Image, ImageOps

def squareAndPadBox(box):
    # Make the crop square by elongating the shortest side
    w = box[2] - box[0]
    h = box[3] - box[1]

    pad_for_square = (max(w, h) - min(w, h)) / 2
    shortest_side = 'w' if w < h else 'h'

    if shortest_side == 'w':
        box[0] = box[0] - pad_for_square
        box[2] = box[2] + pad_for_square
    if shortest_side == 'h':
        box[1] = box[1] - pad_for_square
        box[3] = box[3] + pad_for_square

    # Pad with 10% on all sides
    pad = 0.1 * (box[2] - box[0])
    box[0] = box[0] - pad
    box[1] = box[1] - pad
    box[2] = box[2] + pad
    box[3] = box[3] + pad
    return box

def squareYoloCrop(images,
                   predictions,
                   size=240,
                   multi_crop=False,
                   min_box_score=0.25):        
    for i, image in enumerate(images):
        prediction = predictions[i]
        boxes = prediction[:, :4].cpu().tolist() # x1, y1, x2, y2
        scores = prediction[:, 4]
        box_crops = []
        boxes_valid = len(boxes) >= 1 and max(scores) >= min_box_score
        crop_scores = []
        
        if not boxes_valid:  # Do a centre crop instead
            center_cropped += 1
            w = image.width
            h = image.height
            
            if w > h:
                left = int(w/2-h/2)
                upper = 0
                right = left + h
                lower = h
            if h > w:
                left = 0
                upper = int(h/2-w/2)
                right = w
                lower = upper + w
            if w == h:
                left = 0
                upper = 0
                right = w
                lower = h
            box = (left, upper, right, lower)
            box_crops.append(box)
            crop_scores.append('centre')

        if boxes_valid:
            if multi_crop:
                for score_i, score in enumerate(scores):
                    if score >= min_box_score:
                        box = boxes[score_i]
                        box_crops.append(box)
                        crop_scores.append(score)
            else:
                best_crop_index = torch.argmax(scores).tolist()  # Index with highest score
                box = boxes[best_crop_index]
                box_crops.append(box)
                crop_scores.append(scores[best_crop_index])
            box_crops = list(map(squareAndPadBox, box_crops))
        
        cropped_images = []
        for box_i, box in enumerate(box_crops):
            crop = image.crop(box)
            crop = crop.resize([size, size])
            cropped_images.append(crop)
            
    return box_crops, cropped_images

def crop(model_path, image_path, multi_crop):
    # Load the crop model
    yolo = torch.hub.load("ultralytics/yolov5",
                          "custom",
                          path=model_path/"bestYolo.pt",
                          verbose=False,
                          _verbose=False)

    # Load the image
    im = Image.open(image_path).convert("RGB")
    image_list = [im]

    # Fix image orientation
    im = ImageOps.exif_transpose(im)

    # Predict the crops
    results = yolo(image_list).pred

    # Crop the images
    boxes, images = squareYoloCrop(images=image_list,
                                   predictions=results,
                                   multi_crop=multi_crop)
    return boxes, images
