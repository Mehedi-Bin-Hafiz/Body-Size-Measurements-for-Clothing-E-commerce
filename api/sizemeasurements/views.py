from django.shortcuts import render
## very important code for server ###
# import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

### important ###
from django.http import HttpResponse
import io
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from PIL import Image
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
import numpy as np
import cv2
from ultralytics import YOLO
import torch

## load a pretrained YOLOv8n model
model = YOLO('yolov8m-pose.pt')

Known_distance = 175
# width of face in the real world or Object Plane
# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# def calculate_size(actual_height_cm):
#     # Determine the size category based on actual_height_cm
#     if actual_height_cm < 155:
#         size = "S"
#     elif 155 <= actual_height_cm < 165:
#         size = "M"
#     elif 165 <= actual_height_cm < 175:
#         size = "L"
#     elif 175 <= actual_height_cm < 180:
#         size = "XL"
#     elif 180 <= actual_height_cm < 185:
#         size = "XL"
#     elif 185 <= actual_height_cm < 190:
#         size = "XXL"
#     else:  # 190 cm and above
#         size = "XXXL"
#
#     return size

def calculate_size(height, chest, sleeve, neck):
    if height < 170:
        if chest < 90:
            return 'XS'
        elif 90 <= chest < 96:
            return 'S'
        elif 96 <= chest < 102:
            return 'M'
        elif 102 <= chest < 108:
            return 'L'
        else:
            return 'XL'
    elif 170 <= height < 180:
        if 90 <= chest < 96:
            return 'S'
        elif 96 <= chest < 102:
            return 'M'
        elif 102 <= chest < 108:
            return 'L'
        elif 108 <= chest < 114:
            return 'XL'
        else:
            return 'XXL'
    else:
        if 96 <= chest < 102:
            return 'M'
        elif 102 <= chest < 108:
            return 'L'
        elif 108 <= chest < 114:
            return 'XL'
        elif 114 <= chest < 120:
            return 'XXL'
        else:
            return 'XXXL'

class SizeMeasurementsVisionAI(APIView):
    parser_classes = (MultiPartParser, FormParser)
    def post(self, request, *args, **kwargs):
        # Check if an image is part of the request
        if 'frontPicture' in request.FILES and 'sidePicture' in request.FILES:
            # Convert the uploaded image file (InMemoryUploadedFile) to a NumPy array
            gender = request.data.get('gender')
            front_image = request.FILES['frontPicture']
            front_image_array = np.frombuffer(front_image.read(), np.uint8)
            person_front_image = cv2.imdecode(front_image_array, cv2.IMREAD_COLOR)  # Use cv2.IMREAD_GRAYSCALE for grayscale images
            side_image = request.FILES['sidePicture']
            side_image_array = np.frombuffer(side_image.read(), np.uint8)
            person_side_image = cv2.imdecode(side_image_array, cv2.IMREAD_COLOR)
            # face detector object
            face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            if gender.lower() == 'female':
                Known_width = 13.51
            else:
                # centimeter
                Known_width = 14.15  # 14.15 for men 13.51 for women
            # focal length finder function
            def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
                focal_length = (width_in_rf_image * measured_distance) / real_width
                return focal_length  # 2000 is threshold

            # distance estimation function
            def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
                distance = (real_face_width * Focal_Length) / face_width_in_frame
                # return the distance
                return distance

            def face_data(image):

                face_width = 0  # making face width to zero

                # converting color image to gray scale image
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # detecting face in the image
                faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

                # looping through the faces detect in the image
                # getting coordinates x, y , width and height
                for (x, y, h, w) in faces:
                    # draw the rectangle on the face
                    cv2.rectangle(image, (x, y), (x + w, y + h), GREEN, 2)
                    # getting face width in the pixels
                    face_width = w

                    # return the face width in pixel

                return face_width

                # find the face width(pixels) in the real word human image

            ref_image_face_width = 534.80  # 534.80 for male 510.61 for female

            # get the focal by calling "Focal_Length_Finder"
            # face width in reference(pixels),
            # Known_distance(centimeters),
            # known_width(centimeters)
            Focal_length_found = Focal_Length_Finder(Known_distance, Known_width, ref_image_face_width)

            # calling face_data function to find
            # the width of face(pixels) in the frame
            face_width_in_frame = face_data(person_front_image)
            # check if the face is zero then not
            # find the distance
            Distance = 0
            if face_width_in_frame != 0:
                # finding the distance by calling function
                # Distance finder function need
                # these arguments the Focal_Length,
                # Known_width(centimeters),
                # and Known_distance(centimeters)
                Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame)
            ## workking with YOLO
            results = model(source=person_front_image, show=False, conf=0.70, save=False)
            bounding_box = results[0].boxes.xyxy[0]  # Detections for the first image in batch
            x1, y1, x2, y2 = bounding_box.tolist()
            height = y2 - y1
            # Convert dimensions from pixels to centimeters
            actual_height_cm = (height * Distance) / (Focal_length_found - 1720)
            # dress_size =calculate_size(actual_height_cm)

            ###  side measurements
            side_results = model(source=person_side_image, show=False, conf=0.70, save=False)
            side_bounding_box = side_results[0].boxes.xyxy[0]  # Detections for the first image in batch
            x1, y1, x2, y2 = side_bounding_box.tolist()
            side_width = x2 - x1
            actual_side_width_cm = ((side_width * Distance) / (Focal_length_found - 1720)) / 2.75
            tensor_values = results[0].keypoints.xy[0]
            # Extracting specific keypoints based on the request
            specific_keypoints_tensors = {
                "Left Shoulder": tensor_values[5],
                "Right Shoulder": tensor_values[6],
                "Left Elbow": tensor_values[7],
                "Right Elbow": tensor_values[8],
                "Left Wrist": tensor_values[9],
                "Right Wrist": tensor_values[10],
                "Left Hip": tensor_values[11],
                "Right Hip": tensor_values[12]
            }
            hip_adjustment = torch.norm(
                specific_keypoints_tensors['Left Hip'] - specific_keypoints_tensors['Right Hip']).item()
            hip_adjustment = hip_adjustment + (hip_adjustment / 2)
            shoulder_adjustment = torch.norm(
                specific_keypoints_tensors['Left Shoulder'] - specific_keypoints_tensors['Right Shoulder']).item()
            shoulder_adjustment = shoulder_adjustment + (shoulder_adjustment / 3)
            ## wrist calculation.
            shoulder_to_neck = shoulder_adjustment / 2
            shouldertoelbow = torch.norm(
                specific_keypoints_tensors['Left Shoulder'] - specific_keypoints_tensors['Left Elbow']).item()
            elbowtowrist = torch.norm(
                specific_keypoints_tensors['Left Elbow'] - specific_keypoints_tensors['Left Wrist']).item()
            necktowirst = shoulder_to_neck + shouldertoelbow + elbowtowrist
            # Calculate distances using Pythagorean theorem
            pixel_distances = {
                'chest': shoulder_adjustment,
                'waist': hip_adjustment,
                'sleeve': necktowirst,
            }
            # Convert distances from pixels to cm
            measurements_cm = {key: (value * Distance) / (Focal_length_found - 1720) for key, value in
                               pixel_distances.items()}
            measurements_cm['neck'] = 2 * (measurements_cm['waist'] - 10.16)
            measurements_cm['chest'] = 2 * (measurements_cm['chest'] + actual_side_width_cm)
            measurements_cm['waist'] = 2 * (measurements_cm['waist'] + actual_side_width_cm)
            measurements_cm['height'] = actual_height_cm
            # Calculate T-shirt size
            tshirt_size = calculate_size(measurements_cm['height'], measurements_cm['chest'], measurements_cm['sleeve'],
                                         measurements_cm['neck'])
            return Response({"size": tshirt_size}, status=200)
        else:
            return Response({"message": "No image uploaded"}, status=400)




class VirtualTry(APIView):
    parser_classes = (MultiPartParser, FormParser)
    def post(self, request, *args, **kwargs):
        # Check if 'avatar' and 'selfie' are in the request files
        if 'avatar' in request.FILES and 'selfie' in request.FILES:
            try:
                # Read the avatar image from the request
                avatar = request.FILES['avatar']
                avatar_image_array = np.frombuffer(avatar.read(), np.uint8)
                avatar_img = cv2.imdecode(avatar_image_array, cv2.IMREAD_COLOR)

                # Read the selfie image from the request
                selfie = request.FILES['selfie']
                selfie_image_array = np.frombuffer(selfie.read(), np.uint8)
                selfie_img = cv2.imdecode(selfie_image_array, cv2.IMREAD_COLOR)

                # Initialize the FaceAnalysis app
                app = FaceAnalysis(name='buffalo_l')
                app.prepare(ctx_id=0, det_size=(640, 640))

                # Detect faces in both images
                face1 = app.get(avatar_img)[0]  # Face in the avatar image
                face2 = app.get(selfie_img)[0]  # Face in the selfie image

                # Load the face swapping model
                swapper = get_model('inswapper_128.onnx', download=False)

                # Perform face swapping
                swapped_img = swapper.get(avatar_img, face1, face2, paste_back=True)

                # Convert the swapped image to RGB for PIL compatibility
                swapped_img_rgb = cv2.cvtColor(swapped_img, cv2.COLOR_BGR2RGB)

                # Convert the image to an in-memory file using BytesIO
                buffer = io.BytesIO()
                Image.fromarray(swapped_img_rgb).save(buffer, format='PNG')
                buffer.seek(0)

                # Return the image as a response
                return HttpResponse(buffer, content_type='image/png')

            except Exception as e:
                return HttpResponse(f"An error occurred: {str(e)}", status=500)

        else:
            return HttpResponse("Please provide both 'avatar' and 'selfie' images.", status=400)
