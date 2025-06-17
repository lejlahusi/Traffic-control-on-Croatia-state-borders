import base64
import csv
import os
from datetime import datetime
import requests
import schedule
import time

import cv2
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import paho.mqtt.client as mqtt

places_dict = {
    "gunja": {"ids": [431, 432],"url":"https://m.hak.hr/kamera.asp?g=2&k=206"},
    "zupanja": {"ids": [79, 80],"url":"https://m.hak.hr/kamera.asp?g=2&k=44"},
    "slavonski_brod": {"ids": [195, 196],"url":"https://m.hak.hr/kamera.asp?g=2&k=140"},
    "svilaj": {"ids": [461, 462],"url":"https://m.hak.hr/kamera.asp?g=2&k=211"},
    "stara_gradiska": {"ids": [59, 60],"url":"https://m.hak.hr/kamera.asp?g=2&k=32"},
    "maljevac": {"ids": [429, 430],"url":"https://m.hak.hr/kamera.asp?g=2&k=177"},
    "kamensko": {"ids": [317, 318],"url":"https://m.hak.hr/kamera.asp?g=2&k=192"},
    "arzano": {"ids": [315, 316],"url":"https://m.hak.hr/kamera.asp?g=2&k=193"},
    "vinjani_gornji": {"ids": [994, 995],"url":"https://m.hak.hr/kamera.asp?g=2&k=282"},
    "vinjani_donji": {"ids": [302, 303],"url":"https://m.hak.hr/kamera.asp?g=2&k=39"},
    "novasela/bijaca": {"ids": [201, 202],"url":"https://m.hak.hr/kamera.asp?g=2&k=137"},
    "metkovic": {"ids": [320, 319],"url":"https://m.hak.hr/kamera.asp?g=2&k=136"},
    "klek/neum1": {"ids": [203, 204],"url":"https://m.hak.hr/kamera.asp?g=2&k=138"},
    "zatondoli/neum2": {"ids": [206],"url":"https://m.hak.hr/kamera.asp?g=2&k=139"},
    "brgat": {"ids": [453, 454],"url":"https://m.hak.hr/kamera.asp?g=2&k=208"}
}
mqtt_client_name = "/"
mqtt_broker = "test.mosquitto.org"
mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.connect(mqtt_broker)

# Paths to YOLO files
weights_path = r"yolo_files\yolov4.weights"
config_path = r"yolo_files\yolov4.cfg"
labels_path = r"yolo_files\coco.names"

# Load class labels
with open(labels_path) as f:
    labels = f.read().strip().split("\n")

url = "https://m.hak.hr/kamera.asp?g=2&k=44"

def scrapper():
    """Fetch and save the image."""
    try:
        response = requests.get(url)
    except Exception:
        print("Can't get url.")
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tag = soup.find('img', src=lambda x: x and 'cam.asp?id=79' in x)
    if img_tag:
        img_url = img_tag['src']
        if not img_url.startswith('http'):
            img_url = "https://m.hak.hr/" + img_url.lstrip('/')
        try:
            img_response = requests.get(img_url)
            img_response.raise_for_status()
            with open("static/hak.png", 'wb') as img_file:
                img_file.write(img_response.content)
            print("Image saved successfully as hak.png")
        except Exception as e:
            print(f"Failed to save the image: {e}")
    else:
        print(f"No image found with 'cam.asp?id={id}'.")

def scrapper_ids(granica):
    ids = places_dict.get(granica, {}).get("ids", [])
    
    # If no ids are found for the granica name
    if not ids:
        print(f"No IDs found for {granica}.")
        return

    for id in ids:
        # Construct the URL for the image based on the id
        url = places_dict[granica].get("url")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
        except Exception:
            print("Can't get URL.")
            continue
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the image that corresponds to the 'id'
        img_tag = soup.find('img', src=lambda x: x and f'cam.asp?id={id}' in x)
        
        if img_tag:
            img_url = img_tag['src']
            
            # Ensure the img_url is a full URL
            if not img_url.startswith('http'):
                img_url = "https://m.hak.hr/" + img_url.lstrip('/')
            
            try:
                img_response = requests.get(img_url)
                img_response.raise_for_status()

                # Save the image with the format hak{granica}_{id}.png
                img_filename = f"static/hak_{granica}_{id}.png"
                with open(img_filename, 'wb') as img_file:
                    img_file.write(img_response.content)
                
                print(f"Image saved successfully as {img_filename}")
            except Exception as e:
                print(f"Failed to save the image for id {id}: {e}")
        else:
            print(f"No image found with 'cam.asp?id={id}'.")
def count_cars_ids(granica):
    """
    Count cars in the trapezoidal detection area for each ID in the specified granica.
    :param granica: The name of the granica (key in places_dict).
    """
    if granica not in places_dict:
        print(f"Error: {granica} not found in places_dict.")
        return

    # Fetch URL and IDs for the granica
    granica_data = places_dict[granica]
    ids = granica_data.get("ids", [])

    if not ids:
        print(f"No IDs available for {granica}.")
        return

    vehicle_count_total = 0

    # Load pre-trained YOLO model
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    for image_id in ids:
        image_path = f"static/hak_{granica}_{image_id}.png"

        # Scrape the image for the given granica and ID
        scrapper_ids(granica)  # Assumes this function fetches and saves the image based on ID.

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image for ID {image_id} in {granica}.")
            continue

        image = cv2.resize(image, (1280, 720))
        (H, W) = image.shape[:2]

        # Define trapezoidal points based on the image ID
        trapezoid_points = define_trapezoid(image_id, W, H)  # Define points dynamically.

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []
        vehicles = ["car", "truck", "bus", "motorbike"]
        vehicle_count = 0

        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and labels[class_id] in vehicles:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (center_x, center_y, width, height) = box.astype("int")
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))
                    x2 = x + int(width)
                    y2 = y + int(height)

                    if is_inside_trapezoid((center_x, center_y), trapezoid_points):
                        boxes.append([x, y, x2, y2])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.7)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, x2, y2 = boxes[i]
                cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, labels[class_ids[i]], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                vehicle_count += 1

        # Draw trapezoid and add vehicle count text
        cv2.polylines(image, [np.array(trapezoid_points, dtype=np.int32)], True, (255, 0, 0), 2)
        cv2.putText(image, f"Stationary Vehicles: {vehicle_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Save the processed image
        output_image_path = f"static/output_image_{image_id}.jpg"
        cv2.imwrite(output_image_path, image)

        # Add vehicle count to total
        vehicle_count_total += vehicle_count

        # Log data for this image
        log_file = "traffic_data.csv"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            if csvfile.tell() == 0:
                csvwriter.writerow(["Timestamp", "Granica", "Image ID", "Vehicle Count"])
            csvwriter.writerow([timestamp, granica, image_id, vehicle_count])

    print(f"Total vehicles detected in {granica}: {vehicle_count_total}")
    return vehicle_count_total


def define_trapezoid(image_id, W, H):
    """
    Define trapezoidal detection area points dynamically based on image ID.
    :param image_id: The ID of the image being processed.
    :param W: Width of the image.
    :param H: Height of the image.
    :return: List of points defining the trapezoid.
    """
    # Example logic for defining trapezoid points dynamically
    if image_id in {431, 432}:
        return [
            (int(W * 0.01), int(H)),
            (int(W * 0.9), int(H)),
            (int(W * 0.8), int(H * 0.5)),
            (int(W * 0.4), int(H * 0.5)),
        ]
    elif image_id in {195, 196}:
        return [
            (int(W * 0.2), int(H)),
            (int(W * 1), int(H)),
            (int(W * 1), int(H * 0.3)),
            (int(W * 0.25), int(H * 0.3)),
        ]
    else:
        # Default trapezoid
        return [
            (int(W * 0.09), int(H)),
            (int(W), int(H)),
            (int(W * 0.7), int(H * 0.35)),
            (int(W * 0.35), int(H * 0.35)),
        ]



def is_inside_trapezoid(point, trapezoid_points):
    polygon = np.array(trapezoid_points, dtype=np.int32)
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

def count_cars():
    scrapper()
    """Count cars in the trapezoidal detection area."""
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    image_path = "static/hak.png"
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    image = cv2.resize(image, (1280, 720))
    (H, W) = image.shape[:2]

    trapezoid_points = [
        (int(W * 0.09), int(H)),
        (int(W), int(H)),
        (int(W * 0.7), int(H * 0.35)),
        (int(W * 0.35), int(H * 0.35)),
    ]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    vehicles = ["car", "truck", "bus", "motorbike"]
    vehicle_count = 0

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and labels[class_id] in vehicles:
                box = detection[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype("int")
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))
                x2 = x + int(width)
                y2 = y + int(height)

                if is_inside_trapezoid((center_x, center_y), trapezoid_points):
                    boxes.append([x, y, x2, y2])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.7)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, x2, y2 = boxes[i]
            cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, labels[class_ids[i]], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            vehicle_count += 1

    cv2.polylines(image, [np.array(trapezoid_points, dtype=np.int32)], True, (255, 0, 0), 2)
    cv2.putText(image, f"Stationary Vehicles: {vehicle_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imwrite("static/output_image.jpg", image)

    log_file = "traffic_data.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        if csvfile.tell() == 0:
            csvwriter.writerow(["Timestamp", "Vehicle Count"])
        csvwriter.writerow([timestamp, vehicle_count])

    return vehicle_count

def find_lowest_traffic_intervals():
    log_file = "traffic_data.csv"
    output_file = "lowest_traffic_intervals.csv"
    interval = "30min"
    try:
        data = pd.read_csv(log_file)
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data.set_index('Timestamp', inplace=True)
        data['Week'] = data.index.to_period('W')
        grouped_data = data.groupby('Week')

        results = []
        for week, week_data in grouped_data:
            resampled_data = week_data['Vehicle Count'].resample(interval).sum()
            min_traffic = resampled_data.min()
            min_intervals = resampled_data[resampled_data == min_traffic]
            for interval_start in min_intervals.index:
                results.append({
                    "Week": str(week),
                    "Date": str(interval_start.date()),
                    "Start Time": str(interval_start.time()),
                    "Vehicle Count": int(min_traffic),
                })

        with open(output_file, 'a') as f:
            for entry in results:
                f.write(f"{entry['Week']},{entry['Date']},{entry['Start Time']},{entry['Vehicle Count']}\n")

        return results
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
