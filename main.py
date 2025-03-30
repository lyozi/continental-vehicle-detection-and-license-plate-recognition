import cv2
import torch
from ultralytics import YOLO

from add_missing_data import add_missing_data
from util import get_car_for_license_plate, read_license_plate, write_csv
from sort.sort import *
from visualize import *

endResults = {}

obj_tracker = Sort()

# autó és rendszámtábla modelek betöltése
coco_model = YOLO('yolo11n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./sample.mp4')
if not cap.isOpened():
    print("Hiba: A videófájl nem nyitható meg!")
else:
    print("Siker: A videó betöltődött!")

vehicles = [2, 3, 5, 7]

# ONLY FOR TESTING PURPOSES
frame_nmr = -1
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# max_frames = total_frames // 6  # process only 1/6 of the video

while cap.isOpened():
    frame_nmr += 1
    # if frame_nmr >= max_frames:
    #     break
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        endResults[frame_nmr] = {}
        # Run YOLO inference on the frame
        # autok detektalasa
        results = coco_model(frame)[0]
        results_ = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if int(class_id) in vehicles:
                results_.append([x1, y1, x2, y2, score])

        # jarmuvek kovetese / tracking
        track_ids = obj_tracker.update(np.asarray(results_))

        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # a talalt rendszamtabla hozzarendelese egy kocsihoz
            car_x1, car_y1, car_x2, car_y2, car_id = get_car_for_license_plate(license_plate, track_ids)

            # rendszamtabla negyzet kijelolese
            license_plate_box = frame[int(y1):int(y2), int(x1):int(x2), :]

            # kep filterek alkalmazasa a rendszamtablan, hogy konyebb legyen az OCR-nek
            license_plate_crop_gray = cv2.cvtColor(license_plate_box, cv2.COLOR_BGR2GRAY)
            _, license_plate_box_thresholded = cv2.threshold(license_plate_crop_gray, 64, 255,
                                                             cv2.THRESH_BINARY_INV)  # 64 felett 0-zza, alatta 255-re rakja

            # cv2.imshow('OG CROP', license_plate_box)
            # cv2.imshow('THRESH CROP', license_lplate_box_thresholded)

            # OCR alkalmazasa a rendszamtabla leolvasasahoz
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_box)

            if license_plate_text is not None:
                endResults[frame_nmr][car_id] = {
                    'car':
                        {
                            'bbox': [car_x1, car_y1, car_x2, car_y2]
                        },
                    'license_plate':
                        {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                }

        # Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #    break
    else:
        # Break the loop if the end of the video is reached
        break

write_csv(endResults, './test.csv')

add_missing_data()
visualize()
