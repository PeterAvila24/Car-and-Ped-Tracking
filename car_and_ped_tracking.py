import cv2

img_file = 'Car.jpg'
video = cv2.VideoCapture('carped.mp4')


#Our pre-trained car and ped classifier
car_tracker_file = 'cars.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

#create classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

#Run until video stops
while True:
        
    # Read the current frame
    (read_successful, frame) = video.read()

    # Safe Code
    if read_successful:
            grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect Cars and pedestrians
    carsDetect = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

   # Draw rectangles around cars
    for(x, y, w, h) in carsDetect:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)     

   # Draw rectangles around pedestrians
    for(x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)   


    cv2.imshow('Avila Car Detector', frame)

    key = cv2.waitKey(1)

    if key == 81 or key == 113:
         break


video.release()
print("code complete")