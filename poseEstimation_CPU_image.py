# Pose Estimation and Segmental CoM calculation code

# Segmental CoM calculation (Anthropometric data based on tables found in Winter (2005))

# Segments -> Proximal - Distal (BlazePose Topology segment):
# LForeArm -> LElbowAxis - LWrist (14 to 16)
# RForeArm -> RElbowAxis - RWrist (13 to 15)
# LUpperArm -> LGlenoHumeral - LElbowAxis (12 to 14)
# RUpperArm -> RGlenoHumeral - RElbowAxis (11 to 13)
# LThigh -> LGreaterTrochanter - LFemoralCondyles (24 to 26)
# RThigh -> RGreaterTrochanter - RFemoralCondyles (23 to 25)
# LLeg -> LFemoralCondyles - LMedialMalleolus (26 to 28)
# RLeg -> RFemoralCondyles - RMedialMalleolus (25 to 27)

# LThighCoM = (xThigh, yThigh)
# xThigh = x24 + 0.433(x26 - x24)
# yThigh = y24 - 0.433(y24 - y26)

# Open-source Python libraries needed
import cv2
import mediapipe as mp

# Models used
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic


# Pose estimation for one individual in a static image parameter:
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    image = cv2.imread('pose2.jpg')
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    # Draw the connections between the joints landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Enumerate each landmark according to BlazePose Topology and show them as a blue circle
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = image.shape
            print(idx, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    # Show the output image with the pose estimation
    cv2.imshow("Image", image)
    cv2.waitKey(0)


