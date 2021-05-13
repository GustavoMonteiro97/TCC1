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

# Open-source Python libraries needed (OpenCV and Mediapipe)
import cv2
import mediapipe as mp

# Models used
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# Landmarks dictionary
landmarks_coordinates = {}

# Body segments and their coordinates equations
thigh_segments = {
    'Left Thigh': [24, 26],
    'Right Thigh': [23, 25]
}

upperarm_segments = {
    'Left Upper Arm': [12, 14],
    'Right Upper Arm': [11, 13]
}

forearm_segments = {
    'Left Fore Arm': [14, 16],
    'Right Fore Arm': [13, 15]
}

leg_segments = {
    'Left Leg': [26, 28],
    'Right Leg': [25, 27]
}

foot_segments = {
    'Left foot': [28, 32],
    'Right foot': [27, 31]
}

hands_segments = {
    'Left hand': [16, 20],
    'Right hand': [15, 19]
}

shoulder_segment = {
    'Shoulder': [11, 12]
}

hips_segment = {
    'Hips': [23, 24]
}

# trunk_segments = {
#     'trunk': [33, 34]
# }

# totalbody_segments = {
#     'Total Body': [0, 32]
# }


# Function to calculate CoM coordinates for segments defined in the dictionary
# and return the CoM coordinate in the output picture
def calculate_thigh_CoM(x_a, x_b, y_a, y_b):
    x3 = x_a + 0.433 * (x_b - x_a)
    y3 = y_a - 0.433 * (y_a - y_b)

    return x3, y3


def calculate_UpperArm_CoM(x_c, x_d, y_c, y_d):
    x4 = x_c + 0.436 * (x_d - x_c)
    y4 = y_c - 0.436 * (y_c - y_d)

    return x4, y4


def calculate_ForeArm_CoM(x_e, x_f, y_e, y_f):
    x5 = x_e + 0.430 * (x_f - x_e)
    y5 = y_e - 0.430 * (y_e - y_f)

    return x5, y5


def calculate_Leg_CoM(x_g, x_h, y_g, y_h):
    x2 = x_g + 0.433 * (x_h - x_g)
    y2 = y_g - 0.433 * (y_g - y_h)

    return x2, y2


def calculate_foot_CoM(x_i, x_j, y_i, y_j):
    x1 = (x_i + x_j) * 0.5
    y1 = (y_i + y_j) * 0.5

    return x1, y1


def calculate_hands_CoM(x_o, x_p, y_o, y_p):
    x6 = x_o + 0.506 * (x_p - x_o)
    y6 = y_o - 0.506 * (y_o - y_p)

    return x6, y6


def calculate_shoulder_average(x_q, x_r, y_q, y_r):
    x = (x_q + x_r) * 0.5
    y = (y_q + y_r) * 0.5

    return x, y


def calculate_hips_average(x_s, x_t, y_s, y_t):
    x = (x_s + x_t) * 0.5
    y = (y_s + y_t) * 0.5

    return x, y


# def calculate_trunk_CoM(x_q, x_r, y_q, y_r, x_s, x_t, y_s, y_t):
#     x7 = (x_q + x_r) * 0.5 - 0.66 * (((x_q + x_r) * 0.5) - (x_s + x_t) * 0.5)
#     y7 = (y_q + y_r) * 0.5 - 0.66 * (((y_q + y_r) * 0.5) - (y_s + y_t) * 0.5)
#
#     return x7, y7


# def calculate_totalBody_CoM(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5):
#     xr = 0.145 * x1 + 0.0465 * x2 + 0.1 * x3 + 0.028 * x4 + 0.016 * x5
#     yr = 0.145 * y1 + 0.0465 * y2 + 0.1 * y3 + 0.028 * y4 + 0.016 * y5
#
#     return xr, yr


# Pose estimation for one individual in a static image parameter:
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    image = cv2.imread('pose1.png')
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
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
            landmarks_coordinates[idx] = {'x': lm.x, 'y': lm.y}

    # Calculates body segments COM
    for segment in upperarm_segments:
        print('%s COM:' % segment)
        x_c = landmarks_coordinates[upperarm_segments[segment][0]]['x']
        x_d = landmarks_coordinates[upperarm_segments[segment][1]]['x']
        y_c = landmarks_coordinates[upperarm_segments[segment][0]]['y']
        y_d = landmarks_coordinates[upperarm_segments[segment][1]]['y']

        x4, y4 = calculate_UpperArm_CoM(x_c, x_d, y_c, y_d)
        print('x: %s, y: %s\n' % (str(x4), str(y4)))

        h, w, c = image.shape
        cx, cy = int(x4 * w), int(y4 * h)
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    for segment in forearm_segments:
        print('%s COM:' % segment)
        x_e = landmarks_coordinates[forearm_segments[segment][0]]['x']
        x_f = landmarks_coordinates[forearm_segments[segment][1]]['x']
        y_e = landmarks_coordinates[forearm_segments[segment][0]]['y']
        y_f = landmarks_coordinates[forearm_segments[segment][1]]['y']

        x5, y5 = calculate_ForeArm_CoM(x_e, x_f, y_e, y_f)
        print('x: %s, y: %s\n' % (str(x5), str(y5)))

        h, w, c = image.shape
        cx, cy = int(x5 * w), int(y5 * h)
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    for segment in leg_segments:
        print('%s COM:' % segment)
        x_g = landmarks_coordinates[leg_segments[segment][0]]['x']
        x_h = landmarks_coordinates[leg_segments[segment][1]]['x']
        y_g = landmarks_coordinates[leg_segments[segment][0]]['y']
        y_h = landmarks_coordinates[leg_segments[segment][1]]['y']

        x2, y2 = calculate_Leg_CoM(x_g, x_h, y_g, y_h)
        print('x: %s, y: %s\n' % (str(x2), str(y2)))

        h, w, c = image.shape
        cx, cy = int(x2 * w), int(y2 * h)
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    for segment in thigh_segments:
        print('%s COM:' % segment)
        x_a = landmarks_coordinates[thigh_segments[segment][0]]['x']
        x_b = landmarks_coordinates[thigh_segments[segment][1]]['x']
        y_a = landmarks_coordinates[thigh_segments[segment][0]]['y']
        y_b = landmarks_coordinates[thigh_segments[segment][1]]['y']

        x3, y3 = calculate_thigh_CoM(x_a, x_b, y_a, y_b)
        print('x: %s, y: %s\n' % (str(x3), str(y3)))

        h, w, c = image.shape
        cx, cy = int(x3 * w), int(y3 * h)
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    for segment in foot_segments:
        print('%s COM:' % segment)
        x_i = landmarks_coordinates[foot_segments[segment][0]]['x']
        x_j = landmarks_coordinates[foot_segments[segment][1]]['x']
        y_i = landmarks_coordinates[foot_segments[segment][0]]['y']
        y_j = landmarks_coordinates[foot_segments[segment][1]]['y']

        x1, y1 = calculate_foot_CoM(x_i, x_j, y_i, y_j)
        print('x: %s, y: %s\n' % (str(x1), str(y1)))

        h, w, c = image.shape
        cx, cy = int(x1 * w), int(y1 * h)
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    # for segment in trunk_segments:
    #     print('%s COM:' % segment)
    #     x_q = landmarks_coordinates[trunk_segments[segment][0]]['x']
    #     x_r = landmarks_coordinates[trunk_segments[segment][1]]['x']
    #     y_q = landmarks_coordinates[trunk_segments[segment][0]]['y']
    #     y_r = landmarks_coordinates[trunk_segments[segment][1]]['y']
    #     x_s = landmarks_coordinates[trunk_segments[segment][0]]['x']
    #     x_t = landmarks_coordinates[trunk_segments[segment][1]]['x']
    #     y_s = landmarks_coordinates[trunk_segments[segment][0]]['y']
    #     y_t = landmarks_coordinates[trunk_segments[segment][1]]['y']
    #
    #     x7, y7 = calculate_trunk_CoM(x_q, x_r, y_q, y_r, x_s, x_t, y_s, y_t)
    #     print('x: %s, y: %s\n' % (str(x7), str(y7)))
    #
    #     h, w, c = image.shape
    #     cx, cy = int(x7 * w), int(y7 * h)
    #     cv2.circle(image, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    for segment in hands_segments:
        print('%s COM:' % segment)
        x_o = landmarks_coordinates[hands_segments[segment][0]]['x']
        x_p = landmarks_coordinates[hands_segments[segment][1]]['x']
        y_o = landmarks_coordinates[hands_segments[segment][0]]['y']
        y_p = landmarks_coordinates[hands_segments[segment][1]]['y']

        x6, y6 = calculate_hands_CoM(x_o, x_p, y_o, y_p)
        print('x: %s, y: %s\n' % (str(x6), str(y6)))

        h, w, c = image.shape
        cx, cy = int(x6 * w), int(y6 * h)
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    for segment in shoulder_segment:
        # print('%s COM:' % segment)
        x_q = landmarks_coordinates[shoulder_segment[segment][0]]['x']
        x_r = landmarks_coordinates[shoulder_segment[segment][1]]['x']
        y_q = landmarks_coordinates[shoulder_segment[segment][0]]['y']
        y_r = landmarks_coordinates[shoulder_segment[segment][1]]['y']

        x, y = calculate_shoulder_average(x_q, x_r, y_q, y_r)
        # print('x: %s, y: %s\n' % (str(x), str(y)))

        # h, w, c = image.shape
        # cx, cy = int(x * w), int(y * h)
        # cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    for segment in hips_segment:
        # print('%s COM:' % segment)
        x_s = landmarks_coordinates[hips_segment[segment][0]]['x']
        x_t = landmarks_coordinates[hips_segment[segment][1]]['x']
        y_s = landmarks_coordinates[hips_segment[segment][0]]['y']
        y_t = landmarks_coordinates[hips_segment[segment][1]]['y']

        x, y = calculate_hips_average(x_s, x_t, y_s, y_t)
        # print('x: %s, y: %s\n' % (str(x), str(y)))

        # h, w, c = image.shape
        # cx, cy = int(x * w), int(y * h)
        # cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    # for segment in totalbody_segments:
    #     print('%s COM:' % segment)
    #     x1 = landmarks_coordinates[totalbody_segments[segment][0]]['x']
    #     x2 = landmarks_coordinates[totalbody_segments[segment][1]]['x']
    #     y1 = landmarks_coordinates[totalbody_segments[segment][0]]['y']
    #     y2 = landmarks_coordinates[totalbody_segments[segment][1]]['y']
    #     x3 = landmarks_coordinates[totalbody_segments[segment][0]]['x']
    #     x4 = landmarks_coordinates[totalbody_segments[segment][1]]['x']
    #     y3 = landmarks_coordinates[totalbody_segments[segment][0]]['y']
    #     y4 = landmarks_coordinates[totalbody_segments[segment][1]]['y']
    #     x5 = landmarks_coordinates[totalbody_segments[segment][0]]['x']
    #     y5 = landmarks_coordinates[totalbody_segments[segment][1]]['y']
    #     xr, yr = calculate_totalBody_CoM(x1, x2, y1, y2, x3, x4, y3, y4, x5, y5)
    #     print('x: %s, y: %s\n' % (str(xr), str(yr)))
    #
    #     h, w, c = image.shape
    #     cx, cy = int(xr * w), int(yr * h)
    #     cv2.circle(image, (cx, cy), 5, (153, 0, 151), cv2.FILLED)

    # Show the output image with the pose estimation
    cv2.imshow("Image", image)
    cv2.waitKey(0)
