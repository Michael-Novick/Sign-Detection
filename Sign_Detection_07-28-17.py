# Michael Novick
# July 2017
# Some code sourced from Dr. Steve Mitchell detection.py code
# Some code sourced from Adrian Rosebrock, pyimagesearch.com
# References
# http://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
# http://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/

import cv2
import os
import numpy as np
from PIL import Image
import pytesseract
import imutils


def main():
    video_name = 'testvideo2.mp4'
    f = open('sign_data_' + video_name.rstrip('.mp4') + '.txt', 'w')

    width, height, cap = initialize_video_stream(video_name)
    scale = 1
    width = int(scale * width)
    height = int(scale * height)
    screen_area = width * height
    mask = create_mask(width, height)

    while cap.isOpened():
        count = 0
        ret = 1

        interval = 2
        # Loop causes script to analyze every 'interval' frame. This is only implemented as a recycled segment from
        # the GPS code and may not be necessary here. It can sometimes be handy so I have kept it for now
        while count < interval:
            ret, frame = cap.read()
            count += 1
        # Used to end the loop if there are no more frames. This is a consequence of using the counting loop to control
        # the analyzed frame. Using a VideoCapture class method may eliminate this requirement.
        if ret == 0:
            break

        frame = imutils.resize(frame, width, height)

        contours, edged = detect_signs(mask, frame)

        sign_rect, read = sign_return(screen_area, contours)

        if read is True:
            text = detect_text(sign_rect, edged, f)

        # press the q key to break out of video
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # clear everything once finished
    cap.release()
    cv2.destroyAllWindows()
    f.close()


def initialize_video_stream(video_name):
    # Identify video file (or video stream)
    cap = cv2.VideoCapture(video_name)
    # Print video dimensions, used for determination of snipped window
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(width, height)
    return width, height, cap


def create_mask(width, height):
    mask = np.zeros((height, width), dtype="uint8")
    # These points are used to make a mask of triangles at the left and right sides, with vertices at the corners and
    # a vertex each at roughly the vanishing point on the horizon (vertical position of vanishing point changes with
    # road incline and horizontal position changes with road curvature).
    # pts_left = np.array([[0, 0], [0, height - 1], [width / 2 - 1, 3 * height / 4 - 1]], dtype=np.int32)
    # pts_right = np.array([[width - 1, 0], [width - 1, height - 1], [width / 2, 3 * height / 4]], dtype=np.int32)

    # These points are used to make a mask of trapezoids left and right sides, with vertices at the corners and
    # the shorter parallel side vertical a quarter of the screen in on each side.
    # This is used to just look at the signs when they get close to the camera, when there is sufficient resolution to
    # # read them. It is unconfirmed which mask is better; both are experimental.
    pts_left = np.array([[0, 0], [0, height - 1], [width / 4 - 1, 7 * height / 8 - 1],
                         [width / 4 - 1, 3 * height / 8 - 1]], dtype=np.int32)
    pts_right = np.array([[width - 1, 0], [width - 1, height - 1], [3 * width / 4, 7 * height / 8],
                          [3 * width / 4, 3 * height / 8]], dtype=np.int32)

    cv2.fillConvexPoly(mask, pts_left, 255)
    cv2.fillConvexPoly(mask, pts_right, 255)
    # cv2.imshow("Mask", mask)
    return mask


def detect_signs(mask, frame):
    snip = frame
    masked = cv2.bitwise_and(snip, snip, mask=mask)
    gray = masked.copy()
    cv2.imshow('frame',frame)
    # Convert to grayscale
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # Apply Threshold
    binary_frame = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)[1]
    # binary_frame = cv2.medianBlur(binary_frame, 5)
    # gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # edged = cv2.Canny(binary_frame, 30, 200)
    edged = binary_frame
    # cv2.imshow("gray", binary_frame)
    cv2.imshow("Snip", edged)

    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    screen_cnt = None

    return contours, edged


def detect_text(sign_rect, frame, f):
    # Save as temporary image. This procedure is adapted from Adrian Rosebrock:
    # http://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/
    for rect in sign_rect:
        sign = frame[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0]+rect[2])]
        image = "{}.png".format(os.getpid())
        cv2.imwrite(image, sign)
        binary_image = Image.open(image)
        text = pytesseract.image_to_string(binary_image)
        os.remove(image)
        # Write read text to the open file, named f. This current format outputs "0" as "o" but this can be fixed with
        # some simple string processing.
        # f.write(text + ' \n')
        print(text)
        print()
        return text


def sign_return(screen_area, contours):
    # loop over our contours
    sign_rect = []
    read = False

    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(c)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen

        if len(approx) == 4 and area >= 0.002 * screen_area:
            bounding_rect = cv2.boundingRect(approx)
            sign_rect.append(bounding_rect)
            # break
            read = True

    # cv2.drawContours(snip, [screen_cnt], -1, (0, 255, 0), 3)
    # cv2.imshow("contours", snip)
    # cv2.waitKey(0)
    return sign_rect, read


class Sign:
    def __init__(self):
        pass


class StopSign(Sign):
    pass


class SpeedLimitSign(Sign):
    pass

if __name__ == '__main__':
    main()
