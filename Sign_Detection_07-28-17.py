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
import time

# Define main method
def main():
    # User specifies video file
    video_name = 'testvideo2.mp4'

    # This line initializes a text file to store OCR strings, and is automatically named by referencing the video name.
    # The user must change the video extension in the following line if it varies from the previous video extension.
    f = open('sign_data_' + video_name.rstrip('.mp4') + '.txt', 'w')

    # These are the video dimensions and VideoCapture object that are returned from the method
    width, height, cap = initialize_video_stream(video_name)

    # This scale factor is used to scale the video resolution. There is some rounding or truncation error that can
    # sometimes cause the script to fail, because the mask and frame will be of different sizes. This is an uncommon
    # but possible error.
    scale = 1

    # width and height are scaled, and the screen area is determined.
    width = int(scale * width)
    height = int(scale * height)
    screen_area = width * height

    # A mask is created using the defined create_mask method. A binary image of the mask is returned
    mask = create_mask(width, height)

    # This loop is used to progress through the frames of the VideoCapture object
    while cap.isOpened():

        # count, an iterator, is reset before each new frame. ret is reset to 1.
        count = 0
        ret = 1

        # Interval is user-defined and is used to change the number of frames the script will analyze. It may not always
        # be necessary to check every single frame.

        interval = 1
        # Loop causes script to analyze every 'interval' frame. This is only implemented as a recycled segment from
        # the GPS code and may not be necessary here. It can sometimes be handy so it has been retained for now.
        while count < interval:
            ret, frame = cap.read()
            count += 1

        # Used to end the loop if there are no more frames. This is a consequence of using the counting loop to control
        # the analyzed frame. Using a VideoCapture class method may eliminate this requirement.
        if ret == 0:
            break

        # This line resizes the image using the scaled width and height
        frame = imutils.resize(frame, width, height)

        # The method is called to apply the mask to the frame and return the detected contours and frame
        contours, edged = detect_signs(mask, frame)

        # This method takes the screen area and contours and returns positions for detected signs, and a flag (read)
        # that determines whether or not the script should read for text
        sign_rect, read = sign_return(screen_area, contours)

        # This statement will look for text
        if read is True:
            # a string of text is returned from this method, which gets passed the sign coordinates, frame, and text
            # file
            text = detect_text(sign_rect, edged, f)
            # Text is currently not able to be written to the file, which may refer to different encoding types between
            # Python's write() function. The issue is unclear at the moment

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
    # Retrieve and print video dimensions, used for determination of snipped window
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(width, height)
    return width, height, cap


def create_mask(width, height):
    # Initialize the mask size
    mask = np.zeros((height, width), dtype="uint8")

    # These variables represent the height of the horizon between 0 and 1 and are used for determining the mask
    # geometry. Heights between 0.5 and 0.75 seem typical, and are dependent largely on camera FOV, position, and road
    # incline.
    vanishing_pt_height_percentage = 0.75  # 0.5
    vp = vanishing_pt_height_percentage

    # The correct mask for is unclear at the moment, so below are multiple mask styles commented out for
    # experimentation.

    # ** TRIANGLES **
    # These points are used to make a mask of triangles at the left and right sides, with vertices at the corners and
    # a vertex each at roughly the vanishing point on the horizon (vertical position of vanishing point changes with
    # road incline and horizontal position changes with road curvature).
    #
    # pts_left = np.array([[0, 0], [0, height - 1], [width / 2 - 1, vp * height - 1]], dtype=np.int32)
    # pts_right = np.array([[width - 1, 0], [width - 1, height - 1], [width / 2, vp * height]], dtype=np.int32)

    # ** TRAPEZOIDS **
    # These points are used to make a mask of trapezoids left and right sides, with vertices at the corners and
    # the shorter parallel side vertical a quarter of the screen in on each side.
    # This is used to just look at the signs when they get close to the camera, when there is sufficient resolution to
    # # read them.

    # pts_left = np.array([[0, 0], [0, height - 1], [width / 4 - 1, (0.5 + 0.5 * vp) * height - 1],
    #                      [width / 4 - 1, (1.5 * vp - 0.5) * height - 1]], dtype=np.int32)
    # pts_right = np.array([[width - 1, 0], [width - 1, height - 1], [3 * width / 4, (0.5 + 0.5 * vp) * height],
    #                       [3 * width / 4, (1.5 * vp - 0.5) * height]], dtype=np.int32)

    # ** TRAPEZOIDS, SNIPPED AT HORIZON **
    # pts_left = np.array([[0, 0], [0, vp * height - 1], [width / 2 - 1, vp * height - 1],
    #                      [width / 2 - 1, (1.5 * vp - 0.5) * height - 1]], dtype=np.int32)
    # pts_right = np.array([[width - 1, 0], [width - 1, vp * height - 1], [width / 2, vp * height],
    #                       [width / 2, (1.5 * vp - 0.5) * height]], dtype=np.int32)

    # ** TRAPEZOIDS, 25% IN AND SNIPPED AT HORIZON **
    # These points make a trapezoidal mask with horizontal lower sides parallel with the horizon line as determined
    # by the vanishing point. They extend 25% towards the center of the camera

    pts_left = np.array([[0, 0], [0, vp * height - 1], [width / 4 - 1, vp * height - 1],
                         [width / 4 - 1, (1.5 * vp - 0.5) * height - 1]], dtype=np.int32)
    pts_right = np.array([[width - 1, 0], [width - 1, vp * height - 1], [3 * width / 4, vp * height],
                           [3 * width / 4, (1.5 * vp - 0.5) * height]], dtype=np.int32)

    # These lines fill the mask
    cv2.fillConvexPoly(mask, pts_left, 255)
    cv2.fillConvexPoly(mask, pts_right, 255)

    # This line shows the mask, if desired
    # cv2.imshow("Mask", mask)
    return mask


def detect_signs(mask, frame):
    snip = frame
    # This line applies the mask to the frame
    masked = cv2.bitwise_and(snip, snip, mask=mask)
    # A copy is made
    gray = masked.copy()
    cv2.imshow('frame', frame)
    # Convert to grayscale
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # Apply threshold to pass only white regions
    binary_frame = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]

    # There are a number of available image processing techniques that I have played with. I have left some here as
    # options
    # binary_frame = cv2.medianBlur(binary_frame, 5)
    # gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # edged = cv2.Canny(binary_frame, 30, 200)
    edged = binary_frame
    # cv2.imshow("gray", binary_frame)
    cv2.imshow("Snip", edged)

    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour. pyimagesearch.com was referenced for this segment.
    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    screen_cnt = None

    return contours, edged


def detect_text(sign_rect, frame, f):
    # Save as temporary image. This procedure is adapted from Adrian Rosebrock:
    # http://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/
    for rect in sign_rect:
        # Create a temporary image to process. Rectangle vertices are used to find the window to evaluate
        sign = frame[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0]+rect[2])]
        image = "{}.png".format(os.getpid())
        cv2.imwrite(image, sign)

        # these two lines are used for checking the read image
        cv2.imshow("OCR window", sign)

        # Apply PIL formatting necessary for tesseract
        binary_image = Image.open(image)
        # Read text by applying tesseract
        text = pytesseract.image_to_string(binary_image)
        os.remove(image)
        # Write read text to the open file, named f. This currently does not function correctly for this program,
        # which may be because of different encoding between Tesseract and the write function in Python. This error is
        # still unclear at this point.
        # f.write(text + ' \n')
        print(text)
        print()
        return text


def sign_return(screen_area, contours):
    # loop over our contours. This segment has been adapted from pyimagesearch.com
    sign_rect = []
    read = False

    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        # Shape information is determined of the contour
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # The area is collected, because we wish to compare the shape area to the screen size.
        area = cv2.contourArea(c)

        # The length is checked to see if the contour is a rectangle, and the area is checked as a portion of the screen
        # area to determine whether the rectangle is large enough to bother checking the text. This scaling factor is
        # user-defined.
        if len(approx) == 4 and area >= 0.002 * screen_area:
            # A bounding rectangle for the contour is determined, and appended to the returned list.
            bounding_rect = cv2.boundingRect(approx)
            sign_rect.append(bounding_rect)
            # break
            read = True
    # plotting functions referenced from pyimagesearch.com
    # cv2.drawContours(snip, [screen_cnt], -1, (0, 255, 0), 3)
    # cv2.imshow("contours", snip)
    # cv2.waitKey(0)
    return sign_rect, read


# These classes were created but not yet utilized. Maybe they will become of use at some point in the near future.
class Sign:
    def __init__(self):
        pass


class StopSign(Sign):
    pass


class SpeedLimitSign(Sign):
    pass

if __name__ == '__main__':
    main()
