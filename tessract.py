import cv2
import numpy as np

# C:\Program Files\Tesseract-OCR


def thresh_callback(val):
    threshold = val

    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)

    # contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    print(len(contours))
    additional_bound = 5
    bounds = []
    for i in range(len(contours)):
        color = (0, 0, 255)
        cv2.drawContours(drawing, contours_poly, i, color)
        bounds.append((int(boundRect[i][0] - additional_bound), int(boundRect[i][1] - additional_bound),  # left, up
                      int(boundRect[i][0] + boundRect[i][2]) + additional_bound, int(boundRect[i][1] + boundRect[i][3]) + additional_bound))

        cv2.rectangle(drawing, (int(boundRect[i][0] - additional_bound), int(boundRect[i][1] - additional_bound)),  # left, up
                      (int(boundRect[i][0] + boundRect[i][2]) + additional_bound, int(boundRect[i][1] + boundRect[i][3]) + additional_bound), color, 2)

    print(bounds)
    print(bounds[0][0], bounds[0][1], bounds[0][2], bounds[0][3])  # x1, y1, x2, y2

    for i in range(len(bounds)):
        roi = src_gray[bounds[i][1]:bounds[i][3], bounds[i][0]:bounds[i][2]]  # y1:y2, x1:x2
        cv2.imwrite("data_num_recog/roi/roi{0}.jpg".format(i), roi)
    # cv2.imshow('Contours', drawing)


if __name__ == "__main__":
    src = cv2.imread('data_num_recog/img_100.jpg')
    # src = cv2.resize(src, (600, 360))

    # Convert image to gray and blur it
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.blur(src_gray, (3, 3))


    thresh = 100  # initial threshold
    thresh_callback(thresh)

    # Create Window
    source_window = 'Source'
    cv2.namedWindow(source_window)
    cv2.imshow(source_window, src_gray)

    cv2.waitKey()
