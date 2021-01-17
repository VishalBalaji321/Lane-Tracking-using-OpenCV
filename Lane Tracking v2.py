import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot


class LaneTracking:
    def __init__(self, path, mode='Video', needGridPlot=False, debug=-1):
        self.debug = debug

        if mode == 'Video':
            # Opening the video using the inbuilt OpenCV Function
            cap = cv2.VideoCapture(path)
            while cap.isOpened():
                ret, self.image = cap.read()
                if ret:
                    self.process_frame()
                    if self.waitkey(1):
                        break
            cap.release()
            cv2.destroyAllWindows()

        elif mode == 'Image':
            # Opening the image using the inbuilt OpenCV Function
            self.image = cv2.imread(path)
            self.process_frame()
            dummy = self.waitkey(0)

        else:
            print('Wrong Mode....Please try again')
            # Raise Exception here and add logger data

        if needGridPlot:
            self.grid_plot()

    # Internal Testing Function - 1
    def form_print(self, obj, string):
        print(string)
        print(type(obj))
        print(obj.shape)
        print()

    # Internal Testing Function - 2
    def show_img(self, img):
        cv2.imshow("Test", img)
        cv2.waitKey(0)

    # Pseudo Main Function
    def process_frame(self):
        self.height = int(self.image.shape[0])
        self.width = int(self.image.shape[1])
        self.image = cv2.resize(self.image, (self.width, self.height))
        self.roi_height = int(self.height * 0.6)
        self.list_img = []
        self.masked_image = self.process_image()
        self.line_image = self.process_lines()
        self.final_image()

    def grid_plot(self):
        figure, subplots = plt.subplots(3,3, figsize=(9, 6), num='Lane Tracking')
        figure.suptitle('Lane Tracking Processing steps')

        for index, item in enumerate(self.list_img):
            # OpenCV default colormap is BGR -> Converting to RGB for displaying in Matplotlib
            image_colormap = cv2.cvtColor(item[1], cv2.COLOR_BGR2RGB)
            downsampled_image = cv2.resize(image_colormap, (int(self.width/4), int(self.height/4)))

            subplots[index//3, index % 3].imshow(downsampled_image)
            img_title = f'{index+1}. ' + item[0]
            subplots[index//3, index % 3].title.set_text(img_title)

        plt.show()

    def process_image(self):
        # Converting the image into gray scale, so that processing is faster
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.list_img.append(('Gray', gray))

        # cv2.imshow("gray", gray)
        # cv2.waitKey(0)

        # Blurring the image to smoothen the image and reduce noise present in the image
        blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)
        self.list_img.append(('Blurred', blurred_image))

        # Performing Canny Edge Detection using inbuilt canny function
        canny_image = cv2.Canny(blurred_image, 60, 90)
        self.list_img.append(('Canny', canny_image))

        # (Testing purposes) Thresholding the image
        # ret, thresh_img = cv2.threshold(blurred_image, 130, 200, cv2.THRESH_BINARY)
        # self.list_img.append(('Thresh', thresh_img))

        # Selecting the region of interest (ROI) which happens to be the lower part of the image/video (A trapezium)
        # Note: Adding an unwanted array in the below np.array() function as cv2.fillPoly only accepts in such format
        roi_polygon = np.array([
            [(self.width*0.1, self.height), (self.width*0.4, self.roi_height), (self.width*0.6, self.roi_height), (self.width*0.9, self.height)]
        ])

        # Creating an empty (black mask) with the same shape as of the original image
        mask = np.zeros_like(canny_image, dtype=np.uint8)

        # Making the choosen area of interest as white in the empty mask
        cv2.fillPoly(mask, np.int32([roi_polygon]), color=255)
        self.list_img.append(('Empty Mask ROI', mask))

        # Adding the mask to our original image -> This creates an image where only the ROI is colored and others are blacked out
        masked_image = cv2.bitwise_and(canny_image, mask)
        self.list_img.append(('Masked ROI', masked_image))

        # cv2.imshow('Masked_Image', masked_image)
        # cv2.waitKey(0)
        # print(len(masked_image.shape))

        return masked_image

    def process_lines(self):
        # Obtaining all the lines using Hough Transform method
        Lines = cv2.HoughLinesP(self.masked_image, 2, np.pi/180, 100, np.array([]), minLineLength=10, maxLineGap=250)

        # (Optional) This is for averaging out lines on one side
        Lines_Average = self.average_lines(Lines)

        # Creating an empty image once again to create a mask for the lines
        line_image = np.zeros_like(self.image)

        # Looping through all the lines and drawing them on the mask
        if Lines_Average is not None:
            for line in Lines_Average:
                x1, y1, x2, y2 = line.reshape(4)
                # Sometimes randomly huge values are assigned to the coordinates. This function will take care of that.
                if self.check_ranges(x1, x2, y1, y2):
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), thickness=10)

        self.list_img.append(('Lines Mask', line_image))

        return line_image

    def check_ranges(self, x1, x2, y1, y2):
        if x1 < 0 or x1 > self.width:
            return False
        if x2 < 0 or x2 > self.width:
            return False
        if y1 < 0 or y1 > self.height:
            return False
        if y2 < 0 or y2 > self.height:
            return False
        return True

    def average_lines(self, Lines):
        # This is an optimization function for the lines
        # Instead of displaying multiple overlapping lines, it just show one smooth line
        right_line = []
        left_line = []

        if Lines is not None:
            for line in Lines:
                x1, y1, x2, y2 = line.reshape(4)
                # This is to prevent the lines crossing past the ROI
                if y1 < self.roi_height:
                    y1 = self.roi_height
                if y2 < self.roi_height:
                    y2 = self.roi_height
                # This function returns the co-efficents of polynomial which fits
                # the given points (x1, y1), (x2, y2)
                line_parameters = np.polyfit((x1, x2), (y1, y2), 1)
                # This try block to prevent crash when parameters are not identified
                # from dashed lines
                try:
                    slope, intercept = line_parameters
                except TypeError:
                    slope, intercept = 0.001, 0
                if slope < 0:
                    left_line.append((slope, intercept))
                else:
                    right_line.append((slope, intercept))

        left_fit_avg = np.average(left_line, axis=0)
        right_fit_avg = np.average(right_line, axis=0)

        # Now we have average slope and intercept values for the line on the left side
        # and on the right side. Using this, we have to get two points for each side
        # to plot the lines
        left_line = self.make_coordinates(left_fit_avg)
        right_line = self.make_coordinates(right_fit_avg)

        return np.array([left_line, right_line])

    def make_coordinates(self, line_parameters):
        # We know the height of values (the value we used to create the triangles for the mask)
        # Using that, we are getting the x values
        try:
            slope, intercept = line_parameters
        except TypeError:
            slope, intercept = 0.001, 0

        y1 = self.height
        y2 = int(self.height / 2)

        # This is to prevent the lines crossing past the ROI
        if y1 < self.roi_height:
            y1 = self.roi_height
        if y2 < self.roi_height:
            y2 = self.roi_height

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return np.array([x1, y1, x2, y2])

    def final_image(self):
        # Combining the line image(with red line) and the original image
        final_img = cv2.addWeighted(self.image, 0.8, self.line_image, 1, 1)
        self.list_img.append(('Final Image', final_img))

        # Displaying the final image
        if self.debug == -1:
            cv2.imshow('Final Output', final_img)
        else:
            # This is only meant for testing and error correcting purposes
            cv2.imshow('Testing', self.list_img[self.debug][1])


    def waitkey(self, timeInMilli):
        # This function to moderate how long the image is displayed
        # Depending on Video or Image, the time is switched between
        # 1 or 0 respectively.
        key = cv2.waitKey(timeInMilli)
        if key == ord('q') or key == ord('Q'):
            cv2.destroyWindow('Final Output')
            return True


def videoplayback(path):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Hahaha', frame)
            key = cv2.waitKey(100)
            if key == ord('q') or key == ord('Q'):
                break
    cap.release()

# Image 1
# obj = LaneTracking('Lane.jpeg', mode="Image", needGridPlot=True)

# Video 1
# obj = LaneTracking('test2.mp4', mode="Video", needGridPlot=False, debug=-1)
# Video 2
obj = LaneTracking('Test1.mp4', mode="Video", needGridPlot=False, debug=-1)
# Video 3
# obj = LaneTracking('Test3_Compressed.mp4', mode="Video", needGridPlot=False, debug=2)
# videoplayback('Test3_Compressed.mp4')

# Testing
# 0 -> Gray
# 1 -> Blur
# 2 -> Canny
# 3 -> Empty Mask ROI
# 4 -> Masked ROI
# 5 -> Line Image