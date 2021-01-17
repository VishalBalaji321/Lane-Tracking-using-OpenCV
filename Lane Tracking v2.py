import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot


class LaneTracking:
    def __init__(self, path, mode='Video'):
        if mode == 'Video':
            pass
        elif mode == 'Image':
            self.image = cv2.imread(path)
            self.process_frame()
        else:
            print('Wrong Mode....Please try again')
            # Raise Exception here and add logger data
        pass

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
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.list_img = []
        self.masked_image = self.process_image()
        self.line_image = self.process_lines()
        self.final_image()
        self.grid_plot()

    def grid_plot(self):
        figure, subplots = plt.subplots(3,3, figsize=(9, 6), num='Lane Tracking')
        figure.suptitle('Lane Tracking Processing steps')

        for index, item in enumerate(self.list_img):
            # OpenCV default colormap is BGR -> Converting to RGB for displaying in Matplotlib
            image_colormap = cv2.cvtColor(item[1], cv2.COLOR_BGR2RGB)
            downsampled_image = cv2.resize(image_colormap, (int(self.width/4), int(self.height/4)))

            subplots[index//3, index % 3].imshow(downsampled_image, cmap='CMRmap')
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
        canny_image = cv2.Canny(blurred_image, 35, 175)
        self.list_img.append(('Canny', canny_image))

        # Selecting the region of interest (ROI) which happens to be the lower part of the image/video
        # Note: Adding an unwanted array in the below np.array() function as cv2.fillPoly only accepts in such format
        triangle = np.array([
            [(self.width*0.1, self.height), (self.width*0.5, self.height/2), (self.width*0.9, self.height)]
        ])

        # Creating an empty (black mask) with the same shape as of the original image
        mask = np.zeros_like(canny_image, dtype=np.uint8)

        # Making the choosen area of interest as white in the empty mask
        cv2.fillPoly(mask, np.int32([triangle]), color=255)
        self.list_img.append(('Empty Mask ROI', mask))

        # Adding the mask to our original image -> This creates an image where only the ROI is colored and others are blacked out
        masked_image = cv2.bitwise_and(canny_image, mask)
        self.list_img.append(('Masked ROI', masked_image))

        # cv2.imshow('Masked_Image', masked_image)
        # cv2.waitKey(0)
        # print(len(masked_image.shape))

        return masked_image

    def process_lines(self):
        # conv_grayscale = cv2.cvtColor(self.masked_image, cv2.COLOR_BGR2GRAY)
        # self.form_print(self.masked_image, 'Masked_image')
        # self.show_img(self.masked_image)

        # Obtaining all the lines using Hough Transform method
        Lines = cv2.HoughLinesP(self.masked_image, 2, np.pi/180, 100, np.array([]), minLineLength=50, maxLineGap=50)

        # self.form_print(Lines, 'Lines: Hough Transform')

        # Creating an empty image once again to create a mask for the lines
        line_image = np.zeros_like(self.image)

        # Looping through all the lines and drawing them on the mask
        if Lines is not None:
            for line in Lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), thickness=10)

        self.list_img.append(('Lines Mask', line_image))

        return line_image

    def final_image(self):
        # Combining the line image(with red line) and the original image
        final_img = cv2.addWeighted(self.image, 0.8, self.line_image, 1, 1)
        self.list_img.append(('Final Image', final_img))

        # Displaying the final image
        cv2.imshow('Final Output', final_img)
        self.key = cv2.waitKey(0)
        if self.key == ord('q') or self.key('Q'):
            cv2.destroyWindow('Final Output')


obj = LaneTracking('Lane.jpeg', mode="Image")
