import sys
import torch
import cv2
import numpy as np


def main():
    argv = len(sys.argv)
    print("Total arguments passed:", argv)

    if argv != 3:
        print("Expected 3 args, got", argv)
    else:
        input_filename = sys.argv[1]
        object_class = sys.argv[2]

        output_filename = input_filename[:input_filename.rfind(".")] + "_" + object_class + input_filename[input_filename.rfind("."):]

        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        capture = cv2.VideoCapture(input_filename)
        img_array = []
        obj_crops_array = []

        if not capture.isOpened():
            print("Unable to open:" + input_filename)
            exit(0)

        while capture.isOpened():
            ret, img = capture.read()

            if ret:
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                detections = model(imgRGB)

                # if detections on current img found
                if detections and len(detections.pred[0]) > 0:
                    for i in range(len(detections.pred[0])):
                        classe = detections.names[int(detections.pred[0][i][-1])]

                        if classe == object_class:
                            x1 = int(detections.pred[0][i][0])
                            y1 = int(detections.pred[0][i][1])
                            x2 = int(detections.pred[0][i][2])
                            y2 = int(detections.pred[0][i][3])

                            # crop found object
                            crop_img = img[y1:y2, x1:x2]

                            # shift cropped object
                            height, width = img.shape[:2]
                            translation_matrix = np.float32([[1, 0, x1], [0, 1, y1]])
                            shift_img = cv2.warpAffine(crop_img, translation_matrix, (width, height))

                            # save objects of current img
                            obj_crops_array.append(shift_img)

                            cv2.imshow("Video", shift_img)
                            cv2.waitKey(1)


            else:
                print("No video detected...")
                break

        capture.release()
        cv2.destroyAllWindows()

        # create a new video with cropped objects
        out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

if __name__ == '__main__':
    main()
