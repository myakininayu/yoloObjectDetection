import sys
import torch
import cv2
import functions


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
        capture.set(cv2.CAP_PROP_FPS, 25)
        height, width = 1000, 1200

        img_array = []

        if not capture.isOpened():
            print("Unable to open:", input_filename)
            exit(0)

        while capture.isOpened():
            ret, img = capture.read()

            if ret:
                height, width = img.shape[:2]
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                detections = model(imgRGB)
                img_array = functions.handle_detections(img, detections, object_class)

            else:
                print("No video detected...")
                break

        capture.release()
        cv2.destroyAllWindows()

        # create a new video with cropped objects
        out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


if __name__ == '__main__':
    main()
