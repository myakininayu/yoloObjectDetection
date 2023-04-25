import sys
import torch
import cv2


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

        if not capture.isOpened():
            print("Unable to open:" + input_filename)
            exit(0)

        while capture.isOpened():
            ret, img = capture.read()

            if ret:
                print("Video detected...")
                cv2.imshow("Video", img)
                cv2.waitKey(1)
            else:
                print("No video detected...")
                break

        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
