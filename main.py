import sys
import torch


def main():
    argv = len(sys.argv)
    print("Total arguments passed:", argv)

    if argv != 3:
        print("Expected 3 args, got", argv)
    else:
        input_filename = sys.argv[1]
        object_class = sys.argv[2]

        output_filename = input_filename[:input_filename.rfind(".")] + "_" + object_class + input_filename[input_filename.rfind("."):]
        print(output_filename)

        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


if __name__ == '__main__':
    main()
