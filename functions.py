import numpy as np
import cv2


def handle_detections(img, detections, object_class):
    obj_crops_array = []
    img_array = []

    height, width = img.shape[:2]

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
                translation_matrix = np.float32([[1, 0, x1], [0, 1, y1]])
                shift_img = cv2.warpAffine(crop_img, translation_matrix, (width, height))

                # save objects of current img
                obj_crops_array.append(shift_img)

        if len(obj_crops_array) == 1:
            print("Found 1", object_class)

            img_array.append(obj_crops_array[0])
            obj_crops_array.clear()

        elif len(obj_crops_array) > 1:
            v = cv2.addWeighted(obj_crops_array[0], 1, obj_crops_array[1], 1, 0)
            print("Found", str(len(obj_crops_array)), "objects of class", object_class)

            for i in range(2, len(obj_crops_array)):
                v = cv2.addWeighted(v, 1, obj_crops_array[i], 1, 0)

            img_array.append(v)
            obj_crops_array.clear()

    return img_array
