import numpy as np
import cv2

import numpy as np
import os
import tensorflow as tf
from PIL import Image
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import skimage.io
import keras

LOCAL = True

TEST_IMAGE_PATHS = None
ANS_WRITER = None

INT_TO_CLASS = {
    1: "L",
    2: "R"
}

CLASS_TO_COLOR = {
    "L": "green",
    "R": "red"
}

AREA_THRESHOLD = 0.5


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
    """Adds a bounding box to an image.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
      image: a PIL.Image object.
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin


def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
    """Adds a bounding box to an image (numpy array).

    Args:
      image: a numpy array with shape [height, width, 3].
      ymin: ymin of bounding box in normalized coordinates (same below).
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                               thickness, display_str_list,
                               use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def cal_area(a, b=None):
    if b is None:
        return (a[0] - a[2]) * (a[1] - a[3])
    else:
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        else:
            return 0


def pipeline():
    phase_1_model_name = './models/hand_detection/frozen_inference_graph.pb'
    phase_2_model_name = "./models/hand_classification/hand_classification_model.h5"
    phase_2_model = keras.models.load_model(phase_2_model_name)

    phase_1_graph = tf.Graph()
    with phase_1_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(phase_1_model_name, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with phase_1_graph.as_default():
        with tf.Session(graph=phase_1_graph) as sess:
            image_tensor = phase_1_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = phase_1_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = phase_1_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = phase_1_graph.get_tensor_by_name('detection_classes:0')
            num_detections = phase_1_graph.get_tensor_by_name('num_detections:0')

    cap = cv2.VideoCapture(0)
    size = (64, 64)

    while True:
        ret, image_np = cap.read()

        image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')

        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})

        boxes = boxes[0]
        scores = scores[0]
        threshold = scores[4]

        im_width, im_height = image_np.shape[1], image_np.shape[0]
        box_hist = []
        resized_image_list = []
        for i in range(boxes.shape[0]):
            box = boxes[i]
            ymin, xmin, ymax, xmax = box

            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)

            box_ = (left, top, right, bottom)

            if scores[i] > threshold:

                legal = True

                for box__ in box_hist:
                    area__ = cal_area(box__)
                    area___ = cal_area(box_, box__)
                    if area___ / area__ > AREA_THRESHOLD:
                        legal = False
                        break

                if not legal:
                    continue

                cropped = image_pil.crop(box_)
                cropped.thumbnail(size, Image.ANTIALIAS)
                background = Image.new('RGB', size, (0, 0, 0))
                background.paste(
                    cropped, (int((size[0] - cropped.size[0]) / 2), int((size[1] - cropped.size[1]) / 2))
                )
                resized_image_list.append(load_image_into_numpy_array(background))
                box_hist.append(box_)
            else:
                break
        resized_images_list = np.array(resized_image_list)
        x_data = resized_images_list.astype('float32')
        x_data /= 255
        res = phase_2_model.predict(x_data)
        classes = np.argmax(res, axis=1)
        for idx, box in enumerate(box_hist):
            class_ = "L" if 0 == classes[idx] else "R"
            score_ = res[idx][classes[idx]]
            draw_bounding_box_on_image_array(image_np, int(box[1]), int(box[0]), int(box[3]), int(box[2]),
                                             color=CLASS_TO_COLOR[class_], use_normalized_coordinates=False,
                                             display_str_list=[class_ + ": {0:.3f}".format(score_)])

        cv2.imshow('frame', cv2.resize(image_np, (1200, 900)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


pipeline()
