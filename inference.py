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
import judger_hand

LOCAL = True

PATH_TO_CKPT = './models/hand_detection/frozen_inference_graph.pb'
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


def infer_phase_1(vis=False):
    tf.reset_default_graph()

    # ## Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # # begin test image
    resized_image_list = []
    box_list = []
    filename_list = []

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for image_idx, image_path in enumerate(TEST_IMAGE_PATHS):
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                boxes = boxes[0]
                scores = scores[0]

                # scores_lag = np.roll(scores, shift=-1)
                # idx = np.argmax(scores - scores_lag < 0.1)
                threshold = scores[4]  # np.sort(scores, axis=None)[-5]
                # threshold = threshold if threshold > 0.01 else 0.01

                im_width, im_height = image.size

                box_hist = []
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

                        resized_image_list.append(load_image_into_numpy_array(image.crop(box_).resize((64, 64))))
                        box_list.append((left, top, right, bottom))
                        filename_list.append(image_path)

                        if vis:
                            class_ = INT_TO_CLASS[classes[0][i]]
                            draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color=CLASS_TO_COLOR[class_],
                                                       display_str_list=[class_ + ": {0:.3f}".format(scores[i])])

                        box_hist.append(box_)

                    else:
                        break

    return resized_image_list, box_list, filename_list


def infer_phase_2(resized_images_list):
    tf.reset_default_graph()

    model_name = "./models/hand_classification/hand_classification_model.h5"
    model = keras.models.load_model(model_name)

    x_data = resized_images_list.astype('float32')
    x_data /= 255

    outputs = model.predict(x_data)
    print(outputs.shape)
    classes = np.argmax(outputs, axis=1)
    print(classes)
    return classes  # []


def pipeline():
    resized_image_list, box_list, filename_list = infer_phase_1()
    classes_list = infer_phase_2(np.array(resized_image_list))

    print(len(resized_image_list))

    if not LOCAL:
        ans_writer = judger_hand.get_output_file_object()

        for i, resized_image in enumerate(resized_image_list):
            box = box_list[i]
            str_ = '%s %d %d %d %d %d %f\n' % (
                filename_list[i], int(box[0]), int(box[1]), int(box[2]), int(box[3]), classes_list[i], .8)
            ans_writer.write(str_.encode())

        score, err = judger_hand.judge()
        if err is not None:  # in case we failed to judge your submission
            print(err)
        else:
            print("score", score)
    else:
        if not os.path.isdir("./outputs"):
            os.mkdir("./outputs")

        for i, resized_image in enumerate(resized_image_list):
            filename = filename_list[i].replace(os.path.sep, '/').split("/")[-1]
            skimage.io.imsave("./outputs/{}-{}-{}.jpg".format(filename, i,
                                                              "L" if 0 == classes_list[i] else "R"), resized_image)


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', type=int, default=0, help='if local')
    args = parser.parse_args()

    if args.local == 0:
        LOCAL = True
    else:
        LOCAL = False

    print(LOCAL)

    if LOCAL:
        PATH_TO_TEST_IMAGES_DIR = './test_images/'
        TEST_IMAGE_PATHS = [
            os.path.join(PATH_TO_TEST_IMAGES_DIR, filename) for filename in os.listdir(PATH_TO_TEST_IMAGES_DIR)]
    else:
        try:

            TEST_IMAGE_PATHS = judger_hand.get_file_names()

            print(TEST_IMAGE_PATHS)
        except ImportError:
            print("You need to install judger_hand")

    pipeline()
