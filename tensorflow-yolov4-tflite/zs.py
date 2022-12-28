import time
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, decode
from PIL import Image
from core.config import cfg
import cv2
import numpy as np
import tensorflow as tf

flags.DEFINE_string('framework', 'tf', '(tf, tflite')
flags.DEFINE_string('weights', './checkpoints', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image', 'D:/STUDY/Grad/tfyolov4/tfyolov4/dataset/pictures/test/dayClip1--00000.jpg', 'path to input image')
flags.DEFINE_string('output', 'result.png', 'path to output image')

def main(_argv):
    if FLAGS.tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if FLAGS.model == 'yolov4':
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
        else:
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_V3, FLAGS.tiny)
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    XYSCALE = cfg.YOLO.XYSCALE
    input_size = FLAGS.size
    image_path = FLAGS.image

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    if FLAGS.framework == 'tf':
        # input_layer = tf.keras.layers.Input([input_size, input_size, 3])
        # feature_maps = YOLOv4(input_layer, NUM_CLASS)
        # bbox_tensors = []
        # for i, fm in enumerate(feature_maps):
        #     if i == 0:
        #         bbox_tensor = decode(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        #     elif i == 1:
        #         bbox_tensor = decode(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        #     else:
        #         bbox_tensor = decode(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        #     bbox_tensors.append(bbox_tensor)
        input_layer = tf.keras.layers.Input([input_size, input_size, 3])
        feature_maps = YOLOv4(input_layer, NUM_CLASS)
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = decode(fm, NUM_CLASS, i)
            bbox_tensors.append(bbox_tensor)
        # model = Model(input_layer, bbox_tensors)

        model = tf.keras.models.load_model('./checkpoints/yolov4keras_model.h5')
        pred_bbox = model.predict(image_data)
        #pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
        for key in pred_bbox:
            boxes = decode(key, NUM_CLASS, i=0)

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    image = utils.draw_bbox(original_image, pred_bbox)
    # image = utils.draw_bbox(image_data*255, pred_bbox)
    image = Image.fromarray(image.astype(np.uint8))
    image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(FLAGS.output, image)

    # if FLAGS.model == 'yolov4':
    #     pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
    # else:
    #     pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)
    # bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
    # bboxes = utils.nms(bboxes, 0.213, method='nms')
    #
    # image = utils.draw_bbox(original_image, bboxes)
    # image = Image.fromarray(image)
    # image.show()
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    # cv2.imwrite(FLAGS.output, image)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass