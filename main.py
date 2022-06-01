import pixellib
from pixellib.instance import instance_segmentation
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def object_detection_on_an_image():
    segment_image = instance_segmentation()
    segment_image.load_model('models/mask_rcnn_coco.h5')

    segment_image.segmentImage(
        show_bboxes=True,
        image_path='2.jpg',
        output_image_name='output.jpg'
    )


def main():
    object_detection_on_an_image()


main()