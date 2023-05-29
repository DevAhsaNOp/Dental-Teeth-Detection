import os
import random
import string
import pathlib
import numpy as np
from PIL import Image
import tensorflow as tf
from IPython.display import display
from tensorflow.python.framework.ops import disable_eager_execution
from my_custom_detector.models.research.object_detection.utils import label_map_util
from my_custom_detector.models.research.object_detection.utils import ops as utils_ops
from my_custom_detector.models.research.object_detection.utils import visualization_utils as vis_util


def TeethDetection(testImagesDir, resultImagesDir):
    disable_eager_execution()
    # patch tf1 into `utils.ops`
    utils_ops.tf = tf.compat.v1

    # Patch the location of gfile
    tf.gfile = tf.io.gfile

    def load_model():
        model_dir = "my_custom_detector/trained-inference-graphs/saved_model"
        model = tf.compat.v2.saved_model.load(model_dir, None)
        model = model.signatures['serving_default']

        return model

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'my_custom_detector/training/object-detection.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    print(testImagesDir)
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = pathlib.Path(os.getcwd() + testImagesDir)
    TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))

    detection_model = load_model()

    def run_inference_for_single_image(model, image):
        image = np.asarray(image)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        output_dict = model(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        with tf.compat.v1.Session():
            print(output_dict['num_detections'].eval())  # debugging line
            num_detections = int(output_dict.pop('num_detections').eval()[0])

            output_dict = {key: value[0, :num_detections].eval()  # changed numpy to eval
                           for key, value in output_dict.items()}
            output_dict['num_detections'] = num_detections

            # detection_classes should be ints.
            output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        # Handle models with masks:
        if 'detection_masks' in output_dict:
            # Reframe the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                               tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

        return output_dict

    def show_inference(model, imagePath):
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = np.array(Image.open(imagePath))
        # Actual detection.
        output_dict = run_inference_for_single_image(model, image_np)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=4)
        outputImage = Image.fromarray(image_np)
        randomImageName = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))
        print(resultImagesDir)
        outputImage.save(os.getcwd() + resultImagesDir + "\\" + randomImageName + '.jpg')
        '''
        display image in python using code below
        image_out = Image.new(mode="RGB", size=outputImage.size)
        tuples = [tuple(x) for x in outputImage.getdata()]
        image_out.putdata(tuples)
        image_out.show()
        
        display image in python in jupyter notebook using code below
        display(outputImage)
        '''

    for image_path in TEST_IMAGE_PATHS:
        show_inference(detection_model, image_path)
    print('Image detection completed.....')
