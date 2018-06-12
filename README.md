# Faster R-CNN-Based Glomerular Detector
**This Repository contains the Faster R-CNN-Based Glomerular Detector that detects glomeruli from multistained Whole Slide Images(WSIs) of human renal tissue sections.**

 In this implements, We use the *"[Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)"*
 and it's pretrained network model named "Inception Resnet v2".

 For details of *"Faster R-CNN"* method, please see [here](https://arxiv.org/abs/1506.01497).
 And for details of *"Inception Resnet v2"*, please refer to [here](https://ai.googleblog.com/2016/08/improving-inception-and-image.html).

 This software includes the work that is distributed in the [Apache Licence 2.0](https://github.com/tensorflow/models/blob/master/LICENSE).

## set up

Please set up by the following procedure.

1. Installing Tensorflow and Tensorflow Object Detection API

    Please run through the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) to install *"Tensorflow Object Detection API"* and all it dependencies.

2. Download our tools

    Please download our glomeruli detection programs.

    ```
    git clone https://github.com/jinseikenai/glomeruli_detection.git
    ```

1. dependencies

    Our glomeruli detection programs depends on the following libraries. Please install the following libraries.

    * [OpenSlide](https://openslide.org/)

    Operation Environment：　We confirmed the operation in the following environment.

    * python 3.5
    * tensorflow 1.4.1


## Quick Start:

  * [Quick Start](https://github.com/jinseikenai/glomeruli_detection/blob/master/detecting_glomeruli.md)

    We will explain the procedure to detect glomeruli using the provided learned models and sample WSIs.

## <a name=trained_models>trained models</a>

  We Provide detection models trained on our data sets.

  * [PAS](http://www.m.u-tokyo.ac.jp/medinfo/download/jinai/faster_rcnn/trained_models/pas_train1.tar.gz)
  * [PAM](http://www.m.u-tokyo.ac.jp/medinfo/download/jinai/faster_rcnn/trained_models/pam_train1.tar.gz)
  * [MT](http://www.m.u-tokyo.ac.jp/medinfo/download/jinai/faster_rcnn/trained_models/mt_train1.tar.gz)
  * [Azan](http://www.m.u-tokyo.ac.jp/medinfo/download/jinai/faster_rcnn/trained_models/azan_train1.tar.gz)

  You can un-tar each tar.gz file via, e.g.,:

  ```
  tar -xvfz pas.train1.tar.gz
  ```

## test data

  Whole Slide Images for testing can be download from [here](http://www.m.u-tokyo.ac.jp/medinfo/download/jinai/faster_rcnn/test_data.tar.gz).
  These data are not included in the training data.

  Using these data, you could confirm the glomeruli detection programs and its result.
  Please see [Quick Start](https://github.com/jinseikenai/glomeruli_detection/blob/master/detecting_glomeruli.md) for how to do it.

## Program List
* Glomeruli Detection Programs
  1. detection : detect_glomus_test.py
  2. merging overlapping regions : merge_overlaped_glomus.py
  3. evaluation and visualization: eval_recall_precision_test.py

* Glomeruli Learning Programs
  * learning: my_train.py, my_trainer.py

## <a name='learning'>Transfer Learning or Additional Learning</a>

  Based on our learning model, you could do your transfer learning or additional learning.

  For reference information on how to do learning, Please refer to the following notes, and my_train.py and my_trainer.py.

1. Data Preparation

  Please prepare a set of learning data and annotations showing correct answers.
  And see the TensorFlow's [*Programmer's Guide:"Importing Data"*](https://www.tensorflow.org/programmers_guide/datasets).


2. Configuration

  Please configure a variable *"PATH_TO_BE_CONFIGURED*" in *"config/glomerulus_train.config"* and *"config/input.config*" appropriately to your environment.


3. Learning Execution

  You could execute learning with the following command.

  ```
  python my_train.py --logtostderr \
    --train_dir=${TRAIN_DATA_PATH} \
    --model_config_path=${CONFIG_PATH}/glomerulus_model.config \
    --train_config_path=${CONFIG_PATH}/glomerulus_train.config \
    --input_config_path=${CONFIG_PATH}/input.config \
    --gpu_list=${GPU}
  ```

  * Set the path of learning data to ${TRAIN_DATA_PATH}
  * Set the path of configuration files to ${CONFIG_PATH}.
  * Set the path of directory of output files to ${TRAIN_DATA_PATH}.
  * Set the path of configuration files to ${CONFIG_PATH}. You could specify different paths for each.
    * glomerulus_model.config
    * glomerulus_train.config
    * input.config
  * Set the GPU list you can use to ${GPU} like "--gpu_list=0,1" or "--gpu_list=1".
