# Quick Start: Detecting Glomeruli
  Using pre-trained models and trial data provided [here](https://github.com/jinseikenai/glomeruli_detection#pre-trained_models), you could confirm the [Faster R-CNN-Based Glomerular Detector](https://github.com/jinseikenai/glomeruli_detection) and its results.

  It is shown below that is an example of the result of glomeruli detection for a PAS stain slide.

  ![sample result](https://github.com/jinseikenai/glomeruli_detection/blob/master/OPT_PAS_TEST01_001_pw40_ds8.PNG "SampleResult")

  * <span style="color: red;">□ **Red Frames**</span> are the detection results of the Faster R-CNN-Based Glomerular Detector.
  * <span style="color: yellow;">□ **Yellow Frames**</span> are the Ground Truth(GT) (i.e. correct answer) specified by experts.
  
  The detection procedure is as follows.

  1. [Glomeruli Detection](#detection)
  2. [Merging Overlapping Regions](#merge)
  3. [Evaluation and Visualization](#visualize)


## <a name='detection'>1. Glomeruli Detection</a>

  ```
  python detect_glomus_test.py \
      --target_list=${TARGET_LIST} \
      --data_dir=${DATA_DIR} \
      --staining=${STAINING} \
      --output_dir=${OUTPUT_DIR}
      --output_file_ext=${FILE_EXTENTION} \
      --model=${MODEL_PATH} \
      --window_size=2000 \
      --overlap_ratio=0.1 \
      --conf_threshold=0.2
  ```

  * Set ${TARGET_LIST} points to a path of text file in which the processing target is written.
  * Set ${DATA_DIR} points to a path of the file folder in which there are whole slide images.
  This program assumes that the processing target files indicated by the ${TARGET_LIST} exist under the ${DATA_PATH}/${STAINING}.
  * Set ${STAINING} points to a code of staining method from below.
    * OPT_PAS
    * OPT_PAM
    * OPT_MT
    * OPT_Azan

  * Set ${OUTPUT_DIR} points to a path of the file folder in which this process write output files.
  * Set ${FILE_EXTENSON} for distinguishing execution results. It is added to the tail of result file name.
  * Set ${MODEL_PATH} points to a path of the model file used for detection. You should put a tensorflow's frozen_inference_graph.pb under this path.
  * Set --window_size points to the size(width and height) of sliding window for detection.
  The unit of size(width and height) is micrometer.
  Please set a positive number like 2000.
  * Set --overlap_ratio point to overlapping ration of the sliding window.
  Please set a positive number less than equal 1.0. Namely (0.0, 1.0]
  * Set --conf_threshold points to the minimum confidence value to adopt for candidates.
  Candidates with confidence value lower the threshold will not be included in the detection result.
  Please set a positive number less than equal 1.0. Namely (0.0, 1.0]

## <a name='merge'>2. Merging Overlapping Regions</a>

  ```
  python merge_overlaped_glomus.py \
      --target_list=${TARGET_LIST} \
      --detected_list=${DETECTION_RESULT_FILE} \
      --data_dir=${DATA_DIR} \
      --staining=${STAINING} \
      --output_dir=${OUTPUT_DIR} \
      --output_file_ext=${FILE_EXTENTION} \
      --conf_threshold=0.9 \
      --overlap_threshold=0.35
  ```

  * Set ${TARGET_LIST} points to a path of text file in which the processing target is written.
  * Set ${DETECTION_RESULT_FILE} points to a path of the detection result file.
  * Set ${DATA_DIR} points to a path of the file folder in which there are whole slide images.
  * Set ${STAINING} points to a code of staining method from below.
    * OPT_PAS
    * OPT_PAM
    * OPT_MT
    * OPT_Azan

  * Set ${OUTPUT_DIR} points to a path of the file folder in which this process write a output file to .
  * Set ${FILE_EXTENSON} for distinguishing execution results. It is added to the result file name.
  * Set --conf_threshold points to the minimum confidence value to adopt for merge result.
  Candidates with confidence value lower the threshold will not be included in the merged result.
  Please set a positive number less than equal 1.0. Namely (0.0, 1.0]
  * Set --overlap_threshold points to the minimum duplicate ration that judges overlapped.
  Glomerular region candidates have more overlapped area over this ratio are merged.
  Please set a positive number less than equal 1.0. Namely (0.0, 1.0]


## <a name='visualize'>3. Evaluation and Visualization</a>

  ```
  python -u eval_recall_precision_test.py \
      --staining=${STAINING} \
      --data_dir=${ANNOTATION_FILE_ROOT_DIR} \
      --target_list=${DETECTION_RESULT_FILE} \
      --merged_list=${MERGED_RESULT_FILE} \
      --output_dir=${OUTPUT_DIR} \
      --iou_threshold=0.5 \
      --no_save
  ```

  * Set ${STAINING} points to a code of staining method from below.
    * OPT_PAS
    * OPT_PAM
    * OPT_MT
    * OPT_Azan
  * Set ${DATA_DIR} points to a path of the file folder in which there are whole slide images.
  * Set ${TARGET_LIST} points to a path of text file in which the processing target is written.
  * Set ${MERGED_RESULT_LIST} points to a path of the merged result file.
  * If --no_save flag is set, the result image file is not saved.
  You should not set this flag if you want to save the result image file.
  
  In this evaluation, the correct answer judgment threshold is set to "Intersection over Union(IoU) >= 0.5" (i.e. "--iou_threshold=0.5") of the ground truth(GT) and the detected boundary box
  according to the configuration of the [PASCAL VOC challenge](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf).
