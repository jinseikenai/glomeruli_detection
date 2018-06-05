# Quick Start: Detecting Glomeruli
  ここで提供される学習済みモデルとサンプルWSIを用いて、検出処理を行う手順を説明します。

  検出処理は以下の手順で行われます。

  1. [糸球体検出処理](#detection)
  2. [重複領域のマージ](#merge)
  3. [検出結果の評価と可視化](#visualize)

## <a name='detection'>1. 糸球体検出処理</a>

  ```
  python detect_glomus_test.py \
      --target_list ${TARGET_LIST_PATH} \
      --target_dir ${DATA_PATH} \
      --data_category ${STAINING} \
      --output_dir ${OUTPUT_DIR}
      --output_file_ext ${FILE_EXTENTION} \
      --window_size 2000 \
      --overlap_ratio 0.1 \
      --conf_threshold=0.2
      --model ${MODEL_DIR_PATH}
  ```

## <a name='merge'>2. 重複した糸球体候補領域のマージ処理</a>

  ```
  python merge_overlaped_glomus.py \
      --staining ${STAINING} \
      --input_file ${INPUT_FILE} \
      --output_dir ${OUTPUT_DIR} \
      --training_type ${TRAINING_TYPE} \
      --annotation_dir ${ANNOTATION_FILE_ROOT_DIR} \
      --conf_threshold 0.2 \
      --overlap_threshold 0.35
  ```

## <a name='visualize'>3. 検出結果の評価と可視化</a>

  ```
  python -u eval_recall_precision_test.py \
      --staining_type ${STAINING} \
      --annotation_dir ${ANNOTATION_FILE_ROOT_DIR} \
      --detect_list ${DETECTION_RESULT_FILE} \
      --output_dir ${OUTPUT_DIR} \
      --iou_threshold 0.5 \
      --no_save
  ```

  * 共通変数
    * ${STAINING}

      対象とする染色種別を示す以下のコードを指定する。

      * OPT_PAS
      * OPT_PAM
      * OPT_MT
      * OPT_Azan