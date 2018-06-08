# Quick Start: Detecting Glomeruli
  Using models and data provided [here](https://github.com/jinseikenai/glomeruli_detection#trained_models), you could confirm the glomeruli detection programs and its result.

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

  * Set a path of text file in which the processing target is written to ${TARGET_LIST}.
  * ${DATA_DIR}にはWSIファイルを配置したファイルフォルダを指定します。本プログラムでは${DATA_PATH}/${STAINING}以下に${TARGET_LIST}の内容のファイルが存在することを前提にしています。
  * Set a code of target staining from below to ${STAINING}
    * OPT_PAS
    * OPT_PAM
    * OPT_MT
    * OPT_Azan

  * ${OUTPUT_DIR}には処理結果（検出した糸球体領域の位置情報のリスト）を出力するファイルフォルダを指定します。
  * ${FILE_EXTENSON} には出力ファイルにつける拡張子を指定します。この引数は、複数の条件で抽出処理を実行した場合に、それぞれの結果ファイルを区別するために用意しています。
  * ${MODEL_PATH}には検出に用いる学習済みモデルファイルへのパスを指定します。そのパス以下に、検出に利用する Tensorflow の frozon_inference_graph.pb を置いてください。
  * --window_size には sliding window 方式で糸球体検出領域（"window"）をスライドさせる際の"window"のサイズを指定します。サイズの単位はマイクロメートルです。
  * --overlap_ratio には sliding window 方式で糸球体検出領域をスライドさせる際の重複させる窓の領域を指定します。1.0未満の正の数値を指定してください。
  * --conf_threshold には検出結果に採用する最低限の confidence threshold の値を指定します。この値以下の confidence の糸球体領域候補は検出結果に含まれなくなります。1.0以下の正の数値を指定してください。

## <a name='merge'>2. Merging Overlapping Regions</a>

  ```
  python merge_overlaped_glomus.py \
      --detected_list=${DETECTION_RESULT_FILE} \
      --data_dir=${DATA_DIR} \
      --staining=${STAINING} \
      --output_dir=${OUTPUT_DIR} \
      --output_file_ext=${FILE_EXTENTION} \
      --conf_threshold=0.9 \
      --overlap_threshold=0.35
  ```

  * ${DETECTION_RESULT_FILE}には 1. の検出結果ファイルへのパスを指定します。
  * ${DATA_DIR}にはWSIファイルを配置したファイルフォルダを指定します。
  * Set a code of target staining from below to ${STAINING}
    * OPT_PAS
    * OPT_PAM
    * OPT_MT
    * OPT_Azan

  * ${OUTPUT_DIR}には、マージ結果ファイルを出力するフォルダへのパスを指定します。
  * ${FILE_EXTENTION} にはマージ結果ファイルに付与する名前を指定します。この引数は、複数の条件で抽出処理を実行した場合に、それぞれの結果ファイルを区別するために用意しています。
  * --conf_threshold には、マージ結果に採用する最低限の confidence threshold の値を指定します。この値以下の confidence の糸球体領域候補は検出結果に含まれなくなります。1.0以下の正の数値を指定してください。
  * --overlap_threshold にはオーバーラップしていると判定する重複率の最小値を指定してください。この比率以上に重複している糸球体領域候補はマージされます。(0, 1] の値を指定してください。


## <a name='visualize'>3. Evaluation and Visualization</a>

  ```
  python -u eval_recall_precision_test.py \
      --staining=${STAINING} \
      --data_dir=${ANNOTATION_FILE_ROOT_DIR} \
      --target_list=${DETECTION_RESULT_FILE} \
      --merged_list=${MERGING_RESULT_FILE} \
      --output_dir=${OUTPUT_DIR} \
      --iou_threshold=0.5 \
      --no_save
  ```

  ![sample result](https://github.com/jinseikenai/glomeruli_detection/blob/master/OPT_PAS_TEST01_001_pw40_ds8.PNG "SampleResult")
