# Quick Start: Detecting Glomeruli
  ここで提供される学習済みモデルとサンプルWSIを用いて、検出処理を行う手順を説明します。

  検出処理は以下の手順で行われます。

  1. [糸球体検出処理](#detection)
  2. [重複領域のマージ](#merge)
  3. [検出結果の評価と可視化](#visualize)

## <a name='detection'>1. 糸球体検出処理</a>

  ```
  python detect_glomus_test.py \
      --target_list=${TARGET_LIST} \
      --target_dir=${DATA_PATH} \
      --data_category=${STAINING} \
      --output_dir=${OUTPUT_DIR}
      --output_file_ext=${FILE_EXTENTION} \
      --model=${MODEL_PATH} \
      --window_size=2000 \
      --overlap_ratio=0.1 \
      --conf_threshold=0.2
  ```

  * ${TARGET_LIST}には処理対象のWSIのリストを記したテキストファイルへのパスを指定します。
  * ${DATA_PATH}にはWSIファイルを配置したファイルフォルダを指定します。本プログラムでは${DATA_PATH}/${STAINING}以下に${TARGET_LIST}の内容のファイルが存在することを前提にしています。
  * ${OUTPUT_DIR}には処理結果（検出した糸球体領域の位置情報のリスト）を出力するファイルフォルダを指定します。
  * ${FILE_EXTENSON} には出力ファイルにつける拡張子を指定します。この引数は、複数の条件で実行した場合に、それぞれの結果ファイルを区別するために用意しています。
  * ${MODEL_PATH}には検出に用いる学習済みモデルファイルへのパスを指定します。そのパス以下に、検出に利用する Tensorflow の frozon_inference_graph.pb を置いてください。
  * --window_size には sliding window 方式で糸球体検出領域（"window"）をスライドさせる際の"window"のサイズを指定します。サイズの単位はマイクロメートルです。
  * --overlap_ratio には sliding window 方式で糸球体検出領域をスライドさせる際の重複させる窓の領域を指定します。1.0未満の正の数値を指定してください。
  * --conf_threshold には検出結果に採用する最低限の confidence threshold を指定します。この値以下の confidence の糸球体領域候補は検出結果に含まれなくなります。1.0以下の正の数値を指定してください。

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