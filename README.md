# glomeruli_detection
Faster RCNN を用いた糸球体検出器

detection of glomeruli using faster rcnn

## set up
1. Installing Tensorflow and Tensorflow Object Detection API

    Please run through the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) to install *"Tensorflow Object Detection API"* and all it dependencies.

2. Download our tools

    検出処理プログラムとサンプルデータ、学習済みモデルをダウンロードしてください。

    ```
    git clone https://github.com/jinseikenai/glomeruli_detection.git
    ```

1. dependencies

    我々の糸球体検出付プログラムでは以下のライブラリを用いています。以下のライブラリをインストールしてください。Our Glomeruli Detection Programs depends on the following libraries:

    * [OpenSlide](https://openslide.org/)

## Quick Start: Detecting Glomeruli from sample WSIs.

  * [Quick Start](https://github.com/jinseikenai/glomeruli_detection/blob/master/detecting_glomeruli.md)

## trained models

  We Provide detection models trained on our data sets.

  You can un-tar each tar.gz file via, e.g.,:

  ```
  tar -xvfz pas.train1.tar.gz
  ```

## test data

  各染色の Whole Slide Image が  "test_data" ディレクトリにあります。それらの画像は学習済みモデルの学習に用いていません。

  それらのデータを用いて糸球体検出処理とその結果がどのようなものかを確認することができます。

## Program List
* 糸球体検出処理用プログラム
  * 糸球体検出処理プログラム: detect_glomus_test.py
  * 検出処理結果マージプログラム: merge_overlaped_glomus.py
  * 検出処理結果確認プログラム: eval_recall_precision_test.py

* 学習用プログラム
  * 学習実行処理呼び出しプログラム: my_train.py

## 追加学習
1. Configuration
  * config/glomerulus_train.config と config/input.config の *PATH_TO_BE_CONFIGURED* を環境に合わせて適切に設定する。
　
1. 学習実行

  以下のようにして学習を実行することが出来ます。ここで、${TRAIN_DATA_PATH}には追加学習を行う学習データへのパスを指定し、${CONFIG_PATH}に config ファイルがあるディレクトリへのパスを指定してください。${GPU}には使用するGPUの番号リストを指定してください。

  ```
  python my_train.py --logtostderr \
    --train_dir=${TRAIN_DATA_PATH} \
    --model_config_path=${CONFIG_PATH}/glomerulus_model.config \
    --train_config_path=${CONFIG_PATH}/glomerulus_train.config \
    --input_config_path=${CONFIG_PATH}/input.config \
    --gpu_list=${GPU}
  ```


