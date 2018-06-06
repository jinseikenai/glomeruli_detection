# Glomeruli Detector
**This site introduces an example of a detector that detects glomeruli from Whole Slide Image.**

We uses the [Faster RCNN](https://arxiv.org/abs/1506.01497) method and the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) for its implementation.

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

## Quick Start:

  * [Quick Start](https://github.com/jinseikenai/glomeruli_detection/blob/master/detecting_glomeruli.md)

    提供する学習済みモデルとサンプルWSIを用いて、検出処理を行う手順を説明します。

## trained models

  We Provide detection models trained on our data sets.

  * [pas](http://www.m.u-tokyo.ac.jp/medinfo/download/jinai/faster_rcnn/trained_models/pas_train1.tar.gz)
  * [pam](http://www.m.u-tokyo.ac.jp/medinfo/download/jinai/faster_rcnn/trained_models/pam_train1.tar.gz)
  * [mt](http://www.m.u-tokyo.ac.jp/medinfo/download/jinai/faster_rcnn/trained_models/mt_train1.tar.gz)
  * [azan](http://www.m.u-tokyo.ac.jp/medinfo/download/jinai/faster_rcnn/trained_models/azan_train1.tar.gz)

  You can un-tar each tar.gz file via, e.g.,:

  ```
  tar -xvfz pas.train1.tar.gz
  ```

## test data

  テスト用のWhole Slide Image を[link](http://www.m.u-tokyo.ac.jp/medinfo/download/jinai/faster_rcnn/test_data.tar.gz)からダウンロードすることができます。それらの画像は学習に用いていません。

  それらのデータを用いて糸球体検出処理とその結果がどのようなものかを確認することができます。実行方法については[Quick Start](https://github.com/jinseikenai/glomeruli_detection/blob/master/detecting_glomeruli.md)を参照してください。

## Program List
* 糸球体検出処理用プログラム
  * 糸球体検出処理プログラム: detect_glomus_test.py
  * 検出処理結果マージプログラム: merge_overlaped_glomus.py
  * 検出処理結果確認プログラム: eval_recall_precision_test.py

* 学習用プログラム
  * 学習実行処理呼び出しプログラム: my_train.py, my_trainer.py

## 追加学習

  我々が提供する学習済みモデルを元に、自分たちのデータを追加学習することができます。追加学習を行う方法については、以下のメモと my_train.py, my_trainer.py の内容を参照してください。

1. Configuration

  config/glomerulus_train.config と config/input.config の *PATH_TO_BE_CONFIGURED* を環境に合わせて適切に設定してください。
　
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

  * ${TRAIN_DATA_PATH}には学習結果ファイルを出力するディレクトリを指定します。
  * ${CONFIG_PATH}には以下の configuration file へのパスを指定します。それぞれ別の場所を指定して大丈夫です。
    * glomerulus_model.config
    * glomerulus_train.config
    * input.config
  * 使用する GPU を指定します。"--gpu_list=0,1" or "--gpu_list=1" のように指定します。


