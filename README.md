# glomeruli_detection
Faster RCNN を用いた糸球体検出器

detection of glomeruli using faster rcnn

## set up
1. Installing Tensorflow and Tensorflow Object Detection API

    Please run through the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) to install *"Tensorflow Object Detection API"* and all it dependencies.

2. Download our tools

    検出処理プログラムとサンプルデータ、学習済みモデルをダウンロードする。

    ```
    git clone https://github.com/jinseikenai/glomeruli_detection.git
    ```

1. dependencies

    Our Glomeruli Detection Programs depends on the following libraries:

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

## Program List
* 糸球体検出処理
  * 糸球体検出処理プログラム: detect_glomus_test.py
  * 検出処理結果マージプログラム: merge_overlaped_glomus.py
  * 検出処理結果確認プログラム: eval_recall_precision_test.py

## 追加学習
* Configuration
  * config/glomerulus_train.config と config/input.config の *PATH_TO_BE_CONFIGURED* を環境に合わせて適切に設定する。
　
