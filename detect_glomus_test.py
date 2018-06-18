# Copyright 2018 The University of Tokyo Hospital. All Rights Reserved.
# <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This program is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
r"""
This program is the main unit of Faster R-CNN-Based Glomerular Detector.
This unit detect the reagions of glomeruli from multistained Whole Slide Images(WSIs) of human renal tissue sections.
This detector is the first step of the detection procedure as follows.
  1. Glomeruli Detection
  2. Merging Overlapping Regions
  3. Evaluation and Visualization
"""
import argparse
import os
import math
import tensorflow as tf
import numpy as np
import datetime
import time
from glomus_handler import GlomusHandler, get_staining_type
from PIL import Image
import openslide


class GlomusDetector(GlomusHandler):
    """バーチャルスライドから糸球体を検出する処理を担うクラス"""
    def __init__(self, data_category, args):
        """Initializer"""

        '''identifier of a image file type'''
        self.ndpi_image_ext = ['.ndpi']
        self.png_image_ext = ['.PNG', '.png']
        self.image_type = None

        '''糸球体検出処理クラスの初期化'''
        self.data_category = data_category
        '''データカテゴリに応じて対象ファイル識別パターンを設定する(set_typeはGlomusHanderで定義されている）'''
        self.set_type(data_category)

        self.args = args

        '''標準窓サイズ(micrometre単位)'''
        if args.window_size is None or args.window_size == '':
            self.STD_SIZE = 500  # original -> 200
            self.OVERLAP_RATIO = 0.5
        else:
            self.STD_SIZE = args.window_size
            self.OVERLAP_RATIO = args.overlap_ratio

        self.COPY_EXEC = False

        self.CONF_THRESH = args.conf_threshold
        self.NMS_THRESH = 0.3

        self.CLASSES = ('__background__',  # always index 0
                        'glomerulus')

        self.staining_dir = get_staining_type(self.data_category)
        self.output_root_dir = args.output_dir
        '''ディレクトリを用意しておく'''
        head, _ = os.path.split(self.output_root_dir)
        head2, _ = os.path.split(head)
        if not os.path.isdir(head2):
            os.mkdir(head2)
        if not os.path.isdir(head):
            os.mkdir(head)
        if not os.path.isdir(self.output_root_dir):
            os.mkdir(self.output_root_dir)
        self.output_file_path = os.path.join(self.output_root_dir, self.TYPE + args.output_file_ext + '.csv')

        self.log_file = os.path.join(self.output_root_dir, self.TYPE + args.output_file_ext + '_log.csv')

        '''information of each slide'''
        self.org_slide_width = 0
        self.org_slide_height = 0
        self.org_slide_objective_power = 0.0
        self.slide_downsample = 0.0
        self.mpp_x = 0.0
        self.mpp_y = 0.0

    def split_all(self, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections):
        splited_target_dir = args.target_dir.split('/')
        site_name = splited_target_dir[-2]

        with open(self.output_file_path, "w") as output_file:
            if os.path.isfile(args.target_list):
                with open(args.target_list, 'r') as list_file:
                    with open(self.log_file, 'w') as log_file:
                        log_file.write('file,time\n')
                        lines = list_file.readlines()
                        for line in lines:
                            line_parts = line.strip().split(',')
                            if len(line_parts) < 7:
                                # raise AttributeError('The format of the target_list is inappropriate.')
                                self.org_slide_width = 0
                                self.org_slide_height = 0
                                self.org_slide_objective_power = 0.0
                                self.slide_downsample = 0.0
                                self.mpp_x = 0.0
                                self.mpp_y = 0.0
                            else:
                                self.org_slide_width = int(line_parts[1])
                                self.org_slide_height = int(line_parts[2])
                                self.org_slide_objective_power = float(line_parts[3])
                                self.slide_downsample = float(line_parts[4])
                                self.mpp_x = float(line_parts[5])
                                self.mpp_y = float(line_parts[6])

                            line_parts = line_parts[0].split('/')
                            # data_date = line_parts[0] # data_date は利用しないように変更
                            specimen_id = line_parts[0]
                            if specimen_id.startswith('#'):
                                '''# 始まりの場合はコメントとしてパスする'''
                                pass
                            else:
                                file_name = line_parts[1] #.decode('utf-8')

                                '''ターゲットファイルを取得する'''
                                target_file_path = os.path.join(args.target_dir, self.staining_dir, specimen_id)
                                if os.path.isdir(target_file_path):
                                    for candidate in os.listdir(target_file_path):
                                        candidate_body, ext = os.path.splitext(candidate)
                                        if file_name.find(candidate_body) >= 0 and ext in self.ndpi_image_ext:
                                            self.image_type = 'ndpi'
                                        elif file_name.find(candidate_body) >= 0 and ext in self.png_image_ext:
                                            self.image_type = 'png'
                                        else:
                                            self.image_type = None

                                        if self.image_type is not None:
                                            start_time = time.time()
                                            self.split(sess, site_name, self.staining_dir, specimen_id, candidate, output_file,
                                                       image_tensor, detection_boxes, detection_scores, detection_classes,
                                                       num_detections)
                                            duration = time.time() - start_time
                                            log_file.write('"{}",{}\n'.format(file_name, duration))
                                            log_file.flush()
                                            break

    def split(self, sess, site_name, staining_dir, patient_id, file_name, output_file,
              image_tensor, detection_boxes, detection_scores, detection_classes, num_detections):
        if self.image_type == 'png':
            with Image.open(os.path.join(self.args.target_dir, staining_dir, patient_id, file_name)) as img:
                self.scan_region_from_image(sess, img, site_name, patient_id, file_name, output_file,
                                 image_tensor, detection_boxes, detection_scores, detection_classes, num_detections)
        else:
            with openslide.open_slide(os.path.join(self.args.target_dir, staining_dir, patient_id, file_name)) as slide:
                self.scan_region(sess, slide, site_name, patient_id, file_name, output_file,
                                 image_tensor, detection_boxes, detection_scores, detection_classes, num_detections)

    def scan_region_from_image(self, sess, img, site_name, specimen_id, file_name, output_file,
                               image_tensor, detection_boxes, detection_scores, detection_classes, num_detections):

        '''切り出す窓サイズ(最大画素ベース)(pixel単位)'''
        window_x_org = float(self.STD_SIZE) / self.mpp_x
        window_y_org = float(self.STD_SIZE) / self.mpp_y

        '''切り出しが何回必要なのか（元のサイズを窓サイズで割って、重複比率で割って、天井関数に掛ける）'''
        x_split_times = int(math.ceil(self.org_slide_width / window_x_org / (1.0 - self.OVERLAP_RATIO)))
        y_split_times = int(math.ceil(self.org_slide_height / window_y_org / (1.0 - self.OVERLAP_RATIO)))

        '''ダウンサンプルを考慮した切り出す窓サイズ'''
        window_x = int(math.ceil(window_x_org / self.slide_downsample))
        window_y = int(math.ceil(window_y_org / self.slide_downsample))

        print('WINDOW X SIZE: ' + str(window_x))
        print('WINDOW Y SIZE: ' + str(window_y))

        '''切り出す窓のスライド量(pixel単位)'''
        slide_window_x = int(window_x * (1.0 - self.OVERLAP_RATIO))
        slide_window_y = int(window_y * (1.0 - self.OVERLAP_RATIO))

        # for short test
        # for j in range(4, 6):
        #     for i in range(25, 35):
        for j in range(0, y_split_times):
            for i in range(0, x_split_times):
                x_start = slide_window_x * i
                y_start = slide_window_y * j
                x_end = x_start + window_x
                y_end = y_start + window_y
                region = img.crop((x_start, y_start, x_end, y_end))
                im = np.asarray(region)

                '''検出処理を実行する'''
                bs = self.detect_box(sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections,
                                     im, window_x, window_y, thresh=self.CONF_THRESH)

                self.write_detected_result(bs, i, j, x_start * self.slide_downsample, y_start * self.slide_downsample,
                                           output_file, site_name, specimen_id, file_name)

    def write_detected_result(self, bs, i, j, x_start, y_start, output_file, site_name, specimen_id, file_name):
        if len(bs) == 0:
            print('X:{}, Y:{}'.format(i, j))
        else:
            for k in range(0, len(bs)):
                print('X:{}, Y:{}, RECT:[{}, {}, {}, {}, {}]'.format(i, j,
                                                                     x_start + (bs[k][0] * self.slide_downsample),
                                                                     y_start + (bs[k][1] * self.slide_downsample),
                                                                     x_start + (bs[k][2] * self.slide_downsample),
                                                                     y_start + (bs[k][3] * self.slide_downsample),
                                                                     bs[k][4]))
                if bs[k][4] > 0:
                    date_now = datetime.datetime.today()
                    output_file.write('\"' + site_name + '\",\"' + specimen_id + '\",\"' + file_name + '\",new,'
                                      + date_now.strftime('%Y-%m-%dT%H:%M:%S') + ','
                                      + str(x_start + (bs[k][0] * self.slide_downsample)) + ','
                                      + str(y_start + (bs[k][1] * self.slide_downsample)) + ','
                                      + str(x_start + (bs[k][2] * self.slide_downsample)) + ','
                                      + str(y_start + (bs[k][3] * self.slide_downsample)) + ','
                                      + str(bs[k][4]) + '\n')
                    output_file.flush()

    def scan_region(self, sess, slide, site_name, specimen_id, file_name, output_file,
                    image_tensor, detection_boxes, detection_scores, detection_classes, num_detections):
        self.org_slide_width, self.org_slide_height = slide.dimensions
        # print('W: ' + str(width) + '    H: ' + str(height))

        '''pixelあたりの大きさ(micrometre)'''
        self.mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        self.mpp_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])

        self.org_slide_objective_power = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])

        '''切り出す窓サイズ(pixel単位)'''
        window_x_org = float(self.STD_SIZE) / self.mpp_x
        window_y_org = float(self.STD_SIZE) / self.mpp_y

        '''Decide the level so that the objective magnification is 5 times'''
        self.slide_downsample = 8.0
        target_level = 3
        for level, downsample in enumerate(slide.level_downsamples):
            if self.org_slide_objective_power / downsample <= 5.0:
                target_level = level
                self.slide_downsample = slide.level_downsamples[level]
                break

        '''切り出しが何回必要なのか（元のサイズを窓サイズで割って、重複比率で割って、天井関数に掛ける）'''
        x_split_times = int(math.ceil(self.org_slide_width / window_x_org / (1.0 - self.OVERLAP_RATIO)))
        y_split_times = int(math.ceil(self.org_slide_height / window_y_org / (1.0 - self.OVERLAP_RATIO)))

        '''ダウンサンプルを考慮した切り出す窓サイズ'''
        window_x = int(math.ceil(window_x_org / self.slide_downsample))
        window_y = int(math.ceil(window_y_org / self.slide_downsample))

        print('WINDOW X SIZE: ' + str(window_x))
        print('WINDOW Y SIZE: ' + str(window_y))

        '''切り出す窓のスライド量(pixel単位)'''
        '''注意：スライド幅は最上位レベルのpixel幅で指定する'''
        slide_window_x = int(window_x_org * (1.0 - self.OVERLAP_RATIO))
        slide_window_y = int(window_y_org * (1.0 - self.OVERLAP_RATIO))

        # for short test
        # for j in range(4, 6):
        #     for i in range(25, 35):
        for j in range(0, y_split_times):
            for i in range(0, x_split_times):
                x_start = slide_window_x * i
                y_start = slide_window_y * j
                region = slide.read_region((x_start, y_start), target_level,
                                           (window_x, window_y))
                im = np.asarray(region)
                '''RGBA配列からAを削除する'''
                im = np.delete(im, 3, 2)
                # '''BGR配列をRGB配列に変換する''' <- OpenSlide で　OpenCV を使っていると勘違いしていた。Pillowを使っているのでregionはRGBになっている。
                # im = im[:, :, (2, 1, 0)]

                '''検出処理を実行する'''
                bs = self.detect_box(sess, image_tensor, detection_boxes, detection_scores, detection_classes,
                                     num_detections, im, window_x, window_y, thresh=self.CONF_THRESH)
                self.write_detected_result(bs, i, j, x_start, y_start,
                                           output_file, site_name, specimen_id, file_name)

    @staticmethod
    def detect_box(sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections,
                   im, WINDOW_X, WINDOW_Y, thresh=0.5):

        """検出処理を実行する"""
        # scores, boxes = im_detect(sess, net, im)
        # Actual detection.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(im, axis=0)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        '''一度に処理する画像は1つだけ。無駄な1次元を削除しておく。'''
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        score = np.squeeze(scores)

        '''閾値以上の結果だけを結果として採用する。'''
        inds = np.where(score[:] >= thresh)[0]
        gt_bboxes_score_precision = [[0, 0, 0, 0, 0.0]] * len(inds)

        '''推定結果のboxをリストする'''
        for i in range(0, len(inds)):
            '''bboxの並びに注意する'''
            ymin, xmin, ymax, xmax = boxes[i]
            gt_bboxes_score_precision[i] = [int(WINDOW_X * xmin), int(WINDOW_Y * ymin),
                                            int(WINDOW_X * xmax), int(WINDOW_Y * ymax), score[i]]

        '''以下はデバッグ時の表示用
        if len(gt_bboxes_score_precision) > 0:
            # Size, in inches, of the output images.
            IMAGE_SIZE = (12, 8)
            pil_image = Image.fromarray(im)
            draw = ImageDraw.Draw(pil_image)
            for box in gt_bboxes_score_precision:
                draw.rectangle(((box[0], box[1]), (box[2], box[3])), fill=None, outline='yellow')
            plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(pil_image)
            # pil_image.save('../test_images/test.PNG')
            plt.show()
        '''
        return gt_bboxes_score_precision

def parse_args():
    '''
    Parse input arguments
    :return: args
    '''
    parser = argparse.ArgumentParser(description='Load RoI')
    parser.add_argument('--model', dest='model', help="set learned model", type=str)
    parser.add_argument('--target_list', dest='target_list', help="set target_list", type=str)
    parser.add_argument('--data_dir', dest='target_dir', help="set data_dir", type=str)
    parser.add_argument('--staining', dest='data_category', help="Data Category(Staining Method)", type=str,
                        default='OPT_PAM')
    parser.add_argument('--output_dir', dest='output_dir', help="Please set --output_dir", type=str,
                        default='./output')
    parser.add_argument('--output_file_ext', dest='output_file_ext', help="Please set --output_file_ext", type=str,
                        default='_GlomusList')
    parser.add_argument('--window_size', dest='window_size', help="Please set --window_size", type=int)
    parser.add_argument('--overlap_ratio', dest='overlap_ratio', help="Please set --overlap_ratio", type=float)
    parser.add_argument('--conf_threshold', dest='conf_threshold', help="Please set --conf_threshold", type=float, default=0.6)

    return parser.parse_args()


if __name__ == '__main__':
    print('Tensorflow version:{}'.format(tf.__version__))

    args = parse_args()
    data_category = args.data_category

    # data_category_list = data_category.split('_')

    # load network
    body, ext = os.path.splitext(os.path.basename(args.target_list))
    TEST_SET = body
    TRAIN_SET = body.replace('test', 'train')
    staining_dir = get_staining_type(args.data_category)
    TRAIN_MODEL = args.model
    PATH_TO_CKPT = os.path.join(args.model, TRAIN_MODEL, 'frozen_inference_graph.pb')
    '''Load Tensorflow Model into Memory'''
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # init session
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # FasterRCNN では c のライブラリを利用しているため visible_device_list の設定は効かない。
    # GPUを制限するためには環境変数で CUDA_VISIBLE_DEVICES をセットする必要がある。
    tfConfig = tf.ConfigProto(allow_soft_placement=True,
                              gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.45, allow_growth=True
                                                        # , visible_device_list="0"
                                                        ),
                              log_device_placement=True
                              # , device_count={'GPU': 1}
                              )
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=tfConfig) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            glomus_detector = GlomusDetector(data_category, args)
            glomus_detector.split_all(sess, image_tensor, detection_boxes, detection_scores, detection_classes,
                                      num_detections)
