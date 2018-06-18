# Copyright 2018 The University of Tokyo Hospital. All Rights Reserved.
# <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This program is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
r"""
This program is the second unit of Faster R-CNN-Based Glomerular Detector.
This unit merge Overlapping Regions of detected glomeruli from multi-overlapping sliding windows of detectiong.
This is the second step of the detection procedure as follows.
  1. Glomeruli Detection
  2. Merging Overlapping Regions
  3. Evaluation and Visualization
"""
import csv
import os
import argparse
from glomus_handler import get_staining_type
import time
import openslide


class MargeOverlapedGlomusException(Exception):
    pass


class MargeOverlapedGlomus(object):
    def __init__(self, staining_type, input_file, output_dir, training_type, conf_threshold, annotation_dir,
                 overlap_threshod):
        self.rect_list = [[]]
        self.input_file = input_file
        self.output_dir = output_dir
        '''個別面積が共通面積の35%以上であれば共通化する'''
        '''値は仮決め'''
        self.OVERLAP_THRESHOLD = overlap_threshod
        '''両方のovelap比率が一定以上の場合には大きさに関係なくマージする'''
        self.UNCONDITIONAL_MERGE_THRESHOLD = 0.6
        '''一方の辺がほぼ同じで場合には大きさに関係なくマージする'''
        self.SIDE_LENGTH_MERGE_THRESHOLD = 30 # マイクロメートル
        self.staining_type = staining_type
        self.staining_dir = get_staining_type(staining_type)
        self.training_type = training_type

        self.CONF_THRESH = conf_threshold

        '''mpp値のチェック用にOriginal画像を開くための変数'''
        self.annotation_dir = annotation_dir
        self.slide = None

        '''糸球体は最大350マイクロメートルと想定する'''
        #self.MAX_GLOMUS_SIZE = 240.0
        #self.MAX_GLOMUS_AREA = 220.0 * 220.0
        self.MAX_GLOMUS_SIZE = 350.0
        self.MAX_GLOMUS_AREA = 300.0 * 300.0

        '''target list の mpp 情報を記録しておくための辞書'''
        self.target_list = {}

    def run(self, target_list):
        '''target list の mpp 情報を辞書に記録しておく'''
        if os.path.isfile(target_list):
            with open(target_list, 'r') as target_list_file:
                lines = target_list_file.readlines()
                for line in lines:
                    line_parts = line.strip().split(',')
                    if len(line_parts) < 7:
                        # raise AttributeError('The format of the target_list is inappropriate.')
                        org_slide_width = 0
                        org_slide_height = 0
                        org_slide_objective_power = 0.0
                        slide_downsample = 0.0
                        mpp_x = 0.0
                        mpp_y = 0.0
                    else:
                        org_slide_width = int(line_parts[1])
                        org_slide_height = int(line_parts[2])
                        org_slide_objective_power = float(line_parts[3])
                        slide_downsample = float(line_parts[4])
                        mpp_x = float(line_parts[5])
                        mpp_y = float(line_parts[6])

                    line_parts = line_parts[0].split('/')
                    image_file_id = line_parts[1]

                    self.target_list[image_file_id] = {'org_slide_width': org_slide_width,
                                                       'org_slide_height': org_slide_height,
                                                       'org_slide_objective_power': org_slide_objective_power,
                                                       'slide_downsample': slide_downsample,
                                                       'mpp_x': mpp_x, 'mpp_y': mpp_y}

        with open(self.input_file, "r") as list_file:
            site_name = ''
            # data_date = '' # data_date の利用は廃止
            patient_id = ''
            prev_file_name = ''
            reader = csv.reader(list_file)
            tmp_rect_list = []

            '''処理手順を記録する都合上timestampは廃止する'''
            # date_now = datetime.datetime.today()
            # timestamp = date_now.strftime('%Y-%m-%dT%H%M')

            file_body = self.staining_type + '_GlomusMergedList_' + self.training_type
            log_file_path = os.path.join(self.output_dir, file_body + '_log.csv')
            merged_file_path = os.path.join(self.output_dir, file_body + '.csv')
            with open(merged_file_path, "w") as merged_file:
                with open(log_file_path, "w") as log_file:
                    '''処理時間を計測する'''
                    start_time = time.time()
                    for row in reader:
                        '''ファイルの切り替わりを検出する'''
                        '''ファイルが切り換わったらoverlap検出も初期化する'''
                        '''row[3]がファイル名'''
                        if prev_file_name == '' or prev_file_name != row[2]:
                            '''切り換わった段階で前のファイルの情報を出力する'''
                            if prev_file_name != '':
                                '''スライド全部を一括してoverlapをチェックする'''
                                '''その前にオリジナルスライドの mpp をチェックする'''
                                mpp_x, mpp_y = self.check_mpp(patient_id, prev_file_name)

                                self.check_overlap_from_list(tmp_rect_list, mpp_x, mpp_y)
                                for rect in self.rect_list:
                                    merged_file.write(site_name + ',' + patient_id + ',\"' + prev_file_name + '\",' +
                                                      str(int(rect[0])) + ',' + str(int(rect[1])) + ',' +
                                                      str(int(rect[2])) + ',' + str(int(rect[3])) + ',' +
                                                      str(rect[4]) + '\n')
                                    merged_file.flush()
                                print('"{}":{}').format(prev_file_name, self.rect_list)

                                '''時間記録'''
                                duration = time.time() - start_time
                                log_file.write('"{}",{}\n'.format(prev_file_name, duration))
                                log_file.flush()
                                start_time = time.time()

                            site_name = row[0]
                            # data_date = row[1] # date_date の利用は廃止
                            patient_id = row[1]
                            '''row[3]がファイル名'''
                            prev_file_name = row[2]


                            del self.rect_list[:]
                            del tmp_rect_list[:]

                        '''確信度がしきい値以上の場合のみ領域として採用する'''
                        if float(row[9]) >= self.CONF_THRESH:
                            area = (float(row[7]) - float(row[5])) * (float(row[8]) - float(row[6]))
                            new_rect = list(map(float, row[5:10]))
                            new_rect.append(area)
                            '''overlap を書き込めるようにしておく'''
                            new_rect.append(0.0)
                            tmp_rect_list.append(new_rect)
                            '''overlapをチェックする'''
                            # check_overlap_from_list を用いる場合は個別にチェックしない
                            # self.check_overlap(new_rect)

                    '''最後の周回の結果出力'''
                    '''スライド全部を一括してoverlapをチェックする'''
                    mpp_x, mpp_y = self.check_mpp(patient_id, prev_file_name)
                    self.check_overlap_from_list(tmp_rect_list, mpp_x, mpp_y)
                    for rect in self.rect_list:
                        merged_file.write(site_name + ',' + patient_id + ',\"' + prev_file_name + '\",' +
                                          str(int(rect[0])) + ',' + str(int(rect[1])) + ',' +
                                          str(int(rect[2])) + ',' + str(int(rect[3])) + ',' +
                                          str(rect[4]) + '\n')
                        merged_file.flush()
                    print('{}:{}'.format(prev_file_name, self.rect_list))

                    '''時間記録'''
                    duration = time.time() - start_time
                    log_file.write('"{}",{}\n'.format(prev_file_name, duration))
                    log_file.flush()

    '''スライド分を一括してoverlapをチェックする'''
    def check_overlap_from_list(self, tmp_rect_list, mpp_x, mpp_y):
        '''一概には言えないが面積順にソートしておいた方が適切そう。（面積が大きい -> 糸球体全体を捉えている可能性大）'''
        # tmp_rect_listをscore順にソートしておく
        # tmp_rect_list = sorted(tmp_rect_list, key=lambda x:float(x[4]), reverse=True)
        # tmp_rect_listを面積順にソートしておく
        tmp_rect_list = sorted(tmp_rect_list, key=lambda x:float(x[5]), reverse=True)
        # tmp_rect_list = sorted(tmp_rect_list, key=lambda x:float(x[5]))
        for rect in tmp_rect_list:
            self.check_overlap(rect, mpp_x, mpp_y)

    '''overlapしたrectをマージする'''
    def check_overlap(self, new_rect, mpp_x, mpp_y):
        new_rect_list = []
        mearged_flag = False

        '''overlapを計算してoverlap順にソートしておく'''
        self.calc_overlap_all(new_rect)
        self.rect_list = sorted(self.rect_list, key=lambda x:float(x[6]), reverse=True)

        for rect in self.rect_list:
            '''rect と new_rect のマージを試みる'''
            merged_rect = self.merge_rect(rect, new_rect, mpp_x, mpp_y)
            '''重複比率が一定値未満でnew_rectの長辺がしきい値（マイクロメートル）を越える場合はそれ以上のマージを行わない'''
            # if not(self.merge_decision(new_rect, mpp_x, mpp_y, overlap)):
            #     merged_rect = None

            if not(merged_rect is None):
                '''マージされたrectとさらに他のrectのマージ可能性をチェックする'''
                tmp_merged_rect = self.recheck_overlap(new_rect_list, merged_rect, mpp_x, mpp_y)
                if not(tmp_merged_rect is None):
                    merged_rect = tmp_merged_rect

                '''マージされた領域を引き継ぐ'''
                new_rect_list.append(merged_rect)
                mearged_flag = True

                '''new_rectもmerged_rectにする'''
                new_rect = merged_rect

            else:
                '''既存 rect はそのまま引き継ぐ'''
                new_rect_list.append(rect)

        '''一度もマージされてなければ新規rectを rect_list に加える'''
        if not(mearged_flag):
            new_rect_list.append(new_rect)

        self.rect_list = new_rect_list
        return mearged_flag

    '''self.rect_list に対する overlapを事前に計算しておく'''
    def calc_overlap_all(self, new_rect):
        for rect in self.rect_list:
            overlap_area = self.calc_overlap(new_rect, rect)
            '''overlap'''
            rect[6] = overlap_area

    '''new_rectを作ったときに、再度、他の rect との重複をチェックする'''
    def recheck_overlap(self, new_rect_list, new_rect, mpp_x, mpp_y):
        merged_rect = None
        remove_index = []
        for i in range(0, len(new_rect_list)):
            rect = new_rect_list[i]

            '''rect と new_rect のマージを試みる'''
            merged_rect = self.merge_rect(rect, new_rect, mpp_x, mpp_y)

            '''重複比率が一定値未満でnew_rectの長辺がしきい値（マイクロメートル）を越える場合はそれ以上のマージを行わない'''
            # if not(self.merge_decision(new_rect, mpp_x, mpp_y, overlap)):
            #     merged_rect = None

            if not(merged_rect is None):
                remove_index.append(i)

        for i in remove_index[::-1]:
            new_rect_list.pop(i)

        return merged_rect

    '''重複があればマージする。重複が無ければ None を返す'''
    def merge_rect(self, rect, new_rect, mpp_x, mpp_y):
        merged_rect = None

        '''新しいrectの右辺が既存rectの左辺よりも大きい and 新しいrectの左辺が既存rectの右辺よりも小さい
         and
         新しいrectの下辺が既存rectの上辺よりも大きい and 新しいrectの上辺が既存rectの上辺よりも小さい'''
        overlap_area = self.calc_overlap(new_rect, rect)
        #overlap_area = self.calc_iou(new_rect, rect)
        if  overlap_area > 0.0:

            '''個別面積を求める'''
            area1 = (rect[2] - rect[0]) * (rect[3] - rect[1])
            area2 = (new_rect[2] - new_rect[0]) * (new_rect[3] - new_rect[1])

            '''集約判定を行う'''
            if self.merge_decision(rect, new_rect, area1, area2, overlap_area, mpp_x, mpp_y):
            # if overlap >= self.OVERLAP_THRESHOLD:
            # if False: # 集約しないテスト
                new_x1 = min(new_rect[0], rect[0])
                new_y1 = min(new_rect[1], rect[1])
                new_x2 = max(new_rect[2], rect[2])
                new_y2 = max(new_rect[3], rect[3])

                '''確信度は大きい方を採用することにしてみる'''
                merged_rect = [new_x1, new_y1, new_x2, new_y2, max(new_rect[4], rect[4]), (new_x2-new_x1) * (new_y2-new_y1), 0.0]

                '''if (merged_rect[2] - merged_rect[0] > self.MAX_GLOMUS_SIZE / mpp_x)\
                        or (merged_rect[3] - merged_rect[1] > self.MAX_GLOMUS_SIZE / mpp_y):
                    merged_rect = None
                '''

        return merged_rect

    def calc_overlap(self, rect1, rect2):
        overlap_area = 0.0
        if (rect1[2] >= rect2[0] and rect1[0] <= rect2[2]) and (rect1[3] >= rect2[1] and rect1[1] <= rect2[3]):
            '''共通面積を求める'''
            x1 = max(rect1[0], rect2[0])
            y1 = max(rect1[1], rect2[1])
            x2 = min(rect1[2], rect2[2])
            y2 = min(rect1[3], rect2[3])
            overlap_area = (x2 - x1) * (y2 - y1)

        return overlap_area

    """2つの長方形のIoUを求める"""
    def calc_iou(self, gt, ca):
        dx = min(ca[2], gt[2]) - max(ca[0], gt[0])
        dy = min(ca[3], gt[3]) - max(ca[1], gt[1])

        overlap = 0.0
        score = 0.0
        if (dx > 0) and (dy > 0):
            overlap = dx * dy

        '''重複がある場合はIoU(Intersection over Union)を計算する'''
        if overlap > 0:
            w_ca = ca[2] - ca[0]
            w_gt = gt[2] - gt[0]
            h_ca = ca[3] - ca[1]
            h_gt = gt[3] - gt[1]
            assert w_ca > 0, 'candidate width has invalid value.'
            assert w_gt > 0, 'gt width has invalid value.'
            assert h_ca > 0, 'candidate height has invalid value.'
            assert h_gt > 0, 'gt height has invalid value.'
            area_ca = w_ca * h_ca
            area_gt = w_gt * h_gt

            score = overlap / (area_ca + area_gt - overlap)

        return score

    '''マージするか否かの判定を行う'''
    def merge_decision(self, rect1, rect2, area1, area2, overlap_area, mpp_x, mpp_y):
        """"""
        '''ほぼ同じ領域はマージする'''
        if overlap_area >= area1 * self.UNCONDITIONAL_MERGE_THRESHOLD and overlap_area >= area2 * self.UNCONDITIONAL_MERGE_THRESHOLD:
        # if overlap_area >= self.UNCONDITIONAL_MERGE_THRESHOLD:
            return True

        '''一方の辺がほぼ同じで場合には大きさに関係なくマージする'''
        if abs(rect1[0] - rect2[0]) * mpp_x < self.SIDE_LENGTH_MERGE_THRESHOLD and abs(rect1[2] - rect2[2]) * mpp_x < self.SIDE_LENGTH_MERGE_THRESHOLD \
                and (abs(rect1[1] - rect2[1]) * mpp_y < self.SIDE_LENGTH_MERGE_THRESHOLD or abs(rect1[3] - rect2[3]) * mpp_y < self.SIDE_LENGTH_MERGE_THRESHOLD):
            return True
        elif abs(rect1[1] - rect2[1]) * mpp_y < self.SIDE_LENGTH_MERGE_THRESHOLD and abs(rect1[3] - rect2[3]) * mpp_y < self.SIDE_LENGTH_MERGE_THRESHOLD \
                and (abs(rect1[0] - rect2[0]) * mpp_x < self.SIDE_LENGTH_MERGE_THRESHOLD or abs(rect1[2] - rect2[2]) * mpp_x < self.SIDE_LENGTH_MERGE_THRESHOLD):
            return True

        '''極端に大きな領域とはマージしない'''
        if max(rect1[2]-rect1[0], rect2[2]-rect2[0]) > self.MAX_GLOMUS_SIZE/mpp_x\
                or max(rect1[3]-rect1[1], rect2[3]-rect2[1]) > self.MAX_GLOMUS_SIZE/mpp_y:
            return False
        if max(area1, area2) > self.MAX_GLOMUS_AREA/mpp_x/mpp_y:
            return False

        '''一定以上含有される領域はマージする'''
        if max(overlap_area/area1, overlap_area/area2) >= self.OVERLAP_THRESHOLD:
        # if overlap_area >= self.OVERLAP_THRESHOLD:
            return True

        return False

    '''極端に大きな領域はマージしない
    def merge_decision(self, new_rect, mpp_x, mpp_y, overlap):
        long_side = new_rect[2] - new_rect[0]
        mpp = mpp_x
        if long_side < new_rect[3] - new_rect[1]:
            long_side = new_rect[3] - new_rect[1]
            mpp = mpp_y
        long_side = long_side * mpp
        if long_side > self.MAX_GLOMUS_SIZE:
            return False
        # if overlap >= self.UNCONDITIONAL_MERGE_THRESHOLD or long_side < self.MAX_GLOMUS_SIZE:
        #     # overlap が 0.85 以上、または new_rectの長辺がしきい値（マイクロメートル）を未満の場合はマージを行う
        #     return True

        return True
    '''

    def check_mpp(self, patient_id, file_name):
        file_path = os.path.join(self.annotation_dir, self.staining_dir, patient_id, file_name)
        if os.path.isfile(file_path):
            '''前に開いていたスライドを閉じる'''
            if not (self.slide is None):
                self.slide.close()
            self.slide = openslide.open_slide(file_path)
            '''pixelあたりの大きさ(micrometre)'''
            mpp_x = float(self.slide.properties[openslide.PROPERTY_NAME_MPP_X])
            mpp_y = float(self.slide.properties[openslide.PROPERTY_NAME_MPP_Y])
        else:
            '''pixelあたりの大きさ(micrometre)'''
            body, _ = os.path.splitext(file_name)
            properties = self.target_list[body]
            if properties is not None:
                mpp_x = float(properties['mpp_x'])
                mpp_y = float(properties['mpp_y'])
            else:
                raise MargeOverlapedGlomusException('unknown target file name is given.')

        return mpp_x, mpp_y

def parse_args():
    '''
    Parse input arguments
    :return: args
    '''
    parser = argparse.ArgumentParser(description='MERGE_OVERLAPPED_GLOMUS')
    parser.add_argument('--staining', dest='staining', help="Please set --staining for 染色方法 like OPT_PAS", type=str,
                        default='OPT_PAS')
    parser.add_argument('--target_list', dest='target_list', help="set target_list", type=str)
    parser.add_argument('--detected_list', dest='input_file', help="Please set --input_file", type=str,
                        default='/home/simamoto/work/tmp/ClippedGlomus/PAS')
    parser.add_argument('--output_dir', dest='output_dir', help="Please set --output_dir", type=str,
                        default='/home/simamoto/work/tmp/ClippedGlomus/PAS')
    parser.add_argument('--output_file_ext', dest='training_type', help="Please set --training_type", type=str,
                        default='')
    parser.add_argument('--conf_threshold', dest='conf_threshold', help="Please set --conf_threshold", type=float,
                        default=0.6)
    parser.add_argument('--data_dir', dest='annotation_dir', help="Please set --data_dir", type=str)
    parser.add_argument('--overlap_threshold', dest='overlap_threshold', help="Please set --overlap_threshold", type=float)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    merger = MargeOverlapedGlomus(args.staining, args.input_file, args.output_dir, args.training_type, args.conf_threshold,
                                  args.annotation_dir, args.overlap_threshold)
    merger.run(args.target_list)