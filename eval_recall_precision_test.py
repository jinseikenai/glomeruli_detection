# Copyright 2018 The University of Tokyo Hospital. All Rights Reserved.
# This software includes the work that is distributed in the Apache Licence 2.0.
r"""
This program is unit of compiling the result of Faster R-CNN-Based Glomerular Detector.
This unit detect glomeruli from multistained Whole Slide Images(WSIs) of human renal tissue sections.
This detector is the first step of the detection procedure as follows.
  1. Glomeruli Detection
  2. Merging Overlapping Regions
  3. Evaluation and Visualization
"""
from annotation_handler import AnnotationHandler
import argparse
import os
import xml.etree.ElementTree as ElementTree
import csv
from PIL import Image
from PIL import ImageDraw, ImageFont
import re


class EvalRecallPrecision(AnnotationHandler):
    """領域アノテーションをスキャンするメソッドを提供する"""
    #font = ImageFont.truetype('FreeMono.ttf', 10)
    #font = ImageFont.truetype('NotoSansCJK-Bold.ttc', 12)
    font = ImageFont.truetype('DejaVuSans.ttf', 10)

    def __init__(self, staining_type, annotation_dir, target_list, detect_list_file,
                 iou_threshold, output_dir, no_save=False, start=0, end=0):
        super(EvalRecallPrecision, self).__init__(annotation_dir, staining_type)
        self.iou_threshold = iou_threshold
        self.detect_list_file = detect_list_file
        self.output_dir = output_dir
        self.image_ext = ['.PNG', '.png']
        self.detected_glomus_list = {}
        self.detected_patient_id = []
        self.image = None

        # self.annotation_file_pattern = '(' + self.staining_type + '_(.+)_(.+))_pw(\d\d)_ds(\d{1,2})'
        # self.repattern = re.compile(self.annotation_file_pattern, re.IGNORECASE)
        self.annotation_file_date_pattern = '^\d{8}_(.+)'
        self.re_annotation_file_date_pattern = re.compile(self.annotation_file_date_pattern)

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        self.glomus_category = ['glomerulus', 'glomerulus-kana']

        '''画像を保存しないフラグ'''
        self.no_save = no_save

        self.target_list = target_list
        self.start = start
        self.end = end

    def scan_annotation_file(self):
        """
        アノテーションファイルを見つけて
        1. 糸球体領域の正解を読み込む。
        2. 対応する糸球体領域検出結果を読み込む。
        3. 1. 2. を比較して結果レポートを作成する。
        """

        # for target in self.detected_glomus_list.keys():
        #     print target
        self.print_header()
        with open(self.target_list, "r") as list_file:
            lines = list_file.readlines()
            if self.end == 0 or self.end > len(lines):
                end = len(lines)
            else:
                end = self.end
            for i in range(self.start, end):
                [patient_id, file_body] = lines[i].split(os.sep)
                # print patient_id, file_body
                annotation_dir = os.path.join(self.annotation_dir, self.staining_dir)
        #if os.path.isdir(annotation_dir):
            #for patient_id in self.detected_patient_id:
                dir_path = os.path.join(annotation_dir, patient_id)
                if os.path.isdir(dir_path):
                    for file_name in os.listdir(os.path.join(dir_path,'annotations')):
                        if os.path.isfile(os.path.join(os.path.join(dir_path,'annotations'), file_name)):
                            body, ext = os.path.splitext(file_name)
                            if ext == '.xml' and file_name.find(self.staining_type) == 0:
                                body_list = self.repattern.findall(body)
                                slide_name_body = body_list[0][0].replace(self.staining_type + '_' + patient_id + '_', '')
                                '''スライド名の先頭に日付情報が付与されていることがある（そういうフォーマットのアノテーションファイル名もある）。もし付与されていれば削除する。'''
                                slide_name_body_list = self.re_annotation_file_date_pattern.findall(slide_name_body)
                                if len(slide_name_body_list) == 1:
                                    slide_name_body = slide_name_body_list[0]
                                if slide_name_body in self.detected_glomus_list:
                                    del self.gt_list[:]
                                    '''アノテーションファイルに対応する画像ファイルを読み込む'''
                                    self.read_image(dir_path, body)
                                    try:
                                        self.read_annotation(os.path.join(dir_path,'annotations'), file_name)
                                        # print(self.gt_list)
                                    except ElementTree.ParseError as e:
                                        print('OOPS! {} is not well-formed:{}').format(file_name, e)

                                    # print('{}:{}:{}').format(body_list[0][0], body_list[0][1], body_list[0][2])
                                    recall, recall_hit_num = self.check_recall_precision(slide_name_body, int(body_list[0][2]))
                                    '''ファイル名にカンマ","が入っているファイルが存在する->カンマを無視する'''
                                    self.print_result_record(body.replace(',',''), recall, recall_hit_num,
                                                             str(len(self.gt_list)),
                                                             str(len(self.detected_glomus_list[slide_name_body])))

                                    '''画像記録'''
                                    # plt.savefig(os.path.join(output_dir_path, body + '.PNG'))
                                    self.save_image(self.output_dir, body + '.PNG')

    def print_result_record(self, body, recall, recall_hit_num, num_gt, num_detected):
        print('"{}",{},{},{},{}'.format(body, recall, recall_hit_num, num_gt, num_detected))

    def print_header(self):
        print('data,recall,recall_hit_num,gt_num,detect_num')

    """画像をファイルに記録する"""
    def save_image(self, path, file_name):
        if not self.no_save:
            self.image.save(os.path.join(path, file_name))

    """正解例に対するrecall/precisionを求める"""
    def check_recall_precision(self, file_key, times):
        """正解例に対するrecall/precisionを求める"""
        # print(self.detected_glomus_list[file_key])
        # fig, ax = plt.subplots(figsize=(math.ceil(float(self.image.shape[1])/100.0), math.ceil(float(self.image.shape[0])/100.0)))
        # ax.imshow(self.image, aspect='equal')
        # fig.set_figheight(round(float(self.image.shape[1])/100.0))
        # fig.set_figwidth(round(float(self.image.shape[0])/100.0))

        '''糸球体領域の描画準備'''
        if not self.no_save:
            '''透過を用いるためにRGBAにする'''
            draw = ImageDraw.Draw(self.image, 'RGBA')
        else:
            draw = None

        '''recallチェック'''
        gt_num = len(self.gt_list)
        recall_hit_num = 0
        index = -1
        # text_size = draw.font.getsize('gt')
        for gt in self.gt_list:
            index += 1
            if self.gt_name_list[index] in self.glomus_category:
                # draw.ink('#ffff00')
                if not self.no_save:
                    draw.rectangle([gt[0], gt[1], gt[2], gt[3]], fill=None, outline='yellow')
                # '''領域のラベルも書くようにする'''
                # draw.text((gt[0], gt[1] - text_size), 'gt', fill='yellow')
                gt = list(map(lambda x: x * times, gt))
                iou_list = []
                for found_rect in self.detected_glomus_list[file_key]:
                    iou = self.check_overlap(gt, found_rect)
                    if iou >= self.iou_threshold:
                        iou_list.append(iou)

                '''複数の detected rect と重複していることがある。
                iouが最大の detected rect とだけ重複しているとみなす。'''
                if len(iou_list) > 0:
                    '''hitカウントを一つ上げる'''
                    recall_hit_num += 1
                    # print('{}, {}, {}').format(gt, found_rect, iou)
        if not self.no_save:
            '''precisionチェック'''
            # print(len(self.detected_glomus_list[file_key]))
            for found_rect in self.detected_glomus_list[file_key]:
                rect = list(map(lambda x: x / times, found_rect))
                draw.rectangle([rect[0], rect[1], rect[2], rect[3]], fill=None, outline='red')
                '''領域のラベルも書くようにする'''
                max_iou = 0.0
                for gt in self.gt_list:
                    iou = self.check_overlap(gt, rect)
                    if max_iou < iou:
                        max_iou = iou
                label = 'conf:{:.2f},IoU:{:.2f}'.format(found_rect[4], max_iou)
                (text_w, text_h) = draw.textsize(label)
                draw.rectangle([rect[0], rect[1] - text_h - 4,
                                rect[0] + text_w - 10, rect[1]], fill=(255, 0, 0, 128), outline=None)
                draw.text((rect[0] + 4, rect[1] - text_h - 2), label, fill=(255, 255, 255, 128),
                          font=EvalRecallPrecision.font)
        # self.image.show()
        # plt.show()

        if gt_num != 0:
            return float(recall_hit_num) / float(gt_num), recall_hit_num
        else:
            return 0, recall_hit_num

    def read_detected_glomus_list(self):
        with open(self.detect_list_file, "r") as list_file:
            file_body = ''
            file_key = ''
            reader = csv.reader(list_file)
            for row in reader:
                body, ext = os.path.splitext(row[2].replace(' ', ''))
                if file_body != body:
                    file_body = body
                    # file_key = body
                    self.detected_glomus_list[file_body] = []
                    self.detected_patient_id.append(row[1])

                self.detected_glomus_list[file_body].append([int(row[3]), int(row[4]), int(row[5]), int(row[6]), float(row[7])])
                # print(body)

        # print(self.detected_glomus_list)

    """領域アノテーションの対象画像を読み込む"""
    def read_image(self, dir_path, body):
        for ext in self.image_ext:
            file_path = os.path.join(dir_path, body + ext)
            if os.path.isfile(file_path):
                self.image = Image.open(file_path)
                # self.image = np.asarray(im)
                # plt.imshow(im)
                # plt.show()
                break

def parse_args():
    parser = argparse.ArgumentParser(description='annotation reader')
    parser.add_argument('--staining', dest='staining_type', help="set --staining_type like OPT_PAS", type=str, required=True)
    parser.add_argument('--data_dir', dest='annotation_dir', help="set --annotation_dir", type=str, required=True)
    parser.add_argument('--target_list', dest='target_list', help="set --target_list", type=str, required=True)
    parser.add_argument('--merged_list', dest='detect_list', help="set --detect_list", type=str, required=True)
    parser.add_argument('--iou_threshold', dest='iou_threshold', help="set --iou_threshold", type=float, default=0.5)
    parser.add_argument('--output_dir', dest='output_dir', help="set --output_dir", type=str, default='.')
    parser.add_argument('--no_save', dest='no_save', help="set --no_save for test", action='store_true')
    parser.set_defaults(no_save=False)
    parser.add_argument('--start', dest='start', help="set --start for start line begin 0", type=int, default=0)
    parser.add_argument('--end', dest='end', help="set --end for end line(含まれない）", type=int, default=0)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    evalator = EvalRecallPrecision(args.staining_type, args.annotation_dir,
                                   args.target_list, args.detect_list, args.iou_threshold,
                                   args.output_dir, args.no_save, args.start, args.end)
    evalator.read_detected_glomus_list()
    evalator.scan_annotation_file()