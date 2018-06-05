# coding=utf-8
import os
import xml.etree.ElementTree as ElementTree
import re


class AnnotationHandlerException(BaseException):
    pass


class AnnotationHandler(object):
    def __init__(self, annotation_dir, staining_type):
        self.gt_list = []
        self.gt_name_list = []
        self.annotation_dir = annotation_dir
        self.staining_type = staining_type

        self.annotation_file_pattern = '(.*)_pw(\d{2})_ds(\d{1,2})'
        self.repattern = re.compile(self.annotation_file_pattern, re.IGNORECASE)

        self.set_staining_type()
        self.set_sheet_index()

    def clear_annotation(self):
        del self.gt_list[:]
        del self.gt_name_list[:]

    '''領域アノテーションファイルを読み込む'''
    def read_annotation(self, dir_path, file_name):
        tree = ElementTree.parse(os.path.join(dir_path, file_name))
        objs = tree.findall('object')
        for ix, obj in enumerate(objs):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            if bbox != None:
                # Make pixel indexes 0-based
                x1 = float(bbox.find('xmin').text)
                y1 = float(bbox.find('ymin').text)
                x2 = float(bbox.find('xmax').text)
                y2 = float(bbox.find('ymax').text)
                self.gt_list.append([x1, y1, x2, y2])
                self.gt_name_list.append(name)
                # print('x1:{},y1:{},x2:{},y2:{}').format(x1, y1, x2, y2)
            else:
                raise AnnotationHandlerException('Unknown object is found in:' + file_name)

    def set_staining_type(self):
        if self.staining_type == 'OPT_PAS':
            self.staining_dir = '02_PAS'
        elif self.staining_type == 'OPT_PAM':
            self.staining_dir = '03_PAM'
        elif self.staining_type == 'OPT_MT':
            self.staining_dir = '05_MT'
        elif self.staining_type == 'OPT_Azan':
            self.staining_dir = '06_Azan'
        else:
            raise AnnotationHandlerException('Unknown Augument is given.:' + self.staining_type)

    def set_sheet_index(self):
        if self.staining_type == 'OPT_PAS':
            self.sheet_index = 0
        elif self.staining_type == 'OPT_PAM':
            self.sheet_index = 1
        elif self.staining_type == 'OPT_MT':
            self.sheet_index = 2
        elif self.staining_type == 'OPT_Azan':
            self.sheet_index = 3
        else:
            raise AnnotationHandlerException('Unknown Augument is given.:' + self.staining_type)

    """2つの長方形のIoUを求める"""
    def check_overlap(self, gt, ca):
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


def get_staining_type(staining_type):
    if staining_type == 'OPT_PAS':
        return '02_PAS'
    elif staining_type == 'OPT_PAM':
        return '03_PAM'
    elif staining_type == 'OPT_MT':
        return '05_MT'
    elif staining_type == 'OPT_Azan':
        return '06_Azan'
    else:
        return ''
