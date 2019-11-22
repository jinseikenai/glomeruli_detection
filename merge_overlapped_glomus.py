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
from glomus_handler import GlomusHandler
import time
import openslide


class MargeOverlapedGlomusException(Exception):
    pass


class MargeOverlapedGlomus(object):
    def __init__(self, staining_type, input_file, output_dir, training_type, conf_threshold, annotation_dir,
                 overlap_threshod):

        '''Merge when common area ratio of two individual rectangles is this threshold or more.'''
        self.OVERLAP_THRESHOLD = overlap_threshod
        '''When the common area ratios for both rectangles are equal to or more than this value,
        they are merged regardless of the size.'''
        self.UNCONDITIONAL_MERGE_THRESHOLD = 0.6
        '''If one side of rectangles is nearly the same, merge them regardless of their size.'''
        self.SIDE_LENGTH_MERGE_THRESHOLD = 30 # unit is micrometer

        '''The glomerulus diameter is assumed to be 350 micrometers at the maximum.
        If the merge result exceeds the following value, do not merge them.'''
        #self.MAX_GLOMUS_SIZE = 240.0
        #self.MAX_GLOMUS_AREA = 220.0 * 220.0
        self.MAX_GLOMUS_SIZE = 350.0
        self.MAX_GLOMUS_AREA = 300.0 * 300.0

        self.png = ['.png', '.PNG']
        self.rect_list = [[]]
        self.input_file = input_file
        self.output_dir = output_dir

        self.staining_type = staining_type
        self.staining_dir = GlomusHandler.get_staining_type(staining_type)
        self.training_type = training_type

        self.CONF_THRESH = conf_threshold

        self.annotation_dir = annotation_dir

        '''variable for opening ndpi image file for mpp value check.'''
        self.slide = None
        '''variable for recording mpp value of PNG image recorded in the target list file.'''
        self.target_list = {}

    def run(self, target_list):
        """
        Main loop
        :param target_list:
        :return: None
        """
        '''If the slide is a PNG file, record the mpp information of the target file.'''
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
            patient_id = ''
            prev_file_name = ''
            reader = csv.reader(list_file)
            tmp_rect_list = []

            file_body = self.staining_type + '_GlomusMergedList_' + self.training_type
            log_file_path = os.path.join(self.output_dir, file_body + '_log.csv')
            merged_file_path = os.path.join(self.output_dir, file_body + '.csv')
            with open(merged_file_path, "w") as merged_file:
                with open(log_file_path, "w") as log_file:
                    '''for time record'''
                    start_time = time.time()
                    for row in reader:
                        '''Detect switching of files.'''
                        '''When the file is switched,
                        the precessing results are initialized to prepare for next file.'''
                        '''row[2] is file name'''
                        if prev_file_name == '' or prev_file_name != row[2]:
                            '''When the file is switched, output the processing result of the previous file.'''
                            if prev_file_name != '':
                                '''Check overlap across the whole slide'''
                                mpp_x, mpp_y = self.check_mpp(patient_id, prev_file_name)
                                self.check_overlap_from_list(tmp_rect_list, mpp_x, mpp_y)
                                for rect in self.rect_list:
                                    merged_file.write(site_name + ',' + patient_id + ',\"' + prev_file_name + '\",' +
                                                      str(int(rect[0])) + ',' + str(int(rect[1])) + ',' +
                                                      str(int(rect[2])) + ',' + str(int(rect[3])) + ',' +
                                                      str(rect[4]) + '\n')
                                    merged_file.flush()
                                print('"{}":{}'.format(prev_file_name, self.rect_list))

                                '''for time record'''
                                duration = time.time() - start_time
                                log_file.write('"{}",{}\n'.format(prev_file_name, duration))
                                log_file.flush()
                                start_time = time.time()

                            site_name = row[0]
                            # data_date = row[1] # date_date の利用は廃止
                            patient_id = row[1]
                            '''row[2] is file name'''
                            prev_file_name = row[2]

                            del self.rect_list[:]
                            del tmp_rect_list[:]

                        '''Only candidates whose confidence value is equal to or higher than the threshold
                        are adopted as region candidates.'''
                        if float(row[9]) >= self.CONF_THRESH:
                            area = (float(row[7]) - float(row[5])) * (float(row[8]) - float(row[6]))
                            new_rect = list(map(float, row[5:10]))
                            new_rect.append(area)
                            '''preparation for writing overlap value'''
                            new_rect.append(0.0)
                            tmp_rect_list.append(new_rect)

                    '''Output of the result of the last lap.'''
                    mpp_x, mpp_y = self.check_mpp(patient_id, prev_file_name)
                    self.check_overlap_from_list(tmp_rect_list, mpp_x, mpp_y)
                    for rect in self.rect_list:
                        merged_file.write(site_name + ',' + patient_id + ',\"' + prev_file_name + '\",' +
                                          str(int(rect[0])) + ',' + str(int(rect[1])) + ',' +
                                          str(int(rect[2])) + ',' + str(int(rect[3])) + ',' +
                                          str(rect[4]) + '\n')
                        merged_file.flush()
                    print('{}:{}'.format(prev_file_name, self.rect_list))

                    '''for time record'''
                    duration = time.time() - start_time
                    log_file.write('"{}",{}\n'.format(prev_file_name, duration))
                    log_file.flush()

    def check_overlap_from_list(self, tmp_rect_list, mpp_x, mpp_y):
        """
        Check overlap across the whole slide
        :param tmp_rect_list:
        :param mpp_x:
        :param mpp_y:
        :return: None
        """
        '''Although it cannot be said unconditionally, it seems better tor sort by area.
        （Hypotheses: it is likely that the larger one captures the entire glomerulus）'''
        '''Other possibilities: the higher confidence suggests the higher adequacy.'''
        # tmp_rect_list = sorted(tmp_rect_list, key=lambda x:float(x[4]), reverse=True)
        tmp_rect_list = sorted(tmp_rect_list, key=lambda x:float(x[5]), reverse=True)
        # tmp_rect_list = sorted(tmp_rect_list, key=lambda x:float(x[5]))
        for rect in tmp_rect_list:
            self.check_overlap(rect, mpp_x, mpp_y)

    def check_overlap(self, new_rect, mpp_x, mpp_y):
        """
        Merge overlapped rectangles
        :param new_rect:
        :param mpp_x:
        :param mpp_y:
        :return: boolean value of indicating whether or not merged.
        """
        new_rect_list = []
        mearged_flag = False

        '''Calculate teh overlap ratio and sort in the overlap ratio order.'''
        self.calc_overlap_all(new_rect)
        self.rect_list = sorted(self.rect_list, key=lambda x:float(x[6]), reverse=True)

        for rect in self.rect_list:
            '''Try to merge rect and new_rect.'''
            merged_rect = self.merge_rect(rect, new_rect, mpp_x, mpp_y)

            if not(merged_rect is None):
                '''Check the possibility that the merged rect can be merged with another rect.'''
                tmp_merged_rect = self.recheck_overlap(new_rect_list, merged_rect, mpp_x, mpp_y)
                if not(tmp_merged_rect is None):
                    merged_rect = tmp_merged_rect

                '''Save the merged rect.'''
                new_rect_list.append(merged_rect)
                mearged_flag = True

                '''Replace the merged new_rect with merged_rect.'''
                new_rect = merged_rect

            else:
                '''Save the existing rect that was not merged.'''
                new_rect_list.append(rect)

        '''If it has never been merged, add new_rect to rect_list.'''
        if not(mearged_flag):
            new_rect_list.append(new_rect)

        self.rect_list = new_rect_list
        return mearged_flag

    def calc_overlap_all(self, new_rect):
        """
        Calculate overlap ratio between the new_rect and all other rects.
        :param new_rect:
        :return: None
        """
        for rect in self.rect_list:
            overlap_area = self.calc_overlap(new_rect, rect)
            '''overlap'''
            rect[6] = overlap_area

    def recheck_overlap(self, new_rect_list, new_rect, mpp_x, mpp_y):
        """
        When new_rect is created, check overlap with other rect again.
        :param new_rect_list:
        :param new_rect:
        :param mpp_x:
        :param mpp_y:
        :return: merged_rect: If there is nothing to merge, merged_rect will be None.
        """
        merged_rect = None
        remove_index = []
        for i in range(0, len(new_rect_list)):
            rect = new_rect_list[i]

            merged_rect = self.merge_rect(rect, new_rect, mpp_x, mpp_y)

            if not(merged_rect is None):
                remove_index.append(i)

        for i in remove_index[::-1]:
            new_rect_list.pop(i)

        return merged_rect

    def merge_rect(self, rect, new_rect, mpp_x, mpp_y):
        """
        Merge if there is overlap above a certain amount.
        :param rect:
        :param new_rect:
        :param mpp_x:
        :param mpp_y:
        :return: merged_rect: If there is nothing to merge, merged_rect will be None.
        """
        merged_rect = None

        overlap_area = self.calc_overlap(new_rect, rect)
        if overlap_area > 0.0:

            area1 = (rect[2] - rect[0]) * (rect[3] - rect[1])
            area2 = (new_rect[2] - new_rect[0]) * (new_rect[3] - new_rect[1])

            if self.merge_decision(rect, new_rect, area1, area2, overlap_area, mpp_x, mpp_y):
                new_x1 = min(new_rect[0], rect[0])
                new_y1 = min(new_rect[1], rect[1])
                new_x2 = max(new_rect[2], rect[2])
                new_y2 = max(new_rect[3], rect[3])

                '''adopt the bigger confidence value.'''
                merged_rect = [new_x1, new_y1, new_x2, new_y2, max(new_rect[4], rect[4]),
                               (new_x2-new_x1) * (new_y2-new_y1), 0.0]

        return merged_rect

    @staticmethod
    def calc_overlap(rect1, rect2):
        overlap_area = 0.0
        if (rect1[2] >= rect2[0] and rect1[0] <= rect2[2]) and (rect1[3] >= rect2[1] and rect1[1] <= rect2[3]):
            x1 = max(rect1[0], rect2[0])
            y1 = max(rect1[1], rect2[1])
            x2 = min(rect1[2], rect2[2])
            y2 = min(rect1[3], rect2[3])
            overlap_area = (x2 - x1) * (y2 - y1)

        return overlap_area

    def merge_decision(self, rect1, rect2, area1, area2, overlap_area, mpp_x, mpp_y):
        """
        Judging whether it is better to merge.
        :param rect1:
        :param rect2:
        :param area1:
        :param area2:
        :param overlap_area:
        :param mpp_x:
        :param mpp_y:
        :return: boolean: judgment result
        """
        '''merge the almost same areas'''
        if overlap_area >= area1 * self.UNCONDITIONAL_MERGE_THRESHOLD and overlap_area >= area2 * self.UNCONDITIONAL_MERGE_THRESHOLD:
            return True

        '''merge if one side is nealy the same.'''
        if abs(rect1[0] - rect2[0]) * mpp_x < self.SIDE_LENGTH_MERGE_THRESHOLD and abs(rect1[2] - rect2[2]) * mpp_x < self.SIDE_LENGTH_MERGE_THRESHOLD \
                and (abs(rect1[1] - rect2[1]) * mpp_y < self.SIDE_LENGTH_MERGE_THRESHOLD or abs(rect1[3] - rect2[3]) * mpp_y < self.SIDE_LENGTH_MERGE_THRESHOLD):
            return True
        elif abs(rect1[1] - rect2[1]) * mpp_y < self.SIDE_LENGTH_MERGE_THRESHOLD and abs(rect1[3] - rect2[3]) * mpp_y < self.SIDE_LENGTH_MERGE_THRESHOLD \
                and (abs(rect1[0] - rect2[0]) * mpp_x < self.SIDE_LENGTH_MERGE_THRESHOLD or abs(rect1[2] - rect2[2]) * mpp_x < self.SIDE_LENGTH_MERGE_THRESHOLD):
            return True

        '''We dose not merge with the extremely large area compared with regular size of the glomerulus.'''
        if max(rect1[2]-rect1[0], rect2[2]-rect2[0]) > self.MAX_GLOMUS_SIZE/mpp_x\
                or max(rect1[3]-rect1[1], rect2[3]-rect2[1]) > self.MAX_GLOMUS_SIZE/mpp_y:
            return False
        if max(area1, area2) > self.MAX_GLOMUS_AREA/mpp_x/mpp_y:
            return False

        '''merge areas containing more than a certain ratio.'''
        if max(overlap_area/area1, overlap_area/area2) >= self.OVERLAP_THRESHOLD:
            return True

        return False

    def check_mpp(self, patient_id, file_name):
        body, ext = os.path.splitext(file_name)
        if ext not in self.png:
            file_path = os.path.join(self.annotation_dir, self.staining_dir, patient_id, file_name)
            '''close the slide previously opened'''
            if not (self.slide is None):
                self.slide.close()
            self.slide = openslide.open_slide(file_path)
            '''mpp indicates the number of pixels per micrometer.'''
            mpp_x = float(self.slide.properties[openslide.PROPERTY_NAME_MPP_X])
            mpp_y = float(self.slide.properties[openslide.PROPERTY_NAME_MPP_Y])
        else:
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