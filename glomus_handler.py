# Copyright 2018 The University of Tokyo Hospital. All Rights Reserved.
# <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This program is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

r"""
A super class of handling virtual slide files.
This Unit manage the symbol about representing staining method.
"""

import re


class GlomusHanderException(Exception):
    pass


class GlomusHandler(object):
    """
    Common class of handling stain slide images.
    """
    '''Set the file identification pattern corresponding to the image and stain type.'''
    def set_type(self, data_category):

        if data_category == 'OPT_PAM':
            self.TYPE = 'OPT_PAM'
            self.pattern = r'.*PAM.*\.ndpi'
        elif data_category == 'OPT_MT':
            self.TYPE = 'OPT_MT'
            self.pattern = r'.*MT.*\.ndpi'
        elif data_category == 'OPT_PAS':
            self.TYPE = 'OPT_PAS'
            self.pattern = r'.*PAS.*\.ndpi'
        elif data_category == 'OPT_HE':
            self.TYPE = 'OPT_HE'
            self.pattern = r'.*HE.*\.ndpi|.*\d+ - \d+.*\.ndpi|.*\d+-\d*\.ndpi'
        elif data_category == 'OPT_Azan':
            self.TYPE = 'OPT_Azan'
            self.pattern = r'.*Azan.*\.ndpi'
        else:
            raise GlomusHanderException('Unknown Augument is given.:' + data_category)

        self.repattern = re.compile(self.pattern, re.IGNORECASE)

    @staticmethod
    def get_staining_type(staining_type):
        """
        :param staining_type:
        :return: string of the serial number and string stand for staining method.
        """
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

