"""Decode the MsCelebV1 dataset in TSV (tab separated values) format downloaded from
https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import numpy as np
import base64
import sys
import os
import cv2
import argparse


# File format: text files, each line is an image record containing 6 columns, delimited by TAB.
# Column1: Freebase MID
# Column2: Query/Name
# Column3: ImageSearchRank
# Column4: ImageURL
# Column5: PageURL
# Column6: ImageData_Base64Encoded

def main(args):

    # output_dir = os.path.expanduser(args.output_dir)
    output_dir = r'E:\CelebA'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Store some git revision info in a text file in the output directory
    # src_path, _ = os.path.split(os.path.realpath(__file__))
    file_to_read = open(r'E:\BaiduNetdiskDownload\CelebA\Anno\identity_CelebA.txt')

    src = r'E:\BaiduNetdiskDownload\CelebA\Img\img_align_celeba\img_align_celeba'

    while True:
        line = file_to_read.readline()
        fields = line.split(' ')
        img_name = fields[0]
        class_dir = fields[1][:-1]
        img = cv2.imread(src + '/' + img_name, cv2.IMREAD_COLOR)  #

        full_class_dir = output_dir + '/' + str(class_dir)
        if not os.path.exists(full_class_dir):
            os.mkdir(full_class_dir)

        full_path = os.path.join(full_class_dir, img_name)
        cv2.imwrite(full_path, img)

        # img_name = fields[1] + '-' + fields[4] + '.' + args.output_format
        # img_string = fields[5]
        # img_dec_string = base64.b64decode(img_string)
        # img_data = np.fromstring(img_dec_string, dtype=np.uint8)
        # img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)  # pylint: disable=maybe-no-member
        # if args.size:
        #     img = misc.imresize(img, (args.size, args.size), interp='bilinear')
        # full_class_dir = os.path.join(output_dir, class_dir)
        # if not os.path.exists(full_class_dir):
        #     os.mkdir(full_class_dir)
        # full_path = os.path.join(full_class_dir, img_name.replace('/', '_'))
        # cv2.imwrite(full_path, img)  # pylint: disable=maybe-no-member
        # print('%8d: %s' % (i, full_path))
        # i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--output_dir', type=str, help='Output base directory for the image dataset',
    #                     default=r'E:\CelebA')
    # parser.add_argument('--txt_files', type=argparse.FileType('r'), nargs='+', help='Input TSV file name(s)',
    #                     default=r'E:\BaiduNetdiskDownload\CelebA\Anno\identity_CelebA.txt')
    parser.add_argument('--size', type=int, help='Images are resized to the given size')
    parser.add_argument('--output_format', type=str, help='Format of the output images', default='jpg',
                        choices=['png', 'jpg'])

    main(parser.parse_args())

