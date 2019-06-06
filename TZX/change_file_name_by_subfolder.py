"""
While using facenet lib to do classification or recognition, the name of pictures
include the folder name and its index, for example:

-- Abel_Ken
-----------| Abel_Ken_0001.jpg
-----------| Abel_Ken_0002.jpg

-----------| Abel_Ken_00010.jpg

-- Sam_Jorn
-----------| Sam_Jorn_0001.jpg
-----------| Sam_Jorn_0002.jpg

-----------| Sam_Jorn_00010.jpg

author: Shuai Zhu @TZX
"""
import os
import sys
import argparse


def main(args):
    pic_downloaded = args.data_dir
    print(pic_downloaded)
    for pic_folder in os.listdir(pic_downloaded):
        pic_folder_dir = os.path.join(pic_downloaded, pic_folder)
        if os.path.isdir(pic_folder_dir):
            folder_name = pic_folder
            pic_folder_dir_pics = os.listdir(pic_folder_dir)
            # print('pic_folder is {} and is directionary is {}'.format(folder_name, pic_folder_dir_pics))
            # print('==========================================')
            for i, file_name in enumerate(pic_folder_dir_pics):
                rename_file = folder_name + '_' + '{:04d}'.format(i) + '.png'
                src_ = os.path.join(pic_folder_dir, file_name)
                dst_ = os.path.join(pic_folder_dir, rename_file)
                if os.path.isfile(dst_):
                    continue
                else:
                    os.rename(src_, dst_)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
        help='Directory containing images.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))