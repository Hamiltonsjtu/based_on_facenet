import os
import shutil

scr = r'F:\savePic'
dst = r'F:\SAVE_PIC'
if not os.path.exists(dst):
    os.mkdir(dst)
sub_dir = os.listdir(scr)

# for i in sub_dir:
#     sub_dir_full = scr + '/' + i
#     peos = os.listdir(sub_dir_full)
#
#     for j in peos:
#         peo_dir = sub_dir_full + '/' + j
#         if os.path.isdir(peo_dir):
#             dst_peo = sub_dir_full + '/' + i[:-3] + '_' + j
#             os.rename(peo_dir, dst_peo)
#             for k in os.listdir(dst_peo):
#                 img_scr = dst_peo + '/' + k
#                 img_dst = dst_peo + '/' + i[:-3] + '_' + k
#                 os.rename(img_scr, img_dst)
#         else:
#             print('=== delete not folder files ===')
#             os.remove(peo_dir)

for i in sub_dir:
    sub_dir_full = os.path.join(scr, i)
    peos = os.listdir(sub_dir_full)
    for j in peos:
        sub_sub_dir_full = os.path.join(sub_dir_full, j)
        subsub2sub = os.path.join(dst, j)
        if not os.path.exists(subsub2sub):
            os.mkdir(subsub2sub)
        # os.system('scp sub_sub_dir_full subsub2sub')
        for k in os.listdir(sub_sub_dir_full):
            if k.endswith('.db'):
                continue
            else:
                img_path = os.path.join(sub_sub_dir_full, k)
                img_subsub2sub = os.path.join(subsub2sub, k)
                if os.path.isdir(img_path):
                    continue
                else:
                    shutil.copy2(img_path, img_subsub2sub, follow_symlinks=True)


