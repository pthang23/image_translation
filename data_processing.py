import os
import shutil

data_dir = '/mnt/md1/Sources/thangpvv/translating-images-into-maps_custom/nuscenes_data/samples'
data_dir = '/mnt/md1/Sources/thangpvv/translating-images-into-maps_custom/nuseces_data/samples'

count = 0
for file in os.listdir(data_dir):
    # if file.startswith('('):
    #     file_dir = os.path.join(data_dir, file)
    #     os.remove(file_dir)
    #     count += 1

    if file == 'n015-2018-11-21-19-11-29+0800__CAM_FRONT__1542798825112460.jpg':
        print(file)
print(count)

# add_images_dir = 'nuseces_data/cam_front'
# for file in os.listdir(add_images_dir):
#     file_dir = os.path.join(add_images_dir, file)
#     new_file_dir = os.path.join(data_dir, file)
#     os.rename(file_dir, new_file_dir)