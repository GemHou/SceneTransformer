from subprocess import call
import os.path
import glob
from tqdm import tqdm

# tfrecord_path = './data/tf_example/validation'
# idx_path = './data/idxs_validation_bs_1'

tfrecord_path = './data/tf_example/training'
idx_path = './data/idxs_training_bs_1'
batch_size = 1

for tfrecord in tqdm(glob.glob(tfrecord_path+'/*')):
    idxname = idx_path + '/' + tfrecord.split('/')[-1]
    print("tfrecord: ", tfrecord)
    print("idxname: ", idxname)
    print("python3 -m tfrecord.tools.tfrecord2idx ", tfrecord, idxname)
    call(["python3 -m tfrecord.tools.tfrecord2idx", tfrecord, idxname])
