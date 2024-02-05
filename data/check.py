
import os
pd_path = '/home/install/cjc/dataset/Mc_datasets/IXI/PD/valid/'
t2_path = '/home/install/cjc/dataset/Mc_datasets/IXI/T2/valid/'

pd_files = os.listdir(pd_path)
print(len(pd_files))
t2_files = os.listdir(t2_path)
print(len(t2_files))

for f in pd_files:
    a = f.split('-')
    f = '{}-{}-{}-{}-{}'.format(a[0], a[1], a[2], 'T2', a[4])
    if f not in t2_files:
        print(f)

for f in t2_files:
    a = f.split('-')
    f = '{}-{}-{}-{}-{}'.format(a[0], a[1], a[2], 'PD', a[4])
    if f not in pd_files:
        print(f)

