import os, json, sys
import pandas as pd
import numpy as np

TEST_ROOT = 'D:/ZSQ/logoData/BDCI2017-gsum-Semi-B/test50000/'  # 测试图像根目录
#RESULTS_PATH = 'D:/ZSQ/logoData/code/test/CGWcode12_3/testresults53107_12_9/resultsjson12_9/cnn7iter80000_12_9_07.json'  # 生成结果文件路径

CLASS_IDS = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010',
             '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020',
             '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029', '0030',
             '0031', '0032']

def gen_paths(root):
    f = lambda x: root + x
    for root, dirs, files in os.walk(root):
        pass
    paths = list(map(f, files))
    return paths

def get_data(det_root, thresholds):
    paths = gen_paths(det_root)
    print(len(paths))
    data = pd.DataFrame(columns=['iid', 'score', 'b0', 'b1', 'b2', 'b3', 'label'])
    for path in paths:
        print(path)
        label = path.split('.')[0][-4:]
        if label == '0031':
            real_label = '0019'
        elif label == '0032':
            real_label = '0009'
        else:
            real_label = label
        temp = pd.read_csv(path, header=None, names=['iid', 'score', 'b0', 'b1', 'b2', 'b3'], sep=' ')
        temp['label'] = real_label

        threshold = thresholds[label]
        print(str(threshold))
        temp = temp[temp.score >= threshold]

        data = data.append(temp)
    return data

def txt2json(data):
    results = []
    detIDs = list(map(lambda x: x + '.jpg', list(data.iid.drop_duplicates())))

    print('Len detect data: ', len(data))
    iids = data.iid.drop_duplicates()
    present = 0
    for iid in iids:
        print_task_rate(len(iids), present, 'txt2json')
        present += 1
        temp = data[data.iid == iid]
        iid = iid + '.jpg'
        items = []
        if len(temp) == 0:
            pass
        else:
            for row in range(len(temp)):
                label = temp.iloc[row, -1]
                score = round(float(temp.iloc[row, 1]), 6)
                bbox = list(map(int, map(round, list(temp.iloc[row, 2:6]))))
                items.append({'label_id': label, 'bbox': bbox, 'score': score})
        image = {'image_id': iid, 'type': 'B', 'items': items}
        results.append(image)
    print('Len detect image: ', len(results))
    return results, detIDs c 

def get_testIDs(testRootDir=TEST_ROOT):
    for root, dirs, files in os.walk(testRootDir):
        pass
    testIDs = files
    return testIDs

def gen_need_detResult(need_detIDs, results):
    for iid in need_detIDs:
        items = []
        image = {'image_id': iid, 'type': 'B', 'items': items}
        results.append(image)
    print('Len results: ', len(results))
    return results

def print_task_rate(total, present, task_name):
    if present == 0:
        print(task_name + ' task start')
    display_step = total // 10
    if present % display_step == 0:
        percent = float(present) * 100 / float(total)
        sys.stdout.write("%.2f" % percent);
        sys.stdout.write("%\r");
        sys.stdout.flush();
    if (present + 1) % total == 0:
        print(task_name + ' task finish!')

def to_json(path, data):
    f = open(path, 'w', encoding='utf-8')
    json.dump(data, f, ensure_ascii=False)
    f.close()

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    #RESULTS_PATH = 'D:/12_11/merged_12_11_03_fanxiang.json'

    det_root1 = 'results/results12_7_pycnn_5_7_100000/'
    values1 = [0.35] * 32
    thresholds1 = dict(zip(CLASS_IDS, values1))
    detData1 = get_data(det_root1, thresholds1)

    det_roo2 = 'results/py-faster-rcnn2_15000_iters100000test53107_12_10/Main/'
    values2 = np.array([0.7, 0.76, 0.82, 0.81, 0.65, 0.61, 0.76, 0.76, 0.72, 0.74, 0.86, 0.58, 0.75, 0.92,
               0.6, 0.5, 0.72, 0.7, 0.76, 0.79, 0.89, 0.76, 0.62, 0.71, 0.64, 0.66, 0.66, 0.66, 0.68, 0.57, 0.72, 0.69])

    thresholds2 = dict(zip(CLASS_IDS, values2))
    detData2 = get_data(det_roo2, thresholds2)

    det_root3 = 'results/py-faster-rcnn3_15000_iters100000test53107_12_10/Main/'
    values3 = [0.74, 0.74, 0.69, 0.71, 0.74, 0.78, 0.78, 0.84, 0.89, 0.78, 0.93, 0.57, 0.9, 0.79, 0.71,
               0.57, 0.62, 0.65, 0.79, 0.81, 0.82, 0.64, 0.83, 0.67, 0.79, 0.63, 0.77, 0.83, 0.71, 0.82, 0.85, 0.67]
    thresholds3 = dict(zip(CLASS_IDS, values3))
    detData3 = get_data(det_root3, thresholds3)

    detData1 = detData1.append(detData2)
    detData1 = detData1.append(detData3)


    results, detIDs = txt2json(detData1)
    testIDs = get_testIDs()
    print('Len test image: ', len(testIDs))
    need_detIDs = list(set(detIDs) ^ set(testIDs))
    print('Len need detect image: ', len(need_detIDs))
    results = gen_need_detResult(need_detIDs, results)

    RESULTS_PATH = 'results/result.json'
    to_json(RESULTS_PATH, results)
