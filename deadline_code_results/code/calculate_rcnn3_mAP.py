import os, pickle
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET


TEST_ROOT = 'validate_results/py-faster-rcnn3_15000_iters100000test53107_12_10/Main/'    # 模型测试结果根目录
ANNOTATIONS_ROOT = 'data/validate_annotations/Annotations/'  # 标注根目录
CLASS_IDS = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010',
             '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020',
             '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029', '0030',
             '0031', '0032']

def get_classX_test_paths(class_id, test_root = TEST_ROOT):
    # 获取类别X的测试结果路径
    classX_test_paths, files = [], None
    for root, dirs, files in os.walk(test_root):
        pass
    for file in files:
        if file[-8:-4] == class_id:
            classX_test_paths.append(file)
    f = lambda x: test_root + x
    classX_test_paths = list(map(f, classX_test_paths))
    return classX_test_paths

def read_classX_test_data(classX_test_paths):
    # 获取类别X的测试结果数据
    classX_test_data = pd.DataFrame(columns=['image_id', 'score', 'b0', 'b1', 'b2', 'b3', 'predict_class'])

    for classX_test_path in classX_test_paths:
        temp = pd.read_csv(classX_test_path, header=None, names=['image_id', 'score', 'b0', 'b1', 'b2', 'b3'], sep=' ')
        temp['predict_class'] = class_id
        classX_test_data = classX_test_data.append(temp)

    classX_test_data = classX_test_data.sort_values('score', ascending=False).reset_index(drop=True)
    return classX_test_data

def get_annotation_paths(annotations_root = ANNOTATIONS_ROOT):
    # 获取标注路径
    files = None
    for root, dirs, files in os.walk(annotations_root):
        pass
    f = lambda x: annotations_root + x
    annotation_paths = list(map(f, files))
    return annotation_paths

def read_image_annotations(img_id, class_id, annotations_root = ANNOTATIONS_ROOT):
    # 获取图像的标注
    annotations_path = annotations_root + img_id + '.xml'
    annotations_xml_root = ET.parse(annotations_path)
    object_nodes = annotations_xml_root.findall('object')
    ref_bboxs = []
    for object in object_nodes:
        if object.find('name').text == class_id:
            temp_bbox = []
            bbox_node = object.find('bndbox')
            temp_bbox.append(int(bbox_node.find('xmin').text))
            temp_bbox.append(int(bbox_node.find('ymin').text))
            temp_bbox.append(int(bbox_node.find('xmax').text))
            temp_bbox.append(int(bbox_node.find('ymax').text))
            ref_bboxs.append(temp_bbox)
    return ref_bboxs

def statistic_class_counts(annotations_root = ANNOTATIONS_ROOT):
    # 统计整个验证集中各个类别的数量
    dump_path = 'class_counts.pkl'
    if os.path.exists(dump_path):
        print('class_counts exist')
        with open(dump_path, 'rb') as f:
            class_counts = pickle.load(f)
    else:
        print('class_count not exist')
        class_counts = [0] * 32
        annotation_paths = get_annotation_paths(annotations_root)
        for annotation_path in annotation_paths:
            annotations_xml_root = ET.parse(annotation_path)
            object_nodes = annotations_xml_root.findall('object')
            for object in object_nodes:
                class_id = int(object.find('name').text)
                class_counts[class_id - 1] += 1
        with open(dump_path, 'wb') as f:
            pickle.dump(class_counts, f, protocol=4)
    return class_counts

def calculate_iou(ref_bbox, test_bbox):
    # 计算两个矩形框的iou值
    ref_x1 = ref_bbox[0]
    ref_y1 = ref_bbox[1]
    ref_x2 = ref_bbox[2]
    ref_y2 = ref_bbox[3]
    ref_width = ref_x2 - ref_x1
    ref_height = ref_y2 - ref_y1
    ref_area = ref_width * ref_height

    test_x1 = test_bbox[0]
    test_y1 = test_bbox[1]
    test_x2 = test_bbox[2]
    test_y2 = test_bbox[3]
    test_width = test_x2 - test_x1
    test_height = test_y2 - test_y1
    test_area = test_width * test_height

    start_x = min(ref_x1, test_x1)
    end_x= max(ref_x2, test_x2)
    start_y = min(ref_y1, test_y1)
    end_y = max(ref_y2, test_y2)
    inter_width = ref_width + test_width - (end_x - start_x)
    inter_height = ref_height + test_height - (end_y - start_y)
    inter_area = inter_width * inter_height

    if inter_width < 0 or inter_height < 0:
        iou = 0
    else:
        iou = inter_area * 1.0 / (ref_area + test_area - inter_area)
    return iou

def calculate_gt_precision_recall(classX_test_data, annotations_root):
    # 计算某个类别
    class_id = classX_test_data.ix[0, 'predict_class']
    ground_truth, precision, recall, tp = [], [], [], 0
    for row in range(len(classX_test_data)):#row=0
        predict_count = row + 1
        img_id = classX_test_data.iloc[row, 0]
        test_bbox = list(map(int, map(round, list(classX_test_data.iloc[row, 2:6]))))
        ref_bboxs = read_image_annotations(img_id, class_id, annotations_root)
        if len(ref_bboxs) == 0:
            ground_truth.append(0)
        else:
            IS_TP = False
            for ref_bbox in ref_bboxs:
                iou = calculate_iou(ref_bbox, test_bbox)
                if iou > 0.5:
                    IS_TP = True
            if IS_TP:
                ground_truth.append(1)
                tp += 1
            else:
                ground_truth.append(0)
        precision.append(tp / predict_count)
        recall.append(tp)
    classX_test_data['ground_truth'] = ground_truth
    classX_test_data['precision'] = precision
    classX_test_data['recall'] = recall
    return classX_test_data

def calculate_classX_ap(classX_test_data, threshold, class_count, calculated_max_precisions = None, calculated_rows = 0):
    # 计算类别X的AP 具体计算方法见链接http://blog.sina.com.cn/s/blog_9db078090102whzw.html
    if calculated_rows == 0:
        calculated_max_precisions = []
        predict_count, recall = 0, 0
    else:
        predict_count, recall = calculated_rows, classX_test_data.ix[calculated_rows - 1, 'recall']

    class_id = classX_test_data.ix[0, 'predict_class']
    classX_count = class_count[int(class_id) - 1]

    temp_data = classX_test_data[classX_test_data.score >= threshold]
    max_precisions = calculated_max_precisions[:]

    for row in range(calculated_rows, len(temp_data)):
        predict_count += 1
        temp_recall = temp_data.iloc[row, -1]
        if recall != temp_recall:
            recall = temp_recall
            temp_max_precision = temp_data.ix[row, 'precision']
            max_precisions.append(temp_max_precision)
            calculated_max_precisions.append(temp_max_precision)
    calculated_rows = len(temp_data)

    # 计算真正例没有超过该类别实际数量的max_precisions，相当于有classX_count-tp+1个正例没有预测出来，可以认为预测置信度为0
    # 这时候相当于每增加一个预测都一个真实的标签被预测错误
    tp = recall
    for i in range(classX_count - tp + 1):
        predict_count += 1
        max_precisions.append(tp / predict_count)
    ap = np.sum(np.array(max_precisions)) / len(max_precisions)

    return ap, calculated_max_precisions, calculated_rows

def get_best_threshold(classX_test_data, min_threshold, max_threshold, class_count):
    # 获取类别X的最优阈值
    max_ap, best_threshold = 0, 0
    threshold_range = [x / 100 for x in range(int(min_threshold * 100), int(max_threshold * 100), 1)]
    threshold_range.sort(reverse=True)
    calculated_max_precisions, calculated_rows = [], 0

    for threshold in threshold_range:
        ap, calculated_max_precisions, calculated_rows = \
            calculate_classX_ap(classX_test_data, threshold, class_count, calculated_max_precisions, calculated_rows)
        if ap > max_ap:
            max_ap = ap
            best_threshold = threshold
    return best_threshold, max_ap

if __name__ == '__main__':
    class_count = statistic_class_counts(ANNOTATIONS_ROOT)
    best_thresholds, max_aps = [], []
    for class_id in CLASS_IDS:
        classX_test_paths = get_classX_test_paths(class_id, TEST_ROOT)
        classX_test_data = read_classX_test_data(classX_test_paths)
        classX_test_data = calculate_gt_precision_recall(classX_test_data, ANNOTATIONS_ROOT)
        best_threshold, max_ap  = get_best_threshold(classX_test_data, 0.1, 1, class_count) # 阈值范围设定在 0.1 ~ 1
        best_thresholds.append(best_threshold)
        max_aps.append(max_ap)

        print(class_id + ' best_threshold: ' + str(best_threshold)  + ' max_ap: ' + str(max_ap))
    max_mAP = np.sum(np.array(max_aps)) / len(max_aps)
    print('best thresholds: ', best_thresholds)
    print('max mAP: ', max_mAP)
