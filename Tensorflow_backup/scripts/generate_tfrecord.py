"""
Usage:
  # Create train data:
  python generate_tfrecord.py -x ../workspace/images/train \
        -l ../workspace/annotations/label_map.pbtxt \
        -o ../workspace/annotations/train.record

  # Create test data:
  python generate_tfrecord.py -x ../workspace/images/test \
        -l ../workspace/annotations/label_map.pbtxt \
        -o ../workspace/annotations/test.record
"""

import os
import glob
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET
from object_detection.utils import dataset_util

flags = tf.compat.v1.app.flags
flags.DEFINE_string('x', '', 'Path to the image directory')
flags.DEFINE_string('l', '', 'Path to label map (.pbtxt)')
flags.DEFINE_string('o', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (
                root.find('filename').text,
                int(root.find('size')[0].text),
                int(root.find('size')[1].text),
                member[0].text,
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text)
            )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    return pd.DataFrame(xml_list, columns=column_name)

def class_text_to_int(row_label, label_map_dict):
    return label_map_dict.get(row_label, None)

def create_tf_example(group, path, label_map_dict):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    filename = group.filename.encode('utf8')
    image_format = b'jpg'

    xmins = [group.xmin / group.width]
    xmaxs = [group.xmax / group.width]
    ymins = [group.ymin / group.height]
    ymaxs = [group.ymax / group.height]
    classes_text = [group['class'].encode('utf8')]
    classes = [class_text_to_int(group['class'], label_map_dict)]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(group.height),
        'image/width': dataset_util.int64_feature(group.width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    label_map_dict = {}
    current_label = None
    
    with open(FLAGS.l, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if "name:" in line:
                parts = line.split(':', 1)
                if len(parts) < 2:
                    continue
                current_label = parts[1].strip().replace('"', '').replace("'", "")
            elif "id:" in line and current_label is not None:
                parts = line.split(':', 1)
                if len(parts) < 2:
                    continue
                try:
                    value = int(parts[1].strip())
                    label_map_dict[current_label] = value
                except ValueError:
                    continue

    print(f"Label map dictionary: {label_map_dict}")
    
    if not label_map_dict:
        print("Warning: No labels found in label map file!")
        return

    writer = tf.io.TFRecordWriter(FLAGS.o)
    examples = xml_to_csv(FLAGS.x)
    
    if examples.empty:
        print(f"Warning: No XML files found in {FLAGS.x}")
        return
    
    print(f"Found {len(examples)} annotations")
    grouped = examples.groupby('filename')
    
    record_count = 0
    for filename, group in grouped:
        tf_example = create_tf_example(group.iloc[0], FLAGS.x, label_map_dict)
        writer.write(tf_example.SerializeToString())
        record_count += 1
    
    writer.close()
    print(f"Successfully created TFRecord with {record_count} examples at {FLAGS.o}")

if __name__ == '__main__':
    tf.compat.v1.app.run()
