#!/usr/bin/env python3
"""
Generate recommendation system training data with label and save as TFRecord format.
Feature definitions:
- uid: string
- age: int64
- gender: string
- device: string
- item_id: string
- clicks: list of string (click sequence)
- label: int64 (0 or 1)
"""

import random
import argparse
import tensorflow as tf

# Configurable parameters
AGE_MIN = 0
AGE_MAX = 80
GENDERS = ['M', 'F']
DEVICES = ['mobile', 'desktop', 'tablet']
LABEL_PROB_POS = 0.5  # Probability of positive label (1)

def generate_items(num_items: int) -> list:
    """Generate item ID list"""
    return [f'item_{i}' for i in range(1, num_items + 1)]

def generate_clicks(item_pool: list, main_item: str, max_len: int = 5) -> list:
    """
    Generate click sequence list, dependent on the main item ID.
    The sequence may contain the main item and other random items, deduplicated and keep order.
    Returns a list of strings.
    """
    seq_len = random.randint(1, max_len)
    candidates = random.sample(item_pool, k=min(seq_len, len(item_pool)))
    if main_item not in candidates:
        candidates[-1] = main_item
    return candidates

def generate_label() -> int:
    """Generate binary label (0 or 1)"""
    return 1 if random.random() < LABEL_PROB_POS else 0

def serialize_example(uid, age, gender, device, item_id, clicks, label):
    """
    Serialize a single sample to tf.train.Example
    """
    feature = {
        'uid': tf.train.Feature(bytes_list=tf.train.BytesList(value=[uid.encode('utf-8')])),
        'age': tf.train.Feature(int64_list=tf.train.Int64List(value=[age])),
        'gender': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gender.encode('utf-8')])),
        'device': tf.train.Feature(bytes_list=tf.train.BytesList(value=[device.encode('utf-8')])),
        'item_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item_id.encode('utf-8')])),
        'clicks': tf.train.Feature(bytes_list=tf.train.BytesList(value=[c.encode('utf-8') for c in clicks])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def generate_data(num_records: int, num_items: int, output_file: str):
    """
    Generate data and write to TFRecord file
    """
    items = generate_items(num_items)
    with tf.io.TFRecordWriter(output_file) as writer:
        for i in range(1, num_records + 1):
            uid = f'user_{i}'
            age = random.randint(AGE_MIN, AGE_MAX)
            gender = random.choice(GENDERS)
            device = random.choice(DEVICES)
            item_id = random.choice(items)
            clicks = generate_clicks(items, item_id)
            label = generate_label()
            example_bytes = serialize_example(uid, age, gender, device, item_id, clicks, label)
            writer.write(example_bytes)
            if i % 10000 == 0:
                print(f"Generated {i} records...")
    print(f"Generated {num_records} records, saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate recommendation system training data with label and save as TFRecord')
    parser.add_argument('-n', '--num_records', type=int, default=100,
                        help='Number of records to generate (default: 100)')
    parser.add_argument('-o', '--output', type=str, default='data.tfrecord',
                        help='Output TFRecord file name (default: data.tfrecord)')
    parser.add_argument('--num_items', type=int, default=10000,
                        help='Item pool size (default: 10000)')
    parser.add_argument('--label_pos_prob', type=float, default=0.5,
                        help='Probability of positive label (default: 0.5)')
    args = parser.parse_args()

    # Update global label probability
    global LABEL_PROB_POS
    LABEL_PROB_POS = args.label_pos_prob

    generate_data(args.num_records, args.num_items, args.output)

if __name__ == '__main__':
    main()
