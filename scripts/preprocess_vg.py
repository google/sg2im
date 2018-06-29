#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse, json, os
from collections import Counter, defaultdict

import numpy as np
import h5py
from scipy.misc import imread, imresize


"""
vocab for objects contains a special entry "__image__" intended to be used for
dummy nodes encompassing the entire image; vocab for predicates includes a
special entry "__in_image__" to be used for dummy relationships making the graph
fully-connected.
"""


VG_DIR = 'datasets/vg'

parser = argparse.ArgumentParser()

# Input data
parser.add_argument('--splits_json', default='sg2im/data/vg_splits.json')
parser.add_argument('--images_json',
    default=os.path.join(VG_DIR, 'image_data.json'))
parser.add_argument('--objects_json',
    default=os.path.join(VG_DIR, 'objects.json'))
parser.add_argument('--attributes_json',
    default=os.path.join(VG_DIR, 'attributes.json'))
parser.add_argument('--object_aliases',
    default=os.path.join(VG_DIR, 'object_alias.txt'))
parser.add_argument('--relationship_aliases',
    default=os.path.join(VG_DIR, 'relationship_alias.txt'))
parser.add_argument('--relationships_json',
    default=os.path.join(VG_DIR, 'relationships.json'))

# Arguments for images
parser.add_argument('--min_image_size', default=200, type=int)
parser.add_argument('--train_split', default='train')

# Arguments for objects
parser.add_argument('--min_object_instances', default=2000, type=int)
parser.add_argument('--min_attribute_instances', default=2000, type=int)
parser.add_argument('--min_object_size', default=32, type=int)
parser.add_argument('--min_objects_per_image', default=3, type=int)
parser.add_argument('--max_objects_per_image', default=30, type=int)
parser.add_argument('--max_attributes_per_image', default=30, type=int)

# Arguments for relationships
parser.add_argument('--min_relationship_instances', default=500, type=int)
parser.add_argument('--min_relationships_per_image', default=1, type=int)
parser.add_argument('--max_relationships_per_image', default=30, type=int)

# Output
parser.add_argument('--output_vocab_json',
    default=os.path.join(VG_DIR, 'vocab.json'))
parser.add_argument('--output_h5_dir', default=VG_DIR)


def main(args):
  print('Loading image info from "%s"' % args.images_json)
  with open(args.images_json, 'r') as f:
    images = json.load(f)
  image_id_to_image = {i['image_id']: i for i in images}

  with open(args.splits_json, 'r') as f:
    splits = json.load(f)

  # Filter images for being too small
  splits = remove_small_images(args, image_id_to_image, splits)

  obj_aliases = load_aliases(args.object_aliases)
  rel_aliases = load_aliases(args.relationship_aliases)

  print('Loading objects from "%s"' % args.objects_json)
  with open(args.objects_json, 'r') as f:
    objects = json.load(f)

  # Vocab for objects and relationships
  vocab = {}
  train_ids = splits[args.train_split]
  create_object_vocab(args, train_ids, objects, obj_aliases, vocab)

  print('Loading attributes from "%s"' % args.attributes_json)
  with open(args.attributes_json, 'r') as f:
    attributes = json.load(f)

  # Vocab for attributes
  create_attribute_vocab(args, train_ids, attributes, vocab)

  object_id_to_obj = filter_objects(args, objects, obj_aliases, vocab, splits)
  print('After filtering there are %d object instances'
        % len(object_id_to_obj))

  print('Loading relationshps from "%s"' % args.relationships_json)
  with open(args.relationships_json, 'r') as f:
    relationships = json.load(f)

  create_rel_vocab(args, train_ids, relationships, object_id_to_obj,
                   rel_aliases, vocab)

  print('Encoding objects and relationships ...')
  numpy_arrays = encode_graphs(args, splits, objects, relationships, vocab,
                               object_id_to_obj, attributes)

  print('Writing HDF5 output files')
  for split_name, split_arrays in numpy_arrays.items():
    image_ids = list(split_arrays['image_ids'].astype(int))
    h5_path = os.path.join(args.output_h5_dir, '%s.h5' % split_name)
    print('Writing file "%s"' % h5_path)
    with h5py.File(h5_path, 'w') as h5_file:
      for name, ary in split_arrays.items():
        print('Creating datset: ', name, ary.shape, ary.dtype)
        h5_file.create_dataset(name, data=ary)
      print('Writing image paths')
      image_paths = get_image_paths(image_id_to_image, image_ids)
      path_dtype = h5py.special_dtype(vlen=str)
      path_shape = (len(image_paths),)
      path_dset = h5_file.create_dataset('image_paths', path_shape,
                                         dtype=path_dtype)
      for i, p in enumerate(image_paths):
        path_dset[i] = p
    print()

  print('Writing vocab to "%s"' % args.output_vocab_json)
  with open(args.output_vocab_json, 'w') as f:
    json.dump(vocab, f)

def remove_small_images(args, image_id_to_image, splits):
  new_splits = {}
  for split_name, image_ids in splits.items():
    new_image_ids = []
    num_skipped = 0
    for image_id in image_ids:
      image = image_id_to_image[image_id]
      height, width = image['height'], image['width']
      if min(height, width) < args.min_image_size:
        num_skipped += 1
        continue
      new_image_ids.append(image_id)
    new_splits[split_name] = new_image_ids
    print('Removed %d images from split "%s" for being too small' %
          (num_skipped, split_name))

  return new_splits


def get_image_paths(image_id_to_image, image_ids):
  paths = []
  for image_id in image_ids:
    image = image_id_to_image[image_id]
    base, filename = os.path.split(image['url'])
    path = os.path.join(os.path.basename(base), filename)
    paths.append(path)
  return paths


def handle_images(args, image_ids, h5_file):
  with open(args.images_json, 'r') as f:
    images = json.load(f)
  if image_ids:
    image_ids = set(image_ids)

  image_heights, image_widths = [], []
  image_ids_out, image_paths = [], []
  for image in images:
    image_id = image['image_id']
    if image_ids and image_id not in image_ids:
      continue
    height, width = image['height'], image['width']

    base, filename = os.path.split(image['url'])
    path = os.path.join(os.path.basename(base), filename)
    image_paths.append(path)
    image_heights.append(height)
    image_widths.append(width)
    image_ids_out.append(image_id)

  image_ids_np = np.asarray(image_ids_out, dtype=int)
  h5_file.create_dataset('image_ids', data=image_ids_np)

  image_heights = np.asarray(image_heights, dtype=int)
  h5_file.create_dataset('image_heights', data=image_heights)

  image_widths = np.asarray(image_widths, dtype=int)
  h5_file.create_dataset('image_widths', data=image_widths)

  return image_paths


def load_aliases(alias_path):
  aliases = {}
  print('Loading aliases from "%s"' % alias_path)
  with open(alias_path, 'r') as f:
    for line in f:
      line = [s.strip() for s in line.split(',')]
      for s in line:
        aliases[s] = line[0]
  return aliases


def create_object_vocab(args, image_ids, objects, aliases, vocab):
  image_ids = set(image_ids)

  print('Making object vocab from %d training images' % len(image_ids))
  object_name_counter = Counter()
  for image in objects:
    if image['image_id'] not in image_ids:
      continue
    for obj in image['objects']:
      names = set()
      for name in obj['names']:
        names.add(aliases.get(name, name))
      object_name_counter.update(names)

  object_names = ['__image__']
  for name, count in object_name_counter.most_common():
    if count >= args.min_object_instances:
      object_names.append(name)
  print('Found %d object categories with >= %d training instances' %
        (len(object_names), args.min_object_instances))

  object_name_to_idx = {}
  object_idx_to_name = []
  for idx, name in enumerate(object_names):
    object_name_to_idx[name] = idx
    object_idx_to_name.append(name)

  vocab['object_name_to_idx'] = object_name_to_idx
  vocab['object_idx_to_name'] = object_idx_to_name

def create_attribute_vocab(args, image_ids, attributes, vocab):
  image_ids = set(image_ids)
  print('Making attribute vocab from %d training images' % len(image_ids))
  attribute_name_counter = Counter()
  for image in attributes:
    if image['image_id'] not in image_ids:
      continue
    for attribute in image['attributes']:
      names = set()
      try:
        for name in attribute['attributes']:
          names.add(name)
        attribute_name_counter.update(names)
      except KeyError:
        pass
  attribute_names = []
  for name, count in attribute_name_counter.most_common():
    if count >= args.min_attribute_instances:
      attribute_names.append(name)
  print('Found %d attribute categories with >= %d training instances' %
        (len(attribute_names), args.min_attribute_instances))

  attribute_name_to_idx = {}
  attribute_idx_to_name = []
  for idx, name in enumerate(attribute_names):
    attribute_name_to_idx[name] = idx
    attribute_idx_to_name.append(name)
  vocab['attribute_name_to_idx'] = attribute_name_to_idx
  vocab['attribute_idx_to_name'] = attribute_idx_to_name

def filter_objects(args, objects, aliases, vocab, splits):
  object_id_to_objects = {}
  all_image_ids = set()
  for image_ids in splits.values():
    all_image_ids |= set(image_ids)

  object_name_to_idx = vocab['object_name_to_idx']
  object_id_to_obj = {}

  num_too_small = 0
  for image in objects:
    image_id = image['image_id']
    if image_id not in all_image_ids:
      continue
    for obj in image['objects']:
      object_id = obj['object_id']
      final_name = None
      final_name_idx = None
      for name in obj['names']:
        name = aliases.get(name, name)
        if name in object_name_to_idx:
          final_name = name
          final_name_idx = object_name_to_idx[final_name]
          break
      w, h = obj['w'], obj['h']
      too_small = (w < args.min_object_size) or (h < args.min_object_size)
      if too_small:
        num_too_small += 1
      if final_name is not None and not too_small:
        object_id_to_obj[object_id] = {
          'name': final_name,
          'name_idx': final_name_idx,
          'box': [obj['x'], obj['y'], obj['w'], obj['h']],
        }
  print('Skipped %d objects with size < %d' % (num_too_small, args.min_object_size))
  return object_id_to_obj


def create_rel_vocab(args, image_ids, relationships, object_id_to_obj,
                     rel_aliases, vocab):
  pred_counter = defaultdict(int)
  image_ids_set = set(image_ids)
  for image in relationships:
    image_id = image['image_id']
    if image_id not in image_ids_set:
      continue
    for rel in image['relationships']:
      sid = rel['subject']['object_id']
      oid = rel['object']['object_id']
      found_subject = sid in object_id_to_obj
      found_object = oid in object_id_to_obj
      if not found_subject or not found_object:
        continue
      pred = rel['predicate'].lower().strip()
      pred = rel_aliases.get(pred, pred)
      rel['predicate'] = pred
      pred_counter[pred] += 1

  pred_names = ['__in_image__']
  for pred, count in pred_counter.items():
    if count >= args.min_relationship_instances:
      pred_names.append(pred)
  print('Found %d relationship types with >= %d training instances'
        % (len(pred_names), args.min_relationship_instances))

  pred_name_to_idx = {}
  pred_idx_to_name = []
  for idx, name in enumerate(pred_names):
    pred_name_to_idx[name] = idx
    pred_idx_to_name.append(name)

  vocab['pred_name_to_idx'] = pred_name_to_idx
  vocab['pred_idx_to_name'] = pred_idx_to_name


def encode_graphs(args, splits, objects, relationships, vocab,
                  object_id_to_obj, attributes):

  image_id_to_objects = {}
  for image in objects:
    image_id = image['image_id']
    image_id_to_objects[image_id] = image['objects']
  image_id_to_relationships = {}
  for image in relationships:
    image_id = image['image_id']
    image_id_to_relationships[image_id] = image['relationships']
  image_id_to_attributes = {}
  for image in attributes:
    image_id = image['image_id']
    image_id_to_attributes[image_id] = image['attributes']

  numpy_arrays = {}
  for split, image_ids in splits.items():
    skip_stats = defaultdict(int)
    # We need to filter *again* based on number of objects and relationships
    final_image_ids = []
    object_ids = []
    object_names = []
    object_boxes = []
    objects_per_image = []
    relationship_ids = []
    relationship_subjects = []
    relationship_predicates = []
    relationship_objects = []
    relationships_per_image = []
    attribute_ids = []
    attributes_per_object = []
    object_attributes = []
    for image_id in image_ids:
      image_object_ids = []
      image_object_names = []
      image_object_boxes = []
      object_id_to_idx = {}
      for obj in image_id_to_objects[image_id]:
        object_id = obj['object_id']
        if object_id not in object_id_to_obj:
          continue
        obj = object_id_to_obj[object_id]
        object_id_to_idx[object_id] = len(image_object_ids)
        image_object_ids.append(object_id)
        image_object_names.append(obj['name_idx'])
        image_object_boxes.append(obj['box'])
      num_objects = len(image_object_ids)
      too_few = num_objects < args.min_objects_per_image
      too_many = num_objects > args.max_objects_per_image
      if too_few:
        skip_stats['too_few_objects'] += 1
        continue
      if too_many:
        skip_stats['too_many_objects'] += 1
        continue
      image_rel_ids = []
      image_rel_subs = []
      image_rel_preds = []
      image_rel_objs = []
      for rel in image_id_to_relationships[image_id]:
        relationship_id = rel['relationship_id']
        pred = rel['predicate']
        pred_idx = vocab['pred_name_to_idx'].get(pred, None)
        if pred_idx is None:
          continue
        sid = rel['subject']['object_id']
        sidx = object_id_to_idx.get(sid, None)
        oid = rel['object']['object_id']
        oidx = object_id_to_idx.get(oid, None)
        if sidx is None or oidx is None:
          continue
        image_rel_ids.append(relationship_id)
        image_rel_subs.append(sidx)
        image_rel_preds.append(pred_idx)
        image_rel_objs.append(oidx)
      num_relationships = len(image_rel_ids)
      too_few = num_relationships < args.min_relationships_per_image
      too_many = num_relationships > args.max_relationships_per_image
      if too_few:
        skip_stats['too_few_relationships'] += 1
        continue
      if too_many:
        skip_stats['too_many_relationships'] += 1
        continue

      obj_id_to_attributes = {}
      num_attributes = []
      for obj_attribute in image_id_to_attributes[image_id]:
        obj_id_to_attributes[obj_attribute['object_id']] = obj_attribute.get('attributes', None)
      for object_id in image_object_ids:
        attributes = obj_id_to_attributes.get(object_id, None)
        if attributes is None:
          object_attributes.append([-1] * args.max_attributes_per_image)
          num_attributes.append(0)
        else:
          attribute_ids = []
          for attribute in attributes:
            if attribute in vocab['attribute_name_to_idx']:
              attribute_ids.append(vocab['attribute_name_to_idx'][attribute])
            if len(attribute_ids) >= args.max_attributes_per_image:
              break
          num_attributes.append(len(attribute_ids))
          pad_len = args.max_attributes_per_image - len(attribute_ids)
          attribute_ids = attribute_ids + [-1] * pad_len
          object_attributes.append(attribute_ids)

      # Pad object info out to max_objects_per_image
      while len(image_object_ids) < args.max_objects_per_image:
        image_object_ids.append(-1)
        image_object_names.append(-1)
        image_object_boxes.append([-1, -1, -1, -1])
        num_attributes.append(-1)

      # Pad relationship info out to max_relationships_per_image
      while len(image_rel_ids) < args.max_relationships_per_image:
        image_rel_ids.append(-1)
        image_rel_subs.append(-1)
        image_rel_preds.append(-1)
        image_rel_objs.append(-1)

      final_image_ids.append(image_id)
      object_ids.append(image_object_ids)
      object_names.append(image_object_names)
      object_boxes.append(image_object_boxes)
      objects_per_image.append(num_objects)
      relationship_ids.append(image_rel_ids)
      relationship_subjects.append(image_rel_subs)
      relationship_predicates.append(image_rel_preds)
      relationship_objects.append(image_rel_objs)
      relationships_per_image.append(num_relationships)
      attributes_per_object.append(num_attributes)

    print('Skip stats for split "%s"' % split)
    for stat, count in skip_stats.items():
      print(stat, count)
    print()
    numpy_arrays[split] = {
      'image_ids': np.asarray(final_image_ids),
      'object_ids': np.asarray(object_ids),
      'object_names': np.asarray(object_names),
      'object_boxes': np.asarray(object_boxes),
      'objects_per_image': np.asarray(objects_per_image),
      'relationship_ids': np.asarray(relationship_ids),
      'relationship_subjects': np.asarray(relationship_subjects),
      'relationship_predicates': np.asarray(relationship_predicates),
      'relationship_objects': np.asarray(relationship_objects),
      'relationships_per_image': np.asarray(relationships_per_image),
      'attributes_per_object': np.asarray(attributes_per_object),
      'object_attributes': np.asarray(object_attributes),
    }
    for k, v in numpy_arrays[split].items():
      if v.dtype == np.int64:
        numpy_arrays[split][k] = v.astype(np.int32)
  return numpy_arrays


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
