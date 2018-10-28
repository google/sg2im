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

"""
This script can be used to sample many images from a model for evaluation.
"""


import argparse, json
import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from scipy.misc import imsave, imresize

from sg2im.data import imagenet_deprocess_batch
from sg2im.data.coco import CocoSceneGraphDataset, coco_collate_fn
from sg2im.data.vg import VgSceneGraphDataset, vg_collate_fn
from sg2im.data.utils import split_graph_batch
from sg2im.model import Sg2ImModel
from sg2im.utils import int_tuple, bool_flag
from sg2im.vis import draw_scene_graph


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='sg2im-models/vg64.pt')
parser.add_argument('--checkpoint_list', default=None)
parser.add_argument('--model_mode', default='eval', choices=['train', 'eval'])

# Shared dataset options
parser.add_argument('--dataset', default='vg', choices=['coco', 'vg'])
parser.add_argument('--image_size', default=(64, 64), type=int_tuple)
parser.add_argument('--batch_size', default=24, type=int)
parser.add_argument('--shuffle', default=False, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--num_samples', default=10000, type=int)
parser.add_argument('--save_gt_imgs', default=False, type=bool_flag)
parser.add_argument('--save_graphs', default=False, type=bool_flag)
parser.add_argument('--use_gt_boxes', default=False, type=bool_flag)
parser.add_argument('--use_gt_masks', default=False, type=bool_flag)
parser.add_argument('--save_layout', default=True, type=bool_flag)

parser.add_argument('--output_dir', default='output')

# For VG
VG_DIR = os.path.expanduser('datasets/vg')
parser.add_argument('--vg_h5', default=os.path.join(VG_DIR, 'val.h5'))
parser.add_argument('--vg_image_dir',
        default=os.path.join(VG_DIR, 'images'))

# For COCO
COCO_DIR = os.path.expanduser('~/datasets/coco/2017')
parser.add_argument('--coco_image_dir',
        default=os.path.join(COCO_DIR, 'images/val2017'))
parser.add_argument('--instances_json',
        default=os.path.join(COCO_DIR, 'annotations/instances_val2017.json'))
parser.add_argument('--stuff_json',
        default=os.path.join(COCO_DIR, 'annotations/stuff_val2017.json'))


def build_coco_dset(args, checkpoint):
  checkpoint_args = checkpoint['args']
  print('include other: ', checkpoint_args.get('coco_include_other'))
  dset_kwargs = {
    'image_dir': args.coco_image_dir,
    'instances_json': args.instances_json,
    'stuff_json': args.stuff_json,
    'stuff_only': checkpoint_args['coco_stuff_only'],
    'image_size': args.image_size,
    'mask_size': checkpoint_args['mask_size'],
    'max_samples': args.num_samples,
    'min_object_size': checkpoint_args['min_object_size'],
    'min_objects_per_image': checkpoint_args['min_objects_per_image'],
    'instance_whitelist': checkpoint_args['instance_whitelist'],
    'stuff_whitelist': checkpoint_args['stuff_whitelist'],
    'include_other': checkpoint_args.get('coco_include_other', True),
  }
  dset = CocoSceneGraphDataset(**dset_kwargs)
  return dset


def build_vg_dset(args, checkpoint):
  vocab = checkpoint['model_kwargs']['vocab']
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.vg_h5,
    'image_dir': args.vg_image_dir,
    'image_size': args.image_size,
    'max_samples': args.num_samples,
    'max_objects': checkpoint['args']['max_objects_per_image'],
    'use_orphaned_objects': checkpoint['args']['vg_use_orphaned_objects'],
  }
  dset = VgSceneGraphDataset(**dset_kwargs)
  return dset


def build_loader(args, checkpoint):
  if args.dataset == 'coco':
    dset = build_coco_dset(args, checkpoint)
    collate_fn = coco_collate_fn
  elif args.dataset == 'vg':
    dset = build_vg_dset(args, checkpoint)
    collate_fn = vg_collate_fn

  loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': args.shuffle,
    'collate_fn': collate_fn,
  }
  loader = DataLoader(dset, **loader_kwargs)
  return loader


def build_model(args, checkpoint):
  kwargs = checkpoint['model_kwargs']
  model = Sg2ImModel(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'])
  if args.model_mode == 'eval':
    model.eval()
  elif args.model_mode == 'train':
    model.train()
  model.image_size = args.image_size
  model.cuda()
  return model


def makedir(base, name, flag=True):
  dir_name = None
  if flag:
    dir_name = os.path.join(base, name)
    if not os.path.isdir(dir_name):
      os.makedirs(dir_name)
  return dir_name


def run_model(args, checkpoint, output_dir, loader=None):
  vocab = checkpoint['model_kwargs']['vocab']
  model = build_model(args, checkpoint)
  if loader is None:
    loader = build_loader(args, checkpoint)

  img_dir = makedir(output_dir, 'images')
  graph_dir = makedir(output_dir, 'graphs', args.save_graphs)
  gt_img_dir = makedir(output_dir, 'images_gt', args.save_gt_imgs)
  data_path = os.path.join(output_dir, 'data.pt')

  data = {
    'vocab': vocab,
    'objs': [],
    'masks_pred': [],
    'boxes_pred': [],
    'masks_gt': [],
    'boxes_gt': [],
    'filenames': [],
  }

  img_idx = 0
  for batch in loader:
    masks = None
    if len(batch) == 6:
      imgs, objs, boxes, triples, obj_to_img, triple_to_img = [x.cuda() for x in batch]
    elif len(batch) == 7:
      imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = [x.cuda() for x in batch]

    imgs_gt = imagenet_deprocess_batch(imgs)
    boxes_gt = None
    masks_gt = None
    if args.use_gt_boxes:
      boxes_gt = boxes
    if args.use_gt_masks:
      masks_gt = masks

    # Run the model with predicted masks
    model_out = model(objs, triples, obj_to_img,
                      boxes_gt=boxes_gt, masks_gt=masks_gt)
    imgs_pred, boxes_pred, masks_pred, _ = model_out
    imgs_pred = imagenet_deprocess_batch(imgs_pred)

    obj_data = [objs, boxes_pred, masks_pred]
    _, obj_data = split_graph_batch(triples, obj_data, obj_to_img,
                                    triple_to_img)
    objs, boxes_pred, masks_pred = obj_data

    obj_data_gt = [boxes.data]
    if masks is not None:
      obj_data_gt.append(masks.data)
    triples, obj_data_gt = split_graph_batch(triples, obj_data_gt,
                                       obj_to_img, triple_to_img)
    boxes_gt, masks_gt = obj_data_gt[0], None
    if masks is not None:
      masks_gt = obj_data_gt[1]

    for i in range(imgs_pred.size(0)):
      img_filename = '%04d.png' % img_idx
      if args.save_gt_imgs:
        img_gt = imgs_gt[i].numpy().transpose(1, 2, 0)
        img_gt_path = os.path.join(gt_img_dir, img_filename)
        imsave(img_gt_path, img_gt)

      img_pred = imgs_pred[i]
      img_pred_np = imgs_pred[i].numpy().transpose(1, 2, 0)
      img_path = os.path.join(img_dir, img_filename)
      imsave(img_path, img_pred_np)

      data['objs'].append(objs[i].cpu().clone())
      data['masks_pred'].append(masks_pred[i].cpu().clone())
      data['boxes_pred'].append(boxes_pred[i].cpu().clone())
      data['boxes_gt'].append(boxes_gt[i].cpu().clone())
      data['filenames'].append(img_filename)

      cur_masks_gt = None
      if masks_gt is not None:
        cur_masks_gt = masks_gt[i].cpu().clone()
      data['masks_gt'].append(cur_masks_gt)

      if args.save_graphs:
        graph_img = draw_scene_graph(vocab, objs[i], triples[i])
        graph_path = os.path.join(graph_dir, img_filename)
        imsave(graph_path, graph_img)
      
      img_idx += 1

    torch.save(data, data_path)
    print('Saved %d images' % img_idx)
  

def main(args):
  got_checkpoint = args.checkpoint is not None
  got_checkpoint_list = args.checkpoint_list is not None
  if got_checkpoint == got_checkpoint_list:
    raise ValueError('Must specify exactly one of --checkpoint and --checkpoint_list')

  if got_checkpoint:
    checkpoint = torch.load(args.checkpoint)
    print('Loading model from ', args.checkpoint)
    run_model(args, checkpoint, args.output_dir)
  elif got_checkpoint_list:
    # For efficiency, use the same loader for all checkpoints
    loader = None
    with open(args.checkpoint_list, 'r') as f:
      checkpoint_list = [line.strip() for line in f]
    for i, path in enumerate(checkpoint_list):
      if os.path.isfile(path):
        print('Loading model from ', path)
        checkpoint = torch.load(path)
        if loader is None:
          loader = build_loader(args, checkpoint)
        output_dir = os.path.join(args.output_dir, 'result%03d' % (i + 1))
        run_model(args, checkpoint, output_dir, loader)
      elif os.path.isdir(path):
        # Look for snapshots in this dir
        for fn in sorted(os.listdir(path)):
          if 'snapshot' not in fn:
            continue
          checkpoint_path = os.path.join(path, fn)
          print('Loading model from ', checkpoint_path)
          checkpoint = torch.load(checkpoint_path)
          if loader is None:
            loader = build_loader(args, checkpoint)

          # Snapshots have names like "snapshot_00100K.pt'; we want to
          # extract the "00100K" part
          snapshot_name = os.path.splitext(fn)[0].split('_')[1]
          output_dir = 'result%03d_%s' % (i, snapshot_name)
          output_dir = os.path.join(args.output_dir, output_dir)

          run_model(args, checkpoint, output_dir, loader)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)


