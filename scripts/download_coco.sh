#!/bin/bash -eu
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

COCO_DIR=datasets/coco
mkdir -p $COCO_DIR

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O $COCO_DIR/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip -O $COCO_DIR/stuff_annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/train2017.zip -O $COCO_DIR/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip -O $COCO_DIR/val2017.zip

unzip $COCO_DIR/annotations_trainval2017.zip -d $COCO_DIR
unzip $COCO_DIR/stuff_annotations_trainval2017.zip -d $COCO_DIR
unzip $COCO_DIR/train2017.zip -d $COCO_DIR/images
unzip $COCO_DIR/val2017.zip -d $COCO_DIR/images
