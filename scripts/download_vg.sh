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

VG_DIR=datasets/vg
mkdir -p $VG_DIR

wget https://visualgenome.org/static/data/dataset/objects.json.zip -O $VG_DIR/objects.json.zip
wget https://visualgenome.org/static/data/dataset/attributes.json.zip -O $VG_DIR/attributes.json.zip
wget https://visualgenome.org/static/data/dataset/relationships.json.zip -O $VG_DIR/relationships.json.zip
wget https://visualgenome.org/static/data/dataset/object_alias.txt -O $VG_DIR/object_alias.txt
wget https://visualgenome.org/static/data/dataset/relationship_alias.txt -O $VG_DIR/relationship_alias.txt
wget https://visualgenome.org/static/data/dataset/image_data.json.zip -O $VG_DIR/image_data.json.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -O $VG_DIR/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -O $VG_DIR/images2.zip

unzip $VG_DIR/objects.json.zip -d $VG_DIR
unzip $VG_DIR/attributes.json.zip -d $VG_DIR
unzip $VG_DIR/relationships.json.zip -d $VG_DIR
unzip $VG_DIR/image_data.json.zip -d $VG_DIR
unzip $VG_DIR/images.zip -d $VG_DIR/images
unzip $VG_DIR/images2.zip -d $VG_DIR/images
