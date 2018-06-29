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

# Download all models from the ablation study
mkdir -p sg2im-models

# COCO models
wget https://storage.googleapis.com/sg2im-data/small/coco64_no_gconv.pt -O sg2im-models/coco64_no_gconv.pt
wget https://storage.googleapis.com/sg2im-data/small/coco64_no_relations.pt -O sg2im-models/coco64_no_relations.pt
wget https://storage.googleapis.com/sg2im-data/small/coco64_no_discriminators.pt -O sg2im-models/coco64_no_discriminators.pt
wget https://storage.googleapis.com/sg2im-data/small/coco64_no_img_discriminator.pt -O sg2im-models/coco64_no_img_discriminator.pt
wget https://storage.googleapis.com/sg2im-data/small/coco64_no_obj_discriminator.pt -O sg2im-models/coco64_no_obj_discriminator.pt
wget https://storage.googleapis.com/sg2im-data/small/coco64_gt_layout.pt -O sg2im-models/coco64_gt_layout.pt
wget https://storage.googleapis.com/sg2im-data/small/coco64_gt_layout_no_gconv.pt -O sg2im-models/coco64_gt_layout_no_gconv.pt

# VG models
wget https://storage.googleapis.com/sg2im-data/small/vg64_no_relations.pt -O sg2im-models/vg64_no_relations.pt
wget https://storage.googleapis.com/sg2im-data/small/vg64_no_gconv.pt -O sg2im-models/vg64_no_gconv.pt
wget https://storage.googleapis.com/sg2im-data/small/vg64_no_discriminators.pt -O sg2im-models/vg64_no_discriminators.pt
wget https://storage.googleapis.com/sg2im-data/small/vg64_no_img_discriminator.pt -O sg2im-models/vg64_no_img_discriminator.pt
wget https://storage.googleapis.com/sg2im-data/small/vg64_no_obj_discriminator.pt -O sg2im-models/vg64_no_obj_discriminator.pt
