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

# Download the main models: 64 x 64 for coco and vg, and 128 x 128 for vg
mkdir -p sg2im-models
wget https://storage.googleapis.com/sg2im-data/small/coco64.pt -O sg2im-models/coco64.pt
wget https://storage.googleapis.com/sg2im-data/small/vg64.pt -O sg2im-models/vg64.pt
wget https://storage.googleapis.com/sg2im-data/small/vg128.pt -O sg2im-models/vg128.pt
