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

import argparse
import torch


"""
Tiny utility to print the command-line args used for a checkpoint
"""


parser = argparse.ArgumentParser()
parser.add_argument('checkpoint')


def main(args):
  checkpoint = torch.load(args.checkpoint, map_location='cpu')
  for k, v in checkpoint['args'].items():
    print(k, v)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

