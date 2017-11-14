# Copyright 2017 Medivhna. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

def net_select(name, data_format='NCHW', weight_decay=0.0005):
  if name == 'ResNeXt':
    from resnext import ResNeXt
    network = ResNeXt(num_layers=50, num_card=1, 
                      data_format=data_format, 
                      weight_decay=weight_decay)
  elif name == 'SENet':
    from senet import SENet
    network = SENet(num_layers=50, num_card=1, 
                    data_format=data_format, 
                    weight_decay=weight_decay)
  elif name == 'MobileNet':
    from mobilenet import MobileNet
    network = MobileNet(alpha=1.0, 
                        data_format=data_format, 
                        weight_decay=weight_decay)
  elif name == 'ShuffleNet':
    from shufflenet import ShuffleNet
    network = MobileNet(num_groups=3, alpha=1.0, 
                        data_format=data_format, 
                        weight_decay=weight_decay)
  elif name == 'SphereFace':
    from sphere import SphereFace
    network = SphereFace()
  else:
    raise ValueError('Unsupport network architecture.')

  return network
