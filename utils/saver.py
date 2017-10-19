# Copyright 2017 Guanshuo Wang. All Rights Reserved.
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

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.training.saver import BaseSaverBuilder, _set_cpu0

class DataParallelSaverBuilder(BaseSaverBuilder):
    def __init__(self):
        super(DataParallelSaverBuilder, self).__init__()

    def save_op(self, filename_tensor, saveables):
        tensor_names = []
        tensors = []
        tensor_slices = []
        for saveable in saveables:
            for spec in saveable.specs:
                if spec.name.startswith('replicated_'):
                    if spec.name.startswith('replicated_0') or 'avg' in spec.name:
                        tensor_names.append('/'.join(spec.name.split('/')[1:]))
                        tensors.append(spec.tensor)
                        tensor_slices.append(spec.slice_spec)
                else:
                    tensor_names.append(spec.name)
                    tensors.append(spec.tensor)
                    tensor_slices.append(spec.slice_spec)
        if self._write_version == saver_pb2.SaverDef.V1:
            return io_ops._save(
                filename=filename_tensor,
                tensor_names=tensor_names,
                tensors=tensors,
                tensor_slices=tensor_slices)
        elif self._write_version == saver_pb2.SaverDef.V2:
            # "filename_tensor" is interpreted *NOT AS A FILENAME*, but as a prefix
            # of a V2 checkpoint: e.g. "/fs/train/ckpt-<step>/tmp/worker<i>-<step>".
            return io_ops.save_v2(filename_tensor, tensor_names, tensor_slices,
                                  tensors)
        else:
            raise RuntimeError("Unexpected write_version: " + self._write_version)

    def restore_op(self, filename_tensor, saveable, preferred_shard):
        tensors = []
        for spec in saveable.specs:
            # Ignore the moving_mean and moving_variance in other towers.
            if 'BatchNorm/moving_' in spec.name:
                continue
            if spec.name.startswith('replicated_'):
                if not spec.name.startswith('replicated_0') and 'BatchNorm/moving_' in spec.name:
                    continue
                tensors.append(
                            io_ops.restore_v2(
                                filename_tensor,
                                ['/'.join(spec.name.split('/')[1:])],
                                [spec.slice_spec],
                                [spec.tensor.dtype])[0])
            else:
                tensors.append(
                            io_ops.restore_v2(
                                    filename_tensor,
                                    [spec.name],
                                    [spec.slice_spec],
                                    [spec.tensor.dtype])[0])

        return tensors

    def _AddRestoreOps(self,
                       filename_tensor,
                       saveables,
                       restore_sequentially,
                       reshape,
                       preferred_shard=-1,
                       name="restore_all"):
        assign_ops = []
        for saveable in saveables:
            restore_control_inputs = assign_ops[-1:] if restore_sequentially else []
            with ops.device(_set_cpu0(saveable.device) if saveable.device else None):
                with ops.control_dependencies(restore_control_inputs):
                    tensors = self.restore_op(filename_tensor, saveable, preferred_shard)
                    if len(tensors) == 0:
                        continue
                    shapes = None
                    if reshape:
                        shapes = []
                        for spec in saveable.specs:
                            v = spec.tensor
                            shape = v.get_shape()
                        if not shape.is_fully_defined():
                            shape = array_ops.shape(v)
                        shapes.append(shape)
                assign_ops.append(saveable.restore(tensors, shapes))

        return control_flow_ops.group(*assign_ops, name=name)