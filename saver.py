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

import six
from tensorflow.python.ops import variables
from tensorflow.python.ops import state_ops

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



class DebugSaverBuilder(BaseSaverBuilder):
    def __init__(self):
        super(DebugSaverBuilder, self).__init__()

    def restore_op(self, filename_tensor, saveable, preferred_shard):
        tensors = []
        for spec in saveable.specs:
            print(spec.name)
            tensors.append(
                        io_ops.restore_v2(
                                filename_tensor,
                                [spec.name],
                                [spec.slice_spec],
                                [spec.tensor.dtype])[0])

        return tensors


    # def _ValidateAndSliceInputs(self, names_to_saveables):
    #     if not isinstance(names_to_saveables, dict):
    #         names_to_saveables = super(DebugSaverBuilder, self).OpListToDict(names_to_saveables)

    #     saveables = []
    #     seen_ops = set()
    #     for name in sorted(names_to_saveables.keys()):
    #         if not isinstance(name, six.string_types):
    #             raise TypeError(
    #                 "names_to_saveables must be a dict mapping string names to "
    #                 "checkpointable operations. Name is not a string: %s" % name)
    #         op = names_to_saveables[name]
    #         if isinstance(op, BaseSaverBuilder.SaveableObject):
    #             print(1)
    #             self._AddSaveable(saveables, seen_ops, op)
    #         elif isinstance(op, (list, tuple, variables.PartitionedVariable)):
    #             print(2)
    #             if isinstance(op, variables.PartitionedVariable):
    #                 op = list(op)
    #             # A set of slices.
    #             slice_name = None
    #             # pylint: disable=protected-access
    #             for variable in op:
    #                 print(variable.op.type)
    #                 if (not isinstance(variable, variables.Variable) and
    #                     not isinstance(variable, resource_variable_ops.ResourceVariable)):
    #                     raise ValueError("Slices must all be Variables: %s" % variable)
    #                 if not variable._save_slice_info:
    #                     raise ValueError("Slices must all be slices: %s" % variable)
    #                 if slice_name is None:
    #                     slice_name = variable._save_slice_info.full_name
    #                 elif slice_name != variable._save_slice_info.full_name:
    #                     raise ValueError(
    #                         "Slices must all be from the same tensor: %s != %s" %
    #                         (slice_name, variable._save_slice_info.full_name))
    #                 if variable.op.type in ["Variable", "VariableV2",
    #                                         "AutoReloadVariable"]:
    #                     saveable = DebugSaverBuilder.VariableSaveable(
    #                         variable, variable._save_slice_info.spec, name)
    #                 else:
    #                     saveable = BaseSaverBuilder.ResourceVariableSaveable(
    #                         variable, variable._save_slice_info.spec, name)
    #                 self._AddSaveable(saveables, seen_ops, saveable)
    #                 # pylint: enable=protected-access
    #         else:
    #             # A variable or tensor.
    #             variable = ops.internal_convert_to_tensor(op, as_ref=True)
    #             if not BaseSaverBuilder._IsVariable(variable):
    #                 raise TypeError("names_to_saveables must be a dict mapping string "
    #                               "names to Tensors/Variables. Not a variable: %s" %
    #                               variable)
    #             if variable.op.type in ["Variable", "VariableV2", "AutoReloadVariable"]:
    #                 saveable = DebugSaverBuilder.VariableSaveable(variable, "", name)
    #             else:
    #                 saveable = BaseSaverBuilder.ResourceVariableSaveable(
    #                     variable, "", name)
    #             self._AddSaveable(saveables, seen_ops, saveable)
    #     return saveables


    # class VariableSaveable(BaseSaverBuilder.SaveableObject):
    #     """SaveableObject implementation that handles Variables."""

    #     def __init__(self, var, slice_spec, name):
    #         spec = DebugSaverBuilder.SaveSpec(var, slice_spec, name)
    #         super(DebugSaverBuilder.VariableSaveable, self).__init__(var, [spec], name)

    #     def restore(self, restored_tensors, restored_shapes):
    #         restored_tensor = restored_tensors[0]
    #         if restored_shapes is not None:
    #             restored_tensor = array_ops.reshape(restored_tensor, restored_shapes[0])
    #         return state_ops.assign(
    #             self.op,
    #             restored_tensor,
    #             validate_shape=restored_shapes is None and
    #             self.op.get_shape().is_fully_defined())