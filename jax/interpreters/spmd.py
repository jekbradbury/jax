# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..map_util import map_aval, shard_aval

def batch(fun, in_vals, in_dims, out_dim_dests, axis_size):
  out_vals, out_dims = batch_fun(fun, in_vals, in_dims)
  return map(partial(matchaxis, axis_size), out_dims, out_dim_dests(), out_vals)

def batch_fun(fun, in_vals, in_dims):
  with new_master(BatchTrace) as master:
    fun, out_dims = batch_subtrace(fun, master, in_dims)
    out_vals = fun.call_wrapped(*in_vals)
    del master
  return out_vals, out_dims()

@lu.transformation_with_aux
def batch_subtrace(master, in_dims, *in_vals):
  trace = BatchTrace(master, core.cur_sublevel())
  in_tracers = [BatchTracer(trace, val, dim) if dim is not None else val
                for val, dim in zip(in_vals, in_dims)]
  outs = yield in_tracers, {}
  out_tracers = map(trace.full_raise, outs)
  out_vals, out_dims = unzip2((t.val, t.batch_dim) for t in out_tracers)
  yield out_vals, out_dims


### tracer

class SPMDTracer(Tracer):
  __slots__ = ['val', 'replicated']

  def __init__(self, trace, val, replicated):
    assert core.skip_checks or type(batch_dim) in (int, NotMapped)
    self._trace = trace
    self.val = val
    self.batch_dim = batch_dim

  @property
  def aval(self):
    val_aval = raise_to_shaped(core.get_aval(self.val))
    aval, _ = shard_aval(val_aval, self.batch_dim)
    return aval

  def full_lower(self):
    if self.batch_dim is not_mapped:
      return core.full_lower(self.val)
    else:
      return self

class SPMDTrace(Trace):
  def pure(self, val):
    return BatchTracer(self, val, not_mapped)

  def lift(self, val):
    return BatchTracer(self, val, not_mapped)

  def sublift(self, val):
    return BatchTracer(self, val.val, val.batch_dim)

  def process_primitive(self, primitive, tracers, params):
    vals_in, dims_in = unzip2((t.val, t.batch_dim) for t in tracers)
    if all(bdim is not_mapped for bdim in dims_in):
      return primitive.bind(*vals_in, **params)
    else:
      # TODO(mattjj,phawkins): if no rule implemented, could vmap-via-map here
      batched_primitive = get_primitive_batcher(primitive)
      val_out, dim_out = batched_primitive(vals_in, dims_in, **params)
      if primitive.multiple_results:
        return map(partial(BatchTracer, self), val_out, dim_out)
      else:
        return BatchTracer(self, val_out, dim_out)

  def process_call(self, call_primitive, f, tracers, params):
    assert call_primitive.multiple_results
    name = params.get('name', f.__name__)
    params = dict(params, name=wrap_name(name, 'vmap'))
    if call_primitive in pe.map_primitives:
      return self.process_map(call_primitive, f, tracers, params)
    vals, dims = unzip2((t.val, t.batch_dim) for t in tracers)
    if all(bdim is not_mapped for bdim in dims):
      return call_primitive.bind(f, *vals, **params)
    else:
      f, dims_out = batch_subtrace(f, self.master, dims)
      vals_out = call_primitive.bind(f, *vals, **params)
      return [BatchTracer(self, v, d) for v, d in zip(vals_out, dims_out())]

  def process_map(self, map_primitive, f, tracers, params):
    vals, dims = unzip2((t.val, t.batch_dim) for t in tracers)
    if all(dim is not_mapped for dim in dims):
      return map_primitive.bind(f, *vals, **params)
    else:
      size, = {x.shape[d] for x, d in zip(vals, dims) if d is not not_mapped}
      is_batched = tuple(d is not not_mapped for d in dims)
      vals = [moveaxis(x, d, 1) if d is not not_mapped and d != 1 else x
              for x, d in zip(vals, dims)]
      dims = tuple(not_mapped if d is not_mapped else 0 for d in dims)
      f, dims_out = batch_subtrace(f, self.master, dims)
      vals_out = map_primitive.bind(f, *vals, **params)
      dims_out = tuple(d + 1 if d is not not_mapped else d for d in dims_out())
      return [BatchTracer(self, v, d) for v, d in zip(vals_out, dims_out)]

  def post_process_call(self, call_primitive, out_tracers, params):
    vals, dims = unzip2((t.val, t.batch_dim) for t in out_tracers)
    master = self.master
    def todo(x):
      trace = BatchTrace(master, core.cur_sublevel())
      return map(partial(BatchTracer, trace), x, dims)
    return vals, todo


# lattice

class SPMDArray(ShapedArray):
  # The only information we need to store is which named axes the array
  # might not be replicated over. This is a sorted tuple. (We might want to
  # store more for additional type safety.)
  # Conceptually, a ShapedArray represents the same type as an SPMDArray
  # with the same shape/dtype that might not be replicated over _any_ axes,
  # while every ConcreteArray is a subtype of an SPMDArray with the same
  # shape/dtype known to be replicated over _all_ axes.
  __slots__ = ['mapped_axes']

  def __init__(self, shape, dtype, weak_type=False, mapped_axes=()):
    super(SPMDArray, self).__init__(shape, dtype, weak_type=weak_type)
    self.mapped_axes = tuple(sorted(mapped_axes))

  def __eq__(self, other):
    return (type(self) is type(other)
            and self.dtype == other.dtype and self.shape == other.shape
            and self.weak_type == other.weak_type
            and self.mapped_axes == other.mapped_axes)

  def __hash__(self):
    return hash((self.shape, self.dtype, self.weak_type, self.mapped_axes))

  def join(self, other):
    if self.shape == other.shape and self.dtype == other.dtype:
      weak_type = self.weak_type and other.weak_type
      mapped_axes = set(self.mapped_axes) | set(other.mapped_axes)
      return SPMDArray(self.shape, self.dtype, weak_type, mapped_axes)
    elif self.dtype == other.dtype:
      return UnshapedArray(self.dtype)
    else:
      raise TypeError(self, other)

  def str_short(self):
    shapestr = ','.join(map(str, self.shape))
    return '{}[{}]'.format(self.dtype.name, shapestr)

  def __len__(self):
    try:
      return self.shape[0]
    except IndexError:
      raise TypeError("len() of unsized object")  # same as numpy error

  def _len(self, ignored_tracer):
    return len(self)

  def strip_weak_type(self):
    return ShapedArray(self.shape, self.dtype) if self.weak_type else self

