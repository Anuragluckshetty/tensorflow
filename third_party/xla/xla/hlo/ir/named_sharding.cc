/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/hlo/ir/named_sharding.h"

#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_op_metadata.h"
#include "xla/hlo/ir/mesh_and_axis.h"

namespace xla {

using DimensionSharding = NamedSharding::DimensionSharding;

void DimensionSharding::Append(const DimensionSharding& other,
                               const Mesh& mesh) {
  if (other.axes_.empty()) {
    return;
  }
  if (axes_.empty()) {
    axes_ = other.axes_;
    return;
  }

  // Merge last element of `axes_` with first element of `other.axes_`
  if (!axes_.back().Merge(other.axes_.front(), mesh)) {
    axes_.push_back(other.axes_.front());
  }

  axes_.insert(axes_.end(), other.axes_.begin() + 1, other.axes_.end());
}

std::optional<DimensionSharding> DimensionSharding::Slice(const Mesh& mesh,
                                                          int64_t slice_size) {
  if (slice_size == 1) {
    return DimensionSharding({}, is_closed_);
  }
  if (getShardedSize(mesh) % slice_size != 0) {
    return std::nullopt;
  }

  int64_t axis_index = 0;
  std::vector<AxisRef> sliced_axes, remaining_axes;

  for (; axis_index < axes().size(); ++axis_index) {
    const AxisRef& curr_axis = axes()[axis_index];
    int64_t curr_axis_size = curr_axis.size(mesh);

    if (slice_size == curr_axis_size) {
      sliced_axes =
          std::vector<AxisRef>(axes().begin(), axes().begin() + axis_index + 1);
      slice_size = 1;
      break;
    }
    if (slice_size % curr_axis_size == 0) {
      slice_size /= curr_axis_size;
    } else if (curr_axis_size % slice_size == 0) {
      sliced_axes =
          std::vector<AxisRef>(axes().begin(), axes().begin() + axis_index);
      int64_t sliced_axis_pre_size =
          curr_axis.sub_axis_info() ? curr_axis.sub_axis_info()->pre_size : 1;
      sliced_axes.push_back(AxisRef(curr_axis.mesh_axis_index(),
                                    {sliced_axis_pre_size, slice_size}));
      remaining_axes.push_back(AxisRef(
          curr_axis.mesh_axis_index(),
          {sliced_axis_pre_size * slice_size, curr_axis_size / slice_size}));
      slice_size = 1;
      break;
    } else {
      return std::nullopt;
    }
  }

  if (slice_size != 1) {
    return std::nullopt;
  }

  remaining_axes.insert(remaining_axes.end(), axes().begin() + axis_index + 1,
                        axes().end());
  axes_ = std::move(remaining_axes);
  return DimensionSharding(sliced_axes, is_closed_);
}

int64_t DimensionSharding::getShardedSize(const Mesh& mesh) const {
  return std::accumulate(axes_.begin(), axes_.end(), 1,
                         [&mesh](int64_t cur, const AxisRef& axis) {
                           return cur * axis.size(mesh);
                         });
}

std::string DimensionSharding::ToString(const Mesh* mesh) const {
  std::string result = "{";
  absl::StrAppend(
      &result,
      absl::StrJoin(axes_, ", ", [mesh](std::string* out, const AxisRef& axis) {
        absl::StrAppend(out, axis.ToString(mesh));
      }));

  if (!is_closed_) {
    if (axes_.empty()) {
      absl::StrAppend(&result, "?");
    } else {
      absl::StrAppend(&result, ", ?");
    }
  }

  absl::StrAppend(&result, "}");
  return result;
}

NamedShardingProto::DimensionSharding DimensionSharding::ToProto() const {
  NamedShardingProto::DimensionSharding proto;
  for (const AxisRef& axis : axes_) {
    *proto.add_axes() = axis.ToProto();
  }
  proto.set_is_closed(is_closed_);
  return proto;
}

DimensionSharding DimensionSharding::FromProto(
    const NamedShardingProto::DimensionSharding& proto) {
  DimensionSharding dim_sharding;
  dim_sharding.is_closed_ = proto.is_closed();
  dim_sharding.axes_.reserve(proto.axes_size());
  for (const AxisRefProto& axis_proto : proto.axes()) {
    dim_sharding.axes_.push_back(AxisRef::FromProto(axis_proto));
  }
  return dim_sharding;
}

std::string NamedSharding::ToString(bool include_metadata) const {
  std::string result = "{";

  std::string metadata_str;
  if (include_metadata && !metadata_.empty()) {
    metadata_str = ", metadata={";
    absl::StrAppend(
        &metadata_str,
        absl::StrJoin(
            metadata_, ", ", [&](std::string* out, const auto& metadata) {
              absl::StrAppend(out, "{", OpMetadataToString(metadata), "}");
            }));
    absl::StrAppend(&metadata_str, "}");
  }

  // Special cases.
  if (IsReplicated() && replicated_axes_.empty()) {
    absl::StrAppend(&result, "replicated");
    absl::StrAppend(&result, metadata_str);
    absl::StrAppend(&result, "}");
    return result;
  }

  if (IsMaximal()) {
    absl::StrAppend(&result, "maximal device=");
    absl::StrAppend(&result, *mesh_.device_assignment().array().begin());
    absl::StrAppend(&result, metadata_str);
    absl::StrAppend(&result, "}");
    return result;
  }

  absl::StrAppend(&result, mesh_.ToString());

  // Dimension sharding.
  absl::StrAppend(&result, ", [");
  absl::StrAppend(
      &result,
      absl::StrJoin(dim_shardings_, ", ",
                    [&](std::string* out, const DimensionSharding& ds) {
                      absl::StrAppend(out, ds.ToString(&mesh_));
                    }));
  absl::StrAppend(&result, "]");

  if (!replicated_axes_.empty()) {
    absl::StrAppend(&result, ", replicated={");
    absl::StrAppend(&result,
                    absl::StrJoin(replicated_axes_, ", ",
                                  [&](std::string* out, const AxisRef& axis) {
                                    absl::StrAppend(out, axis.ToString(&mesh_));
                                  }));
    absl::StrAppend(&result, "}");
  }

  if (!unreduced_axes_.empty()) {
    absl::StrAppend(&result, ", unreduced={");
    absl::StrAppend(&result,
                    absl::StrJoin(unreduced_axes_, ", ",
                                  [&](std::string* out, const AxisRef& axis) {
                                    absl::StrAppend(out, axis.ToString(&mesh_));
                                  }));
    absl::StrAppend(&result, "}");
  }

  if (!manual_axes_.empty()) {
    absl::StrAppend(&result, ", manual={");
    absl::StrAppend(&result,
                    absl::StrJoin(manual_axes_, ", ",
                                  [&](std::string* out, const AxisRef& axis) {
                                    absl::StrAppend(out, axis.ToString(&mesh_));
                                  }));
    absl::StrAppend(&result, "}");
  }

  absl::StrAppend(&result, metadata_str);
  absl::StrAppend(&result, "}");

  return result;
}

NamedShardingProto NamedSharding::ToProto() const {
  NamedShardingProto proto;
  *proto.mutable_mesh() = mesh_.ToProto();
  for (const DimensionSharding& dim_sharding : dim_shardings_) {
    *proto.add_dim_shardings() = dim_sharding.ToProto();
  }
  for (const AxisRef& axis : replicated_axes_) {
    *proto.add_replicated_axes() = axis.ToProto();
  }
  for (const AxisRef& axis : unreduced_axes_) {
    *proto.add_unreduced_axes() = axis.ToProto();
  }
  for (const AxisRef& axis : manual_axes_) {
    *proto.add_manual_axes() = axis.ToProto();
  }
  for (const OpMetadata& metadata : metadata_) {
    *proto.add_metadata() = metadata;
  }
  return proto;
}

NamedSharding NamedSharding::FromProto(const NamedShardingProto& proto) {
  NamedSharding named_sharding(Mesh::FromProto(proto.mesh()));

  named_sharding.dim_shardings_.reserve(proto.dim_shardings_size());
  for (const auto& dim_sharding_proto : proto.dim_shardings()) {
    named_sharding.dim_shardings_.push_back(
        DimensionSharding::FromProto(dim_sharding_proto));
  }
  // Canonicalize dim shardings.
  bool all_dims_empty = absl::c_all_of(
      named_sharding.dim_shardings_,
      [](const DimensionSharding& ds) { return ds.axes().empty(); });
  if (all_dims_empty) {
    named_sharding.dim_shardings_.clear();
  }

  named_sharding.replicated_axes_.reserve(proto.replicated_axes_size());
  for (const auto& axis_proto : proto.replicated_axes()) {
    named_sharding.replicated_axes_.push_back(AxisRef::FromProto(axis_proto));
  }
  named_sharding.unreduced_axes_.reserve(proto.unreduced_axes_size());
  for (const auto& axis_proto : proto.unreduced_axes()) {
    named_sharding.unreduced_axes_.push_back(AxisRef::FromProto(axis_proto));
  }
  named_sharding.manual_axes_.reserve(proto.manual_axes_size());
  for (const auto& axis_proto : proto.manual_axes()) {
    named_sharding.manual_axes_.push_back(AxisRef::FromProto(axis_proto));
  }
  named_sharding.metadata_.assign(proto.metadata().begin(),
                                  proto.metadata().end());

  named_sharding.InitShardedSizes();
  return named_sharding;
}

OpSharding NamedSharding::ToOpSharding() const {
  OpSharding op_sharding;
  *op_sharding.mutable_named_sharding() = ToProto();
  return op_sharding;
}

absl::StatusOr<NamedSharding> NamedSharding::FromOpSharding(
    const OpSharding& proto) {
  if (!proto.has_named_sharding()) {
    return absl::InvalidArgumentError(
        "OpSharding does not contain a NamedSharding.");
  }
  return FromProto(proto.named_sharding());
}

std::ostream& operator<<(std::ostream& out, const DimensionSharding& sharding) {
  return out << sharding.ToString();
}

std::ostream& operator<<(std::ostream& out, const NamedSharding& sharding) {
  return out << sharding.ToString();
}

namespace test_utils {
// Construct sharding with given mesh. 'dim_shardings', 'replicated_axes',
// 'unreduced_axes' refer to axis names in the mesh.
// This is a test only helper function.
NamedSharding FromAxisNames(
    Mesh mesh, absl::Span<const std::vector<std::string>> dim_sharding_names,
    absl::Span<const std::string> replicated_axis_names,
    absl::Span<const std::string> unreduced_axis_names,
    absl::Span<const std::string> manual_axis_names,
    absl::Span<const OpMetadata> metadata) {
  std::map<std::string, int64_t> mesh_axis_to_index;
  for (int64_t i = 0; i < mesh.axis_names().size(); ++i) {
    mesh_axis_to_index[mesh.axis_names()[i]] = i;
  }

  std::vector<DimensionSharding> dim_shardings;
  dim_shardings.reserve(dim_sharding_names.size());
  for (const auto& axes_for_dim : dim_sharding_names) {
    std::vector<AxisRef> axis_refs;
    axis_refs.reserve(axes_for_dim.size());
    for (const std::string& axis_name : axes_for_dim) {
      auto it = mesh_axis_to_index.find(axis_name);
      CHECK_NE(it, mesh_axis_to_index.end())
          << "Axis " << axis_name << " not found in mesh " << mesh.ToString();
      axis_refs.push_back(AxisRef(it->second));
    }
    dim_shardings.push_back(
        DimensionSharding(std::move(axis_refs), /*is_closed=*/true));
  }

  std::vector<AxisRef> replicated_axes;
  replicated_axes.reserve(replicated_axis_names.size());
  for (const std::string& axis_name : replicated_axis_names) {
    auto it = mesh_axis_to_index.find(axis_name);
    CHECK_NE(it, mesh_axis_to_index.end())
        << "Axis " << axis_name << " not found in mesh " << mesh.ToString();
    replicated_axes.push_back(AxisRef(it->second));
  }

  std::vector<AxisRef> unreduced_axes;
  unreduced_axes.reserve(unreduced_axis_names.size());
  for (const std::string& axis_name : unreduced_axis_names) {
    auto it = mesh_axis_to_index.find(axis_name);
    CHECK_NE(it, mesh_axis_to_index.end())
        << "Axis " << axis_name << " not found in mesh " << mesh.ToString();
    unreduced_axes.push_back(AxisRef(it->second));
  }

  std::vector<AxisRef> manual_axes;
  manual_axes.reserve(manual_axis_names.size());
  for (const std::string& axis_name : manual_axis_names) {
    auto it = mesh_axis_to_index.find(axis_name);
    CHECK_NE(it, mesh_axis_to_index.end())
        << "Axis " << axis_name << " not found in mesh " << mesh.ToString();
    manual_axes.push_back(AxisRef(it->second));
  }

  return NamedSharding(mesh, dim_shardings, replicated_axes, unreduced_axes,
                       manual_axes, metadata);
}
}  // namespace test_utils
}  // namespace xla
