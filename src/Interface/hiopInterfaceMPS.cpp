// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: BSD-3-Clause

#include "hiopInterfaceMPS.hpp"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace hiop
{
namespace
{
constexpr double kMpsInfinity = 1e20;

enum class Section
{
  none,
  obj_sense,
  obj_name,
  rows,
  columns,
  rhs,
  ranges,
  bounds
};

struct MatrixEntry
{
  std::string column;
  std::string row;
  double value;
};

struct NamedEntry
{
  std::string set;
  std::string row;
  double value;
};

struct BoundEntry
{
  std::string set;
  std::string type;
  std::string column;
  double value;
  bool has_value;
};

std::string trim(const std::string& input)
{
  const auto first = input.find_first_not_of(" \t\r\n");
  if(first == std::string::npos) return std::string();
  const auto last = input.find_last_not_of(" \t\r\n");
  return input.substr(first, last - first + 1);
}

std::string upper(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return std::toupper(ch); });
  return value;
}

std::string unquote(const std::string& value)
{
  if(value.size() >= 2 && ((value.front() == '\'' && value.back() == '\'') ||
                           (value.front() == '"' && value.back() == '"'))) {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

std::vector<std::string> words(const std::string& line)
{
  std::istringstream stream(line);
  std::vector<std::string> result;
  std::string word;
  while(stream >> word) result.push_back(word);
  return result;
}

std::string fixed_field(const std::string& line, std::size_t offset, std::size_t length)
{
  if(offset >= line.size()) return std::string();
  return trim(line.substr(offset, std::min(length, line.size() - offset)));
}

bool looks_fixed(const std::string& line)
{
  if(line.size() < 15 || !std::isspace(static_cast<unsigned char>(line[0]))) return false;
  const std::size_t separators[] = {3, 12, 13};
  for(const auto pos : separators) {
    if(pos < line.size() && !std::isspace(static_cast<unsigned char>(line[pos]))) return false;
  }
  return true;
}

bool parse_number(std::string token, double& value)
{
  std::replace(token.begin(), token.end(), 'D', 'E');
  std::replace(token.begin(), token.end(), 'd', 'e');
  try {
    std::size_t consumed = 0;
    value = std::stod(token, &consumed);
    return consumed == token.size() && std::isfinite(value);
  } catch(...) {
    return false;
  }
}

bool requires_bound_value(const std::string& type)
{
  return type == "LO" || type == "UP" || type == "FX";
}

template<typename T>
void record_set_name(const std::string& name, std::vector<std::string>& order, T& seen)
{
  if(seen.emplace(name, true).second) order.push_back(name);
}

}  // namespace

struct hiopInterfaceMPS::Impl
{
  bool loaded{false};
  std::string error;
  std::string model;
  std::string objective_row;
  ObjectiveSense sense{ObjectiveSense::minimize};
  double objective_scale{1.0};
  double objective_offset{0.0};

  std::vector<std::string> column_names;
  std::vector<std::string> row_names;
  std::vector<char> row_types;
  std::vector<double> costs;
  std::vector<double> column_lower;
  std::vector<double> column_upper;
  std::vector<double> row_lower;
  std::vector<double> row_upper;
  std::vector<index_type> matrix_rows;
  std::vector<index_type> matrix_columns;
  std::vector<double> matrix_values;
  std::vector<std::size_t> matrix_row_offsets;
  size_type nnz_equalities{0};
  size_type nnz_inequalities{0};

  void clear()
  {
    loaded = false;
    error.clear();
    model.clear();
    objective_row.clear();
    sense = ObjectiveSense::minimize;
    objective_scale = 1.0;
    objective_offset = 0.0;
    column_names.clear();
    row_names.clear();
    row_types.clear();
    costs.clear();
    column_lower.clear();
    column_upper.clear();
    row_lower.clear();
    row_upper.clear();
    matrix_rows.clear();
    matrix_columns.clear();
    matrix_values.clear();
    matrix_row_offsets.clear();
    nnz_equalities = 0;
    nnz_inequalities = 0;
  }
};

hiopInterfaceMPS::hiopInterfaceMPS()
    : impl_(new Impl())
{}

hiopInterfaceMPS::~hiopInterfaceMPS() = default;

hiopMPSReadStatus hiopInterfaceMPS::load(const std::string& filename, const hiopMPSReadOptions& options)
{
  impl_->clear();

  std::ifstream input(filename);
  if(!input) {
    impl_->error = "Unable to open MPS file '" + filename + "'.";
    return hiopMPSReadStatus::file_error;
  }

  std::vector<std::pair<char, std::string> > rows;
  std::vector<MatrixEntry> matrix_entries;
  std::vector<NamedEntry> rhs_entries;
  std::vector<NamedEntry> range_entries;
  std::vector<BoundEntry> bound_entries;
  std::vector<std::string> rhs_order, range_order, bound_order;
  std::unordered_map<std::string, bool> rhs_seen, range_seen, bound_seen;

  Section section = Section::none;
  std::string requested_objective;
  std::size_t line_number = 0;
  bool saw_rows = false, saw_columns = false, saw_end = false;

  auto fail = [&](hiopMPSReadStatus status, const std::string& message) {
    std::ostringstream os;
    os << filename;
    if(line_number != 0) os << ':' << line_number;
    os << ": " << message;
    impl_->error = os.str();
    return status;
  };

  std::string line;
  while(std::getline(input, line)) {
    ++line_number;
    if(!line.empty() && line.back() == '\r') line.pop_back();
    const std::string stripped = trim(line);
    if(stripped.empty()) continue;
    if(stripped[0] == '*') {
      const std::string directive = upper(stripped);
      const auto sense_pos = directive.find("SENSE:");
      if(sense_pos != std::string::npos) {
        const std::string value = trim(directive.substr(sense_pos + 6));
        if(value.find("MAX") == 0) impl_->sense = ObjectiveSense::maximize;
        if(value.find("MIN") == 0) impl_->sense = ObjectiveSense::minimize;
      }
      continue;
    }

    const auto tokens = words(line);
    if(tokens.empty()) continue;
    const std::string keyword = upper(tokens[0]);
    const bool header = !std::isspace(static_cast<unsigned char>(line[0]));
    if(header && keyword == "NAME") {
      section = Section::none;
      if(tokens.size() > 1) impl_->model = tokens[1];
      continue;
    }
    if(header && keyword == "OBJSENSE") {
      section = Section::obj_sense;
      if(tokens.size() > 1) {
        const std::string value = upper(tokens[1]);
        if(value == "MAX" || value == "MAXIMIZE") impl_->sense = ObjectiveSense::maximize;
        else if(value == "MIN" || value == "MINIMIZE") impl_->sense = ObjectiveSense::minimize;
        else return fail(hiopMPSReadStatus::parse_error, "unknown objective sense '" + tokens[1] + "'.");
      }
      continue;
    }
    if(header && keyword == "OBJNAME") {
      section = Section::obj_name;
      if(tokens.size() > 1) requested_objective = tokens[1];
      continue;
    }
    if(header && keyword == "ROWS") {
      section = Section::rows;
      saw_rows = true;
      continue;
    }
    if(header && keyword == "COLUMNS") {
      section = Section::columns;
      saw_columns = true;
      continue;
    }
    if(header && keyword == "RHS") {
      section = Section::rhs;
      continue;
    }
    if(header && keyword == "RANGES") {
      section = Section::ranges;
      continue;
    }
    if(header && keyword == "BOUNDS") {
      section = Section::bounds;
      continue;
    }
    if(header && keyword == "ENDATA") {
      saw_end = true;
      break;
    }
    if(header && (keyword == "SOS" || keyword == "SOS1" || keyword == "SOS2" || keyword == "QUADOBJ" ||
                  keyword == "QMATRIX" || keyword == "QSECTION" || keyword == "QCMATRIX" ||
                  keyword == "CSECTION" || keyword == "INDICATORS")) {
      return fail(hiopMPSReadStatus::unsupported_feature, "unsupported MPS section '" + keyword + "'.");
    }

    if(section == Section::obj_sense) {
      const std::string value = upper(tokens[0]);
      if(value == "MAX" || value == "MAXIMIZE") impl_->sense = ObjectiveSense::maximize;
      else if(value == "MIN" || value == "MINIMIZE") impl_->sense = ObjectiveSense::minimize;
      else return fail(hiopMPSReadStatus::parse_error, "unknown objective sense '" + tokens[0] + "'.");
      continue;
    }
    if(section == Section::obj_name) {
      requested_objective = tokens[0];
      continue;
    }
    if(section == Section::rows) {
      if(tokens.size() != 2 || tokens[0].size() != 1) {
        return fail(hiopMPSReadStatus::parse_error, "invalid ROWS record.");
      }
      const char row_type = static_cast<char>(std::toupper(static_cast<unsigned char>(tokens[0][0])));
      if(row_type != 'N' && row_type != 'E' && row_type != 'L' && row_type != 'G') {
        return fail(hiopMPSReadStatus::parse_error, "unknown row type '" + tokens[0] + "'.");
      }
      rows.emplace_back(row_type, tokens[1]);
      continue;
    }
    if(section == Section::columns) {
      if(tokens.size() == 3 && upper(unquote(tokens[1])) == "MARKER" &&
         (upper(unquote(tokens[2])) == "INTORG" || upper(unquote(tokens[2])) == "INTEND")) {
        return fail(hiopMPSReadStatus::unsupported_feature, "integer markers are not supported.");
      }
      if(tokens.size() != 3 && tokens.size() != 5) {
        return fail(hiopMPSReadStatus::parse_error, "invalid COLUMNS record.");
      }
      for(std::size_t pos = 1; pos < tokens.size(); pos += 2) {
        double value;
        if(!parse_number(tokens[pos + 1], value)) {
          return fail(hiopMPSReadStatus::parse_error, "invalid numeric value '" + tokens[pos + 1] + "'.");
        }
        matrix_entries.push_back({tokens[0], tokens[pos], value});
      }
      continue;
    }
    if(section == Section::rhs || section == Section::ranges) {
      std::string set_name;
      std::size_t pos;
      if(tokens.size() == 2 || tokens.size() == 4) {
        set_name.clear();
        pos = 0;
      } else if(tokens.size() == 3 || tokens.size() == 5) {
        set_name = tokens[0];
        pos = 1;
      } else {
        return fail(hiopMPSReadStatus::parse_error, "invalid RHS or RANGES record.");
      }
      if(section == Section::rhs) record_set_name(set_name, rhs_order, rhs_seen);
      else record_set_name(set_name, range_order, range_seen);
      for(; pos < tokens.size(); pos += 2) {
        double value;
        if(!parse_number(tokens[pos + 1], value)) {
          return fail(hiopMPSReadStatus::parse_error, "invalid numeric value '" + tokens[pos + 1] + "'.");
        }
        NamedEntry entry{set_name, tokens[pos], value};
        if(section == Section::rhs) rhs_entries.push_back(entry);
        else range_entries.push_back(entry);
      }
      continue;
    }
    if(section == Section::bounds) {
      std::string type, set_name, column, value_token;
      bool has_value = false;
      if(looks_fixed(line)) {
        type = upper(fixed_field(line, 1, 2));
        set_name = fixed_field(line, 4, 8);
        column = fixed_field(line, 14, 8);
        value_token = fixed_field(line, 24, 12);
        has_value = !value_token.empty();
      } else {
        type = upper(tokens[0]);
        if(requires_bound_value(type)) {
          if(tokens.size() == 4) {
            set_name = tokens[1];
            column = tokens[2];
            value_token = tokens[3];
          } else if(tokens.size() == 3) {
            column = tokens[1];
            value_token = tokens[2];
          } else {
            return fail(hiopMPSReadStatus::parse_error, "invalid BOUNDS record.");
          }
          has_value = true;
        } else {
          if(tokens.size() == 3) {
            set_name = tokens[1];
            column = tokens[2];
          } else if(tokens.size() == 2) {
            column = tokens[1];
          } else {
            return fail(hiopMPSReadStatus::parse_error, "invalid BOUNDS record.");
          }
        }
      }
      if(type == "BV" || type == "LI" || type == "UI" || type == "SC" || type == "SI") {
        return fail(hiopMPSReadStatus::unsupported_feature,
                    "integer and semi-continuous bound type '" + type + "' is not supported.");
      }
      if(type != "LO" && type != "UP" && type != "FX" && type != "FR" && type != "MI" && type != "PL") {
        return fail(hiopMPSReadStatus::parse_error, "unknown bound type '" + type + "'.");
      }
      double value = 0.0;
      if(has_value && !parse_number(value_token, value)) {
        return fail(hiopMPSReadStatus::parse_error, "invalid bound value '" + value_token + "'.");
      }
      if(requires_bound_value(type) && !has_value) {
        return fail(hiopMPSReadStatus::parse_error, "bound type '" + type + "' requires a value.");
      }
      record_set_name(set_name, bound_order, bound_seen);
      bound_entries.push_back({set_name, type, column, value, has_value});
      continue;
    }

    return fail(hiopMPSReadStatus::parse_error, "data record appears outside a supported section.");
  }

  if(!saw_rows || !saw_columns || !saw_end) {
    return fail(hiopMPSReadStatus::parse_error, "MPS file must contain ROWS, COLUMNS, and ENDATA sections.");
  }

  std::unordered_map<std::string, char> all_rows;
  for(const auto& row : rows) {
    if(!all_rows.emplace(row.second, row.first).second) {
      return fail(hiopMPSReadStatus::parse_error, "duplicate row name '" + row.second + "'.");
    }
    if(row.first == 'N' && impl_->objective_row.empty()) impl_->objective_row = row.second;
  }
  if(!requested_objective.empty()) {
    const auto found = all_rows.find(requested_objective);
    if(found == all_rows.end() || found->second != 'N') {
      return fail(hiopMPSReadStatus::parse_error, "OBJNAME does not identify a free row.");
    }
    impl_->objective_row = requested_objective;
  }

  std::unordered_map<std::string, index_type> row_index;
  for(const auto& row : rows) {
    if(row.first != 'N') {
      row_index.emplace(row.second, static_cast<index_type>(impl_->row_names.size()));
      impl_->row_names.push_back(row.second);
      impl_->row_types.push_back(row.first);
    }
  }

  std::unordered_map<std::string, index_type> column_index;
  auto add_column = [&](const std::string& name) {
    const auto found = column_index.find(name);
    if(found != column_index.end()) return found->second;
    const index_type index = static_cast<index_type>(impl_->column_names.size());
    column_index.emplace(name, index);
    impl_->column_names.push_back(name);
    return index;
  };
  for(const auto& entry : matrix_entries) add_column(entry.column);

  const std::string rhs_name = !options.rhs_name.empty() ? options.rhs_name : (rhs_order.empty() ? "" : rhs_order[0]);
  const std::string range_name =
      !options.ranges_name.empty() ? options.ranges_name : (range_order.empty() ? "" : range_order[0]);
  const std::string bounds_name =
      !options.bounds_name.empty() ? options.bounds_name : (bound_order.empty() ? "" : bound_order[0]);

  if(!options.rhs_name.empty() && rhs_seen.find(rhs_name) == rhs_seen.end()) {
    return fail(hiopMPSReadStatus::parse_error, "requested RHS set '" + rhs_name + "' was not found.");
  }
  if(!options.ranges_name.empty() && range_seen.find(range_name) == range_seen.end()) {
    return fail(hiopMPSReadStatus::parse_error, "requested RANGES set '" + range_name + "' was not found.");
  }
  if(!options.bounds_name.empty() && bound_seen.find(bounds_name) == bound_seen.end()) {
    return fail(hiopMPSReadStatus::parse_error, "requested BOUNDS set '" + bounds_name + "' was not found.");
  }

  for(const auto& entry : rhs_entries) {
    if(all_rows.find(entry.row) == all_rows.end()) {
      return fail(hiopMPSReadStatus::parse_error, "RHS record references unknown row '" + entry.row + "'.");
    }
  }
  for(const auto& entry : range_entries) {
    const auto row = all_rows.find(entry.row);
    if(row == all_rows.end() || row->second == 'N') {
      return fail(hiopMPSReadStatus::parse_error,
                  "RANGES record references unknown or non-constraint row '" + entry.row + "'.");
    }
  }
  for(const auto& bound : bound_entries) {
    if(column_index.find(bound.column) == column_index.end()) {
      return fail(hiopMPSReadStatus::parse_error,
                  "BOUNDS record references unknown column '" + bound.column + "'.");
    }
  }

  impl_->costs.assign(impl_->column_names.size(), 0.0);
  std::map<std::pair<index_type, index_type>, double> matrix;
  for(const auto& entry : matrix_entries) {
    const auto row = all_rows.find(entry.row);
    if(row == all_rows.end()) {
      return fail(hiopMPSReadStatus::parse_error, "COLUMNS record references unknown row '" + entry.row + "'.");
    }
    const index_type col = column_index[entry.column];
    if(row->second == 'N') {
      if(entry.row == impl_->objective_row) impl_->costs[col] += entry.value;
    } else {
      matrix[std::make_pair(row_index[entry.row], col)] += entry.value;
    }
  }

  std::vector<double> rhs(impl_->row_names.size(), 0.0);
  std::vector<double> ranges_value(impl_->row_names.size(), 0.0);
  std::vector<bool> has_range(impl_->row_names.size(), false);
  double objective_rhs = 0.0;
  for(const auto& entry : rhs_entries) {
    if(entry.set != rhs_name) continue;
    const auto row = all_rows.find(entry.row);
    if(row->second == 'N') {
      if(entry.row == impl_->objective_row) objective_rhs += entry.value;
    } else {
      rhs[row_index[entry.row]] += entry.value;
    }
  }
  for(const auto& entry : range_entries) {
    if(entry.set != range_name) continue;
    const auto row = row_index.find(entry.row);
    // All range row references were validated above.
    assert(row != row_index.end());
    ranges_value[row->second] += entry.value;
    has_range[row->second] = true;
  }

  impl_->row_lower.resize(impl_->row_names.size());
  impl_->row_upper.resize(impl_->row_names.size());
  for(std::size_t i = 0; i < impl_->row_names.size(); ++i) {
    const double b = rhs[i];
    const double range = std::abs(ranges_value[i]);
    if(impl_->row_types[i] == 'E') {
      impl_->row_lower[i] = has_range[i] && ranges_value[i] < 0.0 ? b - range : b;
      impl_->row_upper[i] = has_range[i] && ranges_value[i] >= 0.0 ? b + range : b;
    } else if(impl_->row_types[i] == 'L') {
      impl_->row_lower[i] = has_range[i] ? b - range : -kMpsInfinity;
      impl_->row_upper[i] = b;
    } else {
      impl_->row_lower[i] = b;
      impl_->row_upper[i] = has_range[i] ? b + range : kMpsInfinity;
    }
  }

  impl_->column_lower.assign(impl_->column_names.size(), 0.0);
  impl_->column_upper.assign(impl_->column_names.size(), kMpsInfinity);
  for(const auto& bound : bound_entries) {
    if(bound.set != bounds_name) continue;
    const index_type col = column_index[bound.column];
    if(bound.type == "LO") {
      impl_->column_lower[col] = bound.value;
    } else if(bound.type == "UP") {
      impl_->column_upper[col] = bound.value;
      if(bound.value < 0.0 && impl_->column_lower[col] == 0.0) impl_->column_lower[col] = -kMpsInfinity;
    } else if(bound.type == "FX") {
      impl_->column_lower[col] = bound.value;
      impl_->column_upper[col] = bound.value;
    } else if(bound.type == "FR") {
      impl_->column_lower[col] = -kMpsInfinity;
      impl_->column_upper[col] = kMpsInfinity;
    } else if(bound.type == "MI") {
      impl_->column_lower[col] = -kMpsInfinity;
    } else if(bound.type == "PL") {
      impl_->column_upper[col] = kMpsInfinity;
    }
  }
  for(std::size_t col = 0; col < impl_->column_names.size(); ++col) {
    if(impl_->column_lower[col] > impl_->column_upper[col]) {
      return fail(hiopMPSReadStatus::parse_error,
                  "inconsistent bounds for column '" + impl_->column_names[col] + "'.");
    }
  }

  impl_->objective_scale = impl_->sense == ObjectiveSense::maximize ? -1.0 : 1.0;
  impl_->objective_offset = impl_->objective_scale * (-objective_rhs);
  for(auto& cost : impl_->costs) cost *= impl_->objective_scale;

  for(const auto& entry : matrix) {
    if(entry.second == 0.0) continue;
    impl_->matrix_rows.push_back(entry.first.first);
    impl_->matrix_columns.push_back(entry.first.second);
    impl_->matrix_values.push_back(entry.second);
    if(impl_->row_lower[entry.first.first] == impl_->row_upper[entry.first.first]) ++impl_->nnz_equalities;
    else ++impl_->nnz_inequalities;
  }
  impl_->matrix_row_offsets.assign(impl_->row_names.size() + 1, 0);
  for(const auto row : impl_->matrix_rows) {
    ++impl_->matrix_row_offsets[static_cast<std::size_t>(row) + 1];
  }
  for(std::size_t row = 1; row < impl_->matrix_row_offsets.size(); ++row) {
    impl_->matrix_row_offsets[row] += impl_->matrix_row_offsets[row - 1];
  }

  impl_->loaded = true;
  impl_->error.clear();
  return hiopMPSReadStatus::success;
}

bool hiopInterfaceMPS::is_loaded() const { return impl_->loaded; }
const std::string& hiopInterfaceMPS::last_error() const { return impl_->error; }
const std::string& hiopInterfaceMPS::model_name() const { return impl_->model; }
const std::vector<std::string>& hiopInterfaceMPS::variable_names() const { return impl_->column_names; }
const std::vector<std::string>& hiopInterfaceMPS::constraint_names() const { return impl_->row_names; }
hiopInterfaceMPS::ObjectiveSense hiopInterfaceMPS::objective_sense() const { return impl_->sense; }

double hiopInterfaceMPS::original_objective_value(double hiop_objective_value) const
{
  return impl_->objective_scale * hiop_objective_value;
}

bool hiopInterfaceMPS::get_prob_sizes(size_type& n, size_type& m)
{
  if(!impl_->loaded) return false;
  n = static_cast<size_type>(impl_->column_names.size());
  m = static_cast<size_type>(impl_->row_names.size());
  return true;
}

bool hiopInterfaceMPS::get_prob_info(NonlinearityType& type)
{
  type = hiopLinear;
  return impl_->loaded;
}

bool hiopInterfaceMPS::get_vars_info(const size_type& n,
                                     double* xlow,
                                     double* xupp,
                                     NonlinearityType* type)
{
  if(!impl_->loaded || n != static_cast<size_type>(impl_->column_names.size())) return false;
  for(size_type i = 0; i < n; ++i) {
    xlow[i] = impl_->column_lower[i];
    xupp[i] = impl_->column_upper[i];
    type[i] = hiopLinear;
  }
  return true;
}

bool hiopInterfaceMPS::get_cons_info(const size_type& m,
                                     double* clow,
                                     double* cupp,
                                     NonlinearityType* type)
{
  if(!impl_->loaded || m != static_cast<size_type>(impl_->row_names.size())) return false;
  for(size_type i = 0; i < m; ++i) {
    clow[i] = impl_->row_lower[i];
    cupp[i] = impl_->row_upper[i];
    type[i] = hiopLinear;
  }
  return true;
}

bool hiopInterfaceMPS::get_sparse_blocks_info(size_type& nx,
                                               size_type& nnz_sparse_Jaceq,
                                               size_type& nnz_sparse_Jacineq,
                                               size_type& nnz_sparse_Hess_Lagr)
{
  if(!impl_->loaded) return false;
  nx = static_cast<size_type>(impl_->column_names.size());
  nnz_sparse_Jaceq = impl_->nnz_equalities;
  nnz_sparse_Jacineq = impl_->nnz_inequalities;
  nnz_sparse_Hess_Lagr = 0;
  return true;
}

bool hiopInterfaceMPS::eval_f(const size_type& n, const double* x, bool, double& obj_value)
{
  if(!impl_->loaded || n != static_cast<size_type>(impl_->costs.size())) return false;
  obj_value = impl_->objective_offset;
  for(size_type i = 0; i < n; ++i) obj_value += impl_->costs[i] * x[i];
  return true;
}

bool hiopInterfaceMPS::eval_grad_f(const size_type& n, const double*, bool, double* gradf)
{
  if(!impl_->loaded || n != static_cast<size_type>(impl_->costs.size())) return false;
  std::copy(impl_->costs.begin(), impl_->costs.end(), gradf);
  return true;
}

bool hiopInterfaceMPS::eval_cons(const size_type& n,
                                 const size_type& m,
                                 const size_type& num_cons,
                                 const index_type* idx_cons,
                                 const double* x,
                                 bool,
                                 double* cons)
{
  if(!impl_->loaded || n != static_cast<size_type>(impl_->column_names.size()) ||
     m != static_cast<size_type>(impl_->row_names.size()) || num_cons < 0 || num_cons > m ||
     (num_cons > 0 && (idx_cons == nullptr || cons == nullptr))) {
    return false;
  }
  for(size_type k = 0; k < num_cons; ++k) {
    const index_type row = idx_cons[k];
    if(row < 0 || row >= m) return false;
    cons[k] = 0.0;
    for(std::size_t entry = impl_->matrix_row_offsets[static_cast<std::size_t>(row)];
        entry < impl_->matrix_row_offsets[static_cast<std::size_t>(row) + 1];
        ++entry) {
      cons[k] += impl_->matrix_values[entry] * x[impl_->matrix_columns[entry]];
    }
  }
  return true;
}

bool hiopInterfaceMPS::eval_cons(const size_type& n, const size_type& m, const double* x, bool, double* cons)
{
  if(!impl_->loaded || n != static_cast<size_type>(impl_->column_names.size()) ||
     m != static_cast<size_type>(impl_->row_names.size())) {
    return false;
  }
  std::fill(cons, cons + m, 0.0);
  for(std::size_t k = 0; k < impl_->matrix_values.size(); ++k) {
    cons[impl_->matrix_rows[k]] += impl_->matrix_values[k] * x[impl_->matrix_columns[k]];
  }
  return true;
}

bool hiopInterfaceMPS::eval_Jac_cons(const size_type& n,
                                     const size_type& m,
                                     const size_type& num_cons,
                                     const index_type* idx_cons,
                                     const double*,
                                     bool,
                                     const size_type& nnzJacS,
                                     index_type* iJacS,
                                     index_type* jJacS,
                                     double* MJacS)
{
  if(!impl_->loaded || n != static_cast<size_type>(impl_->column_names.size()) ||
     m != static_cast<size_type>(impl_->row_names.size()) || num_cons < 0 || num_cons > m ||
     (num_cons > 0 && idx_cons == nullptr) || ((iJacS == nullptr) != (jJacS == nullptr))) {
    return false;
  }

  size_type expected_nnz = 0;
  for(size_type k = 0; k < num_cons; ++k) {
    const index_type row = idx_cons[k];
    if(row < 0 || row >= m) return false;
    expected_nnz += static_cast<size_type>(impl_->matrix_row_offsets[static_cast<std::size_t>(row) + 1] -
                                          impl_->matrix_row_offsets[static_cast<std::size_t>(row)]);
  }
  if(nnzJacS != expected_nnz) return false;

  size_type output = 0;
  for(size_type k = 0; k < num_cons; ++k) {
    const index_type row = idx_cons[k];
    for(std::size_t entry = impl_->matrix_row_offsets[static_cast<std::size_t>(row)];
        entry < impl_->matrix_row_offsets[static_cast<std::size_t>(row) + 1];
        ++entry, ++output) {
      if(iJacS != nullptr) {
        iJacS[output] = k;
        jJacS[output] = impl_->matrix_columns[entry];
      }
      if(MJacS != nullptr) MJacS[output] = impl_->matrix_values[entry];
    }
  }
  return true;
}

bool hiopInterfaceMPS::eval_Jac_cons(const size_type& n,
                                     const size_type& m,
                                     const double*,
                                     bool,
                                     const size_type& nnzJacS,
                                     index_type* iJacS,
                                     index_type* jJacS,
                                     double* MJacS)
{
  if(!impl_->loaded || n != static_cast<size_type>(impl_->column_names.size()) ||
     m != static_cast<size_type>(impl_->row_names.size()) ||
     nnzJacS != static_cast<size_type>(impl_->matrix_values.size())) {
    return false;
  }
  if((iJacS == nullptr) != (jJacS == nullptr)) return false;
  if(iJacS != nullptr && jJacS != nullptr) {
    std::copy(impl_->matrix_rows.begin(), impl_->matrix_rows.end(), iJacS);
    std::copy(impl_->matrix_columns.begin(), impl_->matrix_columns.end(), jJacS);
  }
  if(MJacS != nullptr) std::copy(impl_->matrix_values.begin(), impl_->matrix_values.end(), MJacS);
  return true;
}

bool hiopInterfaceMPS::eval_Hess_Lagr(const size_type& n,
                                      const size_type& m,
                                      const double*,
                                      bool,
                                      const double&,
                                      const double*,
                                      bool,
                                      const size_type& nnzHSS,
                                      index_type*,
                                      index_type*,
                                      double*)
{
  return impl_->loaded && n == static_cast<size_type>(impl_->column_names.size()) &&
         m == static_cast<size_type>(impl_->row_names.size()) && nnzHSS == 0;
}

}  // namespace hiop
