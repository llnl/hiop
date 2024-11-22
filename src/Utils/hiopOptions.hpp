// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause).
// Please also read “Additional BSD Notice” below.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice, this list
// of conditions and the disclaimer below.
// ii. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the disclaimer (as noted below) in the documentation and/or
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may be used to
// endorse or promote products derived from this software without specific prior written
// permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
// SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S. Department
// of Energy (DOE). This work was produced at Lawrence Livermore National Laboratory under
// Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National Security, LLC
// nor any of their employees, makes any warranty, express or implied, or assumes any
// liability or responsibility for the accuracy, completeness, or usefulness of any
// information, apparatus, product, or process disclosed, or represents that its use would
// not infringe privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or services by
// trade name, trademark, manufacturer or otherwise does not necessarily constitute or
// imply its endorsement, recommendation, or favoring by the United States Government or
// Lawrence Livermore National Security, LLC. The views and opinions of authors expressed
// herein do not necessarily state or reflect those of the United States Government or
// Lawrence Livermore National Security, LLC, and shall not be used for advertising or
// product endorsement purposes.

#ifndef HIOP_OPTIONS
#define HIOP_OPTIONS

#include "hiopLogger.hpp"

#include <string>
#include <vector>
#include <map>

namespace hiop
{
class hiopLogger;

class hiopOptions
{
public:
  hiopOptions();
  virtual ~hiopOptions();

  // Seters for options values that should be self explanatory with the exception of the last parameter.
  //
  // Passing 'setFromFile' with non-default, 'true' value is for expert use-only. It indicates that the option
  // value comes from the options file (hiop.options) and will overwrite any options set at runtime by the
  // user's code. However, passing 'setFromFile' with 'true' at runtime is perfectly fine and will
  // conveniently "overwrite the overwriting" file options

  virtual bool SetNumericValue(const char* name, const double& value, const bool& setFromFile = false);
  virtual bool SetIntegerValue(const char* name, const int& value, const bool& setFromFile = false);
  virtual bool SetStringValue(const char* name, const char* value, const bool& setFromFile = false);

  virtual double GetNumeric(const char* name) const;
  virtual int GetInteger(const char* name) const;
  virtual std::string GetString(const char* name) const;

  void SetLog(hiopLogger* log_in)
  {
    log_ = log_in;
    ensure_consistence();
  }
  virtual void print(FILE* file, const char* msg = NULL) const;

  /**
   * Default name for the options file for the HiOp NLP solver. If the file does not exist,
   * the options object will be created with default options.
   *
   * When the PriDec solver is used and the worker (a.k.a., contingency or recourse) NLP solver
   * is HiOp's NLP solver, this filename will be used by the worker NLP solver to load the options.
   */
  static const char* default_filename;

  /**
   * Default name for the options file for the PriDec solver. If the file does not exist, the
   * options object will be created with default options.
   *
   * This filename is not used by the master or worker NLP solvers employed by the PriDec solver.
   */
  static const char* default_filename_pridec_solver;

  /**
   * Default name for the options file for the master (a.k.a basecase) NLP solver within PriDec solver.
   * This is passed by the PriDec solver to the user's routine that solves the master NLP. The
   * filename can be changed in the PriDec solver options files via the option 'options_file_master_prob'.
   *
   * The behavior for the case when the file does not exist is dependent on the underlying NLP solver
   * used to solve the master. If the file does not exist and HiOp is used as a master solver, HiOp NLP
   * will create an option object with default option values.
   */
  static const char* default_filename_pridec_masterNLP;

  /**
   * Default name for the options file for the feasibility restoration problem. If the file does not exist,
   * the options object will be created with default options.
   */
  static const char* default_filename_fr;

protected:
  /* internal use only */
  void register_num_option(const std::string& name, double defaultValue, double rangeLo, double rangeUp, const char* descr);
  void register_int_option(const std::string& name, int defaultValue, int rangeLo, int rangeUp, const char* descr);

  /// register a string option with a predetermined range
  void register_str_option(const std::string& name,
                           const std::string& defaultValue,
                           const std::vector<std::string>& range,
                           const char* description);
  /// register a string option that can take any value
  void register_str_option(const std::string& name, const std::string& defaultValue, const char* description);

  virtual void register_options() = 0;

  void load_from_file(const char* szFilename);

  virtual void ensure_consistence() = 0;

public:
  // Setter method used to ensure consistence. Does not alter 'specifiedInFile' and 'specifiedAtRuntime'
  virtual bool set_val(const char* name, const double& value);
  // Setter method used to ensure consistence. Does not alter 'specifiedInFile' and 'specifiedAtRuntime'
  virtual bool set_val(const char* name, const int& value);
  // Setter method used to ensure consistence. Does not alter 'specifiedInFile' and 'specifiedAtRuntime'
  virtual bool set_val(const char* name, const char* value);

  // Returns true if an option was set by the user (via options file or at runtime) or false if the option was not set
  // by the user or cannot be found
  virtual bool is_user_defined(const char* option_name);

protected:
  void log_printf(hiopOutVerbosity v, const char* format, ...);

  struct Option
  {  // option entry
    Option(const char* description)
        : descr(description),
          specifiedInFile(false),
          specifiedAtRuntime(false) {};
    virtual ~Option() {};
    std::string descr;
    bool specifiedInFile;
    bool specifiedAtRuntime;
    virtual void print(FILE* f, bool short_ver = false) const = 0;
  };
  struct OptionInt : public Option
  {
    OptionInt(int v, int low, int upp, const char* description)
        : Option(description),
          val(v),
          lb(low),
          ub(upp) {};
    int val, lb, ub;
    void print(FILE* f, bool short_ver = false) const;
  };
  struct OptionNum : public Option
  {
    OptionNum(double v, double low, double upp, const char* description)
        : Option(description),
          val(v),
          lb(low),
          ub(upp) {};
    double val, lb, ub;
    void print(FILE* f, bool short_ver = false) const;
  };

  struct OptionStr : public Option
  {
    OptionStr(std::string v, const std::vector<std::string>& range_, const char* description)
        : Option(description),
          val(v),
          range(range_) {};
    std::string val;
    std::vector<std::string> range;
    void print(FILE* f, bool short_ver = false) const;
  };

  std::map<std::string, Option*> mOptions_;

  hiopLogger* log_;
};

/**
 * @brief Options class specialized for the NLP solver
 *
 */
class hiopOptionsNLP : public hiopOptions
{
public:
  hiopOptionsNLP(const char* opt_filename = nullptr);
  virtual ~hiopOptionsNLP();

protected:
  virtual void register_options();
  virtual void ensure_consistence();
};

/**
 * @brief Options class specialized for the PriDec solver
 *
 */
class hiopOptionsPriDec : public hiopOptions
{
public:
  hiopOptionsPriDec(const char* opt_filename = nullptr);
  virtual ~hiopOptionsPriDec();

  void print(FILE* file, const char* msg = NULL) const;

protected:
  virtual void register_options();
  virtual void ensure_consistence();
};

}  // namespace hiop
#endif
