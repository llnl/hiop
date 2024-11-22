#ifndef HIOP_PERTURB_PD_LINSSYS
#define HIOP_PERTURB_PD_LINSSYS

#include "hiopNlpFormulation.hpp"
#include "hiopVector.hpp"

namespace hiop
{

class hiopPDPerturbation
{
public:
  enum DeltasUpdateType
  {
    None = -2,
    Initialized = -1,
    DualUpdate = 0,
    PrimalUpdate = 1,
    PDUpdate = 2
  };

  /** Default constructor
   * Provides complete initialization, but uses algorithmic parameters from the Ipopt
   * implementation paper. \ref initialize(hiopNlpFormulation*) should be used to initialize
   * this class based on HiOp option file or HiOp user-supplied runtime options.
   */
  hiopPDPerturbation()
      : delta_wx_curr_(nullptr),
        delta_wd_curr_(nullptr),
        delta_cc_curr_(nullptr),
        delta_cd_curr_(nullptr),
        delta_wx_last_(nullptr),
        delta_wd_last_(nullptr),
        delta_cc_last_(nullptr),
        delta_cd_last_(nullptr),
        delta_wx_curr_db_{0.},
        delta_wd_curr_db_{0.},
        delta_cc_curr_db_{0.},
        delta_cd_curr_db_{0.},
        delta_wx_last_db_{0.},
        delta_wd_last_db_{0.},
        delta_cc_last_db_{0.},
        delta_cd_last_db_{0.},
        delta_w_min_bar_(1e-20),
        delta_w_max_bar_(1e+40),
        delta_w_0_bar_(1e-4),
        kappa_w_minus_(1. / 3),
        kappa_w_plus_bar_(100.),
        kappa_w_plus_(8.),
        delta_c_bar_(1e-8),
        kappa_c_(0.25),
        min_uniform_ratio_{0.9},
        max_uniform_ratio_{1.0},
        hess_degenerate_(dtNotEstablished),
        jac_degenerate_(dtNotEstablished),
        num_degen_iters_(0),
        num_degen_max_iters_(3),
        deltas_test_type_(dttNoTest),
        mu_(1e-8),
        nlp_(nullptr)
  {}

  virtual ~hiopPDPerturbation()
  {
    delete delta_wx_curr_;
    delete delta_wd_curr_;
    delete delta_cc_curr_;
    delete delta_cd_curr_;
    delete delta_wx_last_;
    delete delta_wd_last_;
    delete delta_cc_last_;
    delete delta_cd_last_;
  }

  /** Initializes and reinitializes object based on the 'options' parameters of the
   * 'nlp_' object.
   * Returns 'false' if something goes wrong, otherwise 'true'
   */
  virtual bool initialize(hiopNlpFormulation* nlp);

  /** Set log-barrier mu. */
  virtual inline void set_mu(const double& mu) { mu_ = mu; }

  /** Called when a new linear system is attempted to be factorized
   */
  virtual bool compute_initial_deltas() = 0;

  /** Method for correcting inertia */
  virtual bool compute_perturb_wrong_inertia() = 0;

  /** Method for correcting singular Jacobian
   *  (follows Ipopt closely since the paper seems to be outdated)
   */
  virtual bool compute_perturb_singularity() = 0;

  inline bool copy_from_curr_perturbations(hiopVector& delta_wx,
                                           hiopVector& delta_wd,
                                           hiopVector& delta_cc,
                                           hiopVector& delta_cd)
  {
    delta_wx.copyFrom(*delta_wx_curr_);
    delta_wd.copyFrom(*delta_wd_curr_);
    delta_cc.copyFrom(*delta_cc_curr_);
    delta_cd.copyFrom(*delta_cd_curr_);
    return true;
  }

  inline hiopVector* get_curr_delta_wx() const { return delta_wx_curr_; }
  inline hiopVector* get_curr_delta_wd() const { return delta_wd_curr_; }
  inline hiopVector* get_curr_delta_cc() const { return delta_cc_curr_; }
  inline hiopVector* get_curr_delta_cd() const { return delta_cd_curr_; }

  inline DeltasUpdateType get_curr_delta_type() const { return deltas_curr_update_; }

  virtual bool check_consistency() = 0;

protected:
  /** Current and last perturbations, primal is split in x and d, dual in c and d. */
  hiopVector* delta_wx_curr_;
  hiopVector* delta_wd_curr_;
  hiopVector* delta_cc_curr_;
  hiopVector* delta_cd_curr_;

  hiopVector* delta_wx_last_;
  hiopVector* delta_wd_last_;
  hiopVector* delta_cc_last_;
  hiopVector* delta_cd_last_;

  double delta_wx_curr_db_;
  double delta_wd_curr_db_;
  double delta_cc_curr_db_;
  double delta_cd_curr_db_;

  double delta_wx_last_db_;
  double delta_wd_last_db_;
  double delta_cc_last_db_;
  double delta_cd_last_db_;

  /** Algorithmic parameters */

  /** Smallest possible perturbation for Hessian (for primal 'x' and 's' variables). */
  double delta_w_min_bar_;
  /** Maximal perturbation for for Hessian (for primal 'x' and 's' variables). */
  double delta_w_max_bar_;
  /** First trial value for delta_w perturbation. */
  double delta_w_0_bar_;
  /** Decrease factor for delta_w. */
  double kappa_w_minus_;
  /** Increase factor for delta_w for first required perturbation. */
  double kappa_w_plus_bar_;
  /** Increase factor for delta_w for later perturbations. */
  double kappa_w_plus_;

  /** Factor for regularization for potentially rank-deficient Jacobian. */
  double delta_c_bar_;
  /** Exponent of mu when computing regularization for Jacobian. */
  double kappa_c_;

  double min_uniform_ratio_;
  double max_uniform_ratio_;

  /** Degeneracy is handled as in Ipopt*/

  /** Type for degeneracy flags */
  enum DegeneracyType
  {
    dtNotEstablished,
    dtNotDegenerate,
    dtDegenerate
  };
  /** Structural degeneracy of the Hessian */
  DegeneracyType hess_degenerate_;

  /** Structural degeneracy of the Jacobian */
  DegeneracyType jac_degenerate_;

  /* Counter for inertia corrections at which the matrix is suspected to be degenerate. */
  int num_degen_iters_;
  /* Max number of iters after which to conclude matrix is degenerate. */
  const int num_degen_max_iters_;

  /** Status of current trial configuration */
  enum DeltasTestType
  {
    dttNoTest,
    dttDeltac0Deltaw0,
    dttDeltacposDeltaw0,
    dttDeltac0Deltawpos,
    dttDeltacposDeltawpos
  };

  /** Current status */
  DeltasTestType deltas_test_type_;
  DeltasUpdateType deltas_curr_update_;

  /** Log barrier mu in the outer loop. */
  double mu_;

  hiopNlpFormulation* nlp_;

protected:  // methods
  /** Decides degeneracy @hess_degenerate_ and @jac_degenerate_ based on @deltas_test_type_
   *  when the @num_degen_iters_ > @num_degen_max_iters_
   */
  virtual void update_degeneracy_type();
  virtual void set_delta_curr_vec(DeltasUpdateType taskid) = 0;
  virtual void set_delta_last_vec(DeltasUpdateType taskid) = 0;
};

/* method used for quasi newton's method */
class hiopPDPerturbationNull : public hiopPDPerturbation
{
public:
  /** Default constructor
   * Provides complete initialization, but uses algorithmic parameters from the Ipopt
   * implementation paper. \ref initialize(hiopNlpFormulation*) should be used to initialize
   * this class based on HiOp option file or HiOp user-supplied runtime options.
   */
  hiopPDPerturbationNull()
      : hiopPDPerturbation()
  {}
  virtual ~hiopPDPerturbationNull() {}

  /** Called when a new linear system is attempted to be factorized
   */
  virtual bool compute_initial_deltas() { return true; }

  /** Method for correcting inertia */
  virtual bool compute_perturb_wrong_inertia() { return true; }

  /** Method for correcting singular Jacobian
   *  (follows Ipopt closely since the paper seems to be outdated)
   */
  virtual bool compute_perturb_singularity() { return true; }

  virtual bool check_consistency() { return true; }

protected:  // methods
  virtual void set_delta_curr_vec(DeltasUpdateType taskid) {}
  virtual void set_delta_last_vec(DeltasUpdateType taskid) {}
};

/**
 * @brief When inertia correction is required, update primal regularization before dual regularzation.
 *        Both regularizations are computed as \delta*I, i.e., a scalar times an identity matrix
 */
class hiopPDPerturbationPrimalFirstScalar : public hiopPDPerturbation
{
public:
  /** Default constructor
   * Provides complete initialization, but uses algorithmic parameters from the Ipopt
   * implementation paper. \ref initialize(hiopNlpFormulation*) should be used to initialize
   * this class based on HiOp option file or HiOp user-supplied runtime options.
   */
  hiopPDPerturbationPrimalFirstScalar()
      : hiopPDPerturbation()
  {}
  virtual ~hiopPDPerturbationPrimalFirstScalar() {}

  /** Called when a new linear system is attempted to be factorized
   */
  virtual bool compute_initial_deltas();

  /** Method for correcting inertia */
  virtual bool compute_perturb_wrong_inertia();

  /** Method for correcting singular Jacobian
   *  (follows Ipopt closely since the paper seems to be outdated)
   */
  virtual bool compute_perturb_singularity();

  virtual bool check_consistency();

protected:
protected:  // methods
  virtual void set_delta_curr_vec(DeltasUpdateType taskid);
  virtual void set_delta_last_vec(DeltasUpdateType taskid);

  /** Internal method implementing the computation of delta_w's to correct wrong inertia
   *
   */
  virtual bool guts_of_compute_perturb_wrong_inertia(double& delta_wx, double& delta_wd);

  virtual double compute_delta_c(const double& mu) const;
};

/**
 * @brief When inertia correction is required, update primal regularization before dual regularzation.
 *        Both regularizations are computed as a random vector.
 *        v0.7: uniform distribution has been implemented
 */
class hiopPDPerturbationPrimalFirstRand : public hiopPDPerturbationPrimalFirstScalar
{
public:
  hiopPDPerturbationPrimalFirstRand()
      : hiopPDPerturbationPrimalFirstScalar()
  {}

  virtual ~hiopPDPerturbationPrimalFirstRand() {}

protected:  // methods
  virtual void set_delta_curr_vec(DeltasUpdateType taskid);
  virtual void set_delta_last_vec(DeltasUpdateType taskid);
};

/**
 * @brief When inertia correction is required, update dual regularization before primal regularzation.
 *        Both regularizations are computed as \delta*I, i.e., a scalar times an identity matrix
 */
class hiopPDPerturbationDualFirstScalar : public hiopPDPerturbation
{
public:
  hiopPDPerturbationDualFirstScalar();

  virtual ~hiopPDPerturbationDualFirstScalar();

  /** Called when a new linear system is attempted to be factorized
   */
  virtual bool compute_initial_deltas();

  /** Method for correcting inertia */
  virtual bool compute_perturb_wrong_inertia();

  /** Method for correcting singular Jacobian
   *  (follows Ipopt closely since the paper seems to be outdated)
   */
  virtual bool compute_perturb_singularity();

  virtual bool check_consistency();

protected:  // variables
  double delta_c_min_bar_;
  double kappa_c_plus_;

protected:  // methods
  virtual void set_delta_curr_vec(DeltasUpdateType taskid);
  virtual void set_delta_last_vec(DeltasUpdateType taskid);

  /**
   * Internal method implementing the computation of delta_w's to correct wrong inertia
   */
  virtual bool compute_primal_perturb_impl();

  /**
   * Internal method implementing the computation of delta_c's to correct wrong inertia
   */
  virtual bool compute_dual_perturb_impl(const double& mu);
};

/**
 * @brief When inertia correction is required, update dual regularization before primal regularzation.
 *        Both regularizations are computed as a random vector.
 *        v0.7: uniform distribution has been implemented
 */
class hiopPDPerturbationDualFirstRand : public hiopPDPerturbationDualFirstScalar
{
public:
  hiopPDPerturbationDualFirstRand()
      : hiopPDPerturbationDualFirstScalar()
  {}

  virtual ~hiopPDPerturbationDualFirstRand() {}

protected:  // methods
  virtual void set_delta_curr_vec(DeltasUpdateType taskid);
  virtual void set_delta_last_vec(DeltasUpdateType taskid);
};

}  // namespace hiop
#endif
