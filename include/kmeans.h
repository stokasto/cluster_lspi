#ifndef __KMEANS_H__
#define __KMEANS_H__

#include <cstdio>
#include <vector>
#include <omp.h>

#include <Eigen/Core>

// local includes
#include "tools.h"

namespace ckm
{

  /* The classes that will be defined by this module */
  class KMeans;
  class SoftKMeans;

  class KMeans
  {
  protected:
    /* attributes */
    int num_centers_;
    int num_trials_init_;
    int prot_size_;
    std::vector<Eigen::VectorXd*> centers_;

    /* private helper methods */

    /**
     * Initialize the set of prototypes.
     */

    void
    bootstrapInit(const std::vector<Eigen::VectorXd>& pats);

    void
    randomInit(const std::vector<Eigen::VectorXd>& pats);

    void
    smartInit(const std::vector<Eigen::VectorXd>& pats);

    /**
     * Assign prototypes to each pattern in pats (in-place).
     *
     * --> RESULT:
     * The assignment is stored in assignments (which is required to be of size pats.size())
     * The changed flag indicates whether any assignment has changed.
     */
    void
    cluster(int *assignments, int *old_assignments, const std::vector<Eigen::VectorXd>& pats,
        bool &changed);

    /**
     * Update prototype positions (represented by their center).
     * Takes an assignment array and a vector of training patterns as argument.
     */
    void
    updateCenters(int *assignments, const std::vector<Eigen::VectorXd>& pats, bool bootstrap =
        false);

  public:

    /**
     * Construct a new KMeans instance
     *
     * --> Parameters
     * num_centers specifies the number of prototypes
     * data_saze the input dimension
     * num_trials_init is a parameter for the initialization a good value is either 1 or log(k)
     */
    KMeans(int num_centers, int data_size, int num_trials_init = -1) :
      num_centers_(num_centers)
    {
      prot_size_ = data_size;
      if (num_trials_init == -1)
        num_trials_init_ = log(num_centers);
      else
        num_trials_init_ = 1;

      if (num_trials_init_ < 1)
        {
          num_trials_init_ = 1;
        }
      if (num_trials_init_ > 500)
        {
          num_trials_init_ = 500;
        }
    }

    virtual
    ~KMeans();

    /* public methods */

    /**
     * Fit a kmeans model to the training patterns using at most max_iter iterations.
     */
    void
    fit(const std::vector<Eigen::VectorXd>& pats, int max_iter, bool smart = true, bool bootstrap =
        false, void
    (*stat_callback)(int, KMeans *) = 0);

    /**
     * Find the best matching prototype for a given input pattern.
     */
    Eigen::VectorXd
    match(const Eigen::VectorXd &x);

    /**
     * Find the best matching prototype for a given input pattern.
     * And return a Vector of size num_centers_ that has a 1 at the position
     * of the best prototype index.
     */
    virtual Eigen::VectorXd
    getFeature(const Eigen::VectorXd &x);

    /**
     * helper method that calculates the best prototype for a given pattern
     */
    int
    findBestPrototype(const Eigen::VectorXd &x, double &dist, double &dist2nd);

    inline int
    getFeatureSize()
    {
      return num_centers_;
    }

    inline int
    getNumPrototypes()
    {
      return num_centers_;
    }

    inline Eigen::VectorXd &
    getPrototype(int pos)
    {
      assert(pos < num_centers_);
      return *centers_[pos];
    }

    bool
    saveParameters(const char *fname);
    bool
    readParameters(const char *fname);

    bool
    toFile(FILE *f);
    bool
    fromFile(FILE *f);
  };

  class SoftKMeans : public KMeans
  {
  public:
    SoftKMeans(int num_centers, int data_size, int num_trials_init = -1) :
      KMeans(num_centers, data_size, num_trials_init)
    {
    }
    /**
     * Override standard getFeature Method
     */
    virtual Eigen::VectorXd
    getFeature(const Eigen::VectorXd &x);
  };
}

#endif
