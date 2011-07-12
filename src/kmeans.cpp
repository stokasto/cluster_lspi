#include "kmeans.h"

// general includes
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <climits>

#include "tools.h"

using std::min;
using std::max;

#define KM_LVL 8
#define VERBOSE 1
#define BOOTSTRAP_DIM 15

using namespace ckm;

KMeans::~KMeans()
{
  // free memory of prototype centers
  deleteInVector<Eigen::VectorXd> (&centers_);
}

void
KMeans::randomInit(const std::vector<Eigen::VectorXd>& pats)
{
  int center_index = 0;
  int pats_size = (int) pats.size();
  Eigen::VectorXd *center_instance;
  // we chose to use a vector here to store the closest distances to avoid allocation problems
  //  as we expect the number of patterns in pats to be fairly large :)
  // TODO: is this really necessary, or more convient ?
  std::vector<double> closest_dist;
  // clear out old centers
  deleteInVector<Eigen::VectorXd> (&centers_);
  // to assign prototype locations for our num_centers_ protypes
  // we choose the first center randomly
  center_index = drand48() * pats_size;
  center_instance = new Eigen::VectorXd(prot_size_);
  (*center_instance) = pats[center_index];
  LOG(KM_LVL+1, << "Pushing back prot at: " << center_instance->transpose());
  centers_.push_back(center_instance);

  // NO smart init -->
  //    also chose all other centers randomly
  for (int curr_center = 1; curr_center < num_centers_; ++curr_center)
    {
      center_index = drand48() * pats_size;
      center_instance = new Eigen::VectorXd(prot_size_);
      (*center_instance) = pats[center_index];
      LOG(KM_LVL+1, << "Pushing back prot at: " << center_instance->transpose());
      centers_.push_back(center_instance);
    }
}

/**
 * Initialize Prototypes.
 * NOTE: instead of simple random initialization we use the method outlined in
 * Arthur, D. and Vassilvitskii, S. k-means++: the advantages of careful seeding. - 2007
 **/
void
KMeans::smartInit(const std::vector<Eigen::VectorXd>& pats)
{
  int center_index = 0;
  int pats_size = (int) pats.size();
  double potential = 0.;
  Eigen::VectorXd *center_instance;
  // we chose to use a vector here to store the closest distances to avoid allocation problems
  //  as we expect the number of patterns in pats to be fairly large :)
  // TODO: is this really necessary, or more convient ?
  std::vector<double> closest_dist;
  // clear out old centers
  deleteInVector<Eigen::VectorXd> (&centers_);
  // to assign prototype locations for our num_centers_ protypes
  // we choose the first center randomly
  center_index = drand48() * pats_size;
  center_instance = new Eigen::VectorXd(prot_size_);
  (*center_instance) = pats[center_index];
  LOG(KM_LVL+1, << "Pushing back prot at: " << center_instance->transpose());
  centers_.push_back(center_instance);

  // compute distances to this initial center for all other patterns
  for (int i = 0; i < pats_size; i++)
    {
      double tmpNorm2 = (pats[i] - *center_instance).squaredNorm();
      closest_dist.push_back(tmpNorm2);
      potential += tmpNorm2;
    }

  // chose all other centers
  for (int curr_center = 1; curr_center < num_centers_; ++curr_center)
    {
      // we do this by searching for the best next potential at which a prototype
      // should be placed
      double best_potential = -1.;
      int best_idx = 0;
      for (int trial = 0; trial < num_trials_init_; ++trial)
        {
          // chose a random decision boundary for the potential
          double rand_pot = drand48() * potential;
          for (center_index = 0; center_index < pats_size - 1; ++center_index)
            {
              if (rand_pot <= closest_dist[center_index])
                {
                  LOG(KM_LVL+3, << "BREAK rand pot: " << rand_pot << " dist: " << closest_dist[center_index]);
                  break;
                }
              else
                {
                  LOG(KM_LVL+3, << "rand pot: " << rand_pot << " dist: " << closest_dist[center_index]);
                  rand_pot -= closest_dist[center_index];
                }
            }
          // calculate correct potential for center_index
          double curr_potential = 0.;
          for (int i = 0; i < pats_size; i++)
            {
              double tmpNorm2 = (pats[i] - pats[center_index]).squaredNorm();
              curr_potential += min(tmpNorm2, closest_dist[i]);
            }

          // and finaly update best potential if necessary
          if (best_potential < 0. || curr_potential < best_potential)
            {
              best_potential = curr_potential;
              best_idx = center_index;
            }
        }
      LOG(KM_LVL+3, << "Chosing: " << best_idx << " next pot: " << best_potential);
      // add best center to list of prototype centers
      center_instance = new Eigen::VectorXd(prot_size_);
      (*center_instance) = pats[best_idx];
      LOG(KM_LVL+1, << "Pushing back prot at: " << center_instance->transpose());
      centers_.push_back(center_instance);

      // update current potential
      potential = best_potential;
      // update closest distances to the next chosen center
      for (int i = 0; i < pats_size; i++)
        {
          double tmpNorm2 = (pats[i] - *center_instance).squaredNorm();
          closest_dist[i] = min(tmpNorm2, closest_dist[i]);
        }
    }
}

int
KMeans::findBestPrototype(const Eigen::VectorXd &x, double &dist, double &dist2nd)
{
  double best_dist = -1.;
  double tmp_dist = 0.;
  int best_idx = 0;
  for (int j = 0; j < (int) centers_.size(); ++j)
    {
      /*if (bootstrap)
        {
          tmp_dist = (x.block(BOOTSTRAP_DIM, 0, prot_size_ - BOOTSTRAP_DIM, 1)
              - centers_[j]->block(BOOTSTRAP_DIM, 0, prot_size_ - BOOTSTRAP_DIM, 1)).squaredNorm();
        }
      else*/
        {
          tmp_dist = (x - *centers_[j]).squaredNorm();
        }
      if (best_dist < 0 || tmp_dist < best_dist)
        {
          dist2nd = best_dist;
          best_dist = tmp_dist;
          best_idx = j;
        }
    }
  dist = best_dist;
  return best_idx;
}

void
KMeans::cluster(int *assignments, int *old_assignments, const std::vector<Eigen::VectorXd>& pats,
    bool &changed)
{
  int best_idx = 0;
  changed = false; // initially assume no assignment changed
  //omp_set_num_threads(MAX_THREADS);

#pragma omp parallel for
  for (int i = 0; i < (int) pats.size(); ++i)
    { // for each pattern find the best assignment to a prototype
      double dist, dist2nd;
      best_idx = findBestPrototype(pats[i], dist, dist2nd);
      LOG(KM_LVL, << "Best prototype for: " << pats[i].transpose()
          << " is: " << centers_[best_idx]->transpose() << " at idx: " << best_idx);
      /*if (!changed // we have not  yet found and assignment that has changed
       && assignments[i] != best_idx)
       { // check if assignment changed --> if so update changed flag
       changed = true;
       }*/
      assignments[i] = best_idx;
    }
  for (int i = 0; i < (int) pats.size(); ++i)
    {
      if (assignments[i] != old_assignments[i])
        {
          changed = true;
          break;
        }
    }
  LOG(KM_LVL, << "DONE COMPUTING CLUSTERS");
}

void
KMeans::updateCenters(int *assignments, const std::vector<Eigen::VectorXd>& pats, bool bootstrap)
{
  int patterns_per_cluster[num_centers_];
  int curr_assignment = 0;

  for (int i = 0; i < num_centers_; i++)
    { // reset pattern per cluster counts
      patterns_per_cluster[i] = 0;
    }

  for (int i = 0; i < (int) pats.size(); i++)
    { // traverse all training patterns
      curr_assignment = assignments[i];
      // update number of patterns per cluster
      ++patterns_per_cluster[curr_assignment];
      // update center of closest prototype
      if (bootstrap)
        {
          centers_[curr_assignment]->block(BOOTSTRAP_DIM, 0, prot_size_ - BOOTSTRAP_DIM, 1)
              += pats[i].block(BOOTSTRAP_DIM, 0, prot_size_ - BOOTSTRAP_DIM, 1);
        }
      else
        {
          (*centers_[curr_assignment]) += pats[i];
        }
      LOG(KM_LVL+1, << "prototype " << curr_assignment << " has now "
          << patterns_per_cluster[curr_assignment] << " assigned patterns");
    }

#pragma omp parallel for
  for (int i = 0; i < num_centers_; i++)
    { // normalize all prototype positions to get the proper mean of the pattern vectors
      if (patterns_per_cluster[i] > 0.)
        { // beware of the evil division by zero :)
          (*centers_[i]) /= (patterns_per_cluster[i] > 0) ? patterns_per_cluster[i] : 1.;
        }
    }
  LOG(KM_LVL+1, "DONE updating");
}

/**
 * Init kmeans via boostrapping.
 * We essentially execute one kmeans step on a reduced problem here
 * the idea behind it being that this should give reasonably good centers for our prototypes
 * without already pushing them towards a specific solution.
 *
 * WARNING:
 *       This assumes that the dimensions of the patterns in pats are ordered by their
 *       relevance (e.g. last component corresponds to the highest eigenvectors of a PCA).
 */
void
KMeans::bootstrapInit(const std::vector<Eigen::VectorXd>& pats)
{
  Eigen::VectorXd *center_instance;
  int center_index;
  bool changed;

  // init all centers randomly
  for (int curr_center = 0; curr_center < num_centers_; ++curr_center)
    {
      center_index = drand48() * pats.size();
      center_instance = new Eigen::VectorXd(prot_size_);
      center_instance->setZero(prot_size_);
      for (int i = 0; i < BOOTSTRAP_DIM; ++i)
        {
          (*center_instance)(i) = pats[center_index](i);
        }
      LOG(KM_LVL+1, << "Pushing back prot at: " << center_instance->transpose());
      centers_.push_back(center_instance);
    }

  // perform one kmeans update step in the reduced space
  int *assignments = new int[pats.size()]; // cluster assignments from training patterns -> prototypes
  int *old_assignments = new int[pats.size()];

  cluster(assignments, old_assignments, pats, changed);
  updateCenters(assignments, pats, true);
}

void
KMeans::fit(const std::vector<Eigen::VectorXd>& pats, int max_iter, bool smart, bool bootstrap,
    void
    (*stat_callback)(int, KMeans *))
{
  assert(pats.size() > 0);
  assert(pats.size() < INT_MAX);
  assert(num_centers_ > 0); // we obviously need at least one prototype
  int pats_size = (int) pats.size();
  assert(num_centers_ < pats_size); // and we also want more data than prototypes
  int iter = 0;
  int *assignments = new int[pats_size]; // cluster assignments from training patterns -> prototypes
  int *old_assignments = new int[pats_size];
  bool changed; // indicates whether the assignment of one pattern has changed

  // init random seeds --> important for initPrototypes() method
  // NOTE: REMOVED THIS << random seeds should always be set to correct values by the user
  //srand(time(0));
  //srand48(time(0));

  /* choose from the 3 possible init methods */
  if (bootstrap)
    {
      LOG(0, << "kmeans starting bootstrap init")
      bootstrapInit(pats);
    }
  else if (smart)
    {
      LOG(0, << "kmeans starting smart init")
      smartInit(pats);
    }
  else
    {
      LOG(0, << "kmeans starting random init")
      randomInit(pats);
    }

  if (stat_callback != 0)
    {
      stat_callback(-1, this);
    }
  // cluster the training patterns
  cluster(assignments, old_assignments, pats, changed);
  changed = true;

  while (iter < max_iter)
    { // as long as we are not converged and have not reached the iteration limit
      // iterate over all patterns and compute new centers
      LOG(0, << "kmeans updating centers iteration: " << iter)
      updateCenters(assignments, pats);

      // DEBUG ONLY
      if (VERBOSE)
        {
          int all_changed = 0;
          for (int i = 0; i < pats_size; ++i)
            if (assignments[i] != old_assignments[i])
              ++all_changed;
          LOG(0,<< "Num of changed assignments: " << all_changed);
        }

      // swap assignment arrays
      int *tmp = old_assignments;
      old_assignments = assignments;
      assignments = tmp;
      // next cluster again
      LOG(0, << "kmeans clustering iteration: " << iter)
      cluster(assignments, old_assignments, pats, changed);
      LOG(0, << "kmeans changed ? " << changed);
      if (stat_callback != 0)
        {
          stat_callback(iter, this);
        }
      if (!changed)
        break; // converged to optimum
      ++iter;
    }
  // print some statistics
  Eigen::VectorXi counts = Eigen::VectorXi::Zero(num_centers_);
  for (int i = 0; i < pats_size; ++i)
    {
      ++counts[assignments[i]];
    }
  printf("pats_per_center: ");
  for (int i = 0; i < num_centers_; ++i)
    {
      printf("(%d : %d) ", i, counts[i]);
    }
  printf("\n");
  if (assignments)
    delete[] assignments;
  if (old_assignments)
    delete[] old_assignments;
}

Eigen::VectorXd
KMeans::match(const Eigen::VectorXd &x)
{
  double dist, dist2nd;
  int best_idx = findBestPrototype(x, dist, dist2nd);
  return *centers_[best_idx];
}

Eigen::VectorXd
KMeans::getFeature(const Eigen::VectorXd &x)
{
  Eigen::VectorXd result;
  int best_idx;
  double dist, dist2nd;
  result.setZero(num_centers_);
  best_idx = findBestPrototype(x, dist, dist2nd);
  result(best_idx) = 1.;
  return result;
}

bool
KMeans::readParameters(const char *fname)
{
  FILE *f = fopen(fname, "r");
  if (f == 0)
    {
      ERROR("could not read from file %s", fname);
      return false;
    }
  bool res = fromFile(f);
  fclose(f);
  return res;
}

bool
KMeans::fromFile(FILE *f)
{
  if (f == NULL)
    {
      ERROR("Warning could not write to file");
      return false;
    }
  assert(fread(&num_centers_, sizeof(int), 1, f) == 1);
  assert(fread(&num_trials_init_, sizeof(int), 1, f) == 1);
  assert(fread(&prot_size_, sizeof(int), 1, f) == 1);
  // free possibly allocated centers
  deleteInVector<Eigen::VectorXd> (&centers_);
  for (int i = 0; i < num_centers_; ++i)
    {
      Eigen::VectorXd *tmp = new Eigen::VectorXd(prot_size_);
      read_vector(*tmp, f);
      centers_.push_back(tmp);
    }
  return true;
}

bool
KMeans::saveParameters(const char *fname)
{
  FILE *f = fopen(fname, "w");
  bool res = toFile(f);
  fclose(f);
  return res;
}

bool
KMeans::toFile(FILE *f)
{
  if (f == NULL)
    {
      ERROR("Warning could not write to file");
      return false;
    }
  // we chose to write all prototypes to the file in binary format
  // the format is as follows:
  // 32bits - num_centers_
  // 32bits - num_trials_init_
  // 32bits - prot_size_
  // num_centers_ * 64bit * prot_sze_ - prototypes 
  assert(fwrite(&num_centers_, sizeof(int), 1, f) == 1);
  assert(fwrite(&num_trials_init_, sizeof(int), 1, f) == 1);
  assert(fwrite(&prot_size_, sizeof(int), 1, f) == 1);
  for (int i = 0; i < num_centers_; ++i)
    {
      write_vector(*centers_[i], f);
    }
  return true;
}

Eigen::VectorXd
SoftKMeans::getFeature(const Eigen::VectorXd &x)
{
  Eigen::VectorXd result;
  double mean = 0.;
  result.setZero(num_centers_);

  // instead of computing the best feature match
  // apply the soft method here, as described in the paper
//#pragma omp parallel for
  for (int i = 0; i < (int) centers_.size(); ++i)
    {
      double norm = (x - *centers_[i]).norm();
      result(i) = norm;
    }
  mean = result.sum() / result.size();
//#pragma omp parallel for
  for (int i = 0; i < (int) centers_.size(); ++i)
    {
      result(i) = max(0., mean - result(i));
    }
  return result;
}
