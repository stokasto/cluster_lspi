#ifndef __CONTTOOLS_H__
#define __CONTTOOLS_H__

#include <vector>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sys/types.h>
#include <inttypes.h>
#include <iostream>

#include <Eigen/Core>

#define SQR(X) ((X)*(X))

#if 1
#define DEBUG 4
#define LOG(LVL,XX) \
if ( LVL <= DEBUG ) \
{ \
  std::cerr << "LOG " << LVL << ": " XX << std::endl; \
}
#define LOGF(LVL,FORM,...) \
if ( LVL <= DEBUG ) \
{ \
  fprint(stderr,"LOG %d: ", LVL); \
  fprintf(stderr,FORM,__VA_ARGS__); \
  fprint(stderr,"\n"); \
}
#else
#define LOG(LVL,XX)
#define LOGF(LVL,FORM,...)
#endif
 
#define ERROR(...) fprintf(stderr, __VA_ARGS__);

namespace ckm
{
  /* helper function to delete vector elements when a vector of pointers is allocated */
  template<class T>
    void
    deleteInVector(std::vector<T*>* deleteme)
    {
      while (!deleteme->empty())
        {
          delete deleteme->back();
          deleteme->pop_back();
        }
    }

  bool
  compare_pairs(std::pair<uint32_t, float> p1, std::pair<uint32_t, float> p2);
  bool
  write_vector(Eigen::VectorXd &x, FILE *f);
  bool
  read_vector(Eigen::VectorXd &x, FILE *f);
  bool
  write_matrix(Eigen::MatrixXd &A, FILE *f);
  bool
  read_matrix(Eigen::MatrixXd &A, FILE *f);

}

#endif

