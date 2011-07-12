#include "tools.h"


    bool
    ckm::write_vector(Eigen::VectorXd &x, FILE *f)
    {
      assert(x.size() > 0);
      int chunck_size = 1024;
      double buf[chunck_size];
      int p;
      int size = x.size();
      if (chunck_size > x.size())
        {
          chunck_size = x.size();
        }
      // write as many chuncks as possible
      for (p = 0; p + chunck_size <= size; p += chunck_size)
        {
          for (int j = 0; j < chunck_size; ++j)
            {
              buf[j] = x(p + j);
            }
          assert((int) fwrite(buf, sizeof(double), chunck_size, f) == chunck_size);
        }
      // write rest of vector
      if (p < size)
        {
          for (int j = 0; j < size - p; ++j)
            {
              buf[j] = x(p + j);
            }
          assert((int) fwrite(buf, sizeof(double), size - p, f) == size - p);
        }
      return true;
    }

    bool
    ckm::read_vector(Eigen::VectorXd &x, FILE *f)
    {
      assert(x.size() > 0);
      int chunck_size = 1024;
      double buf[chunck_size];
      int p;
      int size = x.size();
      // read as many chunkgs as possible
      for (p = 0; p + chunck_size <= size; p += chunck_size)
        {
          assert((int) fread(buf, sizeof(double), chunck_size, f) == chunck_size);
          for (int j = 0; j < chunck_size; ++j)
            {
              x(p + j) = buf[j];
            }
        }
      // read rest of vector
      if (p < size)
        {
          assert((int) fread(buf, sizeof(double), size - p, f) == size - p);
          for (int j = 0; j < size - p; ++j)
            {
              x(p + j) = buf[j];
            }
        }
      return true;
    }
    bool
    ckm::write_matrix(Eigen::MatrixXd &A, FILE *f)
    { // WARNING: we read and store in column major format!
      assert(A.rows() > 0);
      assert(A.cols() > 0);
      int chunck_size = 1024;
      double buf[chunck_size];
      int p;
      int size = A.rows() * A.cols();
      if (chunck_size > A.size())
        {
          chunck_size = A.size();
        }
      // write as many chuncks as possible
      for (p = 0; p + chunck_size <= size; p += chunck_size)
        {
          for (int j = 0; j < chunck_size; ++j)
            {
              buf[j] = A(p + j);
            }
          assert((int) fwrite(buf, sizeof(double), chunck_size, f) == chunck_size);
        }
      // write rest of vector
      if (p < size)
        {
          for (int j = 0; j < size - p; ++j)
            {
              buf[j] = A(p + j);
            }
          assert((int) fwrite(buf, sizeof(double), size - p, f) == size - p);
        }
      return true;
    }

    bool
    ckm::read_matrix(Eigen::MatrixXd &A, FILE *f)
    { // WARNING: we read and store in column major format!
      assert(A.rows() > 0);
      assert(A.cols() > 0);
      int chunck_size = 1024;
      double buf[chunck_size];
      int p;
      int size = A.rows() * A.cols();
      // read as many chunks as possible
      for (p = 0; p + chunck_size <= size; p += chunck_size)
        {
          assert((int) fread(buf, sizeof(double), chunck_size, f) == chunck_size);
          for (int j = 0; j < chunck_size; ++j)
            {
              A(p + j) = buf[j];
            }
        }
      // read rest of vector
      if (p < size)
        {
          assert((int) fread(buf, sizeof(double), size - p, f) == size - p);
          for (int j = 0; j < size - p; ++j)
            {
              A(p + j) = buf[j];
            }
        }
      return true;
    }

