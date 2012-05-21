#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <Eigen/Core>

#include <kmeans.h>
#include <lspi.h>

class SoftKmeansBasis : public LSPI::Basis
{
	private:
	ckm::SoftKMeans &kmeans_;
	public:
	
	SoftKmeansBasis(ckm::SoftKMeans &kmeans) : kmeans_(kmeans)
	{
	}
	
	virtual int getNumFeatures()
	{
		return kmeans_.getNumPrototypes();
	}
	virtual Eigen::VectorXd getFeatures(Eigen::VectorXd &xu)
	{
		return kmeans_.getFeature(xu);
	}
};

int 
main(int args, char * argv[])
{
  if (args < 2)
    {
      fprintf(stderr, "usage: %s INPUT_FILE\n", argv[0]);
      return 1;
    }

  ckm::SoftKMeans *kmeans = NULL;
  FILE *f = fopen(argv[1], "r");
  
  const int x_size = 4;
  const int input_dim = 5;
  int data_size;
  float xu[input_dim];
  float xrt[6];
  std::vector<LSPI::xux> data;
  Eigen::VectorXd tmp(input_dim);
  
  if (f == NULL)
    {
      fprintf(stderr, "Error: could not read data from %s\n", argv[1]);
      return 1;
    }
  
  if(fscanf(f, "#%d\n", &data_size) != 1 || data_size <= 0)
    {
      fprintf(stderr, "Error: could not read data size %s\n", argv[1]);
      return 1;
    }

  printf("data_size: %d\n", data_size);
  // read all the data
  while(fscanf(f, "%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f\n", xu, xu+1, xu+2, xu+3, xu+4, 
        xrt, xrt+1, xrt+2, xrt+3, xrt+4, xrt+5) == 11)
    {
	  struct LSPI::xux next;
      for (int i = 0; i < x_size; ++i)
        {
		  next.x.resize(x_size);
		  next.xn.resize(x_size);
          next.x(i) = xu[i];
          next.xn(i) = xrt[i];
        }
      next.u = xu[x_size];
      next.r = xrt[x_size];
      data.push_back(next);
    }
  fclose(f);
  
  // allocate kmeans instance
  kmeans = new ckm::SoftKMeans(1,1,1); // will be initialized properly by fromFile 
  // read from file
  FILE *kmf = fopen("test_xu.kmeans", "r");
  if (kmf == NULL)
    {
	  fprintf(stderr, "Error, could not open kmeans file for reading \n");
	  return 1;
	}
  kmeans->fromFile(kmf);
  fclose(kmf);
  
  // configure lspi
  SoftKmeansBasis basis(*kmeans);
  std::vector<double> actions;
  actions.push_back(-10.);
  actions.push_back(10.);
  assert(kmeans->getNumPrototypes() == 40);
  Eigen::Matrix<double, 40, 1> w0;
  // chose initial policy to be random
  srand48(time(0));
  for (int i = 0; i < w0.size(); ++i)
    w0(i) = drand48();
  // start lspi
  Eigen::VectorXd w = LSPI::lspi<40>(data, basis, 0.8, 0.03, w0, actions);
  
  // and print the resulting policy vector
  std::cout << std::endl << "Resulting policy vector: " << w.transpose() << std::endl;
  
  delete kmeans;
  return 0;
}
