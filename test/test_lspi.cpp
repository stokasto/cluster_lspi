#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <Eigen/Core>

#include <kmeans.h>

struct xux {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	Eigen::VectorXd x;
	double u;
	double r;
	Eigen::VectorXd xn;
};


double
get_action(ckm::SoftKMeans &phi, Eigen::VectorXd &w, Eigen::VectorXd &x, std::vector<double> &actions)
{
	int x_size = x.size();
	double best_q;
	double best_u;
	double tmp_q;
	Eigen::VectorXd tmp(x_size+1);
	Eigen::VectorXd tmp_phi(w.size());
	for (int i = 0; i < (int) actions.size(); ++i)
		{
			tmp.head(x_size) = x;
			tmp(x_size) = actions[i];
			tmp_phi = phi.getFeature(tmp);
			tmp_q = tmp_phi.dot(w);
			// we minimize q << cost function instead of reward
			if (tmp_q < best_q)
			  {
				  best_q = tmp_q;
				  best_u = actions[i];
			  }
		}
	return best_u;
}

Eigen::VectorXd
lstdq(std::vector<xux> &D, int k, ckm::SoftKMeans &phi, double gamma, Eigen::VectorXd &w0, std::vector<double> &actions)
{
	double delta = 0.5;
	double denominator = 1.;
	int x_size = D[0].x.size();
	Eigen::MatrixXd B(k, k);
	Eigen::MatrixXd tmp_B(k, k);
	Eigen::VectorXd tmp_Bdot;
	Eigen::VectorXd b(k);
	Eigen::VectorXd w(k);
	Eigen::VectorXd tmp_xu(x_size + 1);
	Eigen::VectorXd tmpphi_x(k);
	Eigen::VectorXd tmpphi_xn(k);
	
	// init to some multiple of the identity matrix
	for(int i = 0; i < k; ++i)
	  B(i,i) = 1./delta;
	
	for (int i = 0; i < (int) D.size(); ++i)
	  {
		  tmp_xu.head(x_size) = D[i].x;
		  tmp_xu(x_size) = D[i].u;
		  tmpphi_x = phi.getFeature(tmp_xu);
		  
		  tmp_xu.head(x_size) = D[i].x;
		  tmp_xu(x_size) = get_action(phi, w0, D[i].x, actions);
		  tmpphi_xn = phi.getFeature(tmp_xu);
		  
		  tmp_Bdot = ((tmpphi_x - gamma * tmpphi_xn).transpose() * B);
		  // calculate enumerator
		  tmp_B = B * tmp_Bdot;
		  // calculate denominator
		  denominator = 1 + tmp_Bdot.dot(tmpphi_x);
		  tmp_B /= denominator;
		  
		  B -= tmp_B;
		  
		  b += tmpphi_x * D[i].r;
      }
    w = B*b;
    return w;
}

Eigen::VectorXd
lspi(std::vector<xux> &D, int k, ckm::SoftKMeans &phi, double gamma, double epsilon, Eigen::VectorXd &w0, std::vector<double> &actions)
{
	Eigen::VectorXd wprime = w0;
	Eigen::VectorXd w(k);
	int count = 0;
	do 
	  {
		  w = wprime;
		  wprime = lstdq(D, k, phi, gamma, w, actions);
		  printf("lspi iteration: %d\n", count);
		  ++count;
	  } while ( (w-wprime).norm() < epsilon);
	return w;
}

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
  std::vector<xux> data;
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
	  struct xux next;
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
  std::vector<double> actions;
  actions.push_back(-10.);
  actions.push_back(10.);
  Eigen::VectorXd w0(kmeans->getNumPrototypes());
  // chose initial policy to be random
  srand48(time(0));
  for (int i = 0; i < w0.size(); ++i)
    w0(i) = drand48();
  // start lspi
  Eigen::VectorXd w = lspi(data, kmeans->getNumPrototypes(), *kmeans, 0.8, 0.1, w0, actions);
  
  // and print the resulting policy vector
  std::cout << "Resulting policy vector: " << w.transpose() << std::endl;
  
  delete kmeans;
  return 0;
}
