#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

#define VERBOSE 0

namespace LSPI
{

class Basis 
{
	public:
	virtual int getNumFeatures() = 0;
	virtual Eigen::VectorXd getFeatures(Eigen::VectorXd &xu) = 0;
};

struct xux {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	Eigen::VectorXd x;
	double u;
	double r;
	Eigen::VectorXd xn;
};


template<int k>
double
get_action(Basis &phi, Eigen::Matrix<double, k, 1> &w, Eigen::VectorXd &x, std::vector<double> &actions)
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
			tmp_phi = phi.getFeatures(tmp);
			tmp_q = tmp_phi.dot(w);
			// we minimize q << cost function instead of reward
			// TODO: make this changeable
			if (tmp_q < best_q)
			  {
				  best_q = tmp_q;
				  best_u = actions[i];
			  }
		}
	return best_u;
}

template<int k>
typename Eigen::Matrix<double, k, 1>
lstdq(std::vector<xux> &D, Basis &phi, double gamma, Eigen::Matrix<double, k, 1> &w0, std::vector<double> &actions)
{
	assert(phi.getNumFeatures() == k);
	double delta = 0.5;
	double denominator = 1.;
	int x_size = D[0].x.size();
	//Eigen::MatrixXd B2(k, k);
	Eigen::Matrix<double, k, k> B;
	Eigen::Matrix<double, k, k> tmp_B;
	Eigen::Matrix<double, k, 1> b;
	Eigen::Matrix<double, k, 1> w;
	Eigen::Matrix<double, k, 1> tmpphi_x;
	Eigen::Matrix<double, k, 1> tmpphi_xn;
	Eigen::Matrix<double, 1, k> deltaphi;
	Eigen::VectorXd tmp_xu(x_size + 1);
	
	B.setZero();
	b.setZero();
	
	// init to some multiple of the identity matrix
	for(int i = 0; i < k; ++i)
	  B(i,i) = 1./delta;
	
	for (int i = 0; i < (int) D.size(); ++i)
	  {
		  // discounting factor 
		  gamma *= gamma;
		  tmp_xu.head(x_size) = D[i].x;
		  tmp_xu(x_size) = D[i].u;
		  tmpphi_x = phi.getFeatures(tmp_xu);
		  //std::cout << "phi: " << tmpphi_x.transpose() << std::endl;
		  tmp_xu.head(x_size) = D[i].x;
		  tmp_xu(x_size) = get_action<k>(phi, w0, D[i].x, actions);
		  tmpphi_xn = phi.getFeatures(tmp_xu);
#if 1 // optimized
		  deltaphi = (tmpphi_x - gamma * tmpphi_xn).transpose();
		  // calculate enumerator
		  tmp_B = B * ((tmpphi_x * deltaphi) * B);
		  // calculate denominator
		  denominator = 1 + (deltaphi * B) * tmpphi_x;
		  
		  // divide and subtract from B
		  tmp_B /= denominator;
		  
		  B -= tmp_B;
#else // non optimized
         deltaphi = (tmpphi_x - gamma * tmpphi_xn).transpose();
		 B += tmpphi_x * deltaphi;
		 //std::cout  <<  tmpphi_x * deltaphi << std::endl;
#endif 
		  
		  b += tmpphi_x * D[i].r;
      }
#if 1 // optimized
	w = B*b;
#else // non optimized
    for (int i = 0; i < B.rows(); ++i)
		for (int j = 0; j < B.cols(); ++j)
			B2(i,j) = B(i,j);
    w = B2.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b); 
#endif
    std::cout << w.transpose() << std::endl;
    return w;
}

template<int k>
Eigen::Matrix<double, k, 1>
lspi(std::vector<xux> &D, Basis &phi, double gamma, double epsilon, Eigen::Matrix<double, k, 1> &w0, std::vector<double> &actions)
{
	assert(phi.getNumFeatures() == k);
	Eigen::Matrix<double, k, 1> wprime = w0;
	Eigen::Matrix<double, k, 1> w(k);
	int count = 0;
	do 
	  {
		  w = wprime;
		  wprime = lstdq<k>(D, phi, gamma, w, actions);
		  printf("lspi iteration: %d\n", count);
		  printf("norm: %f\n", (w-wprime).norm());
		  if (VERBOSE)
		    {
		      std::cout << "\t" << w.transpose() << std::endl;
		      std::cout << "\t" << wprime.transpose() << std::endl;
		    }
		  ++count;
	  } while ( (w-wprime).norm() > epsilon);
	return w;
}

}
