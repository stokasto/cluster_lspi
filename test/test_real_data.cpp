#include <stdlib.h>
#include <stdio.h>
#include <Eigen/Core>

#include <kmeans.h>


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
  int num_prototypes = 40;
  const int input_dim = 5;
  int data_size;
  float xu[input_dim];
  float dontcare[6];
  std::vector<Eigen::VectorXd> data;
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
        dontcare, dontcare+1, dontcare+2, dontcare+3, dontcare+4, dontcare+5) == 11)
    {
      for (int i = 0; i < input_dim; ++i)
        tmp(i) = xu[i];
      data.push_back(tmp);
    }
  fclose(f);
  
  // allocate kmeans instance
  kmeans = new ckm::SoftKMeans(num_prototypes, data_size);
  // train the kmeans
  kmeans->fit(data, 300);
  
  FILE *kmf = fopen("test_xu.kmeans", "w");
  if (kmf == NULL)
    {
	  fprintf(stderr, "Error, could not open kmeans file for writing \n");
	  return 1;
	}
  
  assert(kmeans->toFile(kmf));
  
  fclose(kmf);
  delete kmeans;
  return 0;
}
