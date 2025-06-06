#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <immintrin.h>

#include "CycleTimer.h"

using namespace std;

typedef struct {
  // Control work assignments
  int start, end;

  // Shared by all functions
  double *data;
  double *clusterCentroids;
  int *clusterAssignments;
  double *currCost;
  int M, N, K;

  // New args
  int threadId;
  double* minDist;
  int mStart, mEnd;
} WorkerArgs;

/**
 * Checks if the algorithm has converged.
 * 
 * @param prevCost Pointer to the K dimensional array containing cluster costs 
 *    from the previous iteration.
 * @param currCost Pointer to the K dimensional array containing cluster costs 
 *    from the current iteration.
 * @param epsilon Predefined hyperparameter which is used to determine when
 *    the algorithm has converged.
 * @param K The number of clusters.
 * 
 * NOTE: DO NOT MODIFY THIS FUNCTION!!!
 */
static bool stoppingConditionMet(double *prevCost, double *currCost,
                                 double epsilon, int K) {
  for (int k = 0; k < K; k++) {
    if (abs(prevCost[k] - currCost[k]) > epsilon)
      return false;
  }
  return true;
}

/**
 * Computes L2 distance between two points of dimension nDim.
 * 
 * @param x Pointer to the beginning of the array representing the first
 *     data point.
 * @param y Poitner to the beginning of the array representing the second
 *     data point.
 * @param nDim The dimensionality (number of elements) in each data point
 *     (must be the same for x and y).
 */
double dist(double *x, double *y, int nDim) {
  double accum = 0.0;
  for (int i = 0; i < nDim; i++) {
    accum += pow((x[i] - y[i]), 2);
  }
  return sqrt(accum);
}

/**
 * Assigns each data point to its "closest" cluster centroid.
 */
void computeAssignments(WorkerArgs *const args) {
  /*
  double *minDist = new double[args->M];
  
  // Initialize arrays
  for (int m =0; m < args->M; m++) {
    minDist[m] = 1e30;
    args->clusterAssignments[m] = -1;
  }
  */
  // Assign datapoints to closest centroids
  for (int k = args->start; k < args->end; k++) {
    for (int m = args->mStart; m < args->mEnd; m++) {
      double d = dist(&args->data[m * args->N],
                      &args->clusterCentroids[k * args->N], args->N);
      if (d < args->minDist[m]) {
        args->minDist[m] = d;
        args->clusterAssignments[m] = k;
      }
    }
  }

  // free(minDist);
}

/**
 * Given the cluster assignments, computes the new centroid locations for
 * each cluster.
 */
void computeCentroids(WorkerArgs *const args) {
  int *counts = new int[args->K];

  // Zero things out
  for (int k = 0; k < args->K; k++) {
    counts[k] = 0;
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] = 0.0;
    }
  }


  // Sum up contributions from assigned examples
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] +=
          args->data[m * args->N + n];
    }
    counts[k]++;
  }

  // Compute means
  for (int k = 0; k < args->K; k++) {
    counts[k] = max(counts[k], 1); // prevent divide by 0
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] /= counts[k];
    }
  }

  free(counts);
}

/**
 * Computes the per-cluster cost. Used to check if the algorithm has converged.
 */
void computeCost(WorkerArgs *const args) {
  double *accum = new double[args->K];

  // Zero things out
  for (int k = 0; k < args->K; k++) {
    accum[k] = 0.0;
  }

  // Sum cost for all data points assigned to centroid
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    accum[k] += dist(&args->data[m * args->N],
                     &args->clusterCentroids[k * args->N], args->N);
  }

  // Update costs
  for (int k = args->start; k < args->end; k++) {
    args->currCost[k] = accum[k];
  }

  free(accum);
}

void workerThreadStart(WorkerArgs *const args)
{
  double startTime {CycleTimer::currentSeconds()};
  computeAssignments(args);
  double endTime {CycleTimer::currentSeconds()};
  printf("Thread %d spends %.2lf ms to finish the task\n", args->threadId, endTime - startTime);
}

void computeAssignmentsThread(WorkerArgs *const args) {
  double *minDist = new double[args->M];
  
  // Initialize arrays
  for (int m =0; m < args->M; m++) {
    minDist[m] = 1e30;
    args->clusterAssignments[m] = -1;
  }

  int NUM_THREADS = 12;

  std::thread workers[NUM_THREADS];
  WorkerArgs thread_args[NUM_THREADS];

  printf("%d %d\n", args->K, args->K / NUM_THREADS);

  for (int i = 0; i < NUM_THREADS; ++i)
  {
    thread_args[i].data = args->data;
    thread_args[i].clusterAssignments = args->clusterAssignments;
    thread_args[i].clusterCentroids = args->clusterCentroids;
    thread_args[i].K = args->K;
    thread_args[i].M = args->M;
    thread_args[i].N = args->N;
    thread_args[i].start = args->start;
    thread_args[i].end = args->end;
    thread_args[i].threadId = i;
    thread_args[i].minDist = minDist;

    thread_args[i].mStart = i * (args->M / NUM_THREADS);
    if (i == NUM_THREADS - 1 && args->M % NUM_THREADS != 0)
    {
      thread_args[i].mEnd = args->M;
    }
    else
    {
      thread_args[i].mEnd = (i + 1) * (args->M / NUM_THREADS);
    }
  }

  for (int i = 1; i < NUM_THREADS; ++i)
  {
    workers[i] = std::thread(workerThreadStart, &thread_args[i]);
  }

  workerThreadStart(&thread_args[0]);

  for (int i = 1; i < NUM_THREADS; ++i)
  {
    workers[i].join();
  }
}

/**
 * Computes the K-Means algorithm, using std::thread to parallelize the work.
 *
 * @param data Pointer to an array of length M*N representing the M different N 
 *     dimensional data points clustered. The data is layed out in a "data point
 *     major" format, so that data[i*N] is the start of the i'th data point in 
 *     the array. The N values of the i'th datapoint are the N values in the 
 *     range data[i*N] to data[(i+1) * N].
 * @param clusterCentroids Pointer to an array of length K*N representing the K 
 *     different N dimensional cluster centroids. The data is laid out in
 *     the same way as explained above for data.
 * @param clusterAssignments Pointer to an array of length M representing the
 *     cluster assignments of each data point, where clusterAssignments[i] = j
 *     indicates that data point i is closest to cluster centroid j.
 * @param M The number of data points to cluster.
 * @param N The dimensionality of the data points.
 * @param K The number of cluster centroids.
 * @param epsilon The algorithm is said to have converged when
 *     |currCost[i] - prevCost[i]| < epsilon for all i where i = 0, 1, ..., K-1
 */
void kMeansThread(double *data, double *clusterCentroids, int *clusterAssignments,
               int M, int N, int K, double epsilon) {

  // Used to track convergence
  double *prevCost = new double[K];
  double *currCost = new double[K];

  // The WorkerArgs array is used to pass inputs to and return output from
  // functions.
  WorkerArgs args;
  args.data = data;
  args.clusterCentroids = clusterCentroids;
  args.clusterAssignments = clusterAssignments;
  args.currCost = currCost;
  args.M = M;
  args.N = N;
  args.K = K;

  // Initialize arrays to track cost
  for (int k = 0; k < K; k++) {
    prevCost[k] = 1e30;
    currCost[k] = 0.0;
  }

  /* Main K-Means Algorithm Loop */
  int iter = 0;
  while (!stoppingConditionMet(prevCost, currCost, epsilon, K)) {
    // Update cost arrays (for checking convergence criteria)
    for (int k = 0; k < K; k++) {
      prevCost[k] = currCost[k];
    }

    // Setup args struct
    args.start = 0;
    args.end = K;

    double startTime {CycleTimer::currentSeconds()};

    // computeAssignments(&args);
    computeAssignmentsThread(&args);
    double endTime {CycleTimer::currentSeconds()};
    printf("Assignments Computation took %lf seconds to finish.\n", endTime - startTime);
    startTime = CycleTimer::currentSeconds();
    computeCentroids(&args);
    endTime = CycleTimer::currentSeconds();
    printf("Centroids Computation took %lf seconds to finish.\n", endTime - startTime);
    startTime = CycleTimer::currentSeconds();
    computeCost(&args);
    endTime = CycleTimer::currentSeconds();
    printf("Cost Computation took %lf seconds to finish.\n", endTime - startTime);

    iter++;
  }

  free(currCost);
  free(prevCost);
}