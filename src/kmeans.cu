#include <stdio.h>
#include <stdlib.h>
#include "kmeans.h"
#include <cfloat>

static inline int nextPowerOfTwo(int n) {
    n--;

    n = n >>  1 | n;
    n = n >>  2 | n;
    n = n >>  4 | n;
    n = n >>  8 | n;
    n = n >> 16 | n;

    return ++n;
}

__host__ __device__ inline static
float calculateDistance(int    numObjs,
                    int    numClusters,
                    float *objects,
                    float *clusters,
                    int    objectId,
                    int    clusterId)
{
	float distance = (objects[objectId] - clusters[clusterId]) *
					 (objects[objectId] - clusters[clusterId]) +
					 (objects[numObjs * 1 + objectId] - clusters[numClusters * 1 + clusterId]) *
					 (objects[numObjs * 1 + objectId] - clusters[numClusters * 1 + clusterId]) +
					 (objects[numObjs * 2 + objectId] - clusters[numClusters * 2 + clusterId]) *
					 (objects[numObjs * 2 + objectId] - clusters[numClusters * 2 + clusterId]);

	return distance;
}

__global__ static
void findNearestCluster(int numObjs,
                          int numClusters,
                          float *objects,
                          float *deviceClusters,
                          int *membership,
                          int *intermediates)
{
    extern __shared__ unsigned int membershipChanged[];

    membershipChanged[threadIdx.x] = 0;

    int objectId = blockDim.x * blockIdx.x + threadIdx.x;

    if (objectId < numObjs) {
        int   index, i;
        float dist, minDist = FLT_MAX;

        for (i=0; i<numClusters; i++) {
            dist = calculateDistance(numObjs, numClusters, objects, deviceClusters, objectId, i);
            if (dist < minDist) {
            	minDist = dist;
                index = i;
            }
        }

        if (membership[objectId] != index) {
            membershipChanged[threadIdx.x] = 1;
        }

        membership[objectId] = index;

        __syncthreads();

        //calculating how many membership changed in this block.
        //warning! blockDim.x must be power of 2
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                membershipChanged[threadIdx.x] += membershipChanged[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
        	//printf("intermediates1:blockIdx.x:%d membershipChanged[0]:%d\n",blockIdx.x, membershipChanged[0]);
            intermediates[blockIdx.x] = membershipChanged[0];
        }
    }
}

__global__ static
void computeDelta(int *deviceIntermediates,
                   int numIntermediates,
                   int numIntermediates2,
                   int startIndex,
                   int *total)
{

    extern __shared__ unsigned int intermediates[];

/*
    if (threadIdx.x == 0) {
    	printf("\nstartIndex:%d:%d\n", startIndex, numIntermediates);
    }
*/
    int objectId = blockDim.x * blockIdx.x + threadIdx.x;
    //copy into shared memory
    intermediates[threadIdx.x] = (objectId < numIntermediates) ? deviceIntermediates[objectId] : 0;

    __syncthreads();

    //calculate how many membership changed. (Sum of block sums)
    //warning! numIntermediates2 must be power of 2.
    for (unsigned int s = numIntermediates2 / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            intermediates[threadIdx.x] += intermediates[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
    	//total[0] = intermediates[0];
    	//printf("intermediates:%d\n", intermediates[0]);
    	total[0] += intermediates[0];
    }
}


float** kmeans(float **objects,
                   int     numObjs,
                   int     numClusters,
                   float   threshold,
                   int    *membership,
                   int    *loop_iterations)
{
    int      i, j, index, loop=0;
    int     *newClusterSize;
    float    delta;
    float  **dimObjects;
    float  **clusters;
    float  **dimClusters;
    float  **newClusters;

    float *deviceObjects;
    float *deviceClusters;
    int *deviceMembership;
    int *deviceIntermediates;
    int *deviceIntermediatesLast;
    int *deviceTotal;


    malloc2D(dimObjects, 3, numObjs, float);
    for (i = 0; i < 3; i++) {
        for (j = 0; j < numObjs; j++) {
            dimObjects[i][j] = objects[j][i];
        }
    }

    malloc2D(dimClusters, 3, numClusters, float);
    for (i = 0; i < 3; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }

    for (i=0; i<numObjs; i++) membership[i] = -1;


    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    malloc2D(newClusters, 3, numClusters, float);
    memset(newClusters[0], 0, 3 * numClusters * sizeof(float));


    const unsigned int numThreadsPerClusterBlock = 1024;
    const unsigned int numClusterBlocks = (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;
    const unsigned int clusterBlockSharedDataSize = numThreadsPerClusterBlock * sizeof(unsigned int);
    //const unsigned int clusterBlockSharedDataSize = 49152;//49152
    //printf("sharedSize:%d\n", sizeof(unsigned char));

    const unsigned int numReductionThreadsMin = 1024;
    const unsigned int numReductionThreadsMax = nextPowerOfTwo(numClusterBlocks);
    const unsigned int numReductionThreads = numReductionThreadsMax < numReductionThreadsMin ? numReductionThreadsMax : numReductionThreadsMin;
    const unsigned int reductionBlockSharedDataSize = numReductionThreads * sizeof(unsigned int);
    const unsigned int reductionRound = (numClusterBlocks + numReductionThreads - 1)/numReductionThreads;//division takes floor.

    printf("numThreadsPerClusterBlock:%d \n", numThreadsPerClusterBlock);
    printf("clusterBlockSharedDataSize:%d \n", clusterBlockSharedDataSize);
    printf("numClusterBlocks:%d \n\n", numClusterBlocks);

    printf("numReductionThreadsMin:%d \n", numReductionThreadsMin);
    printf("numReductionThreadsMax:%d \n", numReductionThreadsMax);
    printf("numReductionThreads:%d \n", numReductionThreads);
    printf("reductionBlockSharedDataSize:%d \n", reductionBlockSharedDataSize);
    printf("reductionRound:%d \n\n", reductionRound);

    checkCuda(cudaMalloc(&deviceObjects, numObjs*3*sizeof(float)));
    checkCuda(cudaMalloc(&deviceClusters, numClusters*3*sizeof(float)));
    checkCuda(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
    checkCuda(cudaMalloc(&deviceIntermediates, numReductionThreadsMax*sizeof(unsigned int)));
    //checkCuda(cudaMemset(&deviceIntermediates, 0, numReductionThreadsMax*sizeof(unsigned int)));
    checkCuda(cudaMalloc(&deviceTotal, sizeof(unsigned int)));


    checkCuda(cudaMemcpy(deviceObjects, dimObjects[0], numObjs*3*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceMembership, membership, numObjs*sizeof(int), cudaMemcpyHostToDevice));

    do {
        checkCuda(cudaMemcpy(deviceClusters, dimClusters[0], numClusters*3*sizeof(float), cudaMemcpyHostToDevice));

        findNearestCluster <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numObjs, numClusters, deviceObjects, deviceClusters, deviceMembership, deviceIntermediates);

        cudaDeviceSynchronize(); checkLastCudaError();

        delta = 0;
        checkCuda(cudaMemcpy(deviceTotal, &delta, sizeof(int), cudaMemcpyHostToDevice));
        for(i = 0; i < reductionRound; ++i)
        {
        	int numIntermediates = (i < reductionRound - 1) || (numClusterBlocks%numReductionThreads==0) ? numReductionThreads : (numClusterBlocks - i*numReductionThreads)%numReductionThreads;
        	//printf("numIntermediates:%d\n", numIntermediates);
        	computeDelta <<< 1, numReductionThreads, reductionBlockSharedDataSize >>>
        	            (deviceIntermediates, numIntermediates, numReductionThreads, i * numReductionThreads, deviceTotal);

        	cudaDeviceSynchronize(); checkLastCudaError();
        }

        int d;
		checkCuda(cudaMemcpy(&d, deviceTotal, sizeof(int), cudaMemcpyDeviceToHost));
		delta = (float)d;

		//printf("d:%d\n\n", d);


/*
        computeDelta <<< 1, numReductionThreads, reductionBlockSharedDataSize >>>
            (deviceIntermediates, numClusterBlocks, numReductionThreads);

        cudaDeviceSynchronize(); checkLastCudaError();

        int d;
        checkCuda(cudaMemcpy(&d, deviceIntermediates, sizeof(int), cudaMemcpyDeviceToHost));
        delta = (float)d;
*/
        checkCuda(cudaMemcpy(membership, deviceMembership, numObjs*sizeof(int), cudaMemcpyDeviceToHost));

        for (i=0; i<numObjs; i++) {
            index = membership[i];

            newClusterSize[index]++;
            for (j=0; j<3; j++)
            {
            	newClusters[j][index] += objects[i][j];
            }

        }

        for (i=0; i<numClusters; i++) {
            for (j=0; j<3; j++) {
                if (newClusterSize[i] > 0)
                    dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
                newClusters[j][i] = 0.0;
            }
            newClusterSize[i] = 0;
        }

        delta /= numObjs;
    } while (delta > threshold && ++loop < 100);

    *loop_iterations = loop;

    printf("delta:%.16f\n", delta);
    malloc2D(clusters, numClusters, 3, float);

    for (i = 0; i < numClusters; i++) {
        for (j = 0; j < 3; j++) {
            clusters[i][j] = dimClusters[j][i];
        }
    }

    checkCuda(cudaFree(deviceObjects));
    checkCuda(cudaFree(deviceClusters));
    checkCuda(cudaFree(deviceMembership));
    checkCuda(cudaFree(deviceIntermediates));

    free(dimObjects[0]);
    free(dimObjects);
    free(dimClusters[0]);
    free(dimClusters);
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return clusters;
}

