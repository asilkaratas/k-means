/*
 * engineSystem.h
 *
 *  Created on: May 28, 2016
 *      Author: asilkaratas
 */

#ifndef ENGINESYSTEM_H_
#define ENGINESYSTEM_H_

#include <helper_functions.h>
#include "vector_functions.h"

class EngineSystem
{
public:
	EngineSystem(uint numPoints, uint numClusters, float **objects, int *membership);
	~EngineSystem();

	int getNumPoints() const
	{
		return m_numPoints;
	}

	void update(float deltaTime);
	void reset();

	float *getArray();
	void setArray(const float *data, int start, int count);


	unsigned int getCurrentReadBuffer() const
	{
		return m_posVbo;
	}

	unsigned int getColorBuffer()       const
	{
		return m_colorVBO;
	}

	void *getCudaPosVBO()              const
	{
		return (void *)m_cudaPosVBO;
	}

	void *getCudaColorVBO()            const
	{
		return (void *)m_cudaColorVBO;
	}


private:
	uint createVBO(uint size);
	void _initialize();
	void _finalize();

private:
	bool m_bInitialized;
	uint m_numPoints;
	uint m_numClusters;

	float **m_objects;
	int	  *m_membership;

	float *m_hPos;
	float *m_dPos;

	uint   m_posVbo;            // vertex buffer object for particle positions
	uint   m_colorVBO;          // vertex buffer object for colors

	float *m_cudaPosVBO;        // these are the CUDA deviceMem Pos
	float *m_cudaColorVBO;      // these are the CUDA deviceMem Color

	struct cudaGraphicsResource *m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource *m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

};

#endif /* ENGINESYSTEM_H_ */
