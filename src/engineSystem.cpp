/*
 * engineSystem.cpp
 *
 *  Created on: May 28, 2016
 *      Author: asilkaratas
 */

#include "engineSystem.h"
#include "engineSystem.cuh"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>

EngineSystem::EngineSystem(uint numPoints, uint numClusters, float **objects, int *membership):
	m_bInitialized(false),
	m_numPoints(numPoints),
	m_numClusters(numClusters),
	m_objects(objects),
	m_membership(membership),
	m_hPos(0),
	m_dPos(0)
{
	_initialize();
}


EngineSystem::~EngineSystem()
{
	_finalize();
	m_numPoints = 0;
}

uint
EngineSystem::createVBO(uint size)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

inline float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}


void
EngineSystem::_initialize()
{
	assert(!m_bInitialized);

	//printf("_initialize:%d\n", m_numPoints);

	m_hPos = new float[m_numPoints*4];
	memset(m_hPos, 0, m_numPoints*4*sizeof(float));

	unsigned int memSize = sizeof(float) * 4 * m_numPoints;

	m_posVbo = createVBO(memSize);
	registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);

	m_colorVBO = createVBO(memSize);
	registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

	glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
	float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	float *ptr = data;

	//printf("m_numClusters:%d \n", m_numClusters);
	float *colors = new float[m_numClusters * 3];
	memset(colors, 0, m_numClusters*3*sizeof(float));

	for(uint i = 0; i < m_numClusters; ++i)
	{
		colors[i * 3] = rand() / (float) RAND_MAX;
		colors[i * 3 + 1] = rand() / (float) RAND_MAX;
		colors[i * 3  + 2] = rand() / (float) RAND_MAX;
	}

	for (uint i=0; i< m_numPoints; i++)
	{
		int colorIndex = m_membership[i] * 3;

		*ptr++ = colors[colorIndex];
		*ptr++ = colors[colorIndex + 1];
		*ptr++ = colors[colorIndex + 2];
		*ptr++ = 1.0f;
	}

	delete [] colors;

	glUnmapBufferARB(GL_ARRAY_BUFFER);

	m_bInitialized = true;
}


void
EngineSystem::reset()
{
	float scale = 0.5f;
	for(int i = 0; i < m_numPoints; ++i) {
		m_hPos[i * 4] = m_objects[i][0] * scale;
		m_hPos[i * 4 + 1] = m_objects[i][1] * scale;
		m_hPos[i * 4 + 2] = m_objects[i][2] * scale;
		m_hPos[i * 4 + 3] = 0.0f;
	}

	setArray(m_hPos, 0, m_numPoints);
}


void
EngineSystem::_finalize()
{
    assert(m_bInitialized);

    unregisterGLBufferObject(m_cuda_colorvbo_resource);
	unregisterGLBufferObject(m_cuda_posvbo_resource);
	glDeleteBuffers(1, (const GLuint *)&m_posVbo);
	glDeleteBuffers(1, (const GLuint *)&m_colorVBO);
}

void
EngineSystem::update(float deltaTime)
{
    assert(m_bInitialized);

    float *dPos;

    dPos = (float *) mapGLBufferObject(&m_cuda_posvbo_resource);

    //dPos[0] = 0.1f;


/*
    for(int i = 0; i < m_numPoints; ++i) {
    	dPos[i * 4] = m_objects[i][0];
    	dPos[i * 4 + 1] = m_objects[i][1];
    	dPos[i * 4 + 2] = m_objects[i][2];
    	dPos[i * 4 + 3] = 0.0f;
    }
*/
    unmapGLBufferObject(m_cuda_posvbo_resource);
}


float *
EngineSystem::getArray()
{
    assert(m_bInitialized);

    float *hdata = 0;
    float *ddata = 0;
    struct cudaGraphicsResource *cuda_vbo_resource = 0;

    hdata = m_hPos;
	ddata = m_dPos;
	cuda_vbo_resource = m_cuda_posvbo_resource;

    copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numPoints*4*sizeof(float));
    return hdata;
}

void
EngineSystem::setArray( const float *data, int start, int count)
{
    assert(m_bInitialized);

    unregisterGLBufferObject(m_cuda_posvbo_resource);
	glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
	glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);

}







