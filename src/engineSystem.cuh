/*
 * engineSystem.cuh
 *
 *  Created on: May 28, 2016
 *      Author: asilkaratas
 */

extern "C"
{
	void cudaGLInit(int argc, char **argv);

	void allocateArray(void **devPtr, int size);
	void freeArray(void *devPtr);

	void threadSync();

	void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
	void copyArrayToDevice(void *device, const void *host, int offset, int size);
	void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
	void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
	void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
	void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
}
