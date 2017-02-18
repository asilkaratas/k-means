#include <GL/glew.h>
#if defined (WIN32)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include "engineSystem.h"
#include "renderer.h"
#include "kmeans.h"
#include "wtime.c"

const uint width = 640, height = 480;

// view
int ox, oy;
int buttonState = 0;
float camera_trans[] = {0, 0, -3};
float camera_rot[]   = {0, 0, 0};
float camera_trans_lag[] = {0, 0, -3};
float camera_rot_lag[] = {0, 0, 0};
const float inertia = 0.1f;

EngineSystem *esystem = 0;
Renderer *renderer = 0;
float modelView[16];

static int fpsCount = 0;
static int fpsLimit = 1;
unsigned int frameCount = 0;
StopWatchInterface *timer = NULL;

float **clusters;
float **objects;
int *membership;
int numPoints;
int numClusters;
float threshold;
int loop_iterations;


extern "C" void cudaGLInit(int argc, char **argv);
extern "C" void copyArrayFromDevice(void *host, const void *device, unsigned int vbo, int size);

void reset();


void cleanup()
{
    sdkDeleteTimer(&timer);

	delete esystem;

    cudaDeviceReset();
    return;
}

void initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("KMeans");

    glewInit();

    if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object"))
    {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(EXIT_FAILURE);
    }

#if defined (WIN32)

    if (wglewIsSupported("WGL_EXT_swap_control"))
    {
        // disable vertical sync
        wglSwapIntervalEXT(0);
    }

#endif

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.25, 0.25, 0.25, 1.0);

    glutReportErrors();
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "KMeans : %3.1f fps", ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(ifps, 1.f);
        sdkResetTimer(&timer);
    }
}

void render() {
	esystem->update(0.0f);
	renderer->setVertexBuffer(esystem->getCurrentReadBuffer(), numPoints);
	renderer->display();
}


void display()
{
	sdkStartTimer(&timer);

	// render
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// view transform
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	for (int c = 0; c < 3; ++c)
	{
		camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
		camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
	}

	glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
	glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
	glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

	// cube
	glColor3f(1.0, 1.0, 1.0);
	glutWireCube(2.0);


	render();


	sdkStopTimer(&timer);

	glutSwapBuffers();
	glutReportErrors();

	computeFPS();
}

void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);

    renderer->setWindowSize(w, h);
    renderer->setFOV(60.0);
}

void mouse(int button, int state, int x, int y)
{
    int mods;

    if (state == GLUT_DOWN)
    {
        buttonState |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    mods = glutGetModifiers();

    if (mods & GLUT_ACTIVE_SHIFT)
    {
        buttonState = 2;
    }
    else if (mods & GLUT_ACTIVE_CTRL)
    {
        buttonState = 3;
    }

    ox = x;
    oy = y;

    glutPostRedisplay();
}


void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 3)
	{
		// left+middle = zoom
		camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
	}
	else if (buttonState & 2)
	{
		// middle = translate
		camera_trans[0] += dx / 100.0f;
		camera_trans[1] -= dy / 100.0f;
	}
	else if (buttonState & 1)
	{
		// left = rotate
		camera_rot[0] += dy / 5.0f;
		camera_rot[1] += dx / 5.0f;
	}

    ox = x;
    oy = y;

    glutPostRedisplay();
}

void key(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case '\033':
        case 'q':
            #if defined(__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
                break;

        case '1':
        	reset();
        	break;
    }

    glutPostRedisplay();
}

void mainMenu(int i)
{
    key((unsigned char) i, 0, 0);
}

void initMenus()
{
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Quit (esc)", '\033');
   // glutAddMenuEntry("Reset random [1]", '1');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void usage() {
	printf("USAGE\n");
	printf("./programName numPoints numClusters");
}

void reset()
{
	float size = 2.0f;
	for(int i = 0; i < numPoints; ++i) {
		//objects[i] = (float*) malloc(numCoords * sizeof(float));
		objects[i][0] = rand()/(float)RAND_MAX * 4 - size;
		objects[i][1] = rand()/(float)RAND_MAX * 4 - size;
		objects[i][2] = rand()/(float)RAND_MAX * 4 - size;
	}

	double startTime = wtime();

	clusters = kmeans(objects, numPoints, numClusters, threshold, membership, &loop_iterations);
	double finishTime = wtime();
	double duration = finishTime - startTime;
	printf("duration:%f \n", duration);
	printf("iterations:%d \n", loop_iterations);

	esystem->reset();
}

int main(int argc, char **argv) {

	if(argc == 3) {
	   numPoints = atoi(argv[1]);
	   numClusters = atoi(argv[2]);
	   assert(numPoints > numClusters);
	} else {
		usage();
		return(0);
	}

	threshold = 0.0f;

	initGL(&argc, argv);
	cudaGLInit(argc, argv);

	printf("numPoints:%d \n", numPoints);
	printf("numClusters:%d \n", numClusters);
	printf("threshold:%.2f \n\n", threshold);

	objects  = (float**)malloc(numPoints * sizeof(float*));
	assert(objects != NULL);

	objects[0] = (float*) malloc(numPoints * 3 * sizeof(float));
	for (int i=1; i< numPoints; i++)
	{
		objects[i] = objects[i-1] + 3;
	}

	srand(100000);

	float size = 2.0f;
	for(int i = 0; i < numPoints; ++i) {
		//objects[i] = (float*) malloc(numCoords * sizeof(float));
		objects[i][0] = rand()/(float)RAND_MAX * 4 - size;
		objects[i][1] = rand()/(float)RAND_MAX * 4 - size;
		objects[i][2] = rand()/(float)RAND_MAX * 4 - size;
	}

    membership = (int*) malloc(numPoints * sizeof(int));
    assert(membership != NULL);

    double startTime = wtime();

    clusters = kmeans(objects, numPoints, numClusters, threshold, membership, &loop_iterations);
    double finishTime = wtime();
    double duration = finishTime - startTime;
    printf("duration:%f \n", duration);
    printf("iterations:%d \n", loop_iterations);



    esystem = new EngineSystem(numPoints, numClusters, objects, membership);
    esystem->reset();

    renderer = new Renderer;
    renderer->setColorBuffer(esystem->getColorBuffer());
    renderer->setParticleRadius(0.01f);

    sdkCreateTimer(&timer);

    initMenus();
    glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(key);
	glutCloseFunc(cleanup);
	glutMainLoop();

	free(objects[0]);
	free(objects);

    free(membership);
    free(clusters[0]);
    free(clusters);

    return(0);
}

