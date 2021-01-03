#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/freeglut.h>

#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>


char datasetText[40];
char networkText[40];

const int WIDTH = 640;
const int HEIGHT = 336;

Network net;

int images, labels;
unsigned char inputPixel;
unsigned char inputLabel;

typedef struct Rect {
	int x1, x2, y1, y2;
	float colour;
}Rect;

Rect pixels[28][28];

double testImage[10000][28*28];
int testLabel[10000];

int imageIndex = -1;
int prediction = 0;

void drawText(float x,float y,float z,char* string){
	glColor3f(1,1,1);
	glRasterPos3f(x,y,z);
	int len,i;
	len=(int)strlen(string);
	for(i=0;i<len;i++){
		glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24,string[i]);
	}
}

int getOutput(Node out[]){
	double max = out[0].value;
	int index = 0;
	for(int i=1;i<10;i++){
		//printf("%lf ",out[i].value);
		if(max<out[i].value){
			max = out[i].value;
			index = i;
		}
	}
	return index;
}

void openMnistDataset(){
	int skip;
	
	images = open("data/t10k-images.idx3-ubyte", O_RDONLY);
	labels = open("data/t10k-labels.idx1-ubyte", O_RDONLY);

	read(images,&skip, sizeof(int ));
	read(labels, &skip, sizeof(int ));
	read(images, &skip, sizeof(int ));
	read(labels, &skip, sizeof(int ));
	read(images, &skip, sizeof(int ));
	read(images, &skip, sizeof(int ));
}

void loadMnistDataset(){
	printf("Loading MNIST Dataset...\n");
	openMnistDataset();
	for(int image=0;image<10000;image++){
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				read(images, &inputPixel, sizeof(unsigned char));
				testImage[image][i * 28 + j] =(double) inputPixel / 255;
			}
		}
		read(labels, &inputLabel, sizeof(unsigned char));
		testLabel[image] = (int)inputLabel;
	}
}

void updatePixels(int index) {
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			pixels[i][j].colour = testImage[index][i * 28 + j];
		}
	}
}

void initGrid() {
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			pixels[i][j].x1 = j * 12;
			pixels[i][j].x2 = (j + 1) * 12;
			pixels[i][j].y1 = i * 12;
			pixels[i][j].y2 = (i + 1) * 12;
			pixels[i][j].colour = 0;
		}
	}
}

void printImage() {
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			glBegin(GL_POLYGON);
				glColor3f(pixels[i][j].colour, pixels[i][j].colour, pixels[i][j].colour);
				glVertex2i(pixels[i][j].x1, pixels[i][j].y1);
				glVertex2i(pixels[i][j].x1, pixels[i][j].y2);
				glVertex2i(pixels[i][j].x2, pixels[i][j].y2);
				glVertex2i(pixels[i][j].x2, pixels[i][j].y1);
			glEnd();
		}
	}
}

void mouseClick(int button, int state, int x, int y) {
	if(state == GLUT_DOWN) {
		if(imageIndex++ >= 10000){
			exit(0);
		}
		forwardPass(&net,testImage[imageIndex]);	
		prediction = getOutput(net.layer[net.numberOfLayers-1].node);
		printf("\nNetwork says it's a: %d\n",prediction);
		printf("Database says it's a %d\n\n", testLabel[imageIndex]);
		
		updatePixels(imageIndex);
		glutPostRedisplay();
	}
}

void normal_keys(unsigned char key, int x, int y) {
   if (key==27) exit(0);
 }

 void special_keys(int keys, int x, int y) {
    switch(keys) {
		case GLUT_KEY_LEFT:
			if(imageIndex>0){
				imageIndex--;
			}
			break;
		case GLUT_KEY_RIGHT:
			if(imageIndex++ >= 10000){
				exit(0);
			}
			break;
		default:
			break;
	}
	if(imageIndex>-1){
		updatePixels(imageIndex);

		forwardPass(&net,testImage[imageIndex]);	
		prediction = getOutput(net.layer[net.numberOfLayers-1].node);
		printf("\nNetwork says it's a: %d\n",prediction);
		printf("Database says it's a %d\n\n", testLabel[imageIndex]);
	}
	glutPostRedisplay();
 }

void display() {
	glClear(GL_COLOR_BUFFER_BIT);
	
	printImage();

	if(imageIndex <= -1){
		sprintf(datasetText,"Database says it's a 0");
	}

	sprintf(networkText,"Network: %d",prediction);
	sprintf(datasetText,"Database: %d", testLabel[imageIndex]);

	drawText(404,150,0.0f,datasetText);
	drawText(404,200,0.0f,networkText);

	glFlush();
}

int main(int argc, char** argv){
	loadMnistDataset();

	char* filename = (char*)"mnist.txt";
	loadModel(&net,filename);

	initGrid();
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutInitWindowPosition(600, 300);
	glutCreateWindow("Test MNIST Dataset");
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0f, WIDTH, HEIGHT, 0.0f, 0.0f, 1.0f);
	glutDisplayFunc(display);
	glutMouseFunc(mouseClick);
	glutKeyboardFunc(normal_keys);
    glutSpecialFunc(special_keys);
	glutMainLoop();

}