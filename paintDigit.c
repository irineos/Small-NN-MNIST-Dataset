#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/freeglut.h>

#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

#define MIN(a,b) (((a)<(b))?(a):(b))

char networkText[40];

const int WIDTH = 640;
const int HEIGHT = 336;

Network net;
double testInput[28*28];

typedef struct Rect {
	int x1, x2, y1, y2;
	float colour;
}Rect;

Rect pixels[28][28];
Rect predictButton;
Rect clearButton;

int prediction = 0;

void drawText(float x,float y,float z,char* string,float c){
	glColor3f(c,c,c);
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
	
	predictButton.x1 = 370;
	predictButton.y1 = 250;
	predictButton.x2 = 470;
	predictButton.y2 = 280;
	predictButton.colour = 0.8;
	

	clearButton.x1 = 500;
	clearButton.y1 = 250;
	clearButton.x2 = 600;
	clearButton.y2 = 280;
	clearButton.colour = 0.8;
}


void showPixels() {
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

void paint(int x, int y) {
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			if (x >= pixels[i][j].x1 && x <= pixels[i][j].x2 && y >= pixels[i][j].y1 && y <= pixels[i][j].y2) {
				pixels[i][j].colour = MIN(pixels[i][j].colour + 0.6, 1);
				if(i>0 && i<27 && j>0 && j<27){
					for (int a = i - 1; a <= i + 1; a++) {
						for (int b = j - 1; b <= j + 1; b++) {
							if (a != i || b != j) {
								pixels[a][b].colour = MIN(pixels[a][b].colour + 0.07, 1);
							}
						}
					}
				}
			}
		}
	}
}

void drawButtons(){
	glBegin(GL_POLYGON);
		glColor3f(predictButton.colour,predictButton.colour,predictButton.colour);
		glVertex2i(predictButton.x1, predictButton.y1);
		glVertex2i(predictButton.x1, predictButton.y2);
		glVertex2i(predictButton.x2, predictButton.y2);
		glVertex2i(predictButton.x2, predictButton.y1);
	glEnd();

	glBegin(GL_POLYGON);
		glColor3f(clearButton.colour,clearButton.colour,clearButton.colour);
		glVertex2i(clearButton.x1, clearButton.y1);
		glVertex2i(clearButton.x1, clearButton.y2);
		glVertex2i(clearButton.x2, clearButton.y2);
		glVertex2i(clearButton.x2, clearButton.y1);
	glEnd();

}

void paintFunc(int x, int y) {	
	paint(x, y);
	glutPostRedisplay();	
}

void normal_keys(unsigned char key, int x, int y) {
   if (key==27) exit(0);
 }

 void mouseClick(int button, int state, int x, int y) {
	if(state == GLUT_DOWN) {
        if (x >= predictButton.x1 && x <= predictButton.x2 && y >= predictButton.y1 && y <= predictButton.y2) {
			predictButton.colour = 0.4;
			glutPostRedisplay();
			for (int i = 0; i < 28; i++) {
				for (int j = 0; j < 28; j++) {
					testInput[i * 28 + j] = (double) pixels[i][j].colour;
				}
			}
			forwardPass(&net,testInput);	
			prediction = getOutput(net.layer[net.numberOfLayers-1].node);
			printf("\nNetwork says it's a: %d\n",prediction);
		}
		else if (x >= clearButton.x1 && x <= clearButton.x2 && y >= clearButton.y1 && y <= clearButton.y2) {
			initGrid();
			prediction=0;
			clearButton.colour = 0.4;
			glutPostRedisplay();
		}
		
	}
	if(state == GLUT_UP) {
    	predictButton.colour = 0.8;
		clearButton.colour = 0.8;
        glutPostRedisplay();
    }
 }

 void display() {
	glClear(GL_COLOR_BUFFER_BIT);

	showPixels();
	drawButtons();

	sprintf(networkText,"Prediction: %d", prediction);
	drawText(420,150,0.0f,networkText,1);

	drawText(385,272,0.0f,"Predict",0);
	drawText(525,272,0.0f,"Clear",0);

	glFlush();
}

int main(int argc, char** argv){
	char* filename = (char*)"mnist_model.txt";
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
	glutMotionFunc(paintFunc);
	glutKeyboardFunc(normal_keys);
	glutMainLoop();
}