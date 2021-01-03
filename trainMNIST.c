#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>



int images, labels;

double trainImage[60000][28*28];
int trainLabel[60000];

unsigned char inputPixel;
unsigned char inputLabel;


double label[10];

void int2arr(double labelArr[], int label){
	for(int i=0;i<10;i++){
		if(i==label){
			labelArr[i]=1.0;
		}
		else{
			labelArr[i]=0.0;
		}
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
	
	images=open("data/train-images.idx3-ubyte", O_RDONLY);
	labels=open("data/train-labels.idx1-ubyte", O_RDONLY);

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
	for(int image=0;image<60000;image++){
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				read(images, &inputPixel, sizeof(unsigned char));
				trainImage[image][i * 28 + j] =(double) inputPixel / 255;
			}
		}
		read(labels, &inputLabel, sizeof(unsigned char));
		trainLabel[image] = (int)inputLabel;
	}
}

int main(int argc, char** argv){
	loadMnistDataset();

	Network net;
	int nlayerSizes[] = {784,100,50,10}; 
	int networkSize = LENGTH(nlayerSizes);

	
	char* activationFunc[3] = {"sigmoid","sigmoid","sigmoid"};

	initNetwork(&net,networkSize, nlayerSizes, activationFunc);
	randomizeWeights(&net);

	double learningRate = 0.1f;
	
	for(int i=1;i<=10;i++){	
		printf("---------------> epoch %d <---------------\n",i);
		for(int images=0;images<60000;images++){
			
			
			forwardPass(&net,trainImage[images]);

			int2arr(label,trainLabel[images]);
			backProp(&net,label);

			updateWeights(&net,learningRate);

			double mserror = mse(net,label);
			//printf("%lf\n",mserror);
		}
	}
	
	saveModel(&net,"mnist.txt");
}