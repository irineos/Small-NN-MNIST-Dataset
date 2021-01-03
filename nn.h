#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include<time.h>
#include<string.h>

#define LENGTH(x)  (sizeof(x) / sizeof((x)[0]))

//network structure//

typedef struct Node{
	double value;
	double *weight;
	double bias;
	double error;
	double dvalue;
}Node;

typedef struct Layer{
	Node *node;
	char *activationFunc;
}Layer;

typedef struct Network{
	int *networkLayerSizes;
	int numberOfLayers;
	int inputSize;
	int outputSize;
	Layer *layer;
}Network;


//activation functions and derivatives//

double sigmoid(double x){
	return 1.0f / (1 + expf(-x));
}
double dsigmoid(double x){
	return sigmoid(x) * (1-sigmoid(x));
}


double tanH(double x)
{
  return ((2 / (1 + exp(-2*x)))-1);
}
double dtanH(double x)
{
  return 1-pow(tanH(x),2);
}


double ReLu(double x){
	if(x>=0)
		return x;
	else
		return 0;
}
double dReLu(double x){
	if(x>=0)
		return 1;
	else
		return 0;
}


void initNetwork(Network *net,int networkSize, int* nlayerSizes, char* activationFunc[]){
	
	//init network
	net->networkLayerSizes = nlayerSizes;
	net->inputSize = net->networkLayerSizes[0];
	net->numberOfLayers = networkSize;
	net->outputSize = net->networkLayerSizes[net->numberOfLayers-1];

	//init layer array
	net->layer = (Layer*)malloc(net->numberOfLayers*sizeof(Layer));

	//init nodes arrays
	for(int i=0;i<net->numberOfLayers;i++){
		net->layer[i].node = (Node*)malloc(net->networkLayerSizes[i]*sizeof(Node));

		if(i>0)
			net->layer[i].activationFunc = activationFunc[i-1];
		//init weights arrays
		for(int j=0;j<net->networkLayerSizes[i];j++){
			if(i>0){
				net->layer[i].node[j].weight = (double*)malloc(net->networkLayerSizes[i-1]*sizeof(double));
			}
		}
	}
}
//random double generator
double randomD(double min, double max){
	double scale = rand()/(double) RAND_MAX;
	return min + scale*(max-min);
}

int max(int arr[], int n) 
{ 
    int i; 
    int max = arr[0]; 
  
    for (i = 1; i < n; i++) 
        if (arr[i] > max) 
            max = arr[i]; 
  
    return max; 
} 
// init weights arrays with random values
void randomizeWeights(Network *net){
	srand(time(NULL));
	for(int layer=0;layer<net->numberOfLayers;layer++){
		for(int node=0;node<net->networkLayerSizes[layer];node++){
			net->layer[layer].node[node].bias = randomD(-1.0,1.0);
			if(layer>0){
				for(int i=0;i<net->networkLayerSizes[layer-1];i++){
					net->layer[layer].node[node].weight[i] = randomD(-1.0,1.0);
				}
			}	
		}
	}
}


void forwardPass(Network *net, double *input){
	for(int i=0;i<net->networkLayerSizes[0];i++){
		net->layer[0].node[i].value = input[i];
		//printf("%f\n",net->layer[0].node[i].value);
	}
	for(int layer=1;layer<net->numberOfLayers;layer++){
		for(int node=0;node<net->networkLayerSizes[layer];node++){

			double x = net->layer[layer].node[node].bias;
			for(int prevNode=0;prevNode<net->networkLayerSizes[layer-1];prevNode++){
				x += net->layer[layer-1].node[prevNode].value * net->layer[layer].node[node].weight[prevNode];
			}

			//strcmp(s1,s2) return 0 if s1 identical with s2
			if(!strcmp(net->layer[layer].activationFunc,"sigmoid")){
				net->layer[layer].node[node].value = sigmoid(x);
				net->layer[layer].node[node].dvalue = dsigmoid(x);
			}
			else if(!strcmp(net->layer[layer].activationFunc,"tanh")){
				net->layer[layer].node[node].value = tanH(x);
				net->layer[layer].node[node].dvalue = dtanH(x);
			}
			else if(!strcmp(net->layer[layer].activationFunc,"ReLu")){
				net->layer[layer].node[node].value = ReLu(x);
				net->layer[layer].node[node].dvalue = dReLu(x);
			}
			else{
				printf("Error in activation function!!");
			}
		}
	}
	
}

void backProp(Network *net, double *target){
	
	for(int node=0;node<net->networkLayerSizes[net->numberOfLayers-1];node++){
		net->layer[net->numberOfLayers-1].node[node].error = (net->layer[net->numberOfLayers-1].node[node].value - target[node]) * net->layer[net->numberOfLayers-1].node[node].dvalue;
	}
	for(int layer=net->numberOfLayers-2;layer>0;layer--){
		for(int node=0;node<net->networkLayerSizes[layer];node++){
			double sum = 0;
			for(int nextNode=0;nextNode<net->networkLayerSizes[layer+1];nextNode++){
				sum += net->layer[layer+1].node[nextNode].weight[node] * net->layer[layer+1].node[nextNode].error;
			}
			net->layer[layer].node[node].error = sum * net->layer[layer].node[node].dvalue;
		}
	}
}

void updateWeights(Network *net,double lr){
	for(int layer=1;layer<net->numberOfLayers;layer++){
		for(int node=0;node<net->networkLayerSizes[layer];node++){
			//calculate once
			double deltaTemp = -lr * net->layer[layer].node[node].error;

			for(int prevNode=0;prevNode<net->networkLayerSizes[layer-1];prevNode++){
				double delta = net->layer[layer-1].node[prevNode].value * deltaTemp;
				net->layer[layer].node[node].weight[prevNode] += delta;
			}
			double delta =  deltaTemp;
			net->layer[layer].node[node].bias = delta;
		}
	}

}
//mean squared error
double mse(Network n, double target[]){
	double val = 0;
	for(int i=0;i<n.outputSize;i++){
		val += pow((target[i] - n.layer[n.numberOfLayers-1].node[i].value),2);
	}
	return val/n.outputSize;
}




/*
void train(Network *net,double *input, double *target,int epochs, double learningRate){
	if(epochs < 1 ){
		printf("epochs must be >0\n");
		exit(0);
	}
	for(int i=1;i<=epochs;i++){
		printf("---------------> epoch %d <---------------\n",i);
		forwardPass(net,input);
		backProp(net,target);
		updateWeights(net,learningRate);

		double mserror = mse(*net,target);
		printf("%f\n",mserror);
	}
}
void fit(Network *net, double input[][net->inputSize], double target[][net->outputSize] ,int epochs, int loops, double learningRate){
	if(epochs < 1 ){
		printf("epochs must be >0\n");
		exit(0);
	}
	for(int i=1;i<=epochs;i++){
		printf("---------------> epoch %d <---------------\n",i);
		for(int j=0;j<loops;j++){
			forwardPass(net,input[j]);
			backProp(net,target[j]);
			updateWeights(net,learningRate);

			double mserror = mse(*net,target[j]);
			printf("%f\n",mserror);
		}
	}
}
*/


void saveModel(Network *net,char *filename){
	printf("Saving model...\n");

	FILE *f = fopen(filename, "wb");
	//write number of layers
	fprintf(f,"%d\n\n",net->numberOfLayers);
	for(int i=0;i<net->numberOfLayers;i++){
		//write number of nodes in each layer
		fprintf(f,"%d",net->networkLayerSizes[i]);
		if(i<net->numberOfLayers-1)
			fprintf(f," ");
	}
	fprintf(f,"\n\n");

	for(int layer=1;layer<net->numberOfLayers;layer++){
		fprintf(f,"%s",net->layer[layer].activationFunc);
		if(layer<net->numberOfLayers-1)
			fprintf(f,"\n");
	}
	
	for(int layer=0;layer<net->numberOfLayers;layer++){
			fprintf(f,"\n");
		for(int node=0;node<net->networkLayerSizes[layer];node++){
			if(layer>0){
				//fprintf(f,"[");
				for(int i=0;i<net->networkLayerSizes[layer-1];i++){
					fprintf(f,"%lf ",net->layer[layer].node[node].weight[i]);
				}	
				fprintf(f,"%lf",net->layer[layer].node[node].bias);
				//fprintf(f,"]");
				fprintf(f,"\n");
			}
		}
		
	}
	
	fclose(f);
	

	printf("Model saved!!\n");
}



void loadModel(Network *n, char *filename){
	printf("Loading model...\n");

	//Network n;
	FILE* filePointer;
	int bufferLength = 1000;
	char buffer[bufferLength];

	filePointer = fopen(filename, "r");
	//read first line
	fgets(buffer, bufferLength, filePointer);
	n->numberOfLayers = atoi(buffer);
	//init array -> number of nodes in each layer
	n->networkLayerSizes = (int*)malloc(n->numberOfLayers*sizeof(int));
	//skip blank line
	fgets(buffer, bufferLength, filePointer);
	//read net->networklayersizes
	fgets(buffer, bufferLength, filePointer);

	//split line into array of ints to fill net->networkLayerSizes
	char *token = strtok(buffer, " "); 
	n->networkLayerSizes[0]=atoi(token);
	int i=1;
	while (token != NULL) { 
		token = strtok(NULL, " "); 
		if(token !=NULL){
			n->networkLayerSizes[i]=atoi(token);
			i=i+1;
		}
	} 
	
	//skip blank line
	fgets(buffer, bufferLength, filePointer);

	//input layer doesn't need activation function
	char* activationFunc[n->numberOfLayers-1];
	//number of chars in activation funcs strings ("sigmoid" has 7 chars)
	for(int i=0;i<n->numberOfLayers-1;i++){
		activationFunc[i] = (char*)malloc(50*sizeof(char));
	}
	//read function of each layer
	for(int i=0;i<n->numberOfLayers-1;i++){
		fscanf(filePointer,"%s",activationFunc[i]);
	}
	
	//init loaded network
	initNetwork(n,n->numberOfLayers,n->networkLayerSizes,activationFunc);

	//read weights
	for(int layer=0;layer<n->numberOfLayers;layer++){
		for(int node=0;node<n->networkLayerSizes[layer];node++){
			if(layer>0){
				for(int i=0;i<n->networkLayerSizes[layer-1];i++){
					fscanf(filePointer,"%lf ",&n->layer[layer].node[node].weight[i]);
				}	
				fscanf(filePointer,"%lf",&n->layer[layer].node[node].bias);
			}
		}
		
	}

	fclose(filePointer);

}
