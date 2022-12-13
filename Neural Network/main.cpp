#include <iostream>
#include <vector>
#include <cstdio>

#include "NeuralNetwork.hpp"


int main()
{
    // creating neural network
    // 2 input neurons, 3 hidden neurons and 1 output neuron
    std::vector<uint32_t> topology = {3, 5, 1};
    std::vector<std::string> activationMethods = {"sigmoid", "sigmoid"};
    sp::NeuralNetwork nn(topology, activationMethods, 0.1);
    
    //sample dataset
    std::vector<std::vector<float>> targetInputs = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 0.0f},
        {1.0f, 1.0f, 1.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 1.0f},
        {0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, 1.0f},
    };
    std::vector<std::vector<float>> targetOutputs = {
        {0.0f},
        {1.0f},
        {1.0f},
        {1.0f},
        {0.0f},
        {1.0f},
        {0.0f},
        {1.0f}
    };

    uint32_t epoch = 500000;
    
    //training the neural network with randomized data
    std::cout << "training start\n";

    for(uint32_t i = 0; i < epoch; i++)
    {
        uint32_t index = rand() % targetInputs.size();
        nn.feedForword(targetInputs[index]);
        nn.backPropagate(targetOutputs[index]);
    }

    std::cout << "training complete\n";


    //testing the neural network
    int i = 0;
    for( std::vector<float> input : targetInputs)
    {
        nn.feedForword(input);
        std::vector<float> preds = nn.getPredictions();
        i++;
        int ans = 0;
        if(input[0] == 1.0f){
            ans = 1;
        }else if(input[1] == 1.0f && input[2] == 1){
            ans = 1;
        }
        std::cout << i << ": " << ans <<" => " << preds[0] << std::endl;
    }

    return 0;
}
