#include <iostream>
#include <vector>
#include <cstdio>

#include "NeuralNetwork.hpp"
#include "utils.hpp"

// A simple learning test case
// The model learns the the relation between the inputs

// Relation: (input0 || (input1 && input2))

int main()
{
    // creating neural network
    // 3 input neurons, 5 hidden neurons and 1 output neuron
    std::vector<uint32_t> topology = {3, 5, 5, 5, 1};
    std::vector<std::string> activationMethods = {"sigmoid", "sigmoid", "sigmoid", "sigmoid"};
    
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
    
    // Create training dataset
    uint32_t training_number = 1000000;
    std::vector<std::vector<float>> training_inputs;
    std::vector<std::vector<float>> training_outputs;
    for(uint32_t i = 0; i < training_number; i++){
        uint32_t index = rand() % targetInputs.size();
        training_inputs.push_back(targetInputs[index]);
        training_outputs.push_back(targetOutputs[index]);
    }
    
    nn.train(training_inputs, training_outputs, 50, true, true);
    
    
    // Create testing dataset
    uint32_t testing_number = 100;
    std::vector<std::vector<float>> testing_inputs;
    std::vector<std::vector<float>> testing_outputs;
    for(uint32_t i = 0; i < testing_number; i++){
        uint32_t index = rand() % targetInputs.size();
        testing_inputs.push_back(targetInputs[index]);
        testing_outputs.push_back(targetOutputs[index]);
    }
    nn.test(testing_inputs, testing_outputs);
    return 0;
}
