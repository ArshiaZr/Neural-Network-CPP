#pragma once
#include "matrix.hpp"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <unordered_map>
#include <functional>

#include "utils.hpp"
#include "activitions.hpp"


std::unordered_map<std::string, std::pair<std::function<float(const float&)>, std::function<float(const float&)>>> activations = {
    {"sigmoid", {Sigmoid, dSigmoid}},
    {"relu", {Relu, dRelu}}
    
};


namespace sp
{
    // calss representing a simple densely connected neural network
    // i.e. every neuron  is connected to every neuron of next layer
    //std::unordered_map<std::string, std::any> activations;
    class NeuralNetwork{
    public:
        std::vector<uint32_t> _topology;
        std::vector<Matrix2D<float>> _weightMatrices;
        std::vector<Matrix2D<float>> _valueMatrices;
        std::vector<Matrix2D<float>> _biasMatrices;
        std::unordered_map<uint32_t, std::pair<std::function<float(const float&)>, std::function<float(const float&)>>> activation_map;
        float _learningRate;
        float _weightUpperBound, _weightLowerBound;
        float _biasUpperBound, _biasLowerBound;
    public:
        
        // topology defines the no.of neurons for each layer
        // learning rate defines how much modification should be done in each backwords propagation i.e. training
        NeuralNetwork(std::vector<uint32_t> topology, std::vector<std::string> activationMethods, float learningRate = 0.1f, float weightUpperBound = 1, float weightLowerBound = 0, float biasUpperBound = 1, float biasLowerBound = 0)
            :_topology(topology),
            _weightMatrices({}),
            _valueMatrices({}),
            _biasMatrices({}),
            _learningRate(learningRate){
            //Map of activation functions
            assert(activationMethods.size() == topology.size() - 1);
            for(int i = 0; i < activationMethods.size(); i++){
                if(activations.find(activationMethods[i]) != activations.end()){
                    
                    activation_map[i] = {activations[activationMethods[i]].first, activations[activationMethods[i]].second};
                }
            }
            // initializing weight and bias matrices with random weights
            for(uint32_t i = 0; i < topology.size() - 1; i++){
                Matrix2D<float> weightMatrix(topology[i], topology[i + 1]);
                weightMatrix = weightMatrix.applyFunction([this](const float &val){
                    return getRandomBetween(1, -1);
                });
                _weightMatrices.push_back(weightMatrix);
                
                Matrix2D<float> biasMatrix(1, topology[i + 1]);
                biasMatrix = biasMatrix.applyFunction([this](const float &val){
                    return getRandomBetween(1, -1);
                });
                _biasMatrices.push_back(biasMatrix);

            }
            _valueMatrices.resize(topology.size());
            
        }

        // function to generate output from given input vector
        bool feedForword(std::vector<float> input){
            if(input.size() != _topology[0])
                return false;
            // creating input matrix
            Matrix2D<float> values(1, (uint32_t)input.size());
            for(uint32_t i = 0; i < input.size(); i++)
                values._vals[0][i] = input[i];
            
            //forwording inputs to next layers
            for(uint32_t i = 0; i < _weightMatrices.size(); i++){
                // y = activationFunc( x1 * w1 + x2 * w2 + ... + b)
                _valueMatrices[i] = values;
                values = values.multiply(_weightMatrices[i]);
                values = values.add(_biasMatrices[i]);
                if(activation_map.find(i) != activation_map.end()){
                    values = values.applyFunction(activation_map[i].first);
                }
                
            }
            _valueMatrices[_weightMatrices.size()] = values;
            return true;
        }


        // function to train with given output vector
        bool backPropagate(std::vector<float> targetOutput){
            if(targetOutput.size() != _topology.back())
                return false;

            // determine the simple error
            // error = target - output
            Matrix2D<float> errors(1, (uint32_t)targetOutput.size());
            for(uint32_t i = 0; i < targetOutput.size(); i++)
                errors._vals[0][i] = targetOutput[i];
            
            Matrix2D<float> neg = _valueMatrices.back().negetive();
            errors = errors.add(neg);
            

            // back propagating the error from output layer to input layer
            // and adjusting weights of weight matrices and bias matrics
            for(int32_t i = (int32_t)_weightMatrices.size() - 1; i >= 0; i--){
                //calculating errrors for previous layer
                Matrix2D<float> trans =_weightMatrices[i].transpose();
                Matrix2D<float> prevErrors = errors.multiply(trans);

                //calculating gradient i.e. delta weight (dw)
                //dw = lr * error * d/dx(activated value)
                Matrix2D<float> dOutputs = _valueMatrices[i + 1];
                if(activation_map.find(i) != activation_map.end()){
                    dOutputs = _valueMatrices[i + 1].applyFunction(activation_map[i].second);
                }
                 
                Matrix2D<float> gradients = errors.multiplyElements(dOutputs);
                gradients = gradients.multiplyScaler(_learningRate);
                Matrix2D<float> weightGradients = _valueMatrices[i].transpose().multiply(gradients);
                
                //adjusting bias and weight
                _biasMatrices[i] = _biasMatrices[i].add(gradients);
                _weightMatrices[i] = _weightMatrices[i].add(weightGradients);
                errors = prevErrors;
            }
            
            return true;
        }
        
        // function to retrive final output
        std::vector<float> getPredictions(){
            return _valueMatrices.back()._vals[0];
        }
        
        float getTotalError(const std::vector<float>& outputs){
            const std::vector<float> predictions = this->getPredictions();
            
            float total_error = 0.0f;
            for(int i = 0; i < predictions.size(); i++){
                total_error += std::abs(outputs[i] - predictions[i]);
            }
            return total_error;
        }
        
        const void printOutputPredictions(const std::vector<float>& outputs){
            std::vector<float> predictions = this->getPredictions();
            assert(outputs.size() == predictions.size());
            for(int i = 0; i < predictions.size(); i++){
                std::cout << "Output/Prediction " << i + 1 << ": " << predictions[i] << ", " << outputs[i] << "\n";
                if(i != predictions.size() - 1){
                    std::cout << ", ";
                }
            }
        }
        
        void train(const std::vector<std::vector<float>>& train_inputs, const std::vector<std::vector<float>>& train_outputs, uint32_t batch, bool verbos = false, bool show_progress = false){
            assert(train_inputs.size() == train_outputs.size() && train_inputs.size() >= batch);
            std::cout << "Training started:" << "\n";
            uint32_t progress = train_inputs.size() / 20;
            for(uint32_t i = 0; i < train_inputs.size(); i++){
                this->feedForword(train_inputs[i]);
                if(verbos && i % batch == 0){
                    this->printOutputPredictions(train_outputs[i]);
                    float total_errors = this->getTotalError(train_outputs[i]);
                    std::cout << "Total Error: " << total_errors<< "\n" << "\n";
                    if(show_progress)
                        std::cout << "Progress: %" << i / (float)train_inputs.size() * 100 << "\n";
                }else if(show_progress && i % progress == 0){
                    std::cout << "Progress: %" << i / (float)train_inputs.size() * 100 << "\n";
                }
                this->backPropagate(train_outputs[i]);
            }
            std::cout << "Training finished" << "\n";
        }
        
        void test(const std::vector<std::vector<float>>& target_inputs, std::vector<std::vector<float>>& target_outputs){
            assert(target_inputs.size() == target_outputs.size());
            std::cout << "Testing started:" << "\n";
            float total_errors = 0;
            for(uint32_t i = 0; i < target_inputs.size(); i++){
                std::cout << "Case " << i+1 << ":\n";
                this->feedForword(target_inputs[i]);
                std::vector<float> predictions = this->getPredictions();
                total_errors += this->getTotalError(target_outputs[i]);
                this->printOutputPredictions(target_outputs[i]);
                std::cout << "\n";
            }
            std::cout << "Testing finished!" << "\n";
            std::cout << "Error Average: %" << (total_errors / (float)target_inputs.size()) * 100 << "\n";
        }
    }; // class NeuralNetwork


}
