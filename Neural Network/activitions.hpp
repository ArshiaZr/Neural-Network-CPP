#pragma once

#include <string>
#include <cmath>


float Sigmoid(float x){
    return 1.0f / (1.0f + exp(-x));
}

float dSigmoid(float x){
    return (x * (1.0f - x));
}

float Relu(float x){
    if(x < 0){
        return 0.0f;
    }
    return x;
}

float dRelu(float x){
    if(x <= 0){
        return 0.0f;
    }
    return 1.0f;
}
