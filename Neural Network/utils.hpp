#pragma once
#include <cmath>

float getRandomBetween(float upper, float lower){
    return lower + ((float)rand() / (float)(RAND_MAX / (upper - lower)));
}

