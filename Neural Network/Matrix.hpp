#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <functional>


namespace sp
{
    template<typename T>
    class Matrix2D
    {
    public:
        uint32_t _cols;
        uint32_t _rows;
        std::vector<std::vector<T>> _vals;


    public:
        Matrix2D(uint32_t rows, uint32_t cols)
            :_rows(rows),
            _cols(cols),
            _vals({})
        {
            this->_vals = std::vector<std::vector<T>>(rows, std::vector<T>(cols));
        }

        Matrix2D()
             : _cols(0),
            _rows(0),
            _vals({})
        {
        }

        bool isSquare()
        {
            return _rows == _cols;
        }


        Matrix2D negetive()
        {
            Matrix2D output(_rows, _cols);
            for (uint32_t i = 0; i < output._rows; i++)
                for (uint32_t j = 0; j < output._cols; j++)
                {
                    output._vals[i][j] = -this->_vals[i][j];
                }
            return output;
        }


        Matrix2D multiply(Matrix2D& target)
        {
            assert(_cols == target._rows);
            Matrix2D output(_rows, target._cols);
            for (uint32_t i = 0; i < output._rows; i++)
                for (uint32_t j = 0; j < output._cols; j++)
                {
                    T result = T();
                    for (uint32_t k = 0; k < _cols; k++)
                        result += this->_vals[i][k] * target._vals[k][j];
                    output._vals[i][j] = result;
                }
            return output;
        }

        Matrix2D multiplyElements(Matrix2D& target)
        {
            assert(_rows == target._rows && _cols == target._cols);
            Matrix2D output(_rows, _cols);
            for (uint32_t i = 0; i < output._rows; i++)
                for (uint32_t j = 0; j < output._cols; j++)
                {
                    
                    output._vals[i][j] = this->_vals[i][j] * target._vals[i][j];
                }
            return output;
        }


        Matrix2D add(Matrix2D& target)
        {
            assert(_rows == target._rows && _cols == target._cols);
            Matrix2D output(_rows, _cols);
            for (uint32_t i = 0; i < output._rows; i++)
                for (uint32_t j = 0; j < output._cols; j++)
                {
                    output._vals[i][j] = this->_vals[i][j] + target._vals[i][j];
                }
            return output;
        }
        Matrix2D applyFunction(std::function<T(const T&)> func)
        {
            Matrix2D output(_rows, _cols);
            for (uint32_t i = 0; i < output._rows; i++)
                for (uint32_t j = 0; j < output._cols; j++)
                {
                    output._vals[i][j] = func(this->_vals[i][j]);
                }
            return output;
        }

        Matrix2D multiplyScaler(float s)
        {
            Matrix2D output(_rows, _cols);
            for (uint32_t i = 0; i < output._rows; i++)
                for (uint32_t j = 0; j < output._cols; j++)
                {
                    output._vals[i][j] = this->_vals[i][j] * s;
                }
            return output;

        }

        Matrix2D addScaler(float s)
        {
            Matrix2D output(_rows, _cols);
            for (uint32_t i = 0; i < output._rows; i++)
                for (uint32_t j = 0; j < output._cols; j++)
                {
                    output._vals[i][j] = this->vals[i][j] + s;
                }
            return output;

        }
        Matrix2D transpose()
        {
            Matrix2D output(_cols, _rows);
            for (uint32_t i = 0; i < _rows; i++)
                for (uint32_t j = 0; j < _cols; j++)
                {
                    output._vals[j][i] = this->_vals[i][j];
                }
            return output;
        }

//        Matrix2D cofactor(uint32_t col, uint32_t row)
//        {
//            Matrix2D output(_rows - 1, _cols - 1);
//            uint32_t i = 0;
//            for (uint32_t y = 0; y < _rows; y++)
//                for (uint32_t x = 0; x < _cols; x++)
//                {
//                    if (x == col || y == row) continue;
//                    output._vals[i++] = at(x, y);
//                }
//
//            return output;
//        }

//        T determinant()
//        {
//            assert(_rows == _cols);
//            T output = T();
//            if (_rows == 1)
//            {
//                return _vals[0];
//            }
//            else
//            {
//                int32_t sign = 1;
//                for (uint32_t x = 0; x < _cols; x++)
//                {
//                    output += sign * at(x, 0) * cofactor(x, 0).determinant();
//                    sign *= -1;
//                }
//            }
//
//            return output;
//        }

//        Matrix2D adjoint()
//        {
//            assert(_rows == _cols);
//            Matrix2D output(_rows, _cols);
//            int32_t sign = 1;
//            for (uint32_t y = 0; y < _rows; y++)
//                for (uint32_t x = 0; x < _cols; x++)
//                {
//                    output.at(x, y) = sign * cofactor(x, y).determinant();
//                    sign *= -1;
//                }
//            output = output.transpose();
//
//            return output;
//        }

//        Matrix2D inverse()
//        {
//            Matrix2D adj = adjoint();
//            T factor = determinant();
//            for (uint32_t y = 0; y < adj._cols; y++)
//                for (uint32_t x = 0; x < adj._rows; x++)
//                {
//                    adj.at(x, y) = adj.at(x, y) / factor;
//                }
//            return adj;
//        }



    }; // class Matrix2D

    template<typename T>
    void LogMatrix2D(Matrix2D<T>& mat)
    {
        for (uint32_t y = 0; y < mat._rows; y++)
        {
            for (uint32_t x = 0; x < mat._cols; x++)
                std::cout << std::setw(10) << mat.at(x, y) << " ";
            std::cout << std::endl;
        }
    }

}
