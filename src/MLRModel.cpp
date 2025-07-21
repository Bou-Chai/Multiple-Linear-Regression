#include "MLRModel.h"

    // Function to estimate target according to features
    double MLRModel::estimate(const std::vector<double>& featureVals) {
        double estimate = constant;
        for (int i = 0; i < coeffs.size(); i++) {
            estimate += coeffs[i] * featureVals[i];
        }
        return estimate;
    }

    void MLRModel::train(tables::Table& featureData, tables::Column<double>& targetData, int epochs) {
        // Initialize constant and coefficients
        constant = 0;
        for (int i = 0; i < featureData.width(); i++) {
            coeffs.push_back(0);
        }

        for (int n = 0; n < epochs; n++) {
            // *Stochastic*
            featureData.reshuffle();
            for (int i = 0; i < featureData.height(); i++) {
                double error = targetData.row(i) - estimate(featureData.getRow<double>(i));
                constant = constant - learningRate * error;
                for (int j = 0; j < coeffs.size(); j++) {
                    coeffs.at(j) = coeffs.at(j) - learningRate * featureData.at<double>(j, i) * error;
                }
            }
        }
    }
    
    int MLRModel::getConstant() {
        return constant;
    }
    
    void MLRModel::setConstant(int constant) {
        this->constant = constant;
    }
    
    std::vector<double> MLRModel::getCoeffs() {
        return coeffs;
    }
    
    void MLRModel::setCoeffs(std::vector<double>) {
        this->coeffs = coeffs;
    }
    
    double MLRModel::getLearningRate() {
        return learningRate;
    }

    void MLRModel::setLearningRate(double learningRate) {
        this->learningRate = learningRate;
    }

