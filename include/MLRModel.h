#ifndef MLRMODEL_H
#define MLRMODEL_H

#include <vector>
#include <Tables/Table.h>

class MLRModel {
public:
    // Function to estimate target according to features
    double estimate(const std::vector<double>& featureVals);
    void train(tables::Table& featureData, tables::Column<double>& targetData, int epochs);
    int getConstant();
    void setConstant(int constant);
    std::vector<double> getCoeffs();
    void setCoeffs(std::vector<double>);
    double getLearningRate();
    void setLearningRate(double learningRate);

private:
    double learningRate = 0.1;
    int constant;
    std::vector<double> coeffs;
};

#endif
