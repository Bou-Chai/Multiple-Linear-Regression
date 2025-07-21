#include "MLRModel.h"
#include "Tables/Table.h"

int main() {
    tables::Table dataset;
    MLRModel model;

    dataset.loadCSV("../../tests/data/winequality-red.csv", ';');
    dataset.reshuffle();
    dataset.print();
    return 0;
}
