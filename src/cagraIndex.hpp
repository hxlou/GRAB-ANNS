#include <iostream>
#include <vector>

#include "./index/cagra.cuh"

class CagraIndex {
public:
    CagraIndex() {
        // Constructor implementation
    }

    void buildIndex(const std::vector<int>& data) {
        indexData = data;
    }

    void queryIndex(int value) {
        for (const auto& item : indexData) {
            if (item == value) {
                std::cout << "Found: " << value << std::endl;
                return;
            }
        }
        std::cout << "Not Found: " << value << std::endl;
    }

private:
    std::vector<int> indexData;

    // TODO
    // auto indexCagra = cagra::cagra();
};