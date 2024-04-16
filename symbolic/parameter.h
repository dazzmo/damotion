#ifndef SYMBOLIC_PARAMETER_H
#define SYMBOLIC_PARAMETER_H

#include <Eigen/Core>
#include <vector>

#include "symbolic/variable.h"

namespace damotion {
namespace symbolic {

class Parameter {
   public:
    // ID type for the variables
    typedef int Id;

    Parameter() = default;

    Parameter(const std::string &name, int rows, int cols = 1) : rows_(rows), cols_(cols) {
        static int next_id_ = 0;
        next_id_++;
        // Set ID for parameter
        id_ = next_id_;
        // Set name for the parameter
        name_ = name;
    }

    ~Parameter() = default;

    /**
     * @brief Number of rows for the parameter
     * 
     * @return const int& 
     */
    const int & rows() const {return rows_;}

    /**
     * @brief Number of columns for the parameter
     * 
     * @return const int& 
     */
    const int & cols() const {return cols_;}

    const Id &id() const { return id_; }
    const std::string &name() const { return name_; }

    bool operator<(const Parameter &p) const { return id() < p.id(); }
    bool operator==(const Parameter &p) const { return id() == p.id(); }

   private:
    Id id_;
    std::string name_;

    int rows_ = 0;
    int cols_ = 0;
};

typedef std::vector<Parameter> ParameterVector;

}
}  // namespace damotion

#endif /* SYMBOLIC_PARAMETER_H */
