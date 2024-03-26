#ifndef SOLVERS_INDEX_H
#define SOLVERS_INDEX_H

#include <cassert>

namespace damotion {
namespace optimisation {

/**
 * @brief Indexing information for a given matrix, including the size of
 *
 */
class BlockIndex {
   public:
    BlockIndex() = default;
    ~BlockIndex() = default;

    BlockIndex(int i_start, int i_sz, int j_start = 0, int j_sz = 1)
        : i_start_(i_start), i_sz_(i_sz), j_start_(j_start), j_sz_(j_sz) {}

    /**
     * @brief Updates the starting row index of the block whilst retaining the
     * size of the block
     *
     * @param i_start
     */
    void UpdateRowStartIndex(const int& i_start) {
        assert(i_start >= 0 && "Index must be non-negative!");
        i_start_ = i_start;
    }

    /**
     * @brief Updates the starting column index of the block whilst retaining
     * the size of the block
     *
     * @param j_start
     */
    void UpdateColumnStartIndex(const int& j_start) {
        assert(j_start >= 0 && "Index must be non-negative!");
        j_start_ = j_start;
    }

    /**
     * @brief Updates the size of the block whilst retaining the starting index
     * of the block
     *
     * @param i_start
     * @param j_start
     */
    void UpdateBlockSize(const int& i_sz, const int& j_sz = 1) {
        assert(i_sz >= 1 && j_sz >= 1 && "Size must be greater than zero!");
        i_sz_ = i_sz;
        j_sz_ = j_sz;
    }

    /**
     * @brief Starting index of the block in the first dimension
     *
     * @return const int&
     */
    const int& i_start() const { return i_start_; }

    /**
     * @brief Size of the block in the first dimension
     *
     * @return const int&
     */
    const int& i_sz() const { return i_sz_; }

    /**
     * @brief Starting index of the block in the second dimension
     *
     * @return const int&
     */
    const int& j_start() const { return j_start_; }

    /**
     * @brief Size of the block in the second dimension
     *
     * @return const int&
     */
    const int& j_sz() const { return j_sz_; }

   private:
    // Starting index of the block in the first dimension
    int i_start_ = 0;
    // Starting index of the block in the second dimension
    int j_start_ = 0;

    // Size of the block in the first dimension
    int i_sz_ = 0;
    // Size of the block in the second dimension
    int j_sz_ = 0;
};

}  // namespace optimisation
}  // namespace damotion

#endif /* SOLVERS_INDEX_H */
