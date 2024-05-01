#ifndef COMMON_PROFILER_H
#define COMMON_PROFILER_H

#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <string>

namespace damotion {
namespace common {

using namespace boost::accumulators;

class Profiler {
 public:
  typedef std::chrono::steady_clock Clock;
  typedef accumulator_set<double,
                          stats<tag::mean, tag::variance, tag::min, tag::max>>
      acc_t;
  /**
   * @brief Generates a profile report for all timers
   *
   */
  Profiler() {
#ifdef DAMOTION_USE_PROFILING
    printf("%20s Calls\tMean (secs)\tStdDev\tMin (sec)\tMax (secs)\n", "Scope");
    for (std::map<std::string, acc_t>::iterator p = map_.begin();
         p != map_.end(); p++) {
      double av = mean(p->second);
      double stdev = sqrt(((double)variance(p->second)));
      double max = boost::accumulators::extract::max(p->second);
      double min = boost::accumulators::extract::min(p->second);
      printf("%20s %ld\t%f\t%f\t%f\t%f\n", p->first.c_str(),
             boost::accumulators::count(p->second), av, stdev, min, max);
    }
#endif
  }

  /**
   * @brief Create a new scoped profiler
   *
   * @param name
   */
  Profiler(const char* name) : name_(name) {
#ifdef DAMOTION_USE_PROFILING
    // Record start time
    start_ = Clock::now();
#endif
  }
  ~Profiler() {
#ifdef DAMOTION_USE_PROFILING
    auto dur = Clock::now() - start_;

    std::map<std::string, acc_t>::iterator p = map_.find(name_);
    if (p == map_.end()) {
      // Create new accumulator
      acc_t acc;
      std::pair<std::string, acc_t> pr(name_, acc);
      p = map_.insert(pr).first;
    }
    // TODO Check what the real time is (make it in seconds)
    (p->second)(dur.count() * 1e-9);
#endif
  }

 private:
  std::string name_;
  std::chrono::steady_clock::time_point start_;

  // Static map
  inline static std::map<std::string, acc_t> map_;
};

}  // namespace common
}  // namespace damotion

#endif /* COMMON_PROFILER_H */
