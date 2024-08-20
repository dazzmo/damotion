#ifndef CORE_PROFILER_HPP
#define CORE_PROFILER_HPP

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

#include "damotion/core/logging.hpp"

namespace damotion {

using namespace boost::accumulators;

/**
 * @brief Profiling class that enables the timing of a specific component.
 * Records the time elapsed over the lifetime of the object and stores the
 * results in a global profiler that can be evaluated using the emptry
 * constructor.
 *
 */
class Profiler {
 public:
  typedef std::chrono::steady_clock Clock;
  typedef accumulator_set<double,
                          stats<tag::mean, tag::variance, tag::min, tag::max>>
      acc_t;
  /**
   * @brief Generates a report for all profilers
   *
   */
  Profiler() {
#ifdef DAMOTION_USE_PROFILING
    LOG(INFO) << "Calls\tMean (secs)\tStdDev\tMin (sec)\tMax (secs)\n";
    for (std::map<std::string, acc_t>::iterator p = map_.begin();
         p != map_.end(); p++) {
      double av = mean(p->second);
      double stdev = sqrt(((double)variance(p->second)));
      double max = boost::accumulators::extract::max(p->second);
      double min = boost::accumulators::extract::min(p->second);
      LOG(INFO) << p->first.c_str() << '\t'
                << boost::accumulators::count(p->second) << '\t' << av << '\t'
                << stdev << '\t' << min << '\t' << max;
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

}  // namespace damotion

#endif /* CORE_PROFILER_HPP */
