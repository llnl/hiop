#ifndef HIOP_STD_UTILS
#define HIOP_STD_UTILS

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <functional>
#include <vector>
#include <list>
#include <numeric>
#include <cassert>
#include <math.h>
#include <chrono>

namespace hiop
{
template<class T>
inline void printvec(const std::vector<T>& v, const std::string& msg = "")
{
  std::cout.precision(6);
  std::cout << msg << " size:" << v.size() << std::endl;
  std::cout << std::scientific;
  typename std::vector<T>::const_iterator it = v.begin();
  for(; it != v.end(); ++it) std::cout << (*it) << " ";
  std::cout << std::endl;
}

template<class T>
inline void printlist(const std::list<T>& v, const std::string& msg = "")
{
  std::cout.precision(6);
  std::cout << msg << " size:" << v.size() << std::endl;
  std::cout << std::scientific;
  typename std::list<T>::const_iterator it = v.begin();
  for(; it != v.end(); ++it) std::cout << (*it) << " ";
  std::cout << std::endl;
}

template<class T>
inline void printvecvec(const std::vector<std::vector<T> >& v, const std::string& msg = "")
{
  std::cout.precision(6);
  std::cout << msg << " size:" << v.size() << std::endl;
  std::cout << std::scientific;
  for(auto& l: v) {
    for(auto& c: l) std::cout << c << " ";
    std::cout << std::endl;
  }
}
template<class T>
inline void hardclear(std::vector<T>& in)
{
  std::vector<T>().swap(in);
}

static inline std::string tolower(const std::string& str_in)
{
  auto str_out = str_in;
  std::transform(str_out.begin(), str_out.end(), str_out.begin(), ::tolower);
  return str_out;
}

static inline void tolower(std::string& str_in) { std::transform(str_in.begin(), str_in.end(), str_in.begin(), ::tolower); }

static inline std::string toupper(const std::string& str_in)
{
  auto str_out = str_in;
  std::transform(str_out.begin(), str_out.end(), str_out.begin(), ::toupper);
  return str_out;
}
static inline void toupper(std::string& str_in) { std::transform(str_in.begin(), str_in.end(), str_in.begin(), ::toupper); }

// Function to reorder elements of arr[] according to index[]
template<class T>
inline void reorder(T* arr, std::vector<int> index, int n)
{
  T temp[n];

  // arr[i] should be present at index[i] index
  for(int i = 0; i < n; i++) temp[i] = arr[index[i]];

  // Copy temp[] to arr[]
  for(int i = 0; i < n; i++) {
    arr[i] = temp[i];
    //       index[i] = i;
  }
}

static inline unsigned long generate_seed()
{
#ifdef NDEBUG
  return std::chrono::system_clock::now().time_since_epoch().count();
#else
  return 0;
#endif
}

}  // namespace hiop
#endif
