#ifndef BACKPROPAGATION_HPP
#define BACKPROPAGATION_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/random.hpp>
#include <ctime>
#include <cstddef>
#include <set>

struct backpropagation {
  typedef double value_type;
  typedef const double c_double_type;
  typedef const int c_int_type;
  typedef typename boost::numeric::ublas::matrix<value_type> matrix_type;
  typedef const matrix_type c_matrix_type;
  typedef typename boost::numeric::ublas::vector<value_type> vector_type;
  typedef const vector_type c_vector_type;

private:
  c_double_type learn_rate;
  c_double_type momentum;
  c_double_type weight_decay;
  c_double_type max_epoch;
  matrix_type weight_input;
  matrix_type weight_hidden;
  matrix_type diff_weight_input;
  matrix_type diff_weight_hidden;
  vector_type hidden;
  vector_type masked_hidden;

public:
  backpropagation(c_int_type input_type, c_int_type hidden_layer,
    c_int_type output_layer, c_double_type learn_rate, c_double_type momentum,
    c_double_type weight_decay, c_double_type max_epoch) :
    learn_rate(learn_rate), momentum(momentum), weight_decay(weight_decay), 
    max_epoch(max_epoch), hidden(hidden_layer), masked_hidden(hidden_layer) {

  }

  void do_training(const std::vector<std::pair<
  vector_type, vector_type> >& dataset) {

  }

  vector_type predict(c_vector_type& input) {

  }

private:
  vector_type forwardpropagate(c_vector_type& input, c_vector_type& mask) {

  }

  void backpropagate(c_vector_type& answer, c_vector_type& input,
    c_vector_type& output, c_vector_type& mask) {

  }

  void update_weight() {

  }

  vector_type generate_dropout_mask(c_int_type max) {

  }
};

template <typename T = double>
inline T random_number_generator(const T min = 0.0, const T max = 1.0) {
  typedef unsigned long value_type;
  static boost::mt19937 r(static_cast<value_type>(time(0)));
  boost::random::uniform_real_distribution<> range(min, max);

  return range;
}

#endif
