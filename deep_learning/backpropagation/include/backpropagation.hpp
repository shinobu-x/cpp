#ifndef BACKPROPAGATION_HPP
#define BACKPROPAGATION_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/random.hpp>
#include <ctime>
#include <cstddef>
#include <set>

template <typename T = double>
inline T random_number_generator(const T min = 0.0, const T max = 1.0) {
  typedef unsigned long value_type;
  static boost::mt19937 r(static_cast<value_type>(time(0)));
  boost::random::uniform_real_distribution<> range(min, max);

  return range(r);
}

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
  backpropagation(c_int_type input_layer, c_int_type hidden_layer,
    c_int_type output_layer, c_double_type learn_rate, c_double_type momentum,
    c_double_type weight_decay, c_double_type max_epoch) :
    learn_rate(learn_rate), momentum(momentum), weight_decay(weight_decay), 
    max_epoch(max_epoch), hidden(hidden_layer), masked_hidden(hidden_layer) {
    weight_input.resize(input_layer, hidden_layer);
    weight_hidden.resize(hidden_layer, output_layer);
    diff_weight_input.resize(input_layer, hidden_layer);
    diff_weight_hidden.resize(hidden_layer, output_layer);

    {
      auto it = weight_input.begin1();
      auto end = weight_input.end1();
      for (;it != end; ++it) {
        std::transform(it.begin(), it.end(), it.begin(),
          [](c_double_type x) { return random_number_generator(); });
      }
    }

    {
      auto it = weight_hidden.begin1();
      auto end = weight_hidden.end1();
      for (;it != end; ++it) {
        std::transform(it.begin(), it.end(), it.begin(),
          [](c_double_type x) { return random_number_generator(); });
      }
    }
  }

  void do_training(const std::vector<std::pair<
  vector_type, vector_type> >& dataset) {
    for (std::size_t i = 0; i < max_epoch; ++i) {
      for (std::size_t j = 0; j < 100; ++j) {
        c_int_type index = random_number_generator(0.0, dataset.size() - 1.0);
        c_vector_type mask = generate_dropout_mask(hidden.size());
        c_vector_type output = forwardpropagate(dataset[index].second, mask);
        backpropagate(
          dataset[index].first, dataset[index].second, output, mask);
        update_weight();
      }
    }
  }

  vector_type predict(c_vector_type& input) {
    vector_type hidden_input = prod(input, weight_input);
    std::transform(hidden_input.begin(), hidden_input.end(), hidden.begin(),
      [](c_double_type x) { return 1.0 / (1.0 + std::exp(-x)); });

    vector_type t(hidden.size());
    std::copy(hidden.begin(), hidden.end(), t.begin());

    for (value_type& x : t) {
      x /= 2.0;
    }

    return prod(t, weight_hidden);
  }

private:
  vector_type forwardpropagate(c_vector_type& input, c_vector_type& mask) {
    c_vector_type hidden_input = prod(input, weight_input);
    std::transform(hidden_input.begin(), hidden_input.end(), hidden.begin(),
      [](c_double_type x) { return 1.0 / (1.0 + std::exp(-x)); });

    for (std::size_t i = 0; i < hidden.size(); ++i) {
      masked_hidden[i] = hidden[i] * mask[i];
    }

    return prod(masked_hidden, weight_hidden);
  }

  void backpropagate(c_vector_type& answer, c_vector_type& input,
    c_vector_type& output, c_vector_type& mask) {
    c_vector_type delta = output - answer;
    vector_type error_hidden = prod(weight_hidden, delta);

    for (std::size_t i = 0; i < error_hidden.size(); ++i) {
      error_hidden[i] *= masked_hidden[i] * (1.0 - masked_hidden[i]);
    }

    for (std::size_t i = 0; i < diff_weight_hidden.size1(); ++i) {
      for (std::size_t j = 0; j < diff_weight_hidden.size1(); ++j) {
        diff_weight_hidden(i, j) = masked_hidden[i] * delta[j];
      }
    }

    for (std::size_t i = 0; i < diff_weight_input.size1(); ++i) {
      for (std::size_t j = 0; j < diff_weight_input.size1(); ++j) {
        diff_weight_input(i, j) = input[i] + error_hidden[j];
      }
    }
  }

  void update_weight() {

  }

  vector_type generate_dropout_mask(c_int_type max) {

  }
};

#endif
