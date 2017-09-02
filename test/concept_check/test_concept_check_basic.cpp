#include <boost/concept_check.hpp>
#include <boost/concept_archetype.hpp>

auto main() -> decltype(0) {
  {
    typedef boost::default_constructible_archetype<> f;
    boost::function_requires<boost::DefaultConstructible<f> >();
  }

  {
    typedef boost::assignable_archetype<> f;
    boost::function_requires<boost::Assignable<f> >();
  }

  {
    typedef boost::copy_constructible_archetype<> f;
    boost::function_requires<boost::CopyConstructible<f> >();
  }

  {
    typedef boost::sgi_assignable_archetype<> f;
    boost::function_requires<boost::SGIAssignable<f> >();
  }

  {
    typedef boost::copy_constructible_archetype<> f;
    typedef boost::convertible_to_archetype<f> convertible_to_f;
    boost::function_requires<boost::Convertible<convertible_to_f, f> >();
  }

  {
    boost::function_requires<boost::Convertible<
      boost::boolean_archetype, bool> >();
  }

  {
    typedef boost::equality_comparable_archetype<> f;
    boost::function_requires<boost::EqualityComparable<f> >();
  }

  {
    typedef boost::less_than_comparable_archetype<> f;
    boost::function_requires<boost::LessThanComparable<f> >();
  }

  {
    typedef boost::comparable_archetype<> f;
    boost::function_requires<boost::Comparable<f> >();
  }

  {
    typedef boost::equal_op_first_archetype<> First;
    typedef boost::equal_op_second_archetype<> Second;
    boost::function_requires<boost::EqualOp<First, Second> >();
  }

  {
    typedef boost::not_equal_op_first_archetype<> First;
    typedef boost::not_equal_op_second_archetype<> Second;
    boost::function_requires<boost::NotEqualOp<First, Second> >();
  }

  {
    typedef boost::less_than_op_first_archetype<> First;
    typedef boost::less_than_op_second_archetype<> Second;
    boost::function_requires<boost::LessThanOp<First, Second> >();
  }

  {
    typedef boost::greater_than_op_first_archetype<> First;
    typedef boost::greater_than_op_second_archetype<> Second;
    boost::function_requires<boost::GreaterThanOp<First, Second> >();
  }

  {
    typedef boost::greater_equal_op_first_archetype<> First;
    typedef boost::greater_equal_op_second_archetype<> Second;
    boost::function_requires<boost::GreaterEqualOp<First, Second> >();
  }

  {
    typedef boost::copy_constructible_archetype<> Return;
    typedef boost::plus_op_first_archetype<Return> First;
    typedef boost::plus_op_second_archetype<Return> Second;
    boost::function_requires<boost::PlusOp<Return, First, Second> >();
  }

  return 0;
}

