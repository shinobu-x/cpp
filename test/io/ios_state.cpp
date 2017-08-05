#include <boost/cstdlib.hpp>
#include <boost/io/ios_state.hpp>

#include <cstddef>   /* size_t                    */
#include <iomanip>   /* setw                      */
#include <ios>       /* ios_base, streamsize, etc */
#include <iostream>  /* cout, etc                 */
#include <istream>   /* istream                   */
#include <locale>    /* numpunct locale           */
#include <ostream>   /* endl, ostream             */
#include <streambuf> /* streambuf                 */
#include <string>    /* string                    */
#include <stdexcept>

#include <cassert>
#include <cstdlib>

#define ERROR(d) std::cerr << d; std::cerr << "\n"; std::abort();

class backward_bool_names : public std::numpunct<char> {
  typedef std::numpunct<char> base_type;
public:
  explicit backward_bool_names(std::size_t refs = 0) : base_type(refs) {}

protected:
  virtual ~backward_bool_names() {}
  virtual base_type::string_type do_truename() const {
    return "eurt";
  }
  virtual base_type::string_type do_falsename() const {
    return "eslaf";
  }
};

int const index = std::ios_base::xalloc();

char const test_string[] = "Foo Bar";
int const test_num1 = -16;
double const test_num2 = 34.5678901234;
bool const test_bool = true;

void test_1(std::istream& i, std::ostream& o, std::ostream& e) {
  boost::io::ios_flags_saver const ifls(o);
  boost::io::ios_precision_saver const iprs(o);
  boost::io::ios_width_saver const iws(o);
  boost::io::ios_tie_saver const its(i);
  boost::io::ios_rdbuf_saver const irs(o);
  boost::io::ios_fill_saver const ifis(o);
  boost::io::ios_locale_saver const ils(o);
  boost::io::ios_iword_saver const iis(o, index);
  boost::io::ios_pword_saver const ipws(o, index);

  std::locale loc(std::locale::classic(), new backward_bool_names);

  i.tie(&e);
  o.rdbuf(e.rdbuf());
  o.iword(index) = 69L;
  o.pword(index) = &e;

  o << "The data is (again):\n";
  o.setf(std::ios_base::showpos | std::ios_base::boolalpha);
  o.setf(std::ios_base::internal, std::ios_base::adjustfield);
  o.fill('@');
  o.precision(9);
  o << '\t' << test_string << '\n';
  o << '\t' << std::setw(10) << test_num1 << '\n';
  o << '\t' << std::setw(15) << test_num2 << '\n';
  o.imbue(loc);
  o << '\t' << test_bool << '\n';

  assert(&e == o.pword(index));
  assert(69L == o.iword(index));

  try {
    boost::io::ios_exception_saver const ies(o);
    boost::io::ios_iostate_saver const iis(o);

    o.exceptions(std::ios_base::eofbit | std::ios_base::badbit);
    o.setstate(std::ios_base::eofbit);
    e << __func__ << '\n';
  }
#if defined(BOOST_GCC) || (defined(BOOST_CLANG) && defined(BOOST_GNU_STDLIB))
  catch (std::exception& ex)
#else
  catch (std::ios_base::failure& ex)
#endif
  {
    e << "Got the expected I/O failure: \"" << ex.what() << "\".\n";
    ERROR("Previous line should have thrown");
    assert(o.exceptions() == std::ios_base::goodbit);
  } catch (...) {
    e << "Got an unknown error when doing exception test\n";
    throw;
  }
}

void test_2(std::istream& i, std::ostream& o, std::ostream& e) {
  boost::io::ios_tie_saver const its(i, &e);
  boost::io::ios_rdbuf_saver const irs(o, e.rdbuf());
  boost::io::ios_iword_saver const iis(o, index, 69L);
  boost::io::ios_pword_saver const ipws(o, index, &e);
  o << "The data is (a third time, adding the numbers):\n";

  boost::io::ios_flags_saver const ifls(o, (o.flags()
    & ~std::ios_base::adjustfield) | std::ios_base::showpos
    | std::ios_base::boolalpha | (std::ios_base::internal
      & std::ios_base::adjustfield));
  boost::io::ios_precision_saver const iprs(o, 9);
  boost::io::ios_fill_saver const ifis(o, '@');
  o << '\t' << test_string << '\n';

  boost::io::ios_width_saver const iws(o, 12);
  o.put('\t');
  o << test_num1 + test_num2;
  o.put('\n');

  std::locale loc(std::locale::classic(), new backward_bool_names);
  boost::io::ios_locale_saver const ils(o, loc);
  o << '\t' << test_bool << '\n';

  assert(&e == o.pword(index));
  assert(69L == o.iword(index));

  try {
    boost::io::ios_exception_saver const ies(o, std::ios_base::eofbit);
    boost::io::ios_iostate_saver const iis(o, o.rdstate()
      | std::ios_base::eofbit);
    ERROR("Previous line should have thrown");
  }
#if defined(BOOST_GCC) || (defined(BOOST_CLANG) && defined(BOOST_GNU_STDLIB))
  catch (std::exception &ex)
#else
  catch (std::ios_base::failure &ex)
#endif
  {
    e << "Got the expected I/O failure:\"" << ex.what() << "\".\n";
    assert(o.exceptions() == std::ios_base::goodbit);
  } catch (...) {
    e << "Got an unknown error when doing exception test\n";
    throw;
  }
}
    
auto main() -> decltype(0) {
  std::cout << "The original data:\n";
  std::cout << '\t' << test_string << '\n';
  std::cout << '\t' << test_num1 << '\n';
  std::cout << '\t' << test_num2 << '\n';
  std::cout << '\t' << std::boolalpha << test_bool << '\n';

  std::ios_base::fmtflags const flags = std::cout.flags();
  std::streamsize const precision = std::cout.precision();
  std::streamsize const width = std::cout.width();
  std::ios_base::iostate const iostate = std::cout.rdstate();
  std::ios_base::iostate const exceptions = std::cout.exceptions();
  std::ostream* const tie = std::cin.tie();
  std::streambuf* const rdbuf = std::cout.rdbuf();
  char const fill = std::cout.fill();
  std::locale const locale = std::cout.getloc();

  std::cout.iword(index) = 42L;
  std::cout.pword(index) = &std::cin;

  test_1(std::cin, std::cout, std::cerr);

  assert(&std::cin == std::cout.pword(index));
  assert(42L == std::cout.iword(index));
  assert(locale == std::cout.getloc());
  assert(fill == std::cout.fill());
  assert(rdbuf == std::cout.rdbuf());
  assert(tie == std::cout.tie());
  assert(exceptions == std::cout.exceptions());
  assert(iostate == std::cout.rdstate());
  assert(width == std::cout.width());
  assert(precision == std::cout.precision());
  assert(flags == std::cout.flags()); 

  test_2(std::cin, std::cout, std::cerr);

  assert(&std::cin == std::cout.pword(index));
  assert(42L == std::cout.iword(index));
  assert(locale == std::cout.getloc());
  assert(fill == std::cout.fill());
  assert(rdbuf == std::cout.rdbuf());
  assert(tie == std::cin.tie());
  assert(exceptions == std::cout.exceptions());
  assert(iostate == std::cout.rdstate());
  assert(width == std::cout.width());
  assert(precision == std::cout.precision());
  assert(flags == std::cout.flags());

  return 0;
}
