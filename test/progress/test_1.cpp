#include <boost/progress.hpp>

#include <boost/atomic.hpp>
#include <boost/chrono.hpp>
#include <boost/function.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/weak_ptr.hpp>

#include <cstdlib>
#include <list>
#include <utility>

struct progress_group {
  struct item {
    std::size_t _current, _total;

    item(std::size_t current = 0, std::size_t total = 0)
      : _current(current), _total(total) {}

    void tick() {
      if (_current < _total)
        ++_current;
    }
  };

  std::list<boost::weak_ptr<progress_group::item> > members;

  void add(boost::shared_ptr<item> const& i) {
    assert(i);
    members.push_back(i);
  }

  item get_cumulative() {
    item cum(0, 0);

    for (auto& w : members) {
      auto v = w.lock();

      if (v) {
        cum._current += v->_current;
        cum._total += v->_total;
      }
    }
  }
};

struct group_progress_display {
  group_progress_display() :
    display_(1000), entrancy_(0) {}

  void add(boost::shared_ptr<progress_group::item> i) {
    group_.add(i);
  }

  void update() {
    if (1 == ++entrancy_) {
      auto cum = group_.get_cumulative();

      if (cum._total > 0) {
        std::size_t target =
          (1.0 * cum._current)/cum._total * display_.expected_count();
      }
    }
    --entrancy_;
  }
private:
  boost::progress_display display_;
  progress_group group_;
  boost::atomic<int> entrancy_;
};

struct worker {
  explicit worker(std::size_t count) :
    count_(count) {}

  template <typename F>
  void dispatcher(F&& listener) {
    callback_ = std::forward<F>(listener);
  }

  void operator()() {
    for (std::size_t i = 0; i < count_; ++i) {
      boost::this_thread::sleep_for(boost::chrono::microseconds(500));
      callback_();
    }
  }
private:
  boost::function<void()> callback_;
  std::size_t count_;
};

void doit() {
  boost::thread_group workers;
  group_progress_display display;

  for (int i = 0; i < 100; ++i) {
    auto load = (rand() % 5) * 1500;
    auto progress_item = boost::make_shared<progress_group::item>(load);

    worker w(progress_item->_total);
    w.dispatcher([=, &display] {
      progress_item->tick();
      display.update();
    });

    workers.create_thread(w);
  }

  workers.join_all();
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
