#include <rocksdb/db.h>
#include <rocksdb/slice.h>
#include <rocksdb/options.h>

#include <cstdio>
#include <string>

auto main() -> decltype(0) {
  rocksdb::DB* db;
  rocksdb::Options options;

  options.IncreaseParallelism();
  options.OptimizeLevelStyleCompaction();

  options.create_if_missing = true;

  std::string path("/tmp/rockdb.tmp");
  rocksdb::Status s = rocksdb::DB::Open(options, path, &db);
  assert(s.ok());

  s = db->Put(rocksdb::WriteOptions(), "Key1", "value");
  assert(s.ok());

  std::string value;

  s = db->Get(rocksdb::ReadOptions(), "Key1", &value);
  assert(s.ok());
  assert(value == "value");

  {
    rocksdb::WriteBatch batch;
    batch.Delete("Key1");
    batch.Put("Key2", value);
    s = db->Write(rocksdb::WriteOptions(), &batch);
    assert(s.ok());
  }

  s = db->Get(rocksdb::ReadOptions(), "Key1", &value);
  assert(s.IsNotFound());

  db->Get(rocksdb::ReadOptions(), "Key2", &value);
  assert(value == "value");

  {
    rocksdb::PinnableSlice pinnable_value;
    db->Get(rocksdb::ReadOptions(),
      db->DefaultColumnFamily(), "Key2", &pinnable_value);
    assert(pinnable_value == "value");
  }

  {
    std::string string_value;
    rocksdb::PinnableSlice pinnable_value(&string_value);
    db->Get(rocksdb::ReadOptions(), db->DefaultColumnFamily(),
      "Key2", &pinnable_value);
    assert(pinnable_value == "value");
    assert(pinnable_value.IsPinned() || string_value == "value");
  }

  rocksdb::PinnableSlice pinnable_value;
  db->Get(rocksdb::ReadOptions(), db->DefaultColumnFamily(),
    "Key1", &pinnable_value);
  pinnable_value.Reset();
  db->Get(rocksdb::ReadOptions(), db->DefaultColumnFamily(),
    "Key2", &pinnable_value);
  assert(pinnable_value == "value");
  pinnable_value.Reset();

  delete db;

  return 0;
}
