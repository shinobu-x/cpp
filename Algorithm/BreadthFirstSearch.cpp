#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

// 3種類の容器を使用して、いずれかのBucketの状態を4にする
const int Target = 4;
// Bucketの数
const int Bucket = 3;
// Bucketの種類
const int BucketType[Bucket] = {8, 5, 3};

std::string ToString(int Status) {
  std::stringstream SS;
  SS << Status;
  return SS.str();
}

// Bucketの中に、Targetが入っていれば終了
bool IsSatisfied(int* Status) {
  for (int i = 0; i < Bucket; ++i) {
    if (Status[i] == Target) {
      return true;
    }
  }
  return false;
}

struct Status {
  int Status[3]; // それぞれのBucketの状態を保持
  int Cnt;
  std::string State;
};

// Breadth First Search
Status BFS() {
  // それぞれのBucketの状態を保持
  bool Flag[BucketType[0] + 1][BucketType[1] + 1][BucketType[2] + 2];


  // 状態保持フラグの初期化
  for (int i = 0; i < BucketType[0] + 1; ++i) {
    for (int k = 0; k < BucketType[1] + 1; ++k) {
      for (int j = 0; j < BucketType[2] + 1; ++j) {
        Flag[i][k][j] = false;
      }
    }
  }

  std::queue<Status> BFS;
  std::ostringstream SS;
  Status S;

  // 初期状態: 容量8のBucketが満たされている
  S.Status[0] = BucketType[0];
  S.Status[1] = 0;
  S.Status[2] = 0;
  S.Cnt = 0;
  S.State = ToString(S.Status[0]) + " " +
            ToString(S.Status[1]) + " " +
            ToString(S.Status[2]) + "\n";

  BFS.push(S);

  while (!BFS.empty()) {
    // 先頭要素を取得し、状態を確認
    S = BFS.front();
    BFS.pop();

    // 条件を満たしている場合: いずれかのBucketの状態が4である
    if (IsSatisfied(S.Status)) {
      break;  // 処理終了
    }

    // 新しい状態の作成
    Status NewS;

    for (int i = 0; i < Bucket; ++i) {
      for (int j = 0; j < Bucket; ++j) {
        // 同じBucketからの移動は考えない
        if (i == j) {
          continue;
        }

        // 新しい状態の作成
        // NewS.Status[0] = Max(S.Status[0] - (BucketType[1] - S.Status[1]), 0);
        // NewS.Status[0] = Max(8 - (5 - 0), 0);
        // NewS.Status[0] = 3;
        NewS.Status[i] = std::max(
          S.Status[i] - (BucketType[j] - S.Status[j]), 0);
        // NewS.Status[1] = Min(S.Status[1] + S.Status[0], BucketType[1]);
        // NewS.Status[1] = Min(0 + 8, 5);
        // NewS.Status[1] = 5;
        NewS.Status[j] = std::min(S.Status[j] + S.Status[i], BucketType[j]);
        // 残りのBucketの状態
        NewS.Status[3 - i - j] = S.Status[3 - i - j];

        // 処理回数をインクリメント
        NewS.Cnt = S.Cnt + 1;

        // Bucketの状態を記録
        NewS.State = S.State +
                     ToString(NewS.Status[0]) + " " +
                     ToString(NewS.Status[1]) + " " +
                     ToString(NewS.Status[2]) + "\n";

        // 状態に変化がない場合は無視
        if (NewS.Status[i] == S.Status[i]) {
          continue;
        }

        // 新しい状態を保存
        BFS.push(NewS);
      }
    }
  }

  if (IsSatisfied(S.Status)) {
    return S;
  } else {
    S.Cnt = -1;
    S.State = "N/A";
    return S;
  }
}

auto main() -> decltype(0) {
  Status R = BFS();

  std::cout << R.Cnt << "\n"
            << R.State << "\n";

  return 0;
}
