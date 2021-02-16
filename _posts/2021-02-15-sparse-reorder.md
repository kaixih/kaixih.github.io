---
layout: posts
title:  "Sparse Data Structure: Sorting Indices with Any Sorter + Custom Comparators"
published: true
author: kaixi_hou
search                   : true
search_full_content      : true
search_provider          : google
comments: true
---

## Introduction
Recently, I am working on a project regarding sparse tensors in Tensorflow.
Sparse tensors are used to represent tensors with many zeros. To save memory
space, we only store the non-zeros by using a matrix `indices` and an array
`values` (This representation is also called Coordiate Format--COO). By
convention, all sparse operations preserve the canonical ordering of the
non-zeros along increasing dimension number. However, the ordering might be
violated if we manually manipulate the non-zero entries, e.g., appending a new
index and a new value to `indices` and `values` respectively. Therefore, we can
call the `tf.sparse.reorder()` to explicitly enforce the ordering. While digging
into the implementation of this API, I found it very enlightening how to sort
the matrix of indices and corresponding values by using existing library sort
methods.  By doing so, we can avoid writing (and rewriting) the full sort
algorithms, and most importantly, it will be of great benefit when we switch to
other platforms, like GPUs. This post will first describe the problem and then
discuss the method.

## Question Description
Suppose we have a sparse tensor represented by a (_N_, _ndims_) matrix `indices` and
a (N) array `values`, where _N_ is the number of non-zeros and _ndims_ is the tensor
dimension. Each row of `indices` represents an index of a non-zero element and
each position of `values` is a non-zero value. An index and its corresponding
value together are called a non-zero entry. For example, the following inputs
show 3 dimensional sparse tensor with invalid ordering.
```
N = 4;
ndims = 3;
indices matrix:
0, 3, 0,
0, 2, 1,
1, 1, 0,
1, 0, 0,

values array:
'b', 'a', 'd', 'c'
```
Our algorithm takes the sparse tensor as input and output the reordered
non-zero entries, like:
```
indices matrix:
0, 2, 1,
0, 3, 0,
1, 0, 0,
1, 1, 0,

values array:
'a', 'b', 'c', 'd'
```

## Custom Comparator
Our goal is to use the existing sort methods from libraries. To achieve that,
originally, I thought we could simply reinterpret the matrix `indices` as an
array of entries, where each entry is a struct contains _ndims_ of values. Then,
a custom comparator can be designed to recognize this entry struct and cast them
back to integer indices for the actual comparison. The code would be like:
```cpp
struct Entry {
    int val[3]; // 3 is ndims, for example.
};
...
Entry* in = reinterpret_cast<Entry*>(indices);
std::sort(in, in + N, [](Entry& a, Entry& b){
  int* index_a = reinterpret_cast<int*>(&a);
  int* index_b = reinterpret_cast<int*>(&b);
  // Compare index_a and index_b by sweeping each of its dimensions.
})
```
However, there are two major issues of this solution:
1. The _ndims_ needs to be known at compile time to allow the sorter to use the
   correct iterator with correct strides.
2. We only reorder the `indices` matrix but not the `values` array.

---

Under the hood of `tf.sparse.reorder()`, it uses an assistant array `reorder`
filled with values from 0 to _N-1_ which represents the orignal position of each
entry. Then, we apply the sort over this array but with a custom comparitor that
can access the `indices`. The output will be a permuted `reorder` and we can
view it as that the correct `i`th entry should be from `reorder[i]`th `indices`
and `values`. The code structure is like:
```cpp
IndexComparator sorter(indices, ndims);
int reorder[N];
std::iota(reorder, reorder + N, 0);
std::sort(reorder, reorder + N, sorter);
// Apply the permutation.
for (int i = 0; i < N; i++) {
  // Copy reorder[i]th index to ith new_index
  // Copy reorder[i]th value to ith new_value
}
```

With this design, the custom comparator is relatively simple to write: all we
need to do is to use the position values in `reorder` to locate the index
information of interest from `indices` and then conduct the real comparision
from the major to minor dimensions.
```cpp
class IndexComparator {
public:
  IndexComparator(const int *indices, const int ndims) :
      indices_(indices), ndims_(ndims) {}
  inline bool operator()(const int i, const int j) const {
    for (int di = 0; di < ndims_; ++di) {
      if (indices_[i * ndims_ + di] < indices_[j * ndims_ + di]) {
        return true;
      }
      if (indices_[i * ndims_ + di] > indices_[j * ndims_ + di]) {
        return false;
      }
    }
    return false;
  }
private:
  const int *indices_;
  const int ndims_;
};

```

If we use the above inputs, the output is like:
```
permuted reorder:
1, 0, 3, 2,
reordered entries:
( 0, 2, 1, ) -> a
( 0, 3, 0, ) -> b
( 1, 0, 0, ) -> c
( 1, 1, 0, ) -> d
```
The full code of CPU version to use `std::sort()` is
[here](https://github.com/kaixih/dl_samples/blob/main/sparse_reorder/sort_indices.cpp).
By using the same comparator, we can easily port the code onto GPU
with `thrust::sort()`. The GPU version is [here](https://github.com/kaixih/dl_samples/blob/main/sparse_reorder/sort_indices.cu).

## Reference
* [Tensorflow Sparse Tensor](https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor)
* [Tensorflow Sparse Tensor Reorder](https://www.tensorflow.org/api_docs/python/tf/sparse/reorder)


