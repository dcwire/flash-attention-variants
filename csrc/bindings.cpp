// pybind11 entry points for the CUDA leg. Keep all argument validation here so kernels can
// assume contiguous, same-device, same-dtype inputs with the documented layouts.
#include <torch/extension.h>
#include <tuple>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIG(x)

// ---- naive/ ------------------------------------------------------------------------------
torch::Tensor softmax_cuda(torch::Tensor x);
torch::Tensor gemm_nt_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_nn_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor naive_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                   bool causal, double scale);

// ---- flash/ ------------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> flash_forward_cuda(torch::Tensor q, torch::Tensor k,
                                                            torch::Tensor v, bool causal,
                                                            double scale);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flash_backward_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor L,
    torch::Tensor dO, bool causal, double scale);
torch::Tensor flash_decode_cuda(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
                                int64_t seq_len, int64_t num_splits);

static torch::Tensor softmax(torch::Tensor x) {
  CHECK_INPUT(x);
  return softmax_cuda(x);
}
static torch::Tensor gemm_nt(torch::Tensor a, torch::Tensor b) {
  CHECK_INPUT(a); CHECK_INPUT(b);
  TORCH_CHECK(a.size(-1) == b.size(-1), "gemm_nt: inner dims differ");
  return gemm_nt_cuda(a, b);
}
static torch::Tensor gemm_nn(torch::Tensor a, torch::Tensor b) {
  CHECK_INPUT(a); CHECK_INPUT(b);
  TORCH_CHECK(a.size(-1) == b.size(-2), "gemm_nn: inner dims differ");
  return gemm_nn_cuda(a, b);
}
static void check_qkv(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
  CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v);
  TORCH_CHECK(q.dim() == 4, "expected (B, H, N, D)");
  TORCH_CHECK(k.sizes() == v.sizes(), "k and v must match");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(1) == k.size(1) && q.size(3) == k.size(3),
              "q/k must agree on B, H, D");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && k.scalar_type() == v.scalar_type());
}
static torch::Tensor naive_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                     bool causal, double scale) {
  check_qkv(q, k, v);
  return naive_attention_cuda(q, k, v, causal, scale);
}
static std::tuple<torch::Tensor, torch::Tensor> flash_forward(torch::Tensor q, torch::Tensor k,
                                                              torch::Tensor v, bool causal,
                                                              double scale) {
  check_qkv(q, k, v);
  return flash_forward_cuda(q, k, v, causal, scale);
}
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flash_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor L,
    torch::Tensor dO, bool causal, double scale) {
  check_qkv(q, k, v); CHECK_INPUT(o); CHECK_INPUT(L); CHECK_INPUT(dO);
  return flash_backward_cuda(q, k, v, o, L, dO, causal, scale);
}
static torch::Tensor flash_decode(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
                                  int64_t seq_len, int64_t num_splits) {
  check_qkv(q, k_cache, v_cache);
  TORCH_CHECK(q.size(2) == 1, "decode expects a single query position");
  TORCH_CHECK(seq_len <= k_cache.size(2), "seq_len exceeds cache");
  return flash_decode_cuda(q, k_cache, v_cache, seq_len, num_splits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax", &softmax, "row softmax over last dim");
  m.def("gemm_nt", &gemm_nt, "A @ B^T");
  m.def("gemm_nn", &gemm_nn, "A @ B");
  m.def("naive_attention", &naive_attention, "softmax(QK^T*scale)V, full S in HBM");
  m.def("flash_forward", &flash_forward, "fused tiled forward -> (O, L)");
  m.def("flash_backward", &flash_backward, "fused backward -> (dQ, dK, dV)");
  m.def("flash_decode", &flash_decode, "split-KV decode for a single query");
}
