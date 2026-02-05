// Simple CUDA benchmark: naive attention (materialized score matrix) vs
// flash-style online softmax (no score matrix). This is a baseline, not an
// optimized implementation.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

#define CUDA_CHECK(call)                                                     \
	do {                                                                       \
		cudaError_t err = (call);                                                \
		if (err != cudaSuccess) {                                                \
			fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
							cudaGetErrorString(err));                                      \
			std::exit(1);                                                          \
		}                                                                        \
	} while (0)

struct Args {
	int batch = 4;
	int heads = 16;
	int seq = 1024;
	int dim = 128;
	int iters = 20;
	int warmup = 5;
	bool run_naive = true;
	bool run_flash = true;
};

__device__ __forceinline__ float ld_half(const half* p) {
	return __half2float(*p);
}

__global__ void naive_scores_kernel(const half* q, const half* k, float* scores,
																		int B, int H, int S, int D) {
	int b = blockIdx.x;
	int h = blockIdx.y;
	int i = blockIdx.z;
	int tid = threadIdx.x;

	int q_offset = ((b * H + h) * S + i) * D;
	int base = ((b * H + h) * S + i) * S;

	for (int j = tid; j < S; j += blockDim.x) {
		int k_offset = ((b * H + h) * S + j) * D;
		float acc = 0.0f;
		for (int d = 0; d < D; ++d) {
			acc += ld_half(q + q_offset + d) * ld_half(k + k_offset + d);
		}
		scores[base + j] = acc;
	}
}

__global__ void softmax_inplace_kernel(float* scores, int B, int H, int S) {
	int b = blockIdx.x;
	int h = blockIdx.y;
	int i = blockIdx.z;
	int tid = threadIdx.x;

	int base = ((b * H + h) * S + i) * S;

	__shared__ float sh_max;
	__shared__ float sh_sum;

	if (tid == 0) {
		float m = -1e30f;
		for (int j = 0; j < S; ++j) {
			float v = scores[base + j];
			m = v > m ? v : m;
		}
		sh_max = m;

		float s = 0.0f;
		for (int j = 0; j < S; ++j) {
			s += expf(scores[base + j] - m);
		}
		sh_sum = s;
	}
	__syncthreads();

	float m = sh_max;
	float s = sh_sum;
	for (int j = tid; j < S; j += blockDim.x) {
		scores[base + j] = expf(scores[base + j] - m) / s;
	}
}

__global__ void naive_out_kernel(const float* softmax, const half* v, float* out,
																 int B, int H, int S, int D) {
	int b = blockIdx.x;
	int h = blockIdx.y;
	int i = blockIdx.z;
	int tid = threadIdx.x;

	int soft_base = ((b * H + h) * S + i) * S;
	int out_base = ((b * H + h) * S + i) * D;

	for (int d = tid; d < D; d += blockDim.x) {
		float acc = 0.0f;
		for (int j = 0; j < S; ++j) {
			int v_offset = ((b * H + h) * S + j) * D + d;
			acc += softmax[soft_base + j] * ld_half(v + v_offset);
		}
		out[out_base + d] = acc;
	}
}

// Flash-style online softmax (no score matrix). One block per (b,h,i),
// one thread per d. Requires D <= 256.
__global__ void flash_online_kernel(const half* q, const half* k, const half* v,
																		float* out, int B, int H, int S, int D) {
	int b = blockIdx.x;
	int h = blockIdx.y;
	int i = blockIdx.z;
	int d = threadIdx.x;

	if (d >= D) return;

	extern __shared__ float shmem[];
	float* sh_qk = shmem;            // blockDim.x
	float* sh_m = shmem + blockDim.x; // 1
	float* sh_l = sh_m + 1;           // 1

	int q_offset = ((b * H + h) * S + i) * D;
	float qd = ld_half(q + q_offset + d);

	float o = 0.0f;
	float m = -1e30f;
	float l = 0.0f;

	for (int j = 0; j < S; ++j) {
		int k_offset = ((b * H + h) * S + j) * D + d;
		sh_qk[d] = qd * ld_half(k + k_offset);
		__syncthreads();

		// reduce sh_qk to get qk for this j
		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (d < stride) {
				sh_qk[d] += sh_qk[d + stride];
			}
			__syncthreads();
		}

		float qk = sh_qk[0];
		if (d == 0) {
			float m_new = m > qk ? m : qk;
			float l_new = l * expf(m - m_new) + expf(qk - m_new);
			*sh_m = m_new;
			*sh_l = l_new;
		}
		__syncthreads();

		float m_new = *sh_m;
		float l_new = *sh_l;
		float vj = ld_half(v + ((b * H + h) * S + j) * D + d);
		float p = expf(qk - m_new);
		o = o * expf(m - m_new) + p * vj;

		m = m_new;
		l = l_new;
		__syncthreads();
	}

	int out_offset = ((b * H + h) * S + i) * D + d;
	out[out_offset] = o / l;
}

static void parse_args(int argc, char** argv, Args& args) {
	for (int i = 1; i < argc; ++i) {
		if (!strcmp(argv[i], "--batch") && i + 1 < argc) args.batch = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--heads") && i + 1 < argc) args.heads = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--seq") && i + 1 < argc) args.seq = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--dim") && i + 1 < argc) args.dim = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--iters") && i + 1 < argc) args.iters = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) args.warmup = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--naive")) args.run_naive = true;
		else if (!strcmp(argv[i], "--flash")) args.run_flash = true;
		else if (!strcmp(argv[i], "--no-naive")) args.run_naive = false;
		else if (!strcmp(argv[i], "--no-flash")) args.run_flash = false;
	}
}

int main(int argc, char** argv) {
	Args args;
	parse_args(argc, argv, args);

	if (args.dim > 256) {
		fprintf(stderr, "dim > 256 not supported in this demo\n");
		return 1;
	}

	int B = args.batch;
	int H = args.heads;
	int S = args.seq;
	int D = args.dim;

	size_t qkv_elems = (size_t)B * H * S * D;
	size_t scores_elems = (size_t)B * H * S * S;
	size_t out_elems = qkv_elems;

	std::vector<half> h_q(qkv_elems);
	std::vector<half> h_k(qkv_elems);
	std::vector<half> h_v(qkv_elems);

	std::mt19937 rng(123);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	for (size_t i = 0; i < qkv_elems; ++i) {
		h_q[i] = __float2half(dist(rng));
		h_k[i] = __float2half(dist(rng));
		h_v[i] = __float2half(dist(rng));
	}

	half* d_q = nullptr;
	half* d_k = nullptr;
	half* d_v = nullptr;
	float* d_scores = nullptr;
	float* d_out_naive = nullptr;
	float* d_out_flash = nullptr;

	CUDA_CHECK(cudaMalloc(&d_q, qkv_elems * sizeof(half)));
	CUDA_CHECK(cudaMalloc(&d_k, qkv_elems * sizeof(half)));
	CUDA_CHECK(cudaMalloc(&d_v, qkv_elems * sizeof(half)));
	CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), qkv_elems * sizeof(half), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), qkv_elems * sizeof(half), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), qkv_elems * sizeof(half), cudaMemcpyHostToDevice));

	if (args.run_naive) {
		CUDA_CHECK(cudaMalloc(&d_scores, scores_elems * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&d_out_naive, out_elems * sizeof(float)));
	}
	if (args.run_flash) {
		CUDA_CHECK(cudaMalloc(&d_out_flash, out_elems * sizeof(float)));
	}

	dim3 grid(B, H, S);
	int threads = 256;

	cudaEvent_t start, end;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&end));

	if (args.run_naive) {
		for (int i = 0; i < args.warmup; ++i) {
			naive_scores_kernel<<<grid, threads>>>(d_q, d_k, d_scores, B, H, S, D);
			softmax_inplace_kernel<<<grid, threads>>>(d_scores, B, H, S);
			naive_out_kernel<<<grid, threads>>>(d_scores, d_v, d_out_naive, B, H, S, D);
		}
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaEventRecord(start));
		for (int i = 0; i < args.iters; ++i) {
			naive_scores_kernel<<<grid, threads>>>(d_q, d_k, d_scores, B, H, S, D);
			softmax_inplace_kernel<<<grid, threads>>>(d_scores, B, H, S);
			naive_out_kernel<<<grid, threads>>>(d_scores, d_v, d_out_naive, B, H, S, D);
		}
		CUDA_CHECK(cudaEventRecord(end));
		CUDA_CHECK(cudaEventSynchronize(end));
		float ms = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
		printf("naive (materialized scores): %.3f ms\n", ms / args.iters);
	}

	if (args.run_flash) {
		size_t shmem = threads * sizeof(float) + 2 * sizeof(float);
		for (int i = 0; i < args.warmup; ++i) {
			flash_online_kernel<<<grid, threads, shmem>>>(d_q, d_k, d_v, d_out_flash, B, H, S, D);
		}
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaEventRecord(start));
		for (int i = 0; i < args.iters; ++i) {
			flash_online_kernel<<<grid, threads, shmem>>>(d_q, d_k, d_v, d_out_flash, B, H, S, D);
		}
		CUDA_CHECK(cudaEventRecord(end));
		CUDA_CHECK(cudaEventSynchronize(end));
		float ms = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
		printf("flash-style (online softmax): %.3f ms\n", ms / args.iters);
	}

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(end));

	CUDA_CHECK(cudaFree(d_q));
	CUDA_CHECK(cudaFree(d_k));
	CUDA_CHECK(cudaFree(d_v));
	if (d_scores) CUDA_CHECK(cudaFree(d_scores));
	if (d_out_naive) CUDA_CHECK(cudaFree(d_out_naive));
	if (d_out_flash) CUDA_CHECK(cudaFree(d_out_flash));

	return 0;
}
