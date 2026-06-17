use nymph_rs::interpreter::values::tcgen05_datapath::{
    datapath_index_arrays, register_count, supported_atoms,
};
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const THREADS: usize = 128;
const NCOLS: usize = 256;
const MAX_REGS: usize = 128;

#[test]
#[ignore = "requires nvcc and an sm_100a/sm_100f GPU"]
fn tcgen05_ld_st_all_base_b32_matches_hardware() {
    let configs = supported_atoms();
    let out_dir = hardware_output_dir();
    fs::create_dir_all(&out_dir).expect("create hardware test output dir");

    let cu = out_dir.join("tcgen05_ldst_hardware.cu");
    let bin = out_dir.join("tcgen05_ldst_hardware");
    let ld_bin = out_dir.join("ld_out.bin");
    let st_bin = out_dir.join("st_out.bin");
    fs::write(&cu, generate_cuda(&configs)).expect("write generated CUDA harness");

    let nvcc = Command::new("nvcc")
        .args([
            "-gencode",
            "arch=compute_100a,code=sm_100a",
            "-o",
            bin.to_str().expect("utf8 bin path"),
            cu.to_str().expect("utf8 cu path"),
        ])
        .output()
        .expect("run nvcc");
    assert!(
        nvcc.status.success(),
        "nvcc failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&nvcc.stdout),
        String::from_utf8_lossy(&nvcc.stderr)
    );

    let run = Command::new(&bin)
        .args([&ld_bin, &st_bin])
        .output()
        .expect("run generated CUDA harness");
    assert!(
        run.status.success(),
        "CUDA harness failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );

    let ld_words = read_u32_file(&ld_bin);
    let st_words = read_u32_file(&st_bin);
    assert_eq!(ld_words.len(), configs.len() * THREADS * MAX_REGS);
    assert_eq!(st_words.len(), configs.len() * THREADS * NCOLS);

    for (case_id, (shape, num)) in configs.iter().copied().enumerate() {
        let regs = register_count(shape, num).unwrap();
        let (lane_idx, col_idx) = datapath_index_arrays(shape, num).unwrap();

        for tid in 0..THREADS {
            let warp = tid / 32;
            let lane = tid % 32;
            for reg in 0..regs {
                let phys_lane = 32 * warp + lane_idx[[lane, reg]];
                let phys_col = col_idx[[lane, reg]];
                let expected = ((phys_lane as u32) << 16) | phys_col as u32;
                let got = ld_words[case_id * THREADS * MAX_REGS + tid * MAX_REGS + reg];
                assert_eq!(
                    got, expected,
                    "ld {shape}.x{num} tid={tid} reg={reg}: hw={got:#010x} model={expected:#010x}"
                );
            }
        }

        let mut expected_st = vec![0u32; THREADS * NCOLS];
        for tid in 0..THREADS {
            let warp = tid / 32;
            let lane = tid % 32;
            for reg in 0..regs {
                let phys_lane = 32 * warp + lane_idx[[lane, reg]];
                let phys_col = col_idx[[lane, reg]];
                assert!(
                    phys_col < NCOLS,
                    "{shape}.x{num} reaches column {phys_col}, NCOLS={NCOLS}"
                );
                expected_st[phys_lane * NCOLS + phys_col] = ((tid as u32) << 16) | reg as u32;
            }
        }
        let st_base = case_id * THREADS * NCOLS;
        for lane in 0..THREADS {
            for col in 0..NCOLS {
                let got = st_words[st_base + lane * NCOLS + col];
                let expected = expected_st[lane * NCOLS + col];
                assert_eq!(
                    got, expected,
                    "st {shape}.x{num} lane={lane} col={col}: hw={got:#010x} model={expected:#010x}"
                );
            }
        }
        println!("{shape}.x{num:<3} regs={regs:<3} ld/st PASS");
    }
    println!(
        "checked {} base .b32 shape/num configs for both ld and st",
        configs.len()
    );
}

fn hardware_output_dir() -> PathBuf {
    std::env::var_os("NYMPH_TCGEN05_HW_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/tcgen05_ldst_hardware"))
}

fn read_u32_file(path: &PathBuf) -> Vec<u32> {
    let bytes = fs::read(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let chunks = bytes.chunks_exact(4);
    assert!(
        chunks.remainder().is_empty(),
        "{} has non-u32 byte length",
        path.display()
    );
    chunks
        .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn brace_operands(start: usize, n: usize) -> String {
    format!(
        "{{{}}}",
        (0..n)
            .map(|i| format!("%{}", start + i))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn generate_cuda(configs: &[(&str, usize)]) -> String {
    let mut out = String::new();
    out.push_str(
        r#"#include <cstdint>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

"#,
    );
    writeln!(out, "#define THREADS {THREADS}").unwrap();
    writeln!(out, "#define NCOLS {NCOLS}").unwrap();
    writeln!(out, "#define MAX_REGS {MAX_REGS}").unwrap();
    writeln!(out, "#define NCASES {}", configs.len()).unwrap();
    out.push_str(
        r#"
#define CHECK_CUDA(expr) do { \
  cudaError_t err__ = (expr); \
  if (err__ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    return 1; \
  } \
} while (0)

__device__ __forceinline__ void store_identity(uint32_t base, uint32_t tid, uint32_t lanebase,
                                               uint32_t col, uint32_t value) {
  uint32_t taddr = base + (lanebase << 16) + col;
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n" :: "r"(taddr), "r"(value));
}

__device__ __forceinline__ void fill_identity(uint32_t base, uint32_t tid, uint32_t lanebase) {
  for (uint32_t col = 0; col < NCOLS; ++col) {
    store_identity(base, tid, lanebase, col, (tid << 16) | col);
  }
  asm volatile("tcgen05.wait::st.sync.aligned;\n");
}

__device__ __forceinline__ void clear_tmem(uint32_t base, uint32_t tid, uint32_t lanebase) {
  for (uint32_t col = 0; col < NCOLS; ++col) {
    store_identity(base, tid, lanebase, col, 0);
  }
  asm volatile("tcgen05.wait::st.sync.aligned;\n");
}

__device__ __forceinline__ void dump_tmem(uint32_t base, uint32_t tid, uint32_t lanebase, uint32_t* out) {
  for (uint32_t col = 0; col < NCOLS; ++col) {
    uint32_t taddr = base + (lanebase << 16) + col;
    uint32_t val;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n" : "=r"(val) : "r"(taddr));
    asm volatile("tcgen05.wait::ld.sync.aligned;\n");
    out[tid * NCOLS + col] = val;
  }
}

__global__ void kernel(uint32_t* ld_out, uint32_t* st_out) {
  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid >> 5;
  const uint32_t lanebase = warp << 5;
  __shared__ uint32_t tmem_base_smem;

  if (warp == 0) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(&tmem_base_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], %1;\n" :: "r"(smem_addr), "n"(NCOLS));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
  }
  __syncthreads();

  const uint32_t base = tmem_base_smem;

"#,
    );
    for (case_id, (shape, num)) in configs.iter().copied().enumerate() {
        append_ld_case(&mut out, case_id, shape, num);
    }
    for (case_id, (shape, num)) in configs.iter().copied().enumerate() {
        append_st_case(&mut out, case_id, shape, num);
    }
    out.push_str(
        r#"
  __syncthreads();
  if (warp == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" :: "r"(base), "n"(NCOLS));
  }
}

static int write_file(const char* path, const std::vector<uint32_t>& data) {
  FILE* f = std::fopen(path, "wb");
  if (!f) {
    std::perror(path);
    return 1;
  }
  size_t n = std::fwrite(data.data(), sizeof(uint32_t), data.size(), f);
  std::fclose(f);
  return n == data.size() ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <ld_out.bin> <st_out.bin>\n", argv[0]);
    return 2;
  }
  uint32_t *d_ld = nullptr, *d_st = nullptr;
  const size_t ld_words = NCASES * THREADS * MAX_REGS;
  const size_t st_words = NCASES * THREADS * NCOLS;
  CHECK_CUDA(cudaMalloc(&d_ld, ld_words * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_st, st_words * sizeof(uint32_t)));
  CHECK_CUDA(cudaMemset(d_ld, 0, ld_words * sizeof(uint32_t)));
  CHECK_CUDA(cudaMemset(d_st, 0, st_words * sizeof(uint32_t)));
  kernel<<<1, THREADS>>>(d_ld, d_st);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<uint32_t> h_ld(ld_words), h_st(st_words);
  CHECK_CUDA(cudaMemcpy(h_ld.data(), d_ld, ld_words * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_st.data(), d_st, st_words * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_ld));
  CHECK_CUDA(cudaFree(d_st));
  return write_file(argv[1], h_ld) || write_file(argv[2], h_st);
}
"#,
    );
    out
}

fn append_ld_case(out: &mut String, case_id: usize, shape: &str, num: usize) {
    let regs = register_count(shape, num).unwrap();
    let outputs = (0..regs)
        .map(|i| format!("\"=r\"(r{i})"))
        .collect::<Vec<_>>()
        .join(", ");
    let regs_out = brace_operands(0, regs);
    let taddr_operand = format!("%{regs}");
    let split = if shape == "16x32bx2" {
        format!(", {num}")
    } else {
        String::new()
    };
    writeln!(out, "  // ld {shape}.x{num}").unwrap();
    out.push_str("  fill_identity(base, tid, lanebase);\n");
    out.push_str("  __syncthreads();\n");
    out.push_str("  {\n");
    for i in 0..regs {
        writeln!(out, "    uint32_t r{i};").unwrap();
    }
    out.push_str("    uint32_t taddr = base + (lanebase << 16);\n");
    writeln!(
        out,
        "    asm volatile(\"tcgen05.ld.sync.aligned.{shape}.x{num}.b32 {regs_out}, [{taddr_operand}]{split};\\n\""
    )
    .unwrap();
    writeln!(out, "                 : {outputs}").unwrap();
    out.push_str("                 : \"r\"(taddr));\n");
    out.push_str("    asm volatile(\"tcgen05.wait::ld.sync.aligned;\\n\");\n");
    for i in 0..regs {
        writeln!(
            out,
            "    ld_out[{case_id} * THREADS * MAX_REGS + tid * MAX_REGS + {i}] = r{i};"
        )
        .unwrap();
    }
    out.push_str("  }\n");
    out.push_str("  __syncthreads();\n\n");
}

fn append_st_case(out: &mut String, case_id: usize, shape: &str, num: usize) {
    let regs = register_count(shape, num).unwrap();
    let inputs = std::iter::once("\"r\"(taddr)".to_string())
        .chain((0..regs).map(|i| format!("\"r\"(s{i})")))
        .collect::<Vec<_>>()
        .join(", ");
    let regs_in = brace_operands(1, regs);
    let instr = if shape == "16x32bx2" {
        format!("tcgen05.st.sync.aligned.{shape}.x{num}.b32 [%0], {num}, {regs_in};\\n")
    } else {
        format!("tcgen05.st.sync.aligned.{shape}.x{num}.b32 [%0], {regs_in};\\n")
    };
    writeln!(out, "  // st {shape}.x{num}").unwrap();
    out.push_str("  clear_tmem(base, tid, lanebase);\n");
    out.push_str("  __syncthreads();\n");
    out.push_str("  {\n");
    for i in 0..regs {
        writeln!(out, "    uint32_t s{i} = ((tid << 16) | {i}u);").unwrap();
    }
    out.push_str("    uint32_t taddr = base + (lanebase << 16);\n");
    writeln!(out, "    asm volatile(\"{instr}\"").unwrap();
    out.push_str("                 :\n");
    writeln!(out, "                 : {inputs});").unwrap();
    out.push_str("    asm volatile(\"tcgen05.wait::st.sync.aligned;\\n\");\n");
    out.push_str("  }\n");
    out.push_str("  __syncthreads();\n");
    writeln!(
        out,
        "  dump_tmem(base, tid, lanebase, st_out + {case_id} * THREADS * NCOLS);"
    )
    .unwrap();
    out.push_str("  __syncthreads();\n\n");
}
