use nymph_rs::interpreter::values::ldstmatrix::{
    element_coord, pack_b16x2, SUPPORTED_NUMS, SUPPORTED_TRANS,
};
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const THREADS: usize = 32;
const MAX_REGS: usize = 4;
const MAX_ROWS: usize = 32;
const COLS: usize = 8;

#[test]
#[ignore = "requires nvcc and an sm_100a/sm_100f GPU"]
fn ldstmatrix_m8n8_b16_matches_hardware() {
    let configs = configs();
    let out_dir = hardware_output_dir();
    fs::create_dir_all(&out_dir).expect("create hardware test output dir");

    let cu = out_dir.join("ldstmatrix_hardware.cu");
    let bin = out_dir.join("ldstmatrix_hardware");
    let ld_bin = out_dir.join("ldmatrix_out.bin");
    let st_bin = out_dir.join("stmatrix_out.bin");
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
    let st_cells = read_u16_file(&st_bin);
    assert_eq!(ld_words.len(), configs.len() * THREADS * MAX_REGS);
    assert_eq!(st_cells.len(), configs.len() * MAX_ROWS * COLS);

    for (case_id, (num, trans)) in configs.iter().copied().enumerate() {
        for lane in 0..THREADS {
            for matrix_id in 0..num {
                let (lo_row, lo_col) = element_coord(lane, 0, trans);
                let (hi_row, hi_col) = element_coord(lane, 1, trans);
                let expected = pack_b16x2(
                    source_value(matrix_id * 8 + lo_row, lo_col),
                    source_value(matrix_id * 8 + hi_row, hi_col),
                );
                let got = ld_words[case_id * THREADS * MAX_REGS + lane * MAX_REGS + matrix_id];
                assert_eq!(
                    got, expected,
                    "ldmatrix x{num} trans={trans} lane={lane} reg={matrix_id}: hw={got:#010x} model={expected:#010x}"
                );
            }
        }

        let mut expected_st = vec![0u16; MAX_ROWS * COLS];
        for lane in 0..THREADS {
            for matrix_id in 0..num {
                let halves = [
                    st_source_value(matrix_id, lane, 0),
                    st_source_value(matrix_id, lane, 1),
                ];
                for (half, value) in halves.into_iter().enumerate() {
                    let (row, col) = element_coord(lane, half, trans);
                    expected_st[(matrix_id * 8 + row) * COLS + col] = value;
                }
            }
        }
        let st_base = case_id * MAX_ROWS * COLS;
        for row in 0..MAX_ROWS {
            for col in 0..COLS {
                let got = st_cells[st_base + row * COLS + col];
                let expected = expected_st[row * COLS + col];
                assert_eq!(
                    got, expected,
                    "stmatrix x{num} trans={trans} row={row} col={col}: hw={got:#06x} model={expected:#06x}"
                );
            }
        }
        println!("m8n8.x{num} trans={trans}: ld/st PASS");
    }
}

fn configs() -> Vec<(usize, bool)> {
    let mut out = Vec::new();
    for &num in SUPPORTED_NUMS {
        for &trans in SUPPORTED_TRANS {
            out.push((num, trans));
        }
    }
    out
}

fn source_value(row: usize, col: usize) -> u16 {
    ((row << 8) | col) as u16
}

fn st_source_value(matrix_id: usize, lane: usize, half: usize) -> u16 {
    ((matrix_id << 12) | (lane << 1) | half) as u16
}

fn hardware_output_dir() -> PathBuf {
    std::env::var_os("NYMPH_LDSTMATRIX_HW_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/ldstmatrix_hardware"))
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

fn read_u16_file(path: &PathBuf) -> Vec<u16> {
    let bytes = fs::read(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let chunks = bytes.chunks_exact(2);
    assert!(
        chunks.remainder().is_empty(),
        "{} has non-u16 byte length",
        path.display()
    );
    chunks
        .map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
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

fn generate_cuda(configs: &[(usize, bool)]) -> String {
    let mut out = String::new();
    out.push_str(
        r#"#include <cstdint>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

"#,
    );
    writeln!(out, "#define THREADS {THREADS}").unwrap();
    writeln!(out, "#define MAX_REGS {MAX_REGS}").unwrap();
    writeln!(out, "#define MAX_ROWS {MAX_ROWS}").unwrap();
    writeln!(out, "#define COLS {COLS}").unwrap();
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

__device__ __forceinline__ uint16_t source_value(uint32_t row, uint32_t col) {
  return static_cast<uint16_t>((row << 8) | col);
}

__device__ __forceinline__ uint32_t pack_u16x2(uint16_t lo, uint16_t hi) {
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

__device__ __forceinline__ uint32_t st_source_word(uint32_t matrix_id, uint32_t lane) {
  return pack_u16x2(static_cast<uint16_t>((matrix_id << 12) | (lane << 1)),
                    static_cast<uint16_t>((matrix_id << 12) | (lane << 1) | 1));
}

__device__ __forceinline__ void fill_source(uint16_t* smem) {
  const uint32_t tid = threadIdx.x;
  for (uint32_t i = tid; i < MAX_ROWS * COLS; i += THREADS) {
    smem[i] = source_value(i / COLS, i % COLS);
  }
}

__device__ __forceinline__ void clear_source(uint16_t* smem) {
  const uint32_t tid = threadIdx.x;
  for (uint32_t i = tid; i < MAX_ROWS * COLS; i += THREADS) {
    smem[i] = 0;
  }
}

__global__ void kernel(uint32_t* ld_out, uint16_t* st_out) {
  const uint32_t tid = threadIdx.x;
  __shared__ uint16_t smem[MAX_ROWS][COLS];

"#,
    );
    for (case_id, (num, trans)) in configs.iter().copied().enumerate() {
        append_ld_case(&mut out, case_id, num, trans);
    }
    for (case_id, (num, trans)) in configs.iter().copied().enumerate() {
        append_st_case(&mut out, case_id, num, trans);
    }
    out.push_str(
        r#"
}

static int write_u32_file(const char* path, const std::vector<uint32_t>& data) {
  FILE* f = std::fopen(path, "wb");
  if (!f) {
    std::perror(path);
    return 1;
  }
  size_t n = std::fwrite(data.data(), sizeof(uint32_t), data.size(), f);
  std::fclose(f);
  return n == data.size() ? 0 : 1;
}

static int write_u16_file(const char* path, const std::vector<uint16_t>& data) {
  FILE* f = std::fopen(path, "wb");
  if (!f) {
    std::perror(path);
    return 1;
  }
  size_t n = std::fwrite(data.data(), sizeof(uint16_t), data.size(), f);
  std::fclose(f);
  return n == data.size() ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <ld_out.bin> <st_out.bin>\n", argv[0]);
    return 2;
  }
  uint32_t* d_ld = nullptr;
  uint16_t* d_st = nullptr;
  const size_t ld_words = NCASES * THREADS * MAX_REGS;
  const size_t st_words = NCASES * MAX_ROWS * COLS;
  CHECK_CUDA(cudaMalloc(&d_ld, ld_words * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_st, st_words * sizeof(uint16_t)));
  CHECK_CUDA(cudaMemset(d_ld, 0, ld_words * sizeof(uint32_t)));
  CHECK_CUDA(cudaMemset(d_st, 0, st_words * sizeof(uint16_t)));
  kernel<<<1, THREADS>>>(d_ld, d_st);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<uint32_t> h_ld(ld_words);
  std::vector<uint16_t> h_st(st_words);
  CHECK_CUDA(cudaMemcpy(h_ld.data(), d_ld, ld_words * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_st.data(), d_st, st_words * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_ld));
  CHECK_CUDA(cudaFree(d_st));
  return write_u32_file(argv[1], h_ld) || write_u16_file(argv[2], h_st);
}
"#,
    );
    out
}

fn append_ld_case(out: &mut String, case_id: usize, num: usize, trans: bool) {
    let outputs = (0..num)
        .map(|i| format!("\"=r\"(r{i})"))
        .collect::<Vec<_>>()
        .join(", ");
    let regs_out = brace_operands(0, num);
    let taddr_operand = format!("%{num}");
    let trans_suffix = if trans { ".trans" } else { "" };
    writeln!(out, "  // ldmatrix x{num} trans={trans}").unwrap();
    out.push_str("  fill_source(&smem[0][0]);\n");
    out.push_str("  __syncthreads();\n");
    out.push_str("  {\n");
    for i in 0..num {
        writeln!(out, "    uint32_t r{i};").unwrap();
    }
    writeln!(out, "    uint32_t addr_row = tid % {};", 8 * num).unwrap();
    out.push_str("    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[addr_row][0]));\n");
    writeln!(
        out,
        "    asm volatile(\"ldmatrix.sync.aligned.m8n8.x{num}{trans_suffix}.shared.b16 {regs_out}, [{taddr_operand}];\\n\""
    )
    .unwrap();
    writeln!(out, "                 : {outputs}").unwrap();
    out.push_str("                 : \"r\"(addr));\n");
    for i in 0..num {
        writeln!(
            out,
            "    ld_out[{case_id} * THREADS * MAX_REGS + tid * MAX_REGS + {i}] = r{i};"
        )
        .unwrap();
    }
    out.push_str("  }\n");
    out.push_str("  __syncthreads();\n\n");
}

fn append_st_case(out: &mut String, case_id: usize, num: usize, trans: bool) {
    let inputs = std::iter::once("\"r\"(addr)".to_string())
        .chain((0..num).map(|i| format!("\"r\"(s{i})")))
        .collect::<Vec<_>>()
        .join(", ");
    let regs_in = brace_operands(1, num);
    let trans_suffix = if trans { ".trans" } else { "" };
    writeln!(out, "  // stmatrix x{num} trans={trans}").unwrap();
    out.push_str("  clear_source(&smem[0][0]);\n");
    out.push_str("  __syncthreads();\n");
    out.push_str("  {\n");
    for i in 0..num {
        writeln!(out, "    uint32_t s{i} = st_source_word({i}, tid);").unwrap();
    }
    writeln!(out, "    uint32_t addr_row = tid % {};", 8 * num).unwrap();
    out.push_str("    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[addr_row][0]));\n");
    writeln!(
        out,
        "    asm volatile(\"stmatrix.sync.aligned.m8n8.x{num}{trans_suffix}.shared.b16 [%0], {regs_in};\\n\""
    )
    .unwrap();
    out.push_str("                 :\n");
    writeln!(out, "                 : {inputs});").unwrap();
    out.push_str("  }\n");
    out.push_str("  __syncthreads();\n");
    writeln!(
        out,
        "  for (uint32_t i = tid; i < MAX_ROWS * COLS; i += THREADS) st_out[{case_id} * MAX_ROWS * COLS + i] = (&smem[0][0])[i];"
    )
    .unwrap();
    out.push_str("  __syncthreads();\n\n");
}
