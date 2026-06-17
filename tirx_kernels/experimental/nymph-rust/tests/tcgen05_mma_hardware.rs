use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Copy)]
struct MmaCase {
    m: usize,
    n: usize,
    cta_group: usize,
    lane_align: usize,
    cu: &'static str,
}

#[test]
#[ignore = "requires nvcc and an sm_100a/sm_100f GPU"]
fn tcgen05_mma_all_layouts_match_hardware() {
    let cases = [
        MmaCase {
            m: 64,
            n: 128,
            cta_group: 1,
            lane_align: 0,
            cu: "tcgen05_mma_m64_dump.cu",
        },
        MmaCase {
            m: 64,
            n: 128,
            cta_group: 1,
            lane_align: 16,
            cu: "tcgen05_mma_m64_align16_dump.cu",
        },
        MmaCase {
            m: 128,
            n: 128,
            cta_group: 1,
            lane_align: 0,
            cu: "tcgen05_mma_m128_dump.cu",
        },
        MmaCase {
            m: 128,
            n: 32,
            cta_group: 2,
            lane_align: 0,
            cu: "tcgen05_mma_cta2_m128_dump.cu",
        },
        MmaCase {
            m: 256,
            n: 32,
            cta_group: 2,
            lane_align: 0,
            cu: "tcgen05_mma_cta2_dump.cu",
        },
    ];
    for case in cases {
        let hw = run_fixture(case.cu);
        let mut mismatches = 0usize;
        for m in 0..case.m {
            for n in 0..case.n {
                let (cta, lane, col) = placement(case, m, n);
                let got = *hw.get(&(cta, lane, col)).unwrap_or_else(|| {
                    panic!("{} missing cta={cta} lane={lane} col={col}", case.cu)
                });
                let expected = dot(m, n);
                if got != expected {
                    if mismatches < 8 {
                        eprintln!(
                            "{} m={m} n={n} cta={cta} lane={lane} col={col}: hw={got} expected={expected}",
                            case.cu
                        );
                    }
                    mismatches += 1;
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "{} m={} n={} cta_group={} align={}",
            case.cu, case.m, case.n, case.cta_group, case.lane_align
        );
        println!(
            "(m={:3}, cta_group={}, align={:2}): {} cells PASS",
            case.m,
            case.cta_group,
            case.lane_align,
            case.m * case.n
        );
    }
}

#[test]
#[ignore = "requires nvcc and an sm_100a/sm_100f GPU"]
fn tcgen05_mma_accum_matches_hardware() {
    let hw = run_fixture("tcgen05_mma_accum_dump.cu");
    let mut mismatches = 0usize;
    for lane in 0..128 {
        for col in 0..128 {
            let got = *hw
                .get(&(0, lane, col))
                .unwrap_or_else(|| panic!("accum missing lane={lane} col={col}"));
            let expected = 2.0 * dot(lane, col);
            if got != expected {
                if mismatches < 8 {
                    eprintln!("accum lane={lane} col={col}: hw={got} expected={expected}");
                }
                mismatches += 1;
            }
        }
    }
    assert_eq!(mismatches, 0, "tcgen05_mma_accum_dump.cu");
    println!("accum: {} cells PASS", 128 * 128);
}

fn val(row: usize, col: usize, salt: usize) -> f32 {
    (((row * 3 + col * 5 + salt) % 5) as i32 - 2) as f32
}

fn dot(m: usize, n: usize) -> f32 {
    (0..16).map(|k| val(m, k, 0) * val(n, k, 1)).sum()
}

fn placement(case: MmaCase, m: usize, n: usize) -> (usize, usize, usize) {
    if case.cta_group == 1 && case.m == 64 {
        return (0, 32 * (m / 16) + (m % 16) + case.lane_align, n);
    }
    if case.cta_group == 1 {
        return (0, m, n);
    }
    if case.m == 256 {
        return (m / 128, m % 128, n);
    }
    let half = case.n / 2;
    (m / 64, (m % 64) + (n / half) * 64, n % half)
}

fn run_fixture(cu_name: &str) -> HashMap<(usize, usize, usize), f32> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cu = manifest.join("tests").join("cuda").join(cu_name);
    assert!(cu.exists(), "missing CUDA fixture {}", cu.display());

    let out_dir = std::env::var_os("NYMPH_TCGEN05_HW_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("target").join("tcgen05_mma_hardware"));
    fs::create_dir_all(&out_dir).expect("create hardware output dir");
    let bin = out_dir.join(cu_name.trim_end_matches(".cu"));
    compile_fixture(&cu, &bin);
    let run = Command::new(&bin).output().expect("run MMA CUDA fixture");
    assert!(
        run.status.success(),
        "{} failed\nstdout:\n{}\nstderr:\n{}",
        cu_name,
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    parse_dump(&String::from_utf8(run.stdout).expect("fixture stdout is utf8"))
}

fn compile_fixture(cu: &Path, bin: &Path) {
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
        "nvcc failed for {}\nstdout:\n{}\nstderr:\n{}",
        cu.display(),
        String::from_utf8_lossy(&nvcc.stdout),
        String::from_utf8_lossy(&nvcc.stderr)
    );
}

fn parse_dump(text: &str) -> HashMap<(usize, usize, usize), f32> {
    let mut out = HashMap::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let cta: usize = parts.next().expect("cta").parse().expect("cta int");
        let lane: usize = parts.next().expect("lane").parse().expect("lane int");
        let col: usize = parts.next().expect("col").parse().expect("col int");
        let value: f32 = parts.next().expect("value").parse().expect("value f32");
        assert!(
            parts.next().is_none(),
            "unexpected trailing fields in {line}"
        );
        out.insert((cta, lane, col), value);
    }
    out
}
