# 학습한 LSTM의 Val 결과를 팀원에게 넘길 묶음으로 만든다: 결과표, 평가 결과, val_probs.npz, 가중치, README.txt, zip.
# 학습하지 않고 train_lstm.py가 저장한 파일만 읽는다. Test는 읽지 않는다.
#
# 실행: python -m sequence_model.share_lstm --tag v02_last --bundle <폴더> --bundle-name 20261103_84e7bf4 \
#           --compare-results runs/lstm_v01_cpu_cpu_results.json --compare-probs runs/lstm_v01_cpu_seed42_cpu_val_probs.npy \
#           --compare-name v01_cpu --reference-results runs/lstm_diag_cpu_results.json --reference-name diag_cpu

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from eval import window_eval as ev
from eval.window_eval import markdown_table
from eval.metrics import classification_metrics
from rule_based import config as alarm_config
from sequence_model import train_lstm as tl
from sequence_model.build_windows import CLASS_NAMES, EYE_CLOSED


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    """git 값 하나. 값이 없으면 빈 문자열."""
    done = subprocess.run(["git", *args], capture_output=True, encoding="utf-8")
    return done.stdout.strip() if done.returncode == 0 else ""


def code_commit() -> str:
    return _git("rev-parse", "--short", "HEAD") or "알 수 없음"


def private_strings() -> list[str]:
    """묶음에 들어가면 안 되는 문자열: 사용자 폴더 이름, git 이름·이메일, 사용자 폴더 경로."""
    values = [Path.home().name, _git("config", "user.name"), _git("config", "user.email"), "Users\\", "Users/"]
    return [v for v in values if v]


def find_private(folder: Path, strings: list[str]) -> list[str]:
    """폴더 파일에서 찾은 '파일: 문자열' 목록. JSON 안의 \\uXXXX와 CSV의 BOM과 상관없이 바이트로 찾는다."""
    found = []
    for path in sorted(folder.iterdir()):
        data = path.read_bytes()
        found += [f"{path.name}: {s}" for s in strings
                  if s.encode("utf-8") in data or s.encode("unicode_escape") in data or json.dumps(s).strip('"').encode() in data]
    return found


def _settings_pool(result: dict) -> str:
    """풀링 옵션을 넣기 전의 결과에는 값이 없고, 그때는 평균 풀링뿐이었다."""
    return result["settings"].get("pool", "mean")


def make_probs_npz(path: Path, rows: dict, probs: dict[int, np.ndarray]) -> None:
    """시드별 확률 → 공유 .npz (v8과 같은 형식). 행 순서는 windows.npz의 val 행 순서다."""
    np.savez_compressed(path, clip=rows["clip"], start_sec=rows["start_sec"], y=rows["y"], classes=np.array(CLASS_NAMES),
                        **{f"probs_s{seed}": p.astype(np.float32) for seed, p in probs.items()})


def _f1_row(label: object, pool: str, device: str, row: dict) -> list:
    return [label, pool, device, row.get("stopped_epoch"), row.get("best_epoch"), row["macro_f1"],
            *[row["f1_per_class"].get(name) for name in CLASS_NAMES]]


def results_report(args: argparse.Namespace, result: dict, compare: dict, reference: dict | None) -> str:
    """시드별 학습 결과 → results.md. 평균·표준편차는 이 태그의 시드로만 계산하고 비교 행은 넣지 않는다."""
    pool, device, seeds = _settings_pool(result), result["device"], result["seeds"]
    spread = {row[0]: row[1:] for row in tl.seed_spread(seeds)}
    items = ["macro F1", *CLASS_NAMES]
    rows = [_f1_row(s["seed"], pool, device, s) for s in seeds]
    rows += [[name, pool, device, None, None, *[spread[item][i] for item in items]] for i, name in enumerate(("평균", "표준편차"))]
    rows += [_f1_row(f"{s['seed']} ({args.compare_name}, 비교)", _settings_pool(compare), compare["device"], s) for s in compare["seeds"]]
    parts = [
        f"# LSTM {args.tag} Val 결과",
        f"풀링: {pool}, 장치: {device}, 시드: {', '.join(str(s['seed']) for s in seeds)}, 입력 묶음: {args.bundle_name}. Test는 쓰지 않았다.",
        markdown_table(["시드", "풀링", "장치", "멈춘 epoch", "가장 좋았던 epoch", "macro F1", *CLASS_NAMES[:-1], f"{CLASS_NAMES[-1]}(참고)"], rows),
        f"{CLASS_NAMES[-1]}은 창 수가 적어(아래 표) 참고값이다.평균·표준편차는 {args.tag} 시드로만 계산했다(표준편차는 모표준편차). "
        "값은 Val macro F1이 가장 높은 epoch의 것이라 낙관적이고, 표준편차가 작아도 안정적이라는 뜻은 아니다.",
    ]
    if reference:
        mean_f1 = float(np.mean([s["macro_f1"] for s in reference["seeds"]]))
        parts.append(f"참고: 비교 행은 시드 {len(compare['seeds'])}개다. 같은 설정의 {_settings_pool(reference)} 풀링 시드 {len(reference['seeds'])}개"
                     f"({args.reference_name}) 평균 macro F1은 {mean_f1:.3f}이다. 시드 수가 달라 비교 행만 보면 차이가 실제보다 커 보일 수 있다.")
    parts += ["클래스별 창 수",
              markdown_table(["분할", *CLASS_NAMES], [[name, *[count[c] for c in CLASS_NAMES]] for name, count in result["class_counts"].items()])]
    return "\n\n".join(parts) + "\n"


def compare_note(name: str, eye_prob_max: float) -> str:
    """비교 행의 눈 감김 확률이 켜짐 기준에 닿지 않으면 타이머 지표 0을 성능으로 읽지 않게 적는다."""
    note = f"비교 행({name})의 눈 감김 확률 최댓값은 {eye_prob_max:.3f}이다."
    if eye_prob_max < alarm_config.PROB_ON:
        note += (f" 켜짐 기준 {alarm_config.PROB_ON}에 닿지 않아 눈 감김이 한 번도 켜지지 않았으므로, "
                 "구간·창 F1, 재현율, 오경보 0은 경고가 아예 울리지 않았다는 뜻이다. 오경보 0을 장점으로 읽으면 안 된다.")
    return note


def eval_report(args: argparse.Namespace, result: dict, compare: dict, scripted: str, results: dict, summary: dict, compared: dict,
                eye_prob_max: float) -> str:
    """평가 스크립트 보고서 앞에 풀링·장치·시드와 비교 행을 붙인다. 지표 값은 평가 스크립트의 것을 그대로 쓴다."""
    pool, device, seeds = _settings_pool(result), result["device"], result["seeds"]
    rows = [[key, pool, device, s["seed"], *[get(results[key]) for get in ev.SUMMARY_COLUMNS.values()]] for key, s in zip(results, seeds)]
    rows += [[name, pool, device, "-", *summary[stat].values()] for name, stat in (("평균", "mean"), ("표준편차", "std"))]
    compare_seed = compare["seeds"][0]["seed"]
    rows.append([f"{args.compare_name} (비교)", _settings_pool(compare), compare["device"], compare_seed,
                 *[get(compared) for get in ev.SUMMARY_COLUMNS.values()]])
    return "\n\n".join([
        f"# LSTM {args.tag} 평가 (val)",
        f"풀링: {pool}, 장치: {device}, 시드: {', '.join(str(s['seed']) for s in seeds)}, 입력 묶음: {args.bundle_name}. Test는 쓰지 않았다.",
        markdown_table(["키", "풀링", "장치", "시드", *ev.SUMMARY_COLUMNS], rows),
        f"평균·표준편차는 {args.tag} 시드로만 계산했고 비교 행은 넣지 않았다. {compare_note(args.compare_name, eye_prob_max)}",
        "---",
        scripted.strip(),
        ev.to_report(compared, f"{args.compare_name} (비교)").replace("# ", "## ", 1).strip(),
    ]) + "\n"


def readme_text(args: argparse.Namespace, result: dict, compare: dict, results: dict, sources: list[str]) -> str:
    pool, device, seeds = _settings_pool(result), result["device"], result["seeds"]
    first = next(iter(results.values()))
    settings = result["settings"]
    seed_text = " ".join(str(s["seed"]) for s in seeds)
    return f"""LSTM {args.tag} 공유 묶음

풀링 방식: {pool} (시간축 마지막 시점 출력)
장치: {device}
시드: {seed_text}
입력 묶음: {args.bundle_name}
데이터 출처: {', '.join(sources)}
입력 파일 SHA256
  windows.npz: {_sha256(args.bundle / 'windows.npz')}
  norm.npz: {_sha256(args.bundle / 'norm.npz')}
코드
  저장소 커밋: {code_commit()}
  sequence_model/train_lstm.py SHA256: {_sha256(Path(tl.__file__))}
  (train_lstm.py가 저장소에 올라가기 전이면 위 커밋에 없으므로 SHA256으로 확인한다. 줄바꿈이 바뀌면 값도 달라진다)
학습 설정: 은닉 {settings['hidden']}, Dropout {settings['dropout']}, 클래스 가중치 CE, Adam {settings['lr']:g}, 배치 {settings['batch_size']}, 최대 {settings['epochs']} epoch, patience {settings['patience']}, 증강 없음

파일
  results.md: 시드별 학습 결과
  eval_report.md, eval_metrics.json, eval_events.csv: 평가 결과
  val_probs.npz: clip, start_sec, y, classes, probs_s{{시드}}. 각 (N,4), 클래스 순서 {', '.join(CLASS_NAMES)}. 행 순서는 windows.npz의 val 행 순서
  lstm_{args.tag}_seed*_cpu.pt: 시드별 가중치

실행 명령 (<번들 폴더>는 windows.npz와 norm.npz가 있는 폴더)
  python -m sequence_model.train_lstm --tag {args.tag} --bundle <번들 폴더> --pool {pool} --seeds {seed_text} --device {device}
  python -m eval.evaluate --windows <번들 폴더>/windows.npz --probs val_probs.npz --probs-key all --out-dir <결과 폴더>

가중치 불러오기
  saved = torch.load(path, map_location="cpu", weights_only=True)
  model = train_lstm.make_model(saved["pool"], saved["n_features"], saved["n_classes"], saved["hidden"], saved["dropout"])
  model.load_state_dict(saved["state_dict"])
  입력은 saved["mean"], saved["std"]로 train_lstm.normalize를 거친다.

알려진 한계
- CPU로 학습한 값이다. GPU 실행은 검증하지 않았다.
- predictor.py는 model 값이 {tl.POOLS[pool]}인 가중치를 오류로 거부한다. 데모에 쓰려면 models.py와 predictor.py를 고쳐야 한다.
- 값은 Val macro F1이 가장 높은 epoch의 것이라 낙관적이고, 표준편차가 작아도 안정적이라는 뜻은 아니다. Test는 쓰지 않았다.
- 3초 이벤트 {first['drowsy']['n_events']}개, 오경보 분모 {first['open_sec']:g}초로 적어 재현율과 시간당 오경보는 값이 크게 흔들린다.
- {CLASS_NAMES[-1]} 창이 train {result['class_counts']['train'][CLASS_NAMES[-1]]}개, val {result['class_counts']['val'][CLASS_NAMES[-1]]}개로 적어 참고값이다.
- 비교 행({args.compare_name})은 시드 {len(compare['seeds'])}개라 평균·표준편차가 없다.
{f'- {args.bundle_note}' + chr(10) if args.bundle_note else ''}
비공개 팀 공유용이다. 공개하거나 저장소에 올리지 않는다.
"""


def build(args: argparse.Namespace) -> Path:
    """묶음을 임시 폴더에 다 만들고 검사한 뒤 옮긴다. 중간에 멈추면 결과 폴더와 zip은 생기지 않는다."""
    out_dir = args.out_dir or args.runs_dir / f"share_lstm_{args.tag}"
    archive = out_dir.with_suffix(".zip")
    existing = [str(p) for p in (out_dir, archive) if p.exists()]
    if existing:
        raise SystemExit(f"묶음이 이미 있습니다. 다른 --out-dir를 쓰세요: {existing}")

    result = json.loads(tl.results_path(args, "results", "json").read_text(encoding="utf-8"))
    compare = json.loads(args.compare_results.read_text(encoding="utf-8"))
    reference = json.loads(args.reference_results.read_text(encoding="utf-8")) if args.reference_results else None
    seeds = [s["seed"] for s in result["seeds"]]

    windows = args.bundle / "windows.npz"
    rows = ev.load_rows(windows, "val")
    with np.load(windows, allow_pickle=False) as data:
        sources = sorted(str(s) for s in np.unique(data["source"]))
    probs = {seed: np.load(tl.probs_path(args, seed, "val"), allow_pickle=False) for seed in seeds}
    for row in result["seeds"]:
        macro = classification_metrics(rows["y"], probs[row["seed"]].argmax(axis=1))["macro_f1"]
        if not np.isclose(macro, row["macro_f1"]):
            raise SystemExit(f"시드 {row['seed']}: 확률의 macro F1 {macro}가 결과 파일의 {row['macro_f1']}와 다릅니다")

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}_", dir=out_dir.parent))
    try:
        make_probs_npz(work / "val_probs.npz", rows, probs)
        scripted_dir = work / "scripted"
        with contextlib.redirect_stdout(io.StringIO()):
            ev.main(["--windows", str(windows), "--probs", str(work / "val_probs.npz"), "--probs-key", ev.ALL_KEYS, "--out-dir", str(scripted_dir)])
        metrics = json.loads((scripted_dir / "metrics.json").read_text(encoding="utf-8"))
        scripted = (scripted_dir / "report.md").read_text(encoding="utf-8")
        shutil.move(scripted_dir / "events.csv", work / "eval_events.csv")
        shutil.rmtree(scripted_dir)

        compare_probs = ev.load_probs(args.compare_probs, None, rows)
        compared = ev.evaluate(rows["clip"], rows["start_sec"], rows["y"], compare_probs)
        eye_prob_max = float(compare_probs[:, EYE_CLOSED].max())
        metrics["run_info"] = {"tag": args.tag, "pool": _settings_pool(result), "device": result["device"], "seeds": seeds, "bundle": args.bundle_name}
        metrics["comparison"] = {"name": args.compare_name, "pool": _settings_pool(compare), "device": compare["device"],
                                 "seeds": [s["seed"] for s in compare["seeds"]], "eye_prob_max": eye_prob_max, "result": compared}
        (work / "eval_metrics.json").write_text(json.dumps(ev._json_safe(metrics), ensure_ascii=False, indent=1), encoding="utf-8")
        (work / "eval_report.md").write_text(
            eval_report(args, result, compare, scripted, metrics["results"], metrics["summary"], ev._json_safe(compared), eye_prob_max), encoding="utf-8")
        (work / "results.md").write_text(results_report(args, result, compare, reference), encoding="utf-8")
        (work / "README.txt").write_text(readme_text(args, result, compare, metrics["results"], sources), encoding="utf-8")
        for seed in seeds:
            shutil.copy2(tl.weight_path(args, seed), work / tl.weight_path(args, seed).name)

        found = find_private(work, private_strings())
        if found:
            raise SystemExit(f"개인정보로 보이는 문자열이 있어 묶음을 만들지 않습니다: {found}")
        os.replace(work, out_dir)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    shutil.make_archive(str(out_dir), "zip", root_dir=out_dir.parent, base_dir=out_dir.name)
    print(f"저장: {out_dir}, {archive}")
    return out_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSTM Val 결과 공유 묶음")
    parser.add_argument("--tag", required=True, help="train_lstm.py의 --tag")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu", help="train_lstm.py의 --device")
    parser.add_argument("--bundle", type=Path, required=True, help="windows.npz와 norm.npz가 있는 폴더")
    parser.add_argument("--bundle-name", required=True, help="README.txt에 적을 입력 묶음 이름. 경로를 넣지 않는다")
    parser.add_argument("--bundle-note", default="", help="이 입력 묶음의 한계")
    parser.add_argument("--compare-results", type=Path, required=True, help="비교 행의 학습 결과 JSON")
    parser.add_argument("--compare-probs", type=Path, required=True, help="비교 행의 Val 확률 .npy")
    parser.add_argument("--compare-name", required=True, help="비교 행 이름")
    parser.add_argument("--reference-results", type=Path, help="비교 행의 시드 수가 적을 때 참고 문장에 쓸 학습 결과 JSON")
    parser.add_argument("--reference-name", default="", help="참고 문장에 적을 이름")
    parser.add_argument("--out-dir", type=Path, help="기본값은 runs/share_lstm_<tag>")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    return build(parse_args(argv))


if __name__ == "__main__":
    main()
