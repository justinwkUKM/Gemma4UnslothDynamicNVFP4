#!/usr/bin/env python3
import json, os, re, signal, statistics, subprocess, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("GEMMA_ROOT", "/workspace/gemma4-benchmark"))
PY = ROOT / "env" / "bin" / "python"
VLLM = ROOT / "env" / "bin" / "vllm"
HF_HOME = ROOT / "cache" / "huggingface"
RATE = float(os.environ.get("GEMMA_HOURLY_RATE", "0.69"))
RUNTIME_ENV = {**os.environ,
               "HF_HOME": str(HF_HOME),
               "TRANSFORMERS_CACHE": str(HF_HOME),
               "CUDA_HOME": "/usr/local/cuda-13.0",
               "CUDA_PATH": "/usr/local/cuda-13.0",
               "PATH": "/usr/local/cuda-13.0/bin:" + str(ROOT / "env" / "bin") + ":" + os.environ.get("PATH", "")}
MODELS = [
    ("gemma-4-E4B-it-NVFP4", "unsloth/gemma-4-E4B-it-NVFP4"),
    ("gemma-4-12b-it-NVFP4", "unsloth/gemma-4-12b-it-NVFP4"),
    ("gemma-4-26B-A4B-it-NVFP4", "unsloth/gemma-4-26B-A4B-it-NVFP4"),
]
requested_models = {x.strip() for x in os.environ.get("GEMMA_MODEL_IDS", "").split(",") if x.strip()}
if requested_models:
    MODELS = [item for item in MODELS if item[0] in requested_models]
START = time.monotonic()
DEADLINE = START + 5 * 3600
RUN_UTC = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
summary_dir = ROOT / "summary"
summary_dir.mkdir(parents=True, exist_ok=True)

def now():
    return datetime.now(timezone.utc).isoformat()

def run_id(model_id, workload, rep):
    return f"{model_id}__{workload}__{rep}"

def gpu_sample(path, stop):
    with path.open("w") as f:
        f.write("utc,memory_used_mib,memory_total_mib,utilization_gpu\n")
        while not stop.is_set():
            try:
                out = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
                f.write(f"{now()},{out.replace(', ', ',')}\n"); f.flush()
            except Exception as e:
                f.write(f"{now()},ERROR,{e}\n"); f.flush()
            stop.wait(2)

def peak_memory(path):
    vals=[]
    if path.exists():
        for line in path.read_text(errors="replace").splitlines()[1:]:
            try: vals.append(float(line.split(",")[1]))
            except Exception: pass
    return max(vals) if vals else None

def kill_group(proc):
    if proc is None or proc.poll() is not None: return
    try: os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError: return
    end=time.monotonic()+30
    while proc.poll() is None and time.monotonic()<end: time.sleep(.5)
    if proc.poll() is None:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass

def health(timeout=2700):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as r:
                if r.status == 200:
                    with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as m:
                        if m.status == 200: return True
        except Exception: pass
        time.sleep(3)
    return False

def metric(obj, *names):
    for n in names:
        v=obj.get(n)
        if isinstance(v,(int,float)): return float(v)
    return None

def normalize(raw, model_id, checkpoint, workload, rep, elapsed, peak, failed=None):
    try: obj=json.loads(raw.read_text())
    except Exception: obj={}
    return {
        "model_id":model_id,"checkpoint":checkpoint,"workload":workload,"repetition":rep,
        "raw_file":str(raw),"wall_seconds":elapsed,"peak_gpu_memory_mib":peak,
        "median_output_tps":metric(obj,"output_throughput"),
        "median_total_tps":metric(obj,"total_token_throughput"),
        "median_request_throughput":metric(obj,"request_throughput"),
        "median_ttft_ms":metric(obj,"median_ttft_ms","mean_ttft_ms"),
        "median_tpot_ms":metric(obj,"median_tpot_ms","mean_tpot_ms"),
        "prompt_tokens":metric(obj,"input_tokens","prompt_tokens"),
        "generated_tokens":metric(obj,"output_tokens","generated_tokens"),
        "failures": int(obj.get("failed_requests", obj.get("num_failed_requests", 0)) or 0),
        "error": failed,
        "estimated_cost_usd": RATE*elapsed/3600,
    }

def invoke(model_id, checkpoint, workload, rep, n, c, model_dir):
    rid=run_id(model_id,workload,rep); raw=model_dir/(rid+".json")
    cmd=[str(VLLM),"bench","serve","--backend","openai-chat","--base-url","http://127.0.0.1:8000",
         "--endpoint","/v1/chat/completions","--model",checkpoint,"--dataset-name","random",
         "--random-input-len","512","--random-output-len","512","--random-range-ratio","0",
         "--num-prompts",str(n),"--max-concurrency",str(c),"--seed","0","--ignore-eos",
         "--save-result","--save-detailed","--result-dir",str(model_dir),"--result-filename",raw.name,
         "--percentile-metrics","ttft,tpot,e2el","--metric-percentiles","50,90,99"]
    meta=model_dir/(rid+".meta.json"); started=time.monotonic(); err=None
    try:
        p=subprocess.run(cmd,env=RUNTIME_ENV,capture_output=True,text=True,timeout=1200)
        (model_dir/(rid+".stdout.log")).write_text(p.stdout+p.stderr)
        if p.returncode: err=f"exit {p.returncode}: {p.stderr[-1000:]}"
    except subprocess.TimeoutExpired as e:
        (model_dir/(rid+".stdout.log")).write_text((e.stdout or "")+(e.stderr or "")); err="benchmark timeout"
    elapsed=time.monotonic()-started
    rec=normalize(raw,model_id,checkpoint,workload,rep,elapsed,peak_memory(model_dir/"gpu.csv"),err)
    rec["command"]=cmd; meta.write_text(json.dumps(rec,indent=2)+"\n")
    return rec

all_records=[]; failures=[]
existing = ROOT / "results" / "normalized.json"
if existing.exists():
    try:
        all_records = json.loads(existing.read_text())
        rerun_ids = {model_id for model_id, _ in MODELS}
        all_records = [r for r in all_records if r.get("model_id") not in rerun_ids]
    except Exception:
        all_records = []
for model_id, checkpoint in MODELS:
    if time.monotonic()+900 > DEADLINE: break
    model_dir=ROOT/"results"/model_id; model_dir.mkdir(parents=True,exist_ok=True)
    log=ROOT/"logs"/(model_id+".server.log"); gpu=ROOT/"logs"/(model_id+".gpu.csv")
    cmd=[str(VLLM),"serve",checkpoint,"--host","127.0.0.1","--port","8000","--served-model-name",checkpoint,
         "--max-model-len","8192","--gpu-memory-utilization","0.90","--reasoning-parser","gemma4",
         "--linear-backend","auto","--moe-backend","flashinfer_cutlass","--seed","0"]
    log.parent.mkdir(parents=True,exist_ok=True); lf=log.open("w")
    proc=subprocess.Popen(cmd,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True,env=RUNTIME_ENV)
    stop=__import__('threading').Event(); th=__import__('threading').Thread(target=gpu_sample,args=(gpu,stop),daemon=True); th.start()
    loaded=health()
    text=log.read_text(errors="replace")
    # Reject only an actually selected fallback/error line. vLLM routinely
    # prints MARLIN/EMULATION in a list of potential backends, which is not a
    # fallback and must not fail the gate.
    selected_lines = [line for line in text.splitlines()
                      if re.search(r"(?i)(selected|using|fallback|error|offload|out of memory|incompatible)", line)
                      and "potential backends" not in line.lower()]
    bad=None
    for line in selected_lines:
        match = re.search(r"(?i)(?:selected|using|fallback|error).*?(marlin|emulation|cpu offload|out of memory|incompatible.*(?:kernel|compute))", line)
        if match:
            bad=match
            break
    good=("flashinfer_cutlass" in text.lower() or "flashinfercutlassnvfp4" in text.lower())
    gate={"model_id":model_id,"checkpoint":checkpoint,"loaded":loaded,"kernel_positive":good,"kernel_bad_match":bad.group(0) if bad else None,"startup_seconds":None,"command":cmd,"utc":now()}
    (model_dir/"compatibility.json").write_text(json.dumps(gate,indent=2)+"\n")
    if not loaded or bad or not good:
        stop.set(); th.join(5); kill_group(proc); lf.close()
        failures.append({"model_id":model_id,"class":"kernel_selection" if (bad or not good) else "model_loading","detail":gate}); continue
    for rep in range(1,6): all_records.append(invoke(model_id,checkpoint,"warmup",rep,5,1,model_dir))
    for workload,n,c in (("interactive",10,1),("throughput",100,16)):
        for rep in range(1,4): all_records.append(invoke(model_id,checkpoint,workload,rep,n,c,model_dir))
    stop.set(); th.join(5); kill_group(proc); lf.close()

(ROOT/"results"/"normalized.json").write_text(json.dumps(all_records,indent=2)+"\n")
(ROOT/"results"/"failures.json").write_text(json.dumps(failures,indent=2)+"\n")
lines=[f"# Gemma 4 NVFP4 benchmark report\n\nGenerated: {now()}  ",f"Rate: ${RATE:.2f}/GPU-hour\n", "\n## Results\n", "| Model | Workload | Median output TPS | Median total TPS | TTFT ms | TPOT ms | Request TPS | Failures |\n|---|---|---:|---:|---:|---:|---:|---:|\n"]
for model_id,_ in MODELS:
  for w in ("interactive","throughput"):
    rs=[r for r in all_records if r["model_id"]==model_id and r["workload"]==w]
    def med(k):
      x=[r[k] for r in rs if isinstance(r.get(k),(int,float))]; return f"{statistics.median(x):.3f}" if x else "n/a"
    lines.append(f"| {model_id} | {w} | {med('median_output_tps')} | {med('median_total_tps')} | {med('median_ttft_ms')} | {med('median_tpot_ms')} | {med('median_request_throughput')} | {sum(r.get('failures',0) for r in rs)} |\n")
lines += ["\n## Failures\n\n```json\n",json.dumps(failures,indent=2),"\n```\n"]
(summary_dir/"benchmark-report.md").write_text("".join(lines))
print(json.dumps({"records":len(all_records),"failures":len(failures),"summary":str(summary_dir/"benchmark-report.md")},indent=2))
