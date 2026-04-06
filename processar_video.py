#!/usr/bin/env python3
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
from faster_whisper import WhisperModel

@dataclass
class Segment:
    start: float
    end: float
    text: str

def ffprobe_video_info(video_path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration",
        "-of", "json",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    stream = data["streams"][0]
    duration = float(data["format"]["duration"])
    width = int(stream["width"])
    height = int(stream["height"])
    orientation = "vertical" if height > width else "horizontal"
    return {"width": width, "height": height, "duration": duration, "orientation": orientation}

def transcribe(video_path: Path, model_name: str) -> List[Segment]:
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(video_path), vad_filter=True, word_timestamps=False)
    result: List[Segment] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            result.append(Segment(float(seg.start), float(seg.end), text))
    return result

def score_text(text: str) -> float:
    t = text.lower()
    words = re.findall(r"\w+", t)
    if not words:
        return 0.0
    hooks = {"agora", "segredo", "erro", "resultado", "passo", "importante", "atenção"}
    n_hooks = sum(1 for w in words if w in hooks)
    score = 0.0
    score += min(len(words), 24) * 0.04
    score += n_hooks * 0.8
    if "?" in text:
        score += 0.5
    if "!" in text:
        score += 0.3
    if len(words) < 5:
        score -= 0.6
    return score

def select_best_segments(segments: List[Segment], max_final_seconds: float) -> List[Segment]:
    scored = sorted(segments, key=lambda s: score_text(s.text), reverse=True)
    selected: List[Segment] = []
    total = 0.0
    for seg in scored:
        dur = max(0.0, seg.end - seg.start)
        if dur < 1.0:
            continue
        if total + dur > max_final_seconds:
            continue
        selected.append(seg)
        total += dur
        if total >= max_final_seconds * 0.95:
            break
    selected.sort(key=lambda s: s.start)
    return selected

def to_srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000; ms %= 60000
    s = ms // 1000; ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def split_tight_caption(text: str, max_words: int = 6) -> List[str]:
    words = re.findall(r"\S+", text.strip())
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def build_srt_from_selected(selected: List[Segment]) -> str:
    lines = []
    idx = 1
    for seg in selected:
        chunks = split_tight_caption(seg.text, max_words=6)
        if not chunks:
            continue
        seg_dur = max(0.1, seg.end - seg.start)
        chunk_dur = seg_dur / len(chunks)
        for i, chunk in enumerate(chunks):
            st = seg.start + (i * chunk_dur)
            en = min(seg.end, st + chunk_dur)
            lines += [str(idx), f"{to_srt_timestamp(st)} --> {to_srt_timestamp(en)}", chunk, ""]
            idx += 1
    return "\n".join(lines).strip() + "\n"

def extract_letterings(selected: List[Segment], per_minute: int) -> List[Dict[str, Any]]:
    letterings: List[Dict[str, Any]] = []
    timeline_cursor = 0.0
    for seg in selected:
        seg_duration = max(0.1, seg.end - seg.start)
        words = re.findall(r"\w+", seg.text)
        if len(words) < 3:
            timeline_cursor += seg_duration
            continue
        snippet = " ".join(words[: min(5, max(3, len(words)))])
        letterings.append({
            "start": timeline_cursor + 0.2,
            "end": timeline_cursor + min(seg_duration, 2.4),
            "text": snippet,
            "position": "center_mid",
        })
        timeline_cursor += seg_duration
    if not selected:
        return []
    total_seconds = sum(max(0.0, s.end - s.start) for s in selected)
    allowed = max(1, math.ceil((total_seconds / 60.0) * per_minute))
    return letterings[:allowed]

def generate_xmeml(video_path: Path, selected: List[Segment], width: int, height: int) -> str:
    clipitems = []
    timeline_cursor = 0
    for i, seg in enumerate(selected, start=1):
        start_frames = int(round(seg.start * 30))
        end_frames = int(round(seg.end * 30))
        dur_frames = max(1, end_frames - start_frames)
        clipitems.append(f"""
        <clipitem id="clipitem-{i}">
          <name>{video_path.name}</name>
          <start>{timeline_cursor}</start>
          <end>{timeline_cursor + dur_frames}</end>
          <in>{start_frames}</in>
          <out>{end_frames}</out>
          <file id="file-1">
            <name>{video_path.name}</name>
            <pathurl>file://{video_path}</pathurl>
          </file>
        </clipitem>""")
        timeline_cursor += dur_frames
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE xmeml>
<xmeml version="5">
  <sequence id="sequence-1">
    <name>Decupagem_AI</name>
    <rate><timebase>30</timebase><ntsc>FALSE</ntsc></rate>
    <media>
      <video>
        <format>
          <samplecharacteristics>
            <width>{width}</width>
            <height>{height}</height>
          </samplecharacteristics>
        </format>
        <track>
          {''.join(clipitems)}
        </track>
      </video>
    </media>
  </sequence>
</xmeml>
"""

def generate_jsx(plan_json_name: str, video_name: str, mogrt_path: str, mogrt_text_param: str) -> str:
    return f"""var PLAN_PATH = "{plan_json_name}";
var VIDEO_PATH = "{video_name}";
var MOGRT_PATH = "{mogrt_path}";
var MOGRT_TEXT_PARAM = "{mogrt_text_param}";
function readFile(path) {{
  var f = new File(path);
  if (!f.exists) return null;
  f.open("r"); var txt = f.read(); f.close();
  return txt;
}}
function importMedia(path) {{
  app.project.importFiles([path], true, app.project.getInsertionBin(), false);
}}
function findProjectItemByName(name, bin) {{
  if (!bin) bin = app.project.rootItem;
  for (var i = 0; i < bin.children.numItems; i++) {{
    var it = bin.children[i];
    if (it.name === name) return it;
    if (it.type === 2) {{ var nested = findProjectItemByName(name, it); if (nested) return nested; }}
  }}
  return null;
}}
function secToTicks(sec) {{ return String(Math.floor(sec * 254016000000.0)); }}
function setMogrtText(trackItem, paramName, newText) {{
  if (!trackItem) return false;
  var mgt = trackItem.getMGTComponent();
  if (!mgt) return false;
  for (var i = 0; i < mgt.properties.numItems; i++) {{
    var p = mgt.properties[i];
    if (p && p.displayName === paramName) {{ p.setValue(newText, 1); return true; }}
  }}
  return false;
}}
function run() {{
  var seq = app.project.activeSequence;
  if (!seq) {{ alert("Selecione uma sequência ativa antes de rodar."); return; }}
  importMedia(VIDEO_PATH);
  var planTxt = readFile(PLAN_PATH);
  if (!planTxt) {{ alert("Plano não encontrado."); return; }}
  var plan = JSON.parse(planTxt);
  var item = findProjectItemByName(plan.video_file_name);
  if (!item) {{ alert("Não encontrei o vídeo no projeto: " + plan.video_file_name); return; }}
  var vTrack = seq.videoTracks[0];
  var timelineSec = 0.0;
  for (var i = 0; i < plan.selected_segments.length; i++) {{
    var s = plan.selected_segments[i];
    item.setInPoint(s.start, 4); item.setOutPoint(s.end, 4);
    vTrack.insertClip(item, secToTicks(timelineSec));
    timelineSec += (s.end - s.start);
  }}
  if (MOGRT_PATH && MOGRT_PATH.length > 0) {{
    for (var j = 0; j < plan.letterings.length; j++) {{
      var l = plan.letterings[j];
      var mgtItem = seq.importMGT(MOGRT_PATH, secToTicks(l.start), 1, 0);
      setMogrtText(mgtItem, MOGRT_TEXT_PARAM, l.text);
    }}
  }}
  alert("Montagem concluída. Importe o SRT manualmente.");
}}
run();
"""

def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def run_pipeline(
    video_path: Path, out_dir: Path, model: str = "small",
    max_final_seconds: float = 180.0, letterings_per_minute: int = 5,
    mogrt_path: str = "", mogrt_text_param: str = "TEXT",
) -> Dict[str, str]:
    video_path = video_path.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    info = ffprobe_video_info(video_path)
    segments = transcribe(video_path, model)
    selected = select_best_segments(segments, max_final_seconds)
    srt_text = build_srt_from_selected(selected)
    letterings = extract_letterings(selected, letterings_per_minute)
    trans_payload = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    selected_payload = [{"start": s.start, "end": s.end, "text": s.text} for s in selected]
    plan = {
        "video_path": str(video_path), "video_file_name": video_path.name,
        "video_info": info, "selected_segments": selected_payload,
        "letterings": letterings,
        "mogrt": {"path": str(Path(mogrt_path).expanduser().resolve()) if mogrt_path else "", "text_param": mogrt_text_param},
    }
    trans_path = out_dir / "transcricao.json"
    plan_path = out_dir / "decupagem.json"
    srt_path = out_dir / "legendas.srt"
    xml_path = out_dir / "timeline.xml"
    jsx_path = out_dir / "montar_premiere.jsx"
    save_json(trans_path, trans_payload)
    save_json(plan_path, plan)
    srt_path.write_text(srt_text, encoding="utf-8")
    xml_path.write_text(generate_xmeml(video_path, selected, info["width"], info["height"]), encoding="utf-8")
    jsx_path.write_text(generate_jsx(
        str(plan_path).replace("\\", "/"), str(video_path).replace("\\", "/"),
        str(Path(mogrt_path).expanduser().resolve()).replace("\\", "/") if mogrt_path else "", mogrt_text_param,
    ), encoding="utf-8")
    return {"transcricao": str(trans_path), "decupagem": str(plan_path), "srt": str(srt_path), "xml": str(xml_path), "jsx": str(jsx_path)}
