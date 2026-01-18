import cgi
import json
import os
import random
import tempfile
import time
import uuid
import threading
from socketserver import ThreadingMixIn
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
import torchaudio
from einops import rearrange

print("Current working directory:", os.getcwd())

from infer_utils import (
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

PORT = 8000
DEFAULT_OUTPUT_DIR = "infer/example/output"
MAX_FRAMES = 6144  # supports up to ~285 seconds

# Only one generation at a time
inference_lock = threading.Lock()

# Current job state (shared across threads - simple & safe for single job)
current_job_state = {
    "active": False,
    "job_id": None,
    "start_time": None,
    "params": None,
    "output_filename": None,
}

# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODELS AT STARTUP
# ──────────────────────────────────────────────────────────────────────────────

print(f"\nLoading models (max_frames={MAX_FRAMES})...")
load_start = time.time()

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'

print(f"Using device: {device}")

cfm, tokenizer, muq, vae = prepare_model(MAX_FRAMES, device)

print(f"Models loaded in {time.time() - load_start:.2f} seconds\n")

# ──────────────────────────────────────────────────────────────────────────────
# CORE INFERENCE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def inference(
    cfm_model,
    vae_model,
    cond,
    text,
    duration,
    style_prompt,
    negative_style_prompt,
    start_time,
    pred_frames,
    batch_infer_num,
    song_duration,
    chunked=False,
):
    with torch.inference_mode():
        latents, _ = cfm_model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            max_duration=duration,
            song_duration=song_duration,
            negative_style_prompt=negative_style_prompt,
            steps=32,
            cfg_strength=4.0,
            start_time=start_time,
            latent_pred_segments=pred_frames,
            batch_infer_num=batch_infer_num
        )

        outputs = []
        print(f"Generated {len(latents)} latent(s)")
        for latent in latents:
            latent = latent.to(torch.float32)
            latent = latent.transpose(1, 2)

            output = decode_audio(latent, vae_model, chunked=chunked)
            output = rearrange(output, "b d n -> d (b n)")
            output = (
                output.to(torch.float32)
                .div(torch.max(torch.abs(output)))
                .clamp(-1, 1)
                .mul(32767)
                .to(torch.int16)
                .cpu()
            )
            outputs.append(output)

        return outputs


# ──────────────────────────────────────────────────────────────────────────────
# BACKGROUND GENERATION TASK
# ──────────────────────────────────────────────────────────────────────────────

def inference_task(params, output_path, job_id):
    global current_job_state
    try:
        current_job_state.update({
            "active": True,
            "job_id": job_id,
            "start_time": time.time(),
            "params": params,
            "output_filename": os.path.basename(output_path),
        })

        audio_length = params['audio_length']

        lrc = params['lrc']
        ref_prompt = params.get('ref_prompt')
        ref_audio_path = params.get('ref_audio_path')
        edit = params['edit']
        ref_song = params.get('ref_song')
        edit_segments = params.get('edit_segments')
        chunked = params['chunked']
        batch_infer_num = params['batch_infer_num']

        lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
            MAX_FRAMES, lrc, tokenizer, audio_length, device
        )

        style_prompt = (
            get_style_prompt(muq, ref_audio_path)
            if ref_audio_path else
            get_style_prompt(muq, prompt=ref_prompt)
        )

        negative_style_prompt = get_negative_style_prompt(device)

        latent_prompt, pred_frames = get_reference_latent(
            device, MAX_FRAMES, edit, edit_segments, ref_song, vae
        )

        print("Starting inference...")
        s_t = time.time()

        generated_songs = inference(
            cfm_model=cfm,
            vae_model=vae,
            cond=latent_prompt,
            text=lrc_prompt,
            duration=end_frame,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            start_time=start_time,
            pred_frames=pred_frames,
            chunked=chunked,
            batch_infer_num=batch_infer_num,
            song_duration=song_duration
        )

        print(f"Inference completed in {time.time() - s_t:.2f} seconds")

        generated_song = random.sample(generated_songs, 1)[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, generated_song, sample_rate=44100)
        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Inference error: {e}")
    finally:
        if params.get('ref_audio_path') and os.path.exists(params['ref_audio_path']):
            try:
                os.remove(params['ref_audio_path'])
            except:
                pass

        # Reset state
        current_job_state.update({
            "active": False,
            "job_id": None,
            "start_time": None,
            "params": None,
            "output_filename": None,
        })

        inference_lock.release()


# ──────────────────────────────────────────────────────────────────────────────
# HTTP HANDLER
# ──────────────────────────────────────────────────────────────────────────────

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class InferenceHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path != '/generate':
            self.send_error(404, 'Not Found')
            return

        if not inference_lock.acquire(blocking=False):
            self.send_response(503)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "Server is busy with another generation"
            }).encode('utf-8'))
            return

        try:
            ctype, _ = cgi.parse_header(self.headers.get('Content-Type', ''))
            if ctype != 'multipart/form-data':
                raise ValueError("Content-Type must be multipart/form-data")

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST',
                         'CONTENT_TYPE': self.headers['Content-Type']}
            )

            params = {
                'lrc': form.getvalue('lrc', ''),
                'ref_prompt': form.getvalue('ref_prompt'),
                'chunked': form.getvalue('chunked', 'false').lower() == 'true',
                'audio_length': int(form.getvalue('audio_length', '95')),
                'output_dir': form.getvalue('output_dir', DEFAULT_OUTPUT_DIR),
                'edit': form.getvalue('edit', 'false').lower() == 'true',
                'ref_song': form.getvalue('ref_song'),
                'edit_segments': form.getvalue('edit_segments'),
                'batch_infer_num': int(form.getvalue('batch_infer_num', '1')),
            }

            ref_audio_path = None
            if 'ref_audio' in form and form['ref_audio'].filename:
                data = form['ref_audio'].file.read()
                fd, path = tempfile.mkstemp(suffix='.wav')
                with os.fdopen(fd, 'wb') as f:
                    f.write(data)
                params['ref_audio_path'] = path

            if not (params['ref_prompt'] or params.get('ref_audio_path')):
                raise ValueError("ref_prompt or ref_audio required")
            if params['ref_prompt'] and params.get('ref_audio_path'):
                raise ValueError("Only one of ref_prompt or ref_audio allowed")
            if params['edit'] and not (params.get('ref_song') and params.get('edit_segments')):
                raise ValueError("edit mode requires ref_song and edit_segments")

            job_id = uuid.uuid4().hex
            file_id = uuid.uuid4().hex
            filename = f"{file_id}.wav"
            output_path = os.path.join(params['output_dir'], filename)
            url = f"/output/{filename}"

            self.send_response(202)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "accepted",
                "url": url,
                "job_id": job_id,
                "message": "Generation started"
            }).encode('utf-8'))

            threading.Thread(
                target=inference_task,
                args=(params, output_path, job_id),
                daemon=True
            ).start()

        except Exception as e:
            inference_lock.release()
            self.send_error(400, f"Bad Request: {str(e)}")

    def do_GET(self):
        if self.path == '/status':
            self._send_status()
        elif self.path == '/files':
            self._send_files_list()
        elif self.path.startswith('/output/'):
            self._send_output_file()
        else:
            self.send_error(404, 'Not Found')

    def _send_status(self):
        global current_job_state
        if current_job_state["active"]:
            elapsed = time.time() - current_job_state["start_time"]
            data = {
                "status": "processing",
                "job_id": current_job_state["job_id"],
                "elapsed_seconds": round(elapsed, 1),
                "filename": current_job_state["output_filename"],
                "params_summary": {
                    "audio_length": current_job_state["params"].get("audio_length"),
                    "edit": current_job_state["params"].get("edit", False),
                    "batch": current_job_state["params"].get("batch_infer_num", 1),
                }
            }
        else:
            data = {
                "status": "idle",
                "message": "No active generation"
            }

        json_data = json.dumps(data, indent=2)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(json_data.encode('utf-8'))

    def _send_files_list(self):
        try:
            wav_files = [
                f for f in os.listdir(DEFAULT_OUTPUT_DIR)
                if f.lower().endswith('.wav')
            ]
            wav_files.sort(key=lambda x: os.path.getmtime(
                os.path.join(DEFAULT_OUTPUT_DIR, x)), reverse=True)

            data = {
                "available_files": wav_files,
                "count": len(wav_files),
                "download_prefix": "/output/",
                "current_job": None
            }

            if current_job_state["active"]:
                elapsed = time.time() - current_job_state["start_time"]
                data["current_job"] = {
                    "status": "processing",
                    "job_id": current_job_state["job_id"],
                    "filename": current_job_state["output_filename"],
                    "elapsed_seconds": round(elapsed, 1)
                }

            json_data = json.dumps(data, indent=2)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json_data.encode('utf-8'))

        except Exception as e:
            self.send_error(500, f"Error listing files: {str(e)}")

    def _send_output_file(self):
        filename = self.path[len('/output/'):]
        full_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)

        if os.path.isfile(full_path):
            self.send_response(200)
            self.send_header('Content-Type', 'audio/wav')
            self.send_header('Content-Length', str(os.path.getsize(full_path)))
            self.send_header('Cache-Control', 'public, max-age=86400')
            self.end_headers()
            with open(full_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            # Only return status if this is the file currently being generated
            if (current_job_state["active"] and
                current_job_state["output_filename"] == filename):
                elapsed = time.time() - current_job_state["start_time"]
                data = {
                    "status": "processing",
                    "job_id": current_job_state["job_id"],
                    "elapsed_seconds": round(elapsed, 1),
                    "message": "File is being generated"
                }
                json_data = json.dumps(data)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json_data.encode('utf-8'))
            else:
                self.send_error(404, "File not found")


if __name__ == "__main__":
    server_address = ('', PORT)
    httpd = ThreadedHTTPServer(server_address, InferenceHandler)
    print(f"Server running on http://localhost:{PORT}")
    print("Endpoints: /generate (POST), /status, /files, /output/<filename>.wav")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        httpd.server_close()

