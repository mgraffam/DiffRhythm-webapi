# Copyright (c) 2025 ASLP-LAB
#               2025 Huakang Chen  (huakang@mail.nwpu.edu.cn)
#               2025 Guobin Ma     (guobin.ma@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

# Choose the model size you want to support
# 2048  → exactly 95 seconds
# 6144  → up to ~285 seconds (recommended for flexibility)
MAX_FRAMES = 6144

# Only one generation runs at a time
inference_lock = threading.Lock()

# Track current job for status reporting
current_job = {
    "active": False,
    "start_time": None,
    "params": None,
    "output_path": None,
    "job_id": None,
    "estimated_duration": None,  # Rough estimate: e.g., 30s base + 1s per output second
}

# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODELS AT STARTUP (eager loading)
# ──────────────────────────────────────────────────────────────────────────────

print(f"\nStarting model loading (max_frames={MAX_FRAMES})...")
load_start = time.time()

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'

print(f"Using device: {device}")

cfm, tokenizer, muq, vae = prepare_model(MAX_FRAMES, device)

print(f"Models loaded successfully in {time.time() - load_start:.2f} seconds\n")

# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE FUNCTION (same as original)
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
            latent = latent.transpose(1, 2)  # [b d t]

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
# BACKGROUND INFERENCE TASK
# ──────────────────────────────────────────────────────────────────────────────

def inference_task(params, output_path, job_id):
    """Runs in background thread - long running generation"""
    global current_job
    try:
        # Update current job status
        current_job["active"] = True
        current_job["start_time"] = time.time()
        current_job["params"] = params
        current_job["output_path"] = output_path
        current_job["job_id"] = job_id
        # Rough estimate: base 20s + 0.5s per output second * batch
        current_job["estimated_duration"] = 20 + (params["audio_length"] * 0.5 * params["batch_infer_num"])

        audio_length = params['audio_length']

        # Optional warning if requested length exceeds loaded model capability
        if audio_length > 95 and MAX_FRAMES == 2048:
            print("Warning: Requested length > 95s but server was loaded with small model (2048 frames)")

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

        if ref_audio_path:
            style_prompt = get_style_prompt(muq, ref_audio_path)
        else:
            style_prompt = get_style_prompt(muq, prompt=ref_prompt)

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

        e_t = time.time() - s_t
        print(f"Inference completed in {e_t:.2f} seconds")

        generated_song = random.sample(generated_songs, 1)[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, generated_song, sample_rate=44100)
        print(f"File saved successfully: {output_path}")

    except Exception as e:
        print(f"Error during inference: {e}")
    finally:
        # Cleanup temporary reference audio file if it exists
        if params.get('ref_audio_path') and os.path.exists(params['ref_audio_path']):
            try:
                os.remove(params['ref_audio_path'])
            except:
                pass

        # Reset current job
        current_job["active"] = False
        current_job["start_time"] = None
        current_job["params"] = None
        current_job["output_path"] = None
        current_job["job_id"] = None
        current_job["estimated_duration"] = None

        # Very important: release the global lock
        inference_lock.release()


# ──────────────────────────────────────────────────────────────────────────────
# HTTP SERVER
# ──────────────────────────────────────────────────────────────────────────────

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded server - handles concurrent GET requests"""
    daemon_threads = True


class InferenceHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path != '/generate':
            self.send_error(404, 'Not Found')
            return

        # Try to acquire lock (non-blocking)
        if not inference_lock.acquire(blocking=False):
            self.send_response(503)  # Service Unavailable
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "Server is busy processing another generation. Please try again later."
            }).encode('utf-8'))
            return

        try:
            ctype, pdict = cgi.parse_header(self.headers.get('Content-Type', ''))
            if ctype != 'multipart/form-data':
                raise ValueError("Content-Type must be multipart/form-data")

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST',
                         'CONTENT_TYPE': self.headers['Content-Type']}
            )

            # Extract all parameters
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

            # Handle optional uploaded reference audio
            ref_audio_path = None
            if 'ref_audio' in form and form['ref_audio'].filename:
                ref_audio_data = form['ref_audio'].file.read()
                fd, ref_audio_path = tempfile.mkstemp(suffix='.wav')
                with os.fdopen(fd, 'wb') as f:
                    f.write(ref_audio_data)
                params['ref_audio_path'] = ref_audio_path

            # Basic validation
            if not (params['ref_prompt'] or params.get('ref_audio_path')):
                raise ValueError("Either 'ref_prompt' or 'ref_audio' file is required")
            if params['ref_prompt'] and params.get('ref_audio_path'):
                raise ValueError("Provide only one of 'ref_prompt' or 'ref_audio'")
            if params['edit'] and not (params.get('ref_song') and params.get('edit_segments')):
                raise ValueError("'ref_song' and 'edit_segments' are required in edit mode")

            # Create unique output filename and job ID
            job_id = uuid.uuid4().hex
            file_id = uuid.uuid4().hex
            output_filename = f"{file_id}.wav"
            output_path = os.path.join(params['output_dir'], output_filename)
            url = f"/output/{output_filename}"

            # Immediate response: generation accepted
            self.send_response(202)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "accepted",
                "url": url,
                "job_id": job_id,
                "message": "Generation started. Check the URL later for the result."
            }).encode('utf-8'))

            # Start background generation
            thread = threading.Thread(
                target=inference_task,
                args=(params, output_path, job_id),
                daemon=True
            )
            thread.start()

        except Exception as e:
            inference_lock.release()
            self.send_error(400, f"Bad Request: {str(e)}")

    def _get_status_json(self, full_path, filename):
        """Helper to get status JSON if file not ready but job matches"""
        global current_job
        if current_job["active"] and current_job["output_path"] == full_path:
            elapsed = time.time() - current_job["start_time"]
            estimated = current_job["estimated_duration"]
            progress = min(100, int((elapsed / estimated) * 100) if estimated else 0)
            return {
                "status": "processing",
                "job_id": current_job["job_id"],
                "file": filename,
                "elapsed_time": f"{elapsed:.1f} seconds",
                "estimated_remaining": f"{max(0, estimated - elapsed):.1f} seconds" if estimated else "unknown",
                "progress_percent": progress,
                "parameters": current_job["params"],
                "server_busy": True,
                "other_goodies": {
                    "device": device,
                    "max_frames": MAX_FRAMES,
                    "current_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                    "uptime": f"{time.time() - load_start:.1f} seconds"
                }
            }
        else:
            return {
                "status": "not_found",
                "message": "No active job for this file or file does not exist."
            }

    def do_GET(self):
        if self.path.startswith('/output/'):
            # Note: we use the default output dir here
            # If you allow custom output_dirs per request, you'd need a mapping system
            filename = self.path[len('/output/'):]
            full_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)

            if os.path.isfile(full_path):
                # File ready - serve audio
                file_size = os.path.getsize(full_path)
                self.send_response(200)
                self.send_header('Content-Type', 'audio/wav')
                self.send_header('Content-Length', str(file_size))
                self.send_header('Cache-Control', 'public, max-age=86400')
                self.end_headers()
                with open(full_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                # Not ready - return status JSON
                status_data = self._get_status_json(full_path, filename)
                status_json = json.dumps(status_data)
                self.send_response(200 if status_data["status"] == "processing" else 404)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(status_json)))
                self.end_headers()
                self.wfile.write(status_json.encode('utf-8'))
        else:
            self.send_error(404, 'Not Found')

    def do_HEAD(self):
        if self.path.startswith('/output/'):
            filename = self.path[len('/output/'):]
            full_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)

            if os.path.isfile(full_path):
                # File ready - HEAD for audio
                file_size = os.path.getsize(full_path)
                self.send_response(200)
                self.send_header('Content-Type', 'audio/wav')
                self.send_header('Content-Length', str(file_size))
                self.send_header('Cache-Control', 'public, max-age=86400')
                self.end_headers()
            else:
                # Not ready - HEAD for status JSON
                status_data = self._get_status_json(full_path, filename)
                status_json = json.dumps(status_data)
                self.send_response(200 if status_data["status"] == "processing" else 404)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(status_json)))
                self.end_headers()
        else:
            self.send_error(404, 'Not Found')


if __name__ == "__main__":
    server_address = ('', PORT)
    httpd = ThreadedHTTPServer(server_address, InferenceHandler)
    print(f"Starting threaded HTTP server on http://localhost:{PORT} ...")
    print("Only one generation allowed at a time.")
    print("Models are already loaded and ready.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()
