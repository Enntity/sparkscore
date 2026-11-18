# Image demos

Three pages that score images with d3-ultra on one DGX Spark through SparkScore's `/v1/systemone`
(`images`: base64 data URLs, as d3-ultra's own server takes them).

- `camera.html`: a camera programmed by a sentence. Up to four typed yes/no questions are asked of
  every frame in one request, so every question reads the same image, encoded once. The page shows a
  live probability trace per question and flags a question while it is above 50%.
  - Sources: the camera (Chrome asks for permission; `localhost` counts as a secure origin), the
    sample photos, or a video file.
  - `?source=photos` starts on the sample photos. `?inflight=N` sets the frames in flight (default 2).
- `photos.html`: ask your photos. One yes/no question is asked of every photo in the library, 16
  requests in flight, which the engine merges into shared passes. The wall re-sorts itself by the
  answer as results arrive.
  - Drop a folder of your own photos on the page to replace the library. Photos larger than 1280 px
    are downscaled in the browser first. `?n=N` uses the first N library photos (default 120).
- `firehose.html`: the library arriving, driven by the presenter. `N` (or →) brings in the next photo,
  large, with its answer; it stays up until the next key. `Space` plays at full speed to the next stop
  (125, 250, 500): the engine answers a pass of photos at a time and the page lets them out one by
  one at the measured rate, each large for a moment, then thrown into a bucket of thumbnails sorted by
  the answer. `Space` again pauses. `1`-`6` (or the question chips, or Enter on a typed question)
  re-sort the whole bucket live while the left panel fills with the top 9 matches. Hovering a
  thumbnail shows it large with its answer. `R` resets, `H` hides the key line.
  - `?stops=125,250,500` the play stops, `?q=` the first question.
  - `python3 demo/record.py --take firehose` records a take to `demo/out/take-firehose.mp4` (headless
    Chromium at 1920x1080, the recorder pressing the keys: three photos one at a time, play to 125,
    two re-sorts and a hover, play to 250, two more re-sorts), with video overlays drawn over the page
    while it records (`take_overlay.js`: a slow-motion title, full-speed transitions, captions whose
    numbers the take measures). Start a fresh server first.
  - The footer counts photos read for the first time apart from photos answered from memory. The
    server remembers every photo it has read (the pack store), so a take that should read every photo
    fresh needs a restarted server.

## Run

The SparkScore harness needs an Atlas build with images in `/v1/score` (Enntity/atlas
`sparkscore/score-images`). On the Spark:

    python3 -m tools.sparkscore.server --backend score --backend-url http://127.0.0.1:8081 --backend-model m \
      --tokenizer <d3-ultra-NVFP4 dir> --template d3-ultra --model-id sparkscore-d3 \
      --temperature choice=1.636,noul=2.366,score=1.588 --port 8015

On the machine with the browser:

    ssh -f -N -L 18015:127.0.0.1:8015 <spark>
    python3 demo/photos/fetch.py 500        # once: COCO val2017 photos, CC BY / CC BY-SA
    python3 demo/serve.py --port 8780 --upstream http://127.0.0.1:18015

Then open http://localhost:8780/camera.html, http://localhost:8780/photos.html or
http://localhost:8780/firehose.html. A browser opens six connections a host, so the image pages spread
their calls over `localhost`, `s1.localhost` and `s2.localhost` (all loopback; `serve.py` answers
them with CORS for loopback origins only).

## Measured (2026-11-18, one DGX Spark)

- Camera, 640x360 frames, 4 questions: 4.3 frames/s and ~420 ms per answer with 2 frames in flight;
  3.2 frames/s and ~250 ms with 1.
- Photos: 120 COCO photos (~640x480), one question, in 20.1 s (6.0 photos/s).
- Firehose, fresh server: new photos stream in at ~5-6 photos/s (one every ~160 ms); re-sorting 125 or
  250 photos by a new question at 33-37 photos/s, every photo answered from memory (500 fit the
  20 GB pack store).

## Photo credits

The sample photos are COCO val2017 images under CC BY 2.0 or CC BY-SA 2.0. `photos/manifest.json`
lists each photo's license and Flickr source, and each tile's tooltip shows them.
