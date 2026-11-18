# Image demos

Two pages that score images with d3-ultra on one DGX Spark through SparkScore's `/v1/systemone`
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
    are downscaled in the browser first.

## Run

The SparkScore harness needs an Atlas build with images in `/v1/score` (Enntity/atlas
`sparkscore/score-images`). On the Spark:

    python3 -m tools.sparkscore.server --backend score --backend-url http://127.0.0.1:8081 --backend-model m \
      --tokenizer <d3-ultra-NVFP4 dir> --template d3-ultra --model-id sparkscore-d3 \
      --temperature choice=1.636,noul=2.366,score=1.588 --port 8015

On the machine with the browser:

    ssh -f -N -L 18015:127.0.0.1:8015 <spark>
    python3 demo/photos/fetch.py            # once: 120 COCO val2017 photos, CC BY / CC BY-SA
    python3 demo/serve.py --port 8780 --upstream http://127.0.0.1:18015

Then open http://localhost:8780/camera.html or http://localhost:8780/photos.html.

## Measured (2026-11-18, Spark 04)

- Camera, 640x360 frames, 4 questions: 4.3 frames/s and ~420 ms per answer with 2 frames in flight;
  3.2 frames/s and ~250 ms with 1.
- Photos: 120 COCO photos (~640x480), one question, in 20.1 s (6.0 photos/s).

## Photo credits

The sample photos are COCO val2017 images under CC BY 2.0 or CC BY-SA 2.0. `photos/manifest.json`
lists each photo's license and Flickr source, and each tile's tooltip shows them.
